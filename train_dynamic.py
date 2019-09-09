import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models
from utils.occlusion import occlusion

model_names = sorted(name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--device', type=str, default='0', help='GPU device (default: 0)')
parser.add_argument('--dataset', default='CHASE', choices=['DRIVE', 'CHASE'])
parser.add_argument('--data_path', type=str, default='datasets/', help='data path')
parser.add_argument('--model', type=str, default='FCN', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: FCN)')
parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', action='store_true', default=False, help='learning rate decay')
parser.add_argument('--threshold_confusion', default=0.5, type=float, help='threshold_confusion')
parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')

parser.add_argument('--patch_num', type=int, default=204800, help='patchs number (default: 800000)')
parser.add_argument('--patch_size', type=int, default=48, help='patch size (default: 48)')

parser.add_argument('--data_augmentation', action='store_true', default=False, help='data augmentation')
parser.add_argument('--occlusion', action='store_true', default=False, help='is add occlusion?')
parser.add_argument('--occ_p', default=0.5, type=float, help='occlusion prob')
parser.add_argument('--occ_length', type=int, default=24, help='length of the occlusion')
parser.add_argument('--occ_func', default='fill_next_tar', choices=['fill_0', 'fill_R', 'fill_next'], help='occ_func')

# use last save model
parser.add_argument('--load_last', action='store_true', default=False, help='load last model')
parser.add_argument('--load_path', type=str, default='logs/', help='load model path')
parser.add_argument('--logs_path', type=str, default='logs/', help='load model path')

args = vars(parser.parse_args())
os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
cudnn.benchmark = True
torch.cuda.is_available()
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

threshold_confusion = args['threshold_confusion']

if str(args['logs_path']).endswith('/') is False:
    args['logs_path'] += '/'

if args['load_path'] is not None and str(args['load_path']).endswith('/') is False:
    args['load_path'] += '/'

if args['load_last'] is False:
    mkdir_p(args['logs_path'] + args['dataset'] + '/' + args['model'] + '/')
    index = np.sort(np.array(os.listdir(args['logs_path'] + args['dataset'] + '/' + args['model'] + '/'), dtype=int))
    index = index.max() + 1 if len(index) > 0 else 1
    basic_path = args['logs_path'] + args['dataset'] + '/' + args['model'] + '/' + str(index) + '/'
    mkdir_p(basic_path)
    args['load_path'] = basic_path
    max_acc, max_F1_score, max_sensitivity = 0., 0., 0.
    cur_epoch = 0
    logs = []
    logs.append(['epoch', 'test_acc', 'max_acc', 'specificity', 'sensitivity', 'max_sensitivity', 'F1_score', 'max_F1_score'])
else:
    basic_path = args['load_path']
    assert os.path.exists(basic_path), '目录不存在'
    assert os.path.isfile(basic_path + 'checkpoints/last.pt'), 'Error: no checkpoint file found!'
    checkpoint = torch.load(basic_path + 'checkpoints/last.pt')
    checkpoint['args']['load_last'] = args['load_last']
    checkpoint['args']['load_path'] = args['load_path']
    args = checkpoint['args']
    max_acc = checkpoint['max_acc']
    max_sensitivity = checkpoint['max_sensitivity']
    max_F1_score = checkpoint['max_F1_score']
    cur_epoch = checkpoint['epoch'] + 1
    logs = checkpoint['logs']
    print('保存模型的最后一次训练结果： %s, 当前训练周期: %4d, ' % (str(logs[-1]), cur_epoch))
    assert cur_epoch < args['epochs'], '已经跑完了，cur_epoch: {}，epochs: {}'.format(cur_epoch, args['epochs'])

print('当前日志目录： ' + basic_path)
mkdir_p(basic_path + 'checkpoints/periods/')
mkdir_p(basic_path + 'tensorboard/')
print(args)
with open(basic_path + 'args.txt', 'w+') as f:
    for arg in args:
        f.write(str(arg) + ': ' + str(args[arg]) + '\n')

vis = get_visdom()
if vis is not None:
    import time

    vis.env = args['dataset'] + '_' + args['model'] + '_' + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))


net = models.__dict__[args['model']]().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=args['lr'])

if args['load_last'] is True and cur_epoch > 0:
    net.load_state_dict(checkpoint['net'], strict=False)
    print('load path: ' + basic_path + 'checkpoints/last.pt')

# 加载数据集
train_orig_imgs, train_orig_gts, train_orig_masks, test_orig_imgs, test_orig_gts, test_orig_masks = get_orig_datasets(args)
test_imgs, test_gts, test_imgs_patches, test_masks_patches = get_testing_patchs(
    test_imgs=test_orig_imgs,
    test_gts=test_orig_gts,
    patch_size=args['patch_size'],
)
test_set = TestDataset(test_imgs_patches)
test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=0)

ts_writer = SummaryWriter(log_dir=basic_path + 'tensorboard/', comment=args['model'])
args_str = ''
for arg in args:
    args_str += str(arg) + ': ' + str(args[arg]) + '<br />'
ts_writer.add_text('args', args_str, cur_epoch)

if args['lr_decay'] is True:
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)

org_images, org_targets, occ_images, occ_targets, vis_mask, vis_outputs = None, None, None, None, None, None


def train():
    global org_images, org_targets, occ_images, occ_targets, vis_mask, vis_outputs, max_acc, max_F1_score, max_sensitivity
    for epoch in range(cur_epoch, args['epochs']):
        if args['lr_decay'] is True:
            scheduler.step(max_F1_score)
        # train network
        train_loss = 0
        train_imgs_patches, train_masks_patches = get_training_patchs(
            train_imgs=train_orig_imgs,
            train_gts=train_orig_gts,
            patch_size=args['patch_size'],
            patch_num=args['patch_num']
        )
        train_set = TrainDataset(train_imgs_patches, train_masks_patches, data_augmentation=args['data_augmentation'])
        train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=0)
        progress_bar = tqdm(train_loader)
        net.train()
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            progress_bar.set_description('Epoch {}-{}'.format(epoch + 1, args['epochs']))

            if vis is not None and batch_idx % 5 == 0:
                org_images = vis.image(make_grid(inputs.data[0:64], nrow=32, normalize=True, scale_each=True, padding=4, pad_value=1),
                                       opts=dict(title='Original Images'),
                                       win=org_images)
                org_targets = vis.image(
                    make_grid(targets[0:64].type_as(torch.FloatTensor()).view(64, 1, args['patch_size'], args['patch_size']), nrow=32,
                              normalize=True,
                              scale_each=True, padding=4, pad_value=1),
                    opts=dict(title='Original Targets'),
                    win=org_targets)

            if args['occlusion'] is True:
                inputs, targets = occlusion(inputs, targets, args['occ_length'], args['occ_func'], args['occ_p'])
                if vis is not None and batch_idx % 5 == 0:
                    occ_images = vis.image(make_grid(inputs.data[0:64], nrow=32, normalize=True, scale_each=True, padding=4, pad_value=1),
                                           opts=dict(title='Occlusion Images'),
                                           win=occ_images)
                    occ_targets = vis.image(
                        make_grid(targets[0:64].type_as(torch.FloatTensor()).view(64, 1, args['patch_size'], args['patch_size']), nrow=32,
                                  padding=4,
                                  pad_value=1),
                        opts=dict(title='Occlusion Targets'),
                        win=occ_targets)

            inputs = Variable(inputs.cuda().detach())
            targets = Variable(targets.cuda().detach())

            optimizer.zero_grad()
            
            output = net(inputs)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss='%.3f' % (train_loss / (batch_idx + 1)))


if __name__ == '__main__':
    train()
    ts_writer.close()

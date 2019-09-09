import argparse
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, f1_score,auc

import torch.backends.cudnn as cudnn

from utils.misc import *
import models

model_names = sorted(name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--device', type=str, default='0', help='GPU device (default: 0)')
parser.add_argument('--stride_size', type=int, default=5, help='stride size (default: 5)')
parser.add_argument('--batch_size', type=int, default=512, help='stride size (default: 1024)')
parser.add_argument('--threshold_confusion', default=0.49, type=float, help='threshold_confusion')
parser.add_argument('--check_path', type=str, default='/home/izhangh/work/python/N2UNet/logs/STARE/N2UNet_v5/2/checkpoints/last.pt',
                    help='load model path')

args = vars(parser.parse_args())
os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
checkpoint = torch.load(args['check_path'])
checkpoint['args']['device'] = args['device']
checkpoint['args']['check_path'] = args['check_path']
checkpoint['args']['stride_size'] = args['stride_size']
checkpoint['args']['threshold_confusion'] = args['threshold_confusion']
checkpoint['args']['batch_size'] = args['batch_size']
checkpoint['args']['data_path'] = 'datasets/'
args = checkpoint['args']
print(args)
max_acc = checkpoint['max_acc']
max_sensitivity = checkpoint['max_sensitivity']
max_F1_score = checkpoint['max_F1_score']
cur_epoch = checkpoint['epoch'] + 1
logs = checkpoint['logs']
threshold_confusion = args['threshold_confusion']
print(logs[-1])

path_arr = args['check_path'].split('/')
basic_path = args['check_path'][:-len(path_arr[-1])] + path_arr[-1].split('.')[0]
basic_path = basic_path + '_result_average/'

cudnn.benchmark = True
torch.cuda.is_available()
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
os.environ['CUDA_VISIBLE_DEVICES'] = args['device']

if args['model'] == 'N2UNet_v5':
    args['model'] = 'M3FCN'
net = models.__dict__[args['model']]()
net.eval().cuda()
net.load_state_dict(checkpoint['net'], strict=False)

data_path = args['data_path'] + args['dataset'] + '/'
test_imgs_original = load_hdf5(data_path + 'imgs_test.hdf5')
test_imgs = preprocessing(test_imgs_original)
test_gts = load_hdf5(data_path + 'ground_truth_test.hdf5')
test_gts = test_gts / 255.
test_masks = load_hdf5(data_path + 'border_masks_test.hdf5')

patch_size = args['patch_size']
stride_size = args['stride_size']
pred_imgs = []
for t in range(len(test_imgs)):
    test_img = test_imgs[t].reshape((1, test_imgs.shape[1], test_imgs.shape[2], test_imgs.shape[3]))
    test_img = paint_border_overlap(test_img, patch_size, stride_size)
    img_h = test_img.shape[2]
    img_w = test_img.shape[3]
    
    preds = []
    patches = []
    for i in range(test_img.shape[0]):
        for h in range((img_h - patch_size) // stride_size + 1):
            for w in range((img_w - patch_size) // stride_size + 1):
                patch = test_img[i, :, h * stride_size:(h * stride_size) + patch_size, w * stride_size:(w * stride_size) + patch_size]
                patches.append(patch)
                if len(patches) == args['batch_size']:
                    test_set = TestDataset(np.asarray(patches))
                    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=0)

                    for batch_idx, inputs in enumerate(test_loader):
                        inputs = inputs.cuda()
                        outputs = net(inputs)
                        outputs = torch.nn.functional.softmax(outputs, dim=1)
                        outputs = outputs.permute(0, 2, 3, 1)
                        outputs = outputs.view(-1, outputs.shape[1] * outputs.shape[2], 2)
                        outputs = outputs.data.cpu().numpy()
                        preds.append(outputs)

                    patches = []
    
    if len(patches) > 0:
        test_set = TestDataset(np.asarray(patches))
        test_loader = DataLoader(test_set, batch_size=len(patches), shuffle=False, num_workers=0)

        for batch_idx, inputs in enumerate(test_loader):
            inputs = inputs.cuda()
            outputs = net(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            outputs = outputs.permute(0, 2, 3, 1)
            outputs = outputs.view(-1, outputs.shape[1] * outputs.shape[2], 2)
            outputs = outputs.data.cpu().numpy()
            preds.append(outputs)
    
    print(np.asarray(preds).shape)
    preds = np.concatenate(preds, axis=0)
    print(preds.shape)
    pred_patches = pred_to_imgs(preds, args['patch_size'])
    pred_img = recompone_overlap(pred_patches, img_h, img_w, stride_size)
    pred_imgs.append(pred_img[:, :, 0:test_imgs.shape[2], 0:test_imgs.shape[3]][0])

pred_imgs = np.array(pred_imgs)
np.save(basic_path + 'pred_imgs.npy', pred_imgs)

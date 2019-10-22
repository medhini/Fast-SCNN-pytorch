import os
import argparse
import torch
import torch.utils.data as data
import timeit

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete

from train import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        # output folder
        self.outdir = '%s_result'%args.split
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        # val_dataset = get_segmentation_dataset(args.dataset, split=args.split, mode='testval',
        #                                        transform=input_transform)
        val_dataset = get_segmentation_dataset(args.dataset, root='./datasets/'+args.dataset, split=args.split,
                                               is_transform=True, img_size=768)                                       
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)
        # create network
        self.model = get_fast_scnn(args.dataset, aux=args.aux, pretrained=True, root=args.save_folder).to(args.device)
        print('Finished loading model!')

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.model.eval()
        mean_miou = 0.0
        mean_pixelAcc = 0.0

        start_time = timeit.default_timer()

        for i, (image, label) in enumerate(self.val_loader):
            image = image.to(self.args.device)

            outputs = self.model(image)

            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            label = label.numpy()

            self.metric.update(pred, label)
            pixAcc, mIoU = self.metric.get()
            print('Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (i + 1, pixAcc * 100, mIoU * 100))
            mean_miou += mIoU
            mean_pixelAcc += pixAcc

            predict = pred.squeeze(0)
            mask = get_color_pallete(predict, self.args.dataset)
            mask.save(os.path.join(self.outdir, 'seg_{}.png'.format(i)))

        end_time = timeit.default_timer()

        print('Time taken:%0.4f s'%(end_time-start_time))
        print('Mean mIoU: %.4f%%, Mean PixelAcc: %.4f%%'%( mean_miou/len(self.val_loader), mean_pixelAcc/len(self.val_loader)))


if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(args)
    print('Testing model: ', args.model)
    evaluator.eval()

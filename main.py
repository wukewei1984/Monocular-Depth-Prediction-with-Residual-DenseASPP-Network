import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import os
import shutil
import time
import sys
import csv
import imgs
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import PIL.Image as im
#from make3d_dataloader import NYUDataset                 #make 3d
from nyu_dataloader import NYUDataset                              #nyu
#from models import Decoder, ResNet
#import  tiramisu
from metrics import AverageMeter, Result
import criteria
import utils
#import training as train_utils
from net.deeplabv3plus import deeplabv3plus
from config import cfg
model_names = ['resnet18', 'resnet50']
loss_names = ['l1', 'l2']
data_names = ['NYUDataset']
#decoder_names = Decoder.names
modality_names = NYUDataset.modality_names

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Sparse-to-Dense Training')
# parser.add_argument('--data', metavar='DIR', help='path to dataset',
#                     default="data/NYUDataset")
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--data', metavar='DATA', default='/media/image/5C808D83808D647C/MHL_Code/Depth/sparse-to-dense/data_nyu/nyudepthv2',
                    choices=data_names,
                    help='dataset: ' +
                        ' | '.join(data_names) +
                        ' (default: nyudepthv2)')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                    choices=modality_names,
                    help='modality: ' +
                        ' | '.join(modality_names) +
                        ' (default: rgb)')
parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                    help='number of sparse depth samples (default: 0)')
'''
parser.add_argument('--decoder', '-d', metavar='DECODER', default='deconv2',
                    choices=decoder_names,
                    help='decoder: ' +
                        ' | '.join(decoder_names) +
                        ' (default: deconv2)')
'''
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='l2', 
                    choices=loss_names,
                    help='loss function: ' +
                        ' | '.join(loss_names) +
                        ' (default: l1)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=True, help='use ImageNet pre-trained weights (default: True)')
parser.add_argument('--add_delta', default=0, type=int, metavar='N',
                    help='train the delta of the image:1-turn on,0-turn off')

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae', 
                'delta1', 'delta2', 'delta3', 
                'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()

def main():
    global args, best_result, output_directory, train_csv, test_csv
    args = parser.parse_args()
    args.data = os.path.join('/media/a2/C0B09857B098562E/深度估计-mhl/data', args.data)
    if args.modality == 'rgb' and args.num_samples != 0:
        print("number of samples is forced to be 0 when input modality is rgb")
        args.num_samples = 0
    # create results folder, if not already exists
    output_directory = os.path.join('results1',
        'NYUDataset.modality={}.nsample={}.arch={}.criterion={}.lr={}.bs={}'.
        format(args.modality, args.num_samples, args.arch,args.criterion, args.lr, args.batch_size))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')
    
    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
        out_channels = 1
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()
        out_channels = 1
    elif args.criterion == 'BerHu':
        criterion = criteria.BerHu().cuda()
        out_channels = 1

    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_dataset = NYUDataset(traindir, type='train', 
        modality=args.modality, num_samples=args.num_samples)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    # set batch size to be 1 for validation
    val_dataset = NYUDataset(valdir, type='val', 
        modality=args.modality, num_samples=args.num_samples)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    print("=> data loaders created.")

    # evaluation mode
    if args.evaluate:
        best_model_filename = os.path.join(output_directory, 'model_best.pth.tar')
        if os.path.isfile(best_model_filename):
            print("=> loading best model '{}'".format(best_model_filename))
            checkpoint = torch.load(best_model_filename)
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no best model found at '{}'".format(best_model_filename))
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return

    # optionally resume from a checkpoint
    elif args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']+1
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create new model
    
    else:
        """
        # define model
        print("=> creating Model ({}) ...".format(args.arch))
        in_channels = len(args.modality)
        if args.arch == 'resnet50':
            model = ResNet(layers=50, decoder=args.decoder, in_channels=in_channels,
                out_channels=out_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet18':
            model = ResNet(layers=18, decoder=args.decoder, in_channels=in_channels,
                out_channels=out_channels, pretrained=args.pretrained)
        print("=> model created.")
        """
       
        
        # create new csv files with only header
        with open(train_csv, 'w') as csvfile:   
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:   
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()


    #model = model.cuda()
    #model = tiramisu.FCDenseNet103(n_classes=1,decoder=args.decoder)
    model = deeplabv3plus(cfg)
    model = torch.nn.DataParallel(model).cuda()
    #model.apply(train_utils.weights_init)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))  
    print(model)
    print("=> model transferred to GPU.")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        result, img_merge = validate(val_loader, model, epoch)

##############################
        save_checkpoint1({
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer' : optimizer,
        },epoch)



        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                    format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)
       
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer' : optimizer,
        }, is_best, epoch)

def get_delta_y(target):
    target_size=target.size()
    heigh=target_size[2]
    max_depth=torch.max(target)
    for i in range(heigh-1):
        target[:,:,i,:]=(target[:,:,i,:]-target[:,:,i+1,:]+max_depth)/2
    target[:,:,heigh-1,:]=0
    #print(target)
    return target

def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    loss_txt = os.path.join(output_directory, 'loss.txt')
    args=parser.parse_args()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        if(args.add_delta==1):
            target=get_delta_y(target)
        input, target = input.cuda(), target.cuda()
        #print (input.shape)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute depth_pred
        end = time.time()
        depth_pred = model(input_var)
        #print(torch.argmax(depth_pred,3))
        #print(depth_pred.shape)
        loss = criterion(depth_pred, target_var)
        with open(loss_txt, 'a') as txtfile:
            txtfile.write("loss={:.3f}\r\n".format(loss))
        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        output1 = torch.index_select(depth_pred.data, 1, torch.cuda.LongTensor([0]))
        result.evaluate(output1, target)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  'loss={loss:.3f} '
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f}) '
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,loss=loss ,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3, 
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if (args.add_delta == 1):
            target = get_delta_y(target)
        input, target = input.cuda(), target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        depth_pred = model(input_var)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        output1 = torch.index_select(depth_pred.data, 1, torch.cuda.LongTensor([0]))
        result.evaluate(output1, target)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = input
            elif args.modality == 'rgbd':
                rgb = input[:,:3,:,:]

            if i == 0:
                img_merge = utils.merge_into_row(rgb, target, depth_pred)
            elif (i < 8*skip) and (i % skip == 0):
                row = utils.merge_into_row(rgb, target, depth_pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename = output_directory + '/comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3, 
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})

    return avg, img_merge

def save_checkpoint1(state, epoch):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
    torch.save(state, checkpoint_filename)

def save_checkpoint(state, is_best, epoch):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()

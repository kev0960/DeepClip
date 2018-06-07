import os
import argparse
import numpy as np
import cv2
from PIL import Image

import torch.multiprocessing as mp

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.model_zoo as model_zoo
from torchvision import transforms as trn
from tqdm import tqdm
from random import shuffle

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

parser = argparse.ArgumentParser(description="Train the bumbumnet")
parser.add_argument('--total_epoch', type=int, default=10)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--num_train_file', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_val_file', type=int, default=50)
parser.add_argument('--is_test', type=bool, default=False)
parser.add_argument('--test_video_name', type=str, default="")
parser.add_argument('--multi_thread', type=bool, default=False)
parser.add_argument('--cache', type=int, default=1000)
parser.add_argument('--allow_test_multiprocess', type=bool, default=False)
args = parser.parse_args()

cache_size = args.cache

global_video_tensor_cache = {}


class VideoDataLoader:
    def __init__(self, keywords, num_keywords=-1, batch_size=5, num_frames=5, num_files=50, num_file_offset=0,
                 is_training=False, is_test=False, test_file_name="", total_epoch=10,
                 is_multithread=False):
        self.current_epoch = 0
        self.current_iter = 0
        self.prepared_batch_iter = 0
        self.total_epoch = total_epoch
        self.is_multithread = is_multithread

        self.batch_size = batch_size
        self.num_frames = num_frames
        self.keywords = keywords
        self.num_files = num_files
        self.num_file_offset = num_file_offset
        self.is_training = is_training
        self.tf = self.transform_frame()
        self.is_test = is_test

        self.dirs_to_check = os.listdir('./training')
        if len(self.keywords) != 0:
            self.dirs_to_check = [dir for dir in self.dirs_to_check if dir in self.keywords]
        elif num_keywords != -1:
            self.dirs_to_check = self.dirs_to_check[:num_keywords]

        self.num_class = len(self.dirs_to_check)

        self.entire_file_list = []

        if not self.is_test:
            dir_index = 0
            for dir in self.dirs_to_check:
                file_names = os.listdir('./training/' + dir)
                for file_name in file_names[self.num_file_offset:self.num_file_offset + self.num_files]:
                    self.entire_file_list.append(('./training/' + dir + '/' + file_name, dir_index))
                dir_index += 1

            if self.is_training:
                shuffle(self.entire_file_list)
        else:
            self.entire_file_list.append(test_file_name)

        print(self.dirs_to_check)

        if is_multithread:
            self.cond = mp.Condition()
            self.worker = mp.Process(target=self.prepare_mini_batch)

            # batches that are ready to be served :)
            # Prepare up to 20 batches.
            self.batch_queue = mp.Queue(maxsize=5)

            # Start the batch processing queue!
            self.worker.start()

    def extract_frames(self, video_file):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Cannot open ", video_file)
            return

        total_frame = (int)(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_stride = total_frame // self.num_frames

        frames = []
        for i in range(0, total_frame, frame_stride)[:self.num_frames]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, frame = cap.read()
            frame_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if i < total_frame :
                cap.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
                _, next_frame = cap.read()
                next_frame = cv2.resize(cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY), (224, 224))

            else :
                cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
                next_frame = frame_gray
                _, frame_gray = cap.read()
                frame_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (224, 224))


            flow = cv2.calcOpticalFlowFarneback(frame_gray, next_frame, None,
                                                pyr_scale=0.5, levels=2,
                                                winsize=15, iterations=1,
                                                poly_n=5, poly_sigma=1.2, flags=0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((224, 224, 3), dtype=frame.dtype)
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            flow = trn.Compose([trn.ToTensor()])(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
            frame = self.tf(Image.fromarray(frame))

            # [flow, frame] : (2, 3, H, W)
            frames.append(torch.stack([frame, flow]))

        # (N, 2, 3, H, W)
        return torch.stack(frames)

    def transform_frame(self):
        tf = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return tf

    # Create the mini batch from videos from video_names
    def create_mini_batch(self, video_names):
        assert len(video_names) == self.batch_size

        batch = []
        for s in video_names:
            batch.append(self.extract_frames(s))

        return torch.stack(batch).cuda()

    def make_batch_labels(self, labels):
        labels = torch.cuda.LongTensor(labels).view(self.batch_size)
        return labels

    # Returns the test batch and labels of entire keywords
    def feed_test_batch(self):
        print(self.entire_file_list)
        batch = [self.extract_frames(f) for f in self.entire_file_list]
        batch = torch.stack(batch).cuda()

        labels = self.dirs_to_check

        return batch, labels

    # Create the mini batch from the current directory.
    def feed_mini_batch(self):
        if self.is_multithread:
            # Wake up the batch preparing process if the queue is not full
            # (If the process were running already, then it does not matter)
            if not self.batch_queue.full():
                self.cond.acquire()
                self.cond.notify()
                self.cond.release()

            res = self.batch_queue.get()
            self.current_iter += self.batch_size

            if self.current_iter >= len(self.entire_file_list):
                self.current_iter = 0
                self.current_epoch += 1

            return res
        else:
            # Note that NEVER update global table for validation case. Only for training
            if self.current_iter < cache_size and self.num_file_offset == 0:
                if self.current_iter in global_video_tensor_cache:
                    (cpu_batch, cpu_labels) = global_video_tensor_cache[self.current_iter]
                    self.current_iter += self.batch_size

                    if self.current_iter >= len(self.entire_file_list):
                        self.current_iter = 0
                        self.current_epoch += 1

                    return (cpu_batch.to(torch.device('cuda')), cpu_labels.to(torch.device('cuda')))

            files = [f for (f, index) in
                     self.entire_file_list[self.current_iter: self.current_iter + self.batch_size]]
            batch = self.create_mini_batch(files)
            labels = self.make_batch_labels(
                [index for (f, index) in
                 self.entire_file_list[self.current_iter: self.current_iter + self.batch_size]])

            if self.current_iter < cache_size and self.num_file_offset == 0:
                cpu_batch = batch.to(torch.device('cpu'))
                cpu_labels = labels.to(torch.device('cpu'))
                global_video_tensor_cache[self.current_iter] = (cpu_batch, cpu_labels)

            self.current_iter += self.batch_size

            if self.current_iter >= len(self.entire_file_list):
                self.current_iter = 0
                self.current_epoch += 1

            return (batch, labels)

    def prepare_mini_batch(self):
        while True:
            while not self.batch_queue.full():
                if self.prepared_batch_iter < cache_size and self.num_file_offset == 0:
                    if self.prepared_batch_iter in global_video_tensor_cache:
                        (cpu_batch, cpu_labels) = global_video_tensor_cache[self.prepared_batch_iter]
                        self.prepared_batch_iter += self.batch_size

                        self.batch_queue.put(
                            (cpu_batch.to(torch.device('cuda')), cpu_labels.to(torch.device('cuda')))
                        )

                        if self.prepared_batch_iter >= len(self.entire_file_list):
                            self.prepared_batch_iter = 0
                            if self.current_epoch + 1 >= self.total_epoch:
                                return

                        continue

                files = [f for (f, index) in
                         self.entire_file_list[self.prepared_batch_iter: self.prepared_batch_iter + self.batch_size]]
                batch = self.create_mini_batch(files)
                labels = self.make_batch_labels(
                    [index for (f, index) in
                     self.entire_file_list[self.prepared_batch_iter: self.prepared_batch_iter + self.batch_size]])

                # Add prepared batch to the queue.
                self.batch_queue.put((batch, labels))

                if self.prepared_batch_iter < cache_size and self.num_file_offset == 0:
                    cpu_batch = batch.to(torch.device('cpu'))
                    cpu_labels = labels.to(torch.device('cpu'))
                    global_video_tensor_cache[self.prepared_batch_iter] = (cpu_batch, cpu_labels)

                self.prepared_batch_iter += self.batch_size

                if self.prepared_batch_iter >= len(self.entire_file_list):
                    self.prepared_batch_iter = 0

                    if self.current_epoch + 1 >= self.total_epoch:
                        return

                    # note that epoch is only increased when actual batch has feeded.

            # When the queue is full, sleep the thread and wait until it is consumed.
            self.cond.acquire()
            self.cond.wait()
            self.cond.release()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, temporal_stream):
        residual = x

        # The residual is passed from above network while
        # the block additionally gets the input from temporal stream
        z = x * F.relu(temporal_stream)
        out = self.conv1(z)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, temporal_stream):
        residual = x

        z = x * F.relu(temporal_stream)
        out = self.conv1(z)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def two_stream_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def two_stream_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def two_stream_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class TwoStreamNetwork(nn.Module):
    def __init__(self, num_class, num_frames, batch_size=5, hidden_lstm=512):
        super(TwoStreamNetwork, self).__init__()
        self.num_class = num_class
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.hidden_size = hidden_lstm
        self.resnet_output = 2048# resnet18, 34 : 512, resnet50 : 2048

        # Use the custom resnet34 that gets the input from the motion stream.
        self.pretrained_resnet = two_stream_resnet50(pretrained=True).cuda()
        self.spatial = nn.Sequential(*list(self.pretrained_resnet.children())[:-1]).cuda()
        for param in self.spatial.parameters():
            param.required_grad = False

        self.plain_resnet = models.resnet50(pretrained=True).cuda()
        self.temporal = nn.Sequential(*list(self.plain_resnet.children())[:-1]).cuda()

        self.lstm = nn.LSTM(input_size=self.resnet_output, hidden_size=hidden_lstm, batch_first=True).cuda()
        self.fc = nn.Linear(hidden_lstm, num_class).cuda()

        torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # x : (batch_size, num_frames, 2, C, H, W)
        image_flow = x[:,:,0,:,:,:]
        motion_flow = x[:,:,1,:,:,:]

        spatial_stream = image_flow.view(self.batch_size * self.num_frames, 3, 224, 224)
        temporal_stream = motion_flow.view(self.batch_size * self.num_frames, 3, 224, 224)

        # First, let it pass the 4 layers.
        for i in range(4) :
            spatial_stream = self.spatial[i](spatial_stream)
            temporal_stream = self.temporal[i](temporal_stream)

        # Add output from the motion stream to the spatial stream
        for i in range(4, 8):
            for j in range(len(self.spatial[i])):
                spatial_stream = self.spatial[i][j](spatial_stream, temporal_stream)
                temporal_stream = self.temporal[i][j](temporal_stream)

        spatial_stream = self.spatial[8](spatial_stream)
        spatial_stream = spatial_stream.view(self.batch_size, self.num_frames, self.resnet_output)

        # Convert (batch_size, num_frames, 2048, 1, 1) to
        # (batch_size, num_frames, 2048)
        spatial_stream, _ = self.lstm(spatial_stream)

        # Take the mean of every output vectors
        # x : (batch_size, 2048)
        x = torch.mean(spatial_stream, dim=1, keepdim=True)

        x = x.view(self.batch_size, self.hidden_size)

        out = self.fc(x)
        return out


def main():
    total_epoch = args.total_epoch
    num_class = args.num_class
    num_train_file = args.num_train_file
    num_val_file = args.num_val_file
    batch_size = args.batch_size
    is_test = args.is_test
    test_video_name = args.test_video_name
    is_multithread = args.multi_thread
    print("Multithread : ", is_multithread, args.multi_thread )
    allow_test_multiprocess = args.allow_test_multiprocess

    if is_test:
        video_data_loader = VideoDataLoader([], batch_size=batch_size, num_keywords=num_class, num_files=num_train_file,
                                            is_training=False, is_test=True, test_file_name=test_video_name)

        model = TwoStreamNetwork(video_data_loader.num_class, num_frames=video_data_loader.num_frames,
                              batch_size=1).cuda()
        if os.path.isfile('checkpoint_{}_{}_{}.pth.tar'.format(num_class, num_train_file, batch_size)):
            checkpoint = torch.load('checkpoint_{}_{}_{}.pth.tar'.format(num_class, num_train_file, batch_size))
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("Train file does not exist!")
            return

        with torch.no_grad():
            model.eval()
            batch, labels = video_data_loader.feed_test_batch()
            outputs = model(batch)

            _, predicted = torch.max(outputs.data, 1)
            _, predicted_k = torch.topk(outputs.data, 5)

            predicted = predicted.item()
            predicted_k = np.reshape(predicted_k.cpu().numpy(), (5))
            print(predicted_k)
            print("Top-1 Keyword :: ", labels[predicted])
            print("Top-5 Keyword :: ", [labels[predicted_k[i]] for i in range(5)])

        return

    video_data_loader = VideoDataLoader([], batch_size=batch_size, num_keywords=num_class, num_files=num_train_file,
                                        is_training=True, total_epoch=total_epoch, is_multithread=is_multithread)
    # summary(model, (3, 224, 224))

    model = TwoStreamNetwork(video_data_loader.num_class, num_frames=video_data_loader.num_frames,
                          batch_size=batch_size).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    total_batch_completed = 0
    pbar = tqdm(total=num_class * num_train_file * total_epoch)

    per_500_loss = []
    per_epoch_loss = []

    if os.path.isfile('checkpoint_{}_{}_{}.pth.tar'.format(num_class, num_train_file, batch_size)):
        checkpoint = torch.load('checkpoint_{}_{}_{}.pth.tar'.format(num_class, num_train_file, batch_size))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    while video_data_loader.current_epoch < total_epoch:
        total_batch_completed += batch_size

        batch, labels = video_data_loader.feed_mini_batch()
        outputs = model(batch)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        per_500_loss.append(loss.item())
        per_epoch_loss.append(loss.item())

        if total_batch_completed % num_train_file == 0:
            print('Current total iteration : {} {:.4f} '.format(total_batch_completed, np.mean(per_500_loss)))
            per_500_loss = []

        # Save the checkpoint per epoch and check train and validation accuracy.
        if total_batch_completed % (num_train_file * num_class) == 0:
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       'checkpoint_{}_{}_{}.pth.tar'.format(num_class, num_train_file, batch_size))
            print('Epoch #{} :: [Iter : {} ] Loss : {:.4f}'.format(video_data_loader.current_epoch + 1,
                                                                   total_batch_completed, np.mean(per_epoch_loss)))
            per_epoch_loss = []

            with torch.no_grad():
                model.eval()

                correct = 0
                correct_k = 0
                total = 0
                pbar_train_acc = tqdm(total=num_class * (num_train_file // 5))
                train_data_loader = VideoDataLoader([], batch_size=batch_size, num_keywords=num_class,
                                                    num_files=num_train_file // 5, num_file_offset=0,
                                                    total_epoch=1, is_multithread=allow_test_multiprocess)
                while train_data_loader.current_epoch < 1:
                    batch, labels = train_data_loader.feed_mini_batch()
                    outputs = model(batch)
                    _, predicted = torch.max(outputs.data, 1)
                    _, predicted_k = torch.topk(outputs.data, 5)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    correct_k += predicted_k.eq(labels.view(1, -1).expand_as(predicted_k)).sum().item()

                    pbar_train_acc.update(train_data_loader.batch_size)
                    pbar_train_acc.set_description("Train Acc : (%d %d %.4f) (%d %d %.4f)" % (
                        correct, total, correct / total, correct_k, total, correct_k / total))
                pbar_train_acc.close()

                print("Train Accuracy : {} / {} ({:.4f})".format(correct, total, correct / total))

                correct, total, correct_k = 0, 0, 0
                eval_data_loader = VideoDataLoader([], batch_size=batch_size, num_keywords=num_class,
                                                   num_files=num_val_file, num_file_offset=num_train_file,
                                                   total_epoch=1, is_multithread=allow_test_multiprocess)
                while eval_data_loader.current_epoch < 1:
                    batch, labels = eval_data_loader.feed_mini_batch()
                    outputs = model(batch)
                    _, predicted = torch.max(outputs.data, 1)
                    _, predicted_k = torch.topk(outputs.data, 5)

                    print(predicted_k, labels)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    correct_k += predicted_k.eq(labels.view(1, -1).expand_as(predicted_k)).sum().item()

                print(
                    "Val Accuracy : {} / {} ({:.4f} ; Top 5 : {} / {} ({:.4f})".format(correct, total, correct / total,
                                                                                       correct_k, total,
                                                                                       correct_k / total))
                model.train()
        pbar.update(batch_size)
        pbar.set_description("Loss : %f" % loss.item())
    pbar.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

import re
import os
import argparse
import numpy as np
import cv2
from PIL import Image

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision import transforms as trn
import torchvision.datasets as datasets
from tqdm import tqdm
from random import shuffle

parser = argparse.ArgumentParser(description="Train the bumbumnet")
parser.add_argument('--total_epoch', type=int, default=10)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--num_train_file', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_val_file', type=int, default=50)
args = parser.parse_args()

class VideoDataLoader:
    def __init__(self, keywords, num_keywords=-1, batch_size=5, num_frames=10, num_files=50, num_file_offset=0, is_training=False):
        self.current_epoch = 0
        self.current_iter = 0

        self.batch_size = batch_size
        self.num_frames = num_frames
        self.keywords = keywords
        self.num_files = num_files
        self.num_file_offset = num_file_offset
        self.is_training = is_training
        self.tf = self.transform_frame()

        self.dirs_to_check = os.listdir('./training')
        if len(self.keywords) != 0:
            self.dirs_to_check = [dir for dir in self.dirs_to_check if dir in self.keywords]
        elif num_keywords != -1:
            self.dirs_to_check = self.dirs_to_check[:num_keywords]

        self.num_class = len(self.dirs_to_check)

        self.entire_file_list = []

        dir_index = 0
        for dir in self.dirs_to_check:
            file_names = os.listdir('./training/' + dir)
            for file_name in file_names[self.num_file_offset:self.num_file_offset + self.num_files]:
                self.entire_file_list.append(('./training/' + dir + '/' + file_name, dir_index))
            dir_index += 1

        if self.is_training:
            shuffle(self.entire_file_list)

        print(self.dirs_to_check)


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
            ret, frame = cap.read()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(self.tf(frame))

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

    # Create the mini batch from the current directory.
    def feed_mini_batch(self):
        files = [f for (f, index) in self.entire_file_list[self.current_iter : self.current_iter + self.batch_size]]
        batch = self.create_mini_batch(files)
        res = batch, self.make_batch_labels([index for (f, index) in self.entire_file_list[self.current_iter : self.current_iter + self.batch_size]])
        self.current_iter += self.batch_size

        if self.current_iter >= len(self.entire_file_list):
            self.current_iter = 0
            self.current_epoch += 1

        return res


class SimpleNetwork(nn.Module):
    def __init__(self, num_class, num_frames, batch_size=5, hidden_lstm=512):
        super(SimpleNetwork, self).__init__()
        self.num_class = num_class
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.hidden_size = hidden_lstm

        # Remove the final FC layer of the pretrained Resnet
        self.pretrained_resnet = models.resnet50(pretrained=True).cuda()
        self.resnet = nn.Sequential(*list(self.pretrained_resnet.children())[:-1]).cuda()
        # for param in self.resnet.parameters():
        #    param.required_grad = False

        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_lstm, batch_first=True).cuda()
        self.fc = nn.Linear(hidden_lstm, num_class).cuda()
        torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # x : (batch_size, num_frames, C, H, W)
        x = x.view(self.batch_size * self.num_frames, 3, 224, 224)
        x = self.resnet(x)

        # Convert (batch_size, num_frames, 2048, 1, 1) to
        # (batch_size, num_frames, 2048)
        x = x.view(self.batch_size, self.num_frames, 2048)
        x, _ = self.lstm(x)

        # Take the mean of every output vectors
        # x : (batch_size, 2048)
        x = torch.mean(x, dim = 1, keepdim = True)

        x = x.view(self.batch_size, self.hidden_size)

        out = self.fc(x)
        return out


def main():
    total_epoch = args.total_epoch
    num_class = args.num_class
    num_train_file = args.num_train_file
    num_val_file = args.num_val_file
    batch_size = args.batch_size

    video_data_loader = VideoDataLoader([], batch_size=batch_size, num_keywords=num_class, num_files=num_train_file, is_training=True)
    # summary(model, (3, 224, 224))

    model = SimpleNetwork(video_data_loader.num_class, num_frames=video_data_loader.num_frames,
                          batch_size=batch_size).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    total_batch_completed = 0
    pbar = tqdm(total=num_class * num_train_file * total_epoch)

    per_500_loss = []
    per_epoch_loss = []

    if os.path.isfile('checkpoint.pth.tar'):
        checkpoint = torch.load('checkpoint.pth.tar')
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
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'checkpoint.pth.tar')
            print('Epoch #{} :: [Iter : {} ] Loss : {:.4f}'.format(video_data_loader.current_epoch + 1,
                                                                   total_batch_completed, np.mean(per_epoch_loss)))
            per_epoch_loss = []

            with torch.no_grad():
                model.eval()

                correct = 0
                correct_k = 0
                total = 0

                pbar_train_acc = tqdm(total=50 * 50)
                train_data_loader = VideoDataLoader([], batch_size=batch_size, num_keywords=num_class, num_files=num_train_file // 5, num_file_offset=0)
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
                eval_data_loader = VideoDataLoader([], batch_size=batch_size, num_keywords=num_class, num_files=num_val_file, num_file_offset=num_train_file)
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
    main()

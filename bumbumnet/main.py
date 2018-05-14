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

parser = argparse.ArgumentParser(description="Train the bumbumnet")
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('--total_epoch', type=int, default=10)

args = parser.parse_args()


class VideoDataLoader:
    def __init__(self, keywords, batch_size=5, num_frames=10, num_files=50, num_file_offset = 0):
        self.current_dir_index = 0
        self.current_file_index = 0
        self.current_epoch = 0
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.keywords = keywords
        self.num_files = num_files
        self.num_file_offset = num_file_offset
        self.tf = self.transform_frame()

        self.dirs_to_check = os.listdir('./training')
        if len(self.keywords) != 0:
            self.dirs_to_check = [dir for dir in self.dirs_to_check if dir in self.keywords]

        self.num_class = len(self.dirs_to_check)

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
        dir_index = 0
        for dir in self.dirs_to_check:
            if dir_index == self.current_dir_index:
                file_names = os.listdir('./training/' + dir)
                for file_index in range(len(file_names)):
                    if file_index == self.current_file_index:
                        files = file_names[file_index + self.num_file_offset:file_index + self.batch_size+ self.num_file_offset]
                        files = ['./training/' + dir + '/' + s for s in files]
                        batch = self.create_mini_batch(files)

                        self.current_dir_index += 1
                        if self.current_dir_index == len(self.dirs_to_check):
                            self.current_file_index += self.batch_size
                            self.current_dir_index = 0

                            if self.current_file_index == len(file_names) or self.current_file_index == self.num_files:
                                self.current_file_index = 0
                                self.current_dir_index = 0
                                self.current_epoch += 1

                        return batch, self.make_batch_labels([dir_index] * self.batch_size)
            dir_index += 1


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
        for param in self.resnet.parameters():
            param.required_grad = False

        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_lstm, batch_first=True).cuda()
        self.fc = nn.Linear(hidden_lstm, num_class).cuda()

    def forward(self, x):
        # x : (batch_size, num_frames, C, H, W)
        x = x.view(self.batch_size * self.num_frames, 3, 224, 224)
        x = self.resnet(x)

        # Convert (batch_size, num_frames, 2048, 1, 1) to
        # (batch_size, num_frames, 2048)
        x = x.view(self.batch_size, self.num_frames, 2048)
        x, _ = self.lstm(x)

        # Take the last output
        # x : (batch_size, 2048)
        x = x[:, -1, :].view(self.batch_size, self.hidden_size)

        out = self.fc(x)
        return out


def main():
    #video_data_loader = VideoDataLoader(["arresting", "ascending", "assembling", "attacking", "baking"])
    video_data_loader = VideoDataLoader([], num_files=100);
    # summary(model, (3, 224, 224))

    model = SimpleNetwork(video_data_loader.num_class, num_frames=video_data_loader.num_frames,
                          batch_size=video_data_loader.batch_size).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_epoch = args.total_epoch

    total_batch_completed = 0
    while video_data_loader.current_epoch < 20:
        total_batch_completed += video_data_loader.batch_size

        batch, labels = video_data_loader.feed_mini_batch()
        outputs = model(batch)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if total_batch_completed % 500 == 0:
            print('Current total iteration : {} {:.4f} '.format(total_batch_completed, loss.item()))

        if total_batch_completed % 10000 == 0:
            torch.save({'state_dict' : model.state_dict() }, 'checkpoint.pth.tar');

        if total_batch_completed % (100 * 125) == 0:
            print('Epoch #{} :: [Iter : {} ] Loss : {:.4f}'.format(video_data_loader.current_epoch + 1,
                                                                   total_batch_completed, loss.item()))

        if total_batch_completed % (200 * 125) == 0:
            with torch.no_grad():
                model.eval()

                correct = 0
                total = 0

                #eval_data_loader = VideoDataLoader(["arresting", "ascending", "assembling", "attacking", "baking"], num_file_offset=50)
                eval_data_loader = VideoDataLoader([], num_files=25, num_file_offset=100);
                while eval_data_loader.current_epoch < 1:
                    batch, labels = eval_data_loader.feed_mini_batch()
                    outputs = model(batch)
                    _, predicted = torch.max(outputs.data, 1)
                    print(predicted, labels)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print("Accuracy : {} / {} ({:.4f})".format(correct, total, correct / total))

                model.train()


if __name__ == "__main__":
    main()

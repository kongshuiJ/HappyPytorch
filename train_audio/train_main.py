import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch import Tensor

from pathlib import Path
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import os

SAVE_MODEL_FILE_PATH = './model_test.pt'
MODEL_NAME = 'JYB_MODEL'


def load_audio_item(filepath: str, path: str):
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label


def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output


class AudioDataset(Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 subset: Optional[str] = None,
                 ) -> None:

        assert subset is None or subset in ["training", "testing"], (
                "When `subset` not None, it must take a value from "
                + "{'training','testing'}."
        )

        folder_in_archive = './AudioData/'
        self._path = os.path.join(root, folder_in_archive)

        if subset == "training":
            self._walker = _load_list(self._path, "training_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt")

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        fileid = self._walker[n]
        return load_audio_item(fileid, self._path)

    def __len__(self) -> int:
        return len(self._walker)


train_set = AudioDataset('./', 'training')
test_set = AudioDataset('./', 'testing')

print(train_set[0])

waveform, sample_rate, label = train_set[0]

print("Shape of waveform: {}".format(waveform.size()))
print('sample_rate', sample_rate)
print('label', label)

labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
print('labels.size', len(labels))
print('labels', labels)

new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 16

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=3, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)

        # log-softmax其实可以理解为对softmax的结果取对数，
        # 在pytorch的实现中，使用了防止溢出的Trick，即log-sum-exp；
        # 使用log的另一个好处是求导更加方便，可以加快反向传播的速度。
        return F.log_softmax(x, dim=2)


model = M5(n_input=transformed.shape[0], n_output=len(labels))
if os.path.exists(SAVE_MODEL_FILE_PATH):
    state_dict = torch.load(SAVE_MODEL_FILE_PATH)
    model.load_state_dict(state_dict[MODEL_NAME])
model.to(device)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


n = count_parameters(model)
print("Number of parameters: %s" % n)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
# reduce the learning after 20 epochs by a factor of 10
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


log_interval = 20
n_epoch = 500

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

print("w222222222222222222222222222222222222222222")

# for i, element in enumerate(test_loader):
#     if i == 1:
#         print('i', i)
#         print('element', element)
#         break
# exit(0)

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()

torch.save({MODEL_NAME: model.state_dict()}, SAVE_MODEL_FILE_PATH)


def test_ont(model, idx):
    model.eval()
    correct = 0

    for i, (data, target) in enumerate(test_loader):
        if i == idx:
            print('test_loader ', idx)
            print('data', data)
            print('target', target)
            data = data.to(device)
            target = target.to(device)
            data = transform(data)
            output = model(data)
            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)

            print('output', output)
            print('pred', pred)
            print(
                f"\nTest One\tAccuracy: {correct}/{len(target)} ({100. * correct / len(target):.0f}%)\n")


print('===========================test1============================')
test_ont(model, 0)
print('===========================test2============================')

# Let's plot the training loss versus the number of iteration.
plt.plot(losses)
plt.title("training loss")
plt.show()


def predict(tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


waveform, sample_rate, utterance, *_ = train_set[-1]

print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")

print("=========================================")

# for i, (waveform, sample_rate, utterance, *_) in enumerate(test_set):
#     output = predict(waveform)
#     if output != utterance:
#         print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
#         break
# else:
#     print("All examples in this dataset were correctly classified!")
#     print("In this case, let's just look at the last data point")
#     print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")

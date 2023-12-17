import numpy as np
from django.conf import settings
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from skimage import measure


class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(16)

        self.conv1_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 1, 2, 1)

        self.conv2_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 1, 2, 1)

        self.conv3_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(64)

        # Feature fusion
        self.conv_fusion1 = nn.Conv2d(16, 64, 1, 4, 2)
        self.conv_fusion2 = nn.Conv2d(32, 64, 1, 2, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x = F.relu(self.bn1_1(self.conv1_1(x1)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x)))
        x = x1 + x_1
        x2 = self.conv2(x)
        x = F.relu(self.bn2_1(self.conv2_1(x2)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x)))
        x = x2 + x_2
        x3 = self.conv3(x)
        x = F.relu(self.bn3_1(self.conv3_1(x3)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x)))
        return x_1, x_2, x_3


class FeatFuse(nn.Module):
    def __init__(self):
        super(FeatFuse, self).__init__()

        self.conv_fusion1 = nn.Conv2d(16, 64, 1, 4, 3)
        self.conv_fusion2 = nn.Conv2d(32, 64, 1, 2, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 64 * 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        batch_size = x1.size(0)
        out_channels = x3.size(1)
        x1 = self.conv_fusion1(x1)
        x2 = self.conv_fusion2(x2)
        output = []
        output.append(x1)
        output.append(x2)
        output.append(x3)
        x = x1 + x2 + x3

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        a_b = x.reshape(batch_size, 3, out_channels, -1)
        a_b = self.softmax(a_b)
        # the part of selection
        a_b = list(a_b.chunk(3, dim=1))  # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.featnet = FeatNet()
        self.featfuse = FeatFuse()
        self.featnet1 = FeatNet()
        self.featfuse1 = FeatFuse()
        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 2)

        self.global_pool1 = nn.AdaptiveAvgPool2d(1)
        self.global_pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 2)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x, y):
        x1_1, x1_2, x1_3 = self.featnet(x)
        x2_1, x2_2, x2_3 = self.featnet1(y)

        feat_11 = self.featfuse(x1_1, x1_2, x1_3)
        feat_22 = self.featfuse1(x2_1, x2_2, x2_3)
        feat_1 = self.global_pool1(feat_11)
        feat_2 = self.global_pool2(feat_22)
        feat_1 = feat_1.view(feat_1.size(0), -1)
        feat_2 = feat_2.view(feat_2.size(0), -1)
        feat_1 = self.fc1(feat_1)
        feat_2 = self.fc2(feat_2)

        feature_corr = self.xcorr_depthwise(feat_11, feat_22)
        feat = feature_corr.view(feature_corr.size(0), -1)
        # feat = global_pool(feature_corr)
        feat = self.fc(feat)
        return feat_1, feat_2, feat

    def xcorr_depthwise(self, x, kernel):
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out


def addZeroPadding(X, margin=2):
    newX = np.zeros((
        X.shape[0] + 2 * margin,
        X.shape[1] + 2 * margin,
        X.shape[2]
    ))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX


def postprocess(res):
    res_new = res
    res = measure.label(res, connectivity=2)
    num = res.max()
    for i in range(1, num + 1):
        idy, idx = np.where(res == i)
        if len(idy) <= 20:
            res_new[idy, idx] = 0
    return res_new


windowSize = 7  # patch size
class_num = 2
testRatio = 0.2  # the ratio of Validation set
trainRatio = 0.9  # the ratio of Training set selected from preclassification
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net().eval().to(device)
model.load_state_dict(torch.load(settings.BASE_DIR / "model.pth", map_location=torch.device(device)))
margin = (windowSize - 1) // 2


def get_result(im1, im2, result_id, cache):
    im1 = im1.reshape(im1.shape[0], im1.shape[1], 1)
    im2 = im2.reshape(im2.shape[0], im2.shape[1], 1)
    height, width, c = im1.shape
    im1 = addZeroPadding(im1, margin=margin)
    im2 = addZeroPadding(im2, margin=margin)
    outputs = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            patch1 = im1[i:i + windowSize, j:j + windowSize, :]
            patch1 = patch1.reshape(1, patch1.shape[0], patch1.shape[1], patch1.shape[2])
            X_test_image = torch.FloatTensor(patch1.transpose(0, 3, 1, 2)).to(device)

            patch2 = im2[i:i + windowSize, j:j + windowSize, :]
            patch2 = patch2.reshape(1, patch2.shape[0], patch2.shape[1], patch2.shape[2])
            X_test_image1 = torch.FloatTensor(patch2.transpose(0, 3, 1, 2)).to(device)

            _, _, prediction = model(X_test_image, X_test_image1)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction
        cache.set(result_id, i * 100 // height)
    # postprocessing if need
    outputs = postprocess(outputs)
    cache.delete(result_id)
    return outputs


def createImgCube(X, gt, pos: list, windowSize=25):
    margin = (windowSize - 1) // 2
    zeroPaddingX = addZeroPadding(X, margin=margin)
    dataPatches = np.zeros((pos.__len__(), windowSize, windowSize, X.shape[2]))
    if (pos[-1][1] + 1 != X.shape[1]):
        nextPos = (pos[-1][0], pos[-1][1] + 1)
    elif (pos[-1][0] + 1 != X.shape[0]):
        nextPos = (pos[-1][0] + 1, 0)
    else:
        nextPos = (0, 0)
    return np.array([zeroPaddingX[i:i + windowSize, j:j + windowSize, :] for i, j in pos]), \
           np.array([gt[i, j] for i, j in pos]), \
           nextPos


def createPosWithoutZero(hsi, gt):
    mask = gt > 0
    return [(i, j) for i, row in enumerate(mask) for j, row_element in enumerate(row) if row_element]
    # return np.argwhere(mask).flatten()


def test(model, device, test_loader):
    model.eval()
    count = 0
    for inputs_1, inputs_2, labels in test_loader:

        inputs_1, inputs_2 = inputs_1.to(device), inputs_2.to(device)
        _, _, outputs = model(inputs_1, inputs_2)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

        if count == 0:
            y_pred_test = outputs
            test_labels = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            test_labels = np.concatenate((test_labels, labels))
    a = 0
    for c in range(len(y_pred_test)):
        if test_labels[c] == y_pred_test[c]:
            a = a + 1
    acc = a / len(y_pred_test) * 100
    return acc


class TestDS(torch.utils.data.Dataset):
    def __init__(self, test_labels, X_test, X_test_2):
        self.len = test_labels.shape[0]
        self.hsi = torch.FloatTensor(X_test)
        self.lidar = torch.FloatTensor(X_test_2)
        self.labels = torch.LongTensor(test_labels - 1)

    def __getitem__(self, index):
        return self.hsi[index], self.lidar[index], self.labels[index]

    def __len__(self):
        return self.len


def get_accuracy(im1, im2, stand_img):
    im1 = im1.reshape(im1.shape[0], im1.shape[1], 1)
    im2 = im2.reshape(im2.shape[0], im2.shape[1], 1)

    # data_path = '/content/drive/MyDrive/sars/test/data'  # 修改为自定义图片路径
    # stand_img = sio.loadmat(os.path.join(data_path, 'san_gt.mat'))['data']
    X_test, test_labels, _ = createImgCube(im1, stand_img, createPosWithoutZero(im1, stand_img),
                                           windowSize=windowSize)
    X_test_2, _, _ = createImgCube(im2, stand_img, createPosWithoutZero(im2, stand_img), windowSize=windowSize)
    X_test = torch.from_numpy(X_test.transpose(0, 3, 1, 2)).float()
    X_test_2 = torch.from_numpy(X_test_2.transpose(0, 3, 1, 2)).float()

    testset = TestDS(test_labels, X_test, X_test_2)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=False, num_workers=0)

    acc = test(model, device, test_loader)
    return acc


if __name__ == '__main__':
    import cv2, os

    im1 = cv2.cvtColor(cv2.imread(os.path.join(os.getcwd(), 'Farm1.bmp')), cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(cv2.imread(os.path.join(os.getcwd(), 'Farm2.bmp')), cv2.COLOR_BGR2GRAY)
    im_gt = cv2.cvtColor(cv2.imread(os.path.join(os.getcwd(), 'Farm_gt.bmp')), cv2.COLOR_BGR2GRAY)
    modified_image = np.where(im_gt == 0, 1, 2)

    print(get_accuracy(im1, im2, modified_image))

import math
import sys

import cv2
import numpy as np
import torch

try:
    from .correlation import correlation
except ImportError:
    sys.path.insert(0, './correlation')
    import correlation

assert (int(str('').join(torch.__version__.split('.')[0:2])) >= 13)

torch.set_grad_enabled(False)

torch.backends.cudnn.enabled = True

output_video_path = 'gt-pred.mp4'
# input_video_path = "/ssd01/zhangyiyang/basketball/data/basketball/raw/videos/0108/2.mp4"  # noqa
input_video_path = "gt.mp4"  # noqa
arguments_strModel = 'default'  # 'default', or 'kitti', or 'sintel'
output_height, output_width = 480, 960
show = True

backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]),
                                1.0 - (1.0 / tenFlow.shape[3]),
                                tenFlow.shape[3]).view(1, 1, 1, -1).expand(
                                    -1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]),
                                1.0 - (1.0 / tenFlow.shape[2]),
                                tenFlow.shape[2]).view(1, 1, -1, 1).expand(
                                    -1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer],
                                                         1).cuda()

    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
        tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)
    ], 1)

    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(
            0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Features(torch.nn.Module):
            def __init__(self):
                super(Features, self).__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3,
                                    out_channels=32,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=96,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96,
                                    out_channels=96,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=192,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super(Matching, self).__init__()

                self.fltBackwarp = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25,
                                    0.625][intLevel]

                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                if intLevel == 6:
                    self.netUpflow = None

                elif intLevel != 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2,
                                                              out_channels=2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1,
                                                              bias=False,
                                                              groups=2)

                if intLevel >= 4:
                    self.netUpcorr = None

                elif intLevel < 4:
                    self.netUpcorr = torch.nn.ConvTranspose2d(in_channels=49,
                                                              out_channels=49,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1,
                                                              bias=False,
                                                              groups=49)

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=49,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=2,
                                    kernel_size=[0, 0, 7, 5, 5, 3,
                                                 3][intLevel],
                                    stride=1,
                                    padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

            def forward(self, tenFirst, tenSecond, tenFeaturesFirst,
                        tenFeaturesSecond, tenFlow):
                tenFeaturesFirst = self.netFeat(tenFeaturesFirst)
                tenFeaturesSecond = self.netFeat(tenFeaturesSecond)

                if tenFlow is not None:
                    tenFlow = self.netUpflow(tenFlow)

                if tenFlow is not None:
                    tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond,
                                                 tenFlow=tenFlow *
                                                 self.fltBackwarp)

                if self.netUpcorr is None:
                    tenCorrelation = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(
                            tenFirst=tenFeaturesFirst,
                            tenSecond=tenFeaturesSecond,
                            intStride=1),
                        negative_slope=0.1,
                        inplace=False)

                elif self.netUpcorr is not None:
                    tenCorrelation = self.netUpcorr(
                        torch.nn.functional.leaky_relu(
                            input=correlation.FunctionCorrelation(
                                tenFirst=tenFeaturesFirst,
                                tenSecond=tenFeaturesSecond,
                                intStride=2),
                            negative_slope=0.1,
                            inplace=False))

                return (tenFlow if tenFlow is not None else
                        0.0) + self.netMain(tenCorrelation)

        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel):
                super(Subpixel, self).__init__()

                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25,
                                    0.625][intLevel]

                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel],
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=2,
                                    kernel_size=[0, 0, 7, 5, 5, 3,
                                                 3][intLevel],
                                    stride=1,
                                    padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

            def forward(self, tenFirst, tenSecond, tenFeaturesFirst,
                        tenFeaturesSecond, tenFlow):
                tenFeaturesFirst = self.netFeat(tenFeaturesFirst)
                tenFeaturesSecond = self.netFeat(tenFeaturesSecond)

                if tenFlow is not None:
                    tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond,
                                                 tenFlow=tenFlow *
                                                 self.fltBackward)

                return (
                    tenFlow if tenFlow is not None else 0.0) + self.netMain(
                        torch.cat(
                            [tenFeaturesFirst, tenFeaturesSecond, tenFlow], 1))

        class Regularization(torch.nn.Module):
            def __init__(self, intLevel):
                super(Regularization, self).__init__()

                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25,
                                    0.625][intLevel]

                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]

                if intLevel >= 5:
                    self.netFeat = torch.nn.Sequential()

                elif intLevel < 5:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel],
                            out_channels=128,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel],
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                if intLevel >= 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=[0, 0, 49, 25, 25, 9,
                                                      9][intLevel],
                                        kernel_size=[0, 0, 7, 5, 5, 3,
                                                     3][intLevel],
                                        stride=1,
                                        padding=[0, 0, 3, 2, 2, 1,
                                                 1][intLevel]))

                elif intLevel < 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=[0, 0, 49, 25, 25, 9,
                                                      9][intLevel],
                                        kernel_size=([0, 0, 7, 5, 5, 3,
                                                      3][intLevel], 1),
                                        stride=1,
                                        padding=([0, 0, 3, 2, 2, 1,
                                                  1][intLevel], 0)),
                        torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9,
                                                     9][intLevel],
                                        out_channels=[0, 0, 49, 25, 25, 9,
                                                      9][intLevel],
                                        kernel_size=(1, [0, 0, 7, 5, 5, 3,
                                                         3][intLevel]),
                                        stride=1,
                                        padding=(0, [0, 0, 3, 2, 2, 1,
                                                     1][intLevel])))

                self.netScaleX = torch.nn.Conv2d(
                    in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.netScaleY = torch.nn.Conv2d(
                    in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)

            def forward(self, tenFirst, tenSecond, tenFeaturesFirst,
                        tenFeaturesSecond, tenFlow):
                tenDifference = (
                    tenFirst -
                    backwarp(tenInput=tenSecond,
                             tenFlow=tenFlow * self.fltBackward)).pow(2.0).sum(
                                 1, True).sqrt().detach()

                tenDist = self.netDist(
                    self.netMain(
                        torch.cat([
                            tenDifference, tenFlow -
                            tenFlow.view(tenFlow.shape[0], 2, -1).mean(
                                2, True).view(tenFlow.shape[0], 2, 1, 1),
                            self.netFeat(tenFeaturesFirst)
                        ], 1)))
                tenDist = tenDist.pow(2.0).neg()
                tenDist = (tenDist - tenDist.max(1, True)[0]).exp()

                tenDivisor = tenDist.sum(1, True).reciprocal()

                tenScaleX = self.netScaleX(
                    tenDist * torch.nn.functional.unfold(
                        input=tenFlow[:, 0:1, :, :],
                        kernel_size=self.intUnfold,
                        stride=1,
                        padding=int((self.intUnfold - 1) /
                                    2)).view_as(tenDist)) * tenDivisor
                tenScaleY = self.netScaleY(
                    tenDist * torch.nn.functional.unfold(
                        input=tenFlow[:, 1:2, :, :],
                        kernel_size=self.intUnfold,
                        stride=1,
                        padding=int((self.intUnfold - 1) /
                                    2)).view_as(tenDist)) * tenDivisor

                return torch.cat([tenScaleX, tenScaleY], 1)

        self.netFeatures = Features()
        self.netMatching = torch.nn.ModuleList(
            [Matching(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.netSubpixel = torch.nn.ModuleList(
            [Subpixel(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.netRegularization = torch.nn.ModuleList(
            [Regularization(intLevel) for intLevel in [2, 3, 4, 5, 6]])

        self.load_state_dict({
            strKey.replace('module', 'net'): tenWeight
            for strKey, tenWeight in torch.hub.load_state_dict_from_url(
                url=  # noqa
                'http://content.sniklaus.com/github/pytorch-liteflownet/network-'  # noqa
                + arguments_strModel + '.pytorch',
                file_name='liteflownet-' + arguments_strModel).items()
        })

    def forward(self, tenFirst, tenSecond):
        tenFirst[:, 0, :, :] = tenFirst[:, 0, :, :] - 0.411618
        tenFirst[:, 1, :, :] = tenFirst[:, 1, :, :] - 0.434631
        tenFirst[:, 2, :, :] = tenFirst[:, 2, :, :] - 0.454253

        tenSecond[:, 0, :, :] = tenSecond[:, 0, :, :] - 0.410782
        tenSecond[:, 1, :, :] = tenSecond[:, 1, :, :] - 0.433645
        tenSecond[:, 2, :, :] = tenSecond[:, 2, :, :] - 0.452793

        tenFeaturesFirst = self.netFeatures(tenFirst)
        tenFeaturesSecond = self.netFeatures(tenSecond)

        tenFirst = [tenFirst]
        tenSecond = [tenSecond]

        for intLevel in [1, 2, 3, 4, 5]:
            tenFirst.append(
                torch.nn.functional.interpolate(
                    input=tenFirst[-1],
                    size=(tenFeaturesFirst[intLevel].shape[2],
                          tenFeaturesFirst[intLevel].shape[3]),
                    mode='bilinear',
                    align_corners=False))
            tenSecond.append(
                torch.nn.functional.interpolate(
                    input=tenSecond[-1],
                    size=(tenFeaturesSecond[intLevel].shape[2],
                          tenFeaturesSecond[intLevel].shape[3]),
                    mode='bilinear',
                    align_corners=False))

        tenFlow = None

        for intLevel in [-1, -2, -3, -4, -5]:
            tenFlow = self.netMatching[intLevel](tenFirst[intLevel],
                                                 tenSecond[intLevel],
                                                 tenFeaturesFirst[intLevel],
                                                 tenFeaturesSecond[intLevel],
                                                 tenFlow)
            tenFlow = self.netSubpixel[intLevel](tenFirst[intLevel],
                                                 tenSecond[intLevel],
                                                 tenFeaturesFirst[intLevel],
                                                 tenFeaturesSecond[intLevel],
                                                 tenFlow)
            tenFlow = self.netRegularization[intLevel](
                tenFirst[intLevel], tenSecond[intLevel],
                tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel],
                tenFlow)

        return tenFlow * 20.0


netNetwork = None


def estimate(tenFirst, tenSecond):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()

    assert (tenFirst.shape[1] == tenSecond.shape[1])
    assert (tenFirst.shape[2] == tenSecond.shape[2])

    intWidth = tenFirst.shape[2]
    intHeight = tenFirst.shape[1]

    # assert (
    #     intWidth == 1024
    # )
    # assert (
    #     intHeight == 436
    # )

    tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tenPreprocessedFirst = torch.nn.functional.interpolate(
        input=tenPreprocessedFirst,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bilinear',
        align_corners=False)
    tenPreprocessedSecond = torch.nn.functional.interpolate(
        input=tenPreprocessedSecond,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bilinear',
        align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(
        tenPreprocessedFirst, tenPreprocessedSecond),
                                              size=(intHeight, intWidth),
                                              mode='bilinear',
                                              align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()


def flow2img(flow_data):
    """
    convert optical flow into color image
    :param flow_data:
    :return: color image
    """
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: horizontal optical flow
    :param v: vertical optical flow
    :return:
    """

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG,
               0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC,
               2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB,
               1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM,
               0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += +BM

    # MR
    colorwheel[col:col + MR,
               2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


if __name__ == '__main__':
    cap = cv2.VideoCapture(input_video_path)  # noqa
    assert cap.isOpened()
    writer = None
    if output_video_path is None:
        writer = None
    else:
        writer = cv2.VideoWriter(output_video_path,
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                 cap.get(cv2.CAP_PROP_FPS),
                                 (output_width, output_height * 2))

    previous = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw = frame = cv2.resize(frame, (output_width, output_height))
        frame = np.ascontiguousarray(
            frame.transpose(2, 0, 1).astype(np.float32) / 255.)
        frame = torch.FloatTensor(frame)
        if previous is None:
            previous = frame
            continue

        current = frame
        flow = estimate(previous, current)
        print(torch.max(flow), torch.min(flow))

        # 光流结果可视化
        img = flow2img(flow.numpy().transpose(1, 2, 0).astype(np.float32))

        # 根据 height 拼接
        concat = np.concatenate([raw, img], axis=0)
        
        # 结果输出
        if writer:
            writer.write(img)
        if show:
            cv2.imshow("demo", concat[:, :, ::-1])
            cv2.waitKey(1)

        previous = current
    if writer:
        writer.close()
    if show:
        cv2.destroyAllWindows()

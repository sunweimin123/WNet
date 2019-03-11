import torch.nn as nn
import torch.nn.functional as F
import torch



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class Wnet(nn.Module):




    def catchup(self,upload,cconv):
        cropidx = (cconv.size(2) - upload.size(2)) // 2
        cconv = cconv[:, :, cropidx:cropidx + upload.size(2), cropidx:cropidx + upload.size(2)]
        xup = torch.cat((cconv, upload), 1)
        return xup

    def __init__(self,colordim=1):
        super(Wnet, self).__init__()

        self.convfirst1 = DoubleConv(colordim, 64)
        self.poolfirst1 = nn.MaxPool2d(2)

        self.convfirst2 = DoubleConv(64, 128)
        self.poolfirst2 = nn.MaxPool2d(2)

        self.convfirst3 = DoubleConv(128, 256)
        self.poolfirst3 = nn.MaxPool2d(2)

        self.convfirst4 = DoubleConv(256, 512)
        self.poolfirst4 = nn.MaxPool2d(2)

        self.convfirst5 = DoubleConv(512, 1024)

        self.upsecond4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.convsecond4 = DoubleConv(1024, 512)

        self.upsecond3 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.convsecond3 = DoubleConv(512,256)

        self.upsecond2 = nn.ConvTranspose2d(256,128,2,stride=2)

        self.convthird2 = DoubleConv(256,128)
        self.poolthird2  =nn.MaxPool2d(2)

        self.convthird3 = DoubleConv(128,256)
        self.poolthird3 = nn.MaxPool2d(2)

        self.convthird4  =DoubleConv(256,512)
        self.poolthird4 = nn.MaxPool2d(2)

        self.convthird5 = DoubleConv(512,1024)

        self.upforth4 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.convforth4 = DoubleConv(1024,512)

        self.upforth3 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.convforth3 = DoubleConv(512,256)


        self.upforth2 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.convforth2 = DoubleConv(256,128)

        self.upforth1 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.convforth1 = DoubleConv(128,64)

        self.output = nn.Conv2d(64,colordim,1)


    def forward(self, input):
        cfirst1 = self.convfirst1(input)
        pfirst1 = self.poolfirst1(cfirst1)

        cfirst2 = self.convfirst2(pfirst1)
        pfirst2 = self.poolfirst2(cfirst2)

        cfirst3 = self.convfirst3(pfirst2)
        pfirst3 = self.poolfirst3(cfirst3)

        cfirst4 = self.convfirst4(pfirst3)
        pfirst4 = self.poolfirst4(cfirst4)

        cfirst5 = self.convfirst5(pfirst4)

        usecond4 = self.upsecond4(cfirst5)
        #secondmerge4 = torch.cat([usecond4, cfirst4], dim=1)
        secondmerge4= catchup(usecond4,cfirst4)
        csecond4 = self.convsecond4(secondmerge4)

        usecond3 = self.upsecond3(csecond4)
        secondmerge3 = catchup(usecond3,cfirst3)

        # secondmerge3 = torch.cat([usecond3, cfirst3], dim=1)
        csecond3 = self.convsecond3(secondmerge3)

        usecond2 = self.upsecond2(csecond3)
        #secondmerge2 = torch.cat([usecond2,cfirst2],dim=1)
        secondmerge2= catchup(usecond2,cfirst2)

        cthird2 = self.convthird2(secondmerge2)
        pthird2 = self.poolthird2(cthird2)

        cthird3 = self.convthird3(pthird2)
        pthird3 = self.poolthird3(cthird3)

        cthird4 = self.convthird4(pthird3)
        pthird4 = self.poolthird4(cthird4)

        cthird5 = self.convthird5(pthird4)

        uforth4 = self.upforth4(cthird5)
        #forthmerge4 = torch.cat([uforth4,cthird4],dim=1)
        forthmerge4= catchup(uforth4,cthird4)
        cforth4 = self.convforth4(forthmerge4)

        uforth3 = self.upforth3(cforth4)
        #forthmerge3 = torch.cat([uforth3,cthird3],dim=1)
        forthmerge3= catchup(uforth3,cthird3)
        cforth3 = self.convforth3(forthmerge3)

        uforth2 = self.upforth2(cforth3)
        #forthmerge2 = torch.cat([uforth2,cthird2],dim=1)
        forthmerge2 = catchup(uforth2,cthird2)
        cforth2 = self.convforth2(forthmerge2)

        uforth1 = self.upforth1(cforth2)
        cforth1 = self.convforth1(uforth1)
        c10 = self.output(cforth1)
        out = nn.Sigmoid()(c10)
        return out











w = Wnet()
print w





























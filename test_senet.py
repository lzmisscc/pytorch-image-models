from timm.models.senet import legacy_seresnet18, legacy_seresnet50
from timm.models.senet import SEBottleneck, SENet, SEResNetBlock, SEResNetBottleneck
from torchsummary import summary
import torch

# net = SENet(SEResNetBlock, [1, 1, 1, 1], 1, 16,
#             num_classes=0, global_pool='', input_3x3=True)
# summary(net, input_size=(3, 32, 100), device='cpu')


net = legacy_seresnet50(pretrained=False, num_classes=0,
                        global_pool='', input_3x3=False, in_chans=1)
# net.load_state_dict(torch.load("seresnet18-4bb0ce65.pth"), strict=False)
summary(net, input_size=(1, 32, 200), device='cpu')

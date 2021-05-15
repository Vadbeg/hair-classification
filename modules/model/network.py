"""Module with network code"""

import timm
import torch
import torchsummary


class HerbariumNet(torch.nn.Module):
    """Neural network class"""

    def __init__(self, model_type: str, pretrained: bool = True,
                 num_of_output_nodes: int = 1000, get_embeddings: bool = False):
        """
        Init class method

        :param model_type: model type (resnet18, effnet)
        :param pretrained: if True uses pretrained weights for network
        :param num_of_output_nodes: number of output network nodes
        :param get_embeddings: if True uses arc face layer instead fc layer
        """

        super().__init__()

        # backbone = timm.create_model(model_type, pretrained=pretrained, output_stride=16)
        backbone = timm.create_model(model_type, pretrained=pretrained, output_stride=32)
        # backbone = timm.create_model(model_type, pretrained=pretrained)

        if 'resnet' in model_type:
            self.n_features = backbone.fc.in_features
        elif 'eff' in model_type:
            self.n_features = backbone.classifier.in_features
        else:
            raise NotImplementedError(f'No num_features for this network type')

        self.backbone = torch.nn.Sequential(*backbone.children())[:-2]

        # self.backbone[-1] = self.backbone[-1][0]
        # self.backbone[-1].downsample[0].stride = 1

        self.classifier = torch.nn.Linear(self.n_features, num_of_output_nodes)

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.get_embeddings = get_embeddings

    def forward_features(self, x: torch.tensor) -> torch.tensor:
        """
        Performs forward propagation to get features from backbone

        :param x: input tensor
        :return: features tensor
        """

        x = self.backbone(x)

        return x

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Performs forward pass threw network

        :param x: input tensor
        :return: network result
        """

        feats = self.forward_features(x)

        x = self.pool(feats).view(x.size(0), -1)

        if not self.get_embeddings:
            x = self.classifier(x)

        return x


if __name__ == '__main__':
    network = HerbariumNet(model_type='resnet18', pretrained=False, num_of_output_nodes=2721).to('cpu')
    print(network)

    # torchsummary.summary(network, input_size=(3, 320, 320), device='cpu')

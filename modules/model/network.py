"""Module with network code"""

import torch
import torchsummary


class FaceNet(torch.nn.Module):
    """Neural network class"""

    def __init__(
            self, model_type: str = 'simple',
            in_channels: int = 3,
            num_of_output_nodes: int = 2
    ):
        """
        Init class method

        :param model_type: type of model to use as backbone
        :param in_channels: number of input network channels
        :param num_of_output_nodes: number of output network nodes
        """

        super().__init__()

        if model_type == 'simple':
            self.backbone = self.__create_simple_backbone_model(
                in_channels=in_channels
            )
            backbone_out_channels = list(self.backbone.children())[-1].out_channels
        elif model_type == 'mobilenet_v2':
            backbone = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)

            self.backbone = torch.nn.Sequential(*backbone.children())[:-1]

            backbone_out_channels = self.backbone[0][-1].out_channels
        else:
            raise NotImplementedError(f'Model {model_type} is not implemented yet.')

        self.pool = torch.nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = torch.nn.Linear(
            backbone_out_channels * 5 * 5,
            num_of_output_nodes
        )

    @staticmethod
    def __create_simple_backbone_model(
            in_channels: int = 3
    ) -> torch.nn.Module:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=(3, 3)
            ),
            torch.nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2
            ),

            torch.nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=(3, 3)
            ),
            torch.nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2
            ),

            torch.nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
        )

        return model

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
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    network = FaceNet(
        model_type='simple',
        in_channels=3,
        num_of_output_nodes=2
    ).to('cpu')

    torchsummary.summary(network, input_size=(3, 320, 320), device='cpu')

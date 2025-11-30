from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """
    Convert an input image into a sequence of patch embeddings.

    This module slices the input image into non-overlapping square patches
    of size `(patch_size x patch_size)`, applies a Conv2D projection to map each
    patch to a `d_model`-dimensional representation, then flattens and
    rearranges the result into a sequence.

    Args:
        d_model (int): Output embedding dimensionality.
        img_size (int): Height/width of the input image (must be square).
        patch_size (int): Size of one square patch.
        n_channels (int): Number of channels in the input image.

    Shape:
        - Input:  (B, C, H, W)
        - Output: (B, N, d_model)
          where N = (H // patch_size) * (W // patch_size)
    """

    def __init__(
        self, d_model: int, img_size: int, patch_size: int, n_channels: int
    ) -> None:
        super().__init__()

        self.d_model: int = d_model
        self.img_size: int = img_size
        self.patch_size: int = patch_size
        self.n_channels: int = n_channels

        self.linear_project: nn.Conv2d = nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply linear patch projection and reshape to a sequence of embeddings.

        Args:
            x (Tensor): Input batch of images of shape (B, C, H, W).

        Returns:
            Tensor: Sequence of flattened patch embeddings of shape
            `(B, N, d_model)`.

        Notes:
            - P_col = H // patch_size
            - P_row = W // patch_size
            - N = P_col * P_row
        """
        x = self.linear_project(x)  # (B, d_model, P_col, P_row)
        x = x.flatten(2)  # (B, d_model, N)
        x = x.transpose(1, 2)  # (B, N, d_model)
        return x

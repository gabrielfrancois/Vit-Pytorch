from torch import nn as nn

"""
For start, let's set up the patch embeeding. This is the first step of our approach, and it consists of preparing the batch size of our dataset. 
For instance, with cifar-10, we'll split each images into 4  patches of 16x16 pixels + 1 cls (prediction token). 

"""

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model # Dimensionality of Model (of embeeding)
        self.img_size = img_size # Image Size
        self.patch_size = patch_size # Patch Size
        self.n_channels = n_channels # Number of Channels

        self.linear_project = nn.Conv2d(in_channels=self.n_channels,out_channels=self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

    # B: Batch Size
    # C: Image Channels
    # H: Image Height
    # W: Image Width
    # P_col: Patch Column
    # P_row: Patch Row
    """
    forward method (with typically :
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
            )
    hidden here by "nn.Module", permits to launch "forward" function only in calling the class method 
    (here, PatchEmbedding(x) will return this forward bellow)
    """
    def forward(self, x):
        """
        return the good patch embeeding required
        """
        x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P), P = (P_colxP_raw)
        x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
        return x

# If you want to try smth, you write bellow, and then, run the programmm will run whatever you want :)
if __name__ == "__main__":
    print("hello world")
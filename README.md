# PointRend-pytorch

This is an unofficial implementation of PointRend function. The paper can be find at <https://arxiv.org/pdf/1912.08193.pdf>

We only define a simple structure of PointRend function with out any segmentation structure.



# Instructions
Build a PointRend block:
```python
from point_rend import PointRend
#use random value
coarse_prediction = torch.rand([32, 3, 128, 128]).cuda()
fine_grained = torch.rand([32, 128, 128, 128]).cuda()

#you can get coarse_prediction and fine_grained by your segmentation
#from your_seg_model import seg_model
#coarse_prediction, fine_grained = seg_model(your_image_input)

net = PointRend(3,1000,[128,128],[128,128],[256,256],128)
output_point, output_mask = net(fine_grained, coarse_prediction)
```



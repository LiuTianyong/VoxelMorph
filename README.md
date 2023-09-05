### VoxelMorph

```text

适配数据集：OASAS IXI 
``` 

### train

```text
python train.py

```
```text
parser.add_argument('--img-list',default='train.txt', help='line-seperated list of training files')
train.txt 包含所有训练数据的路径
示例：
dataset/OASIS_split/train\OASIS_0001_0000.nii.gz
dataset/OASIS_split/train\OASIS_0003_0000.nii.gz
```
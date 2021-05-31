# Write a config yaml

## LFS配置文件的组成

LibFewSHot的配置文件采用了yaml格式的文件，同时也支持从命令行中读取一些全局配置的更改。我们预先定义了一个默认的配置`core/config/default.yaml`，并且用户可以将自定义的配置放在`config/`目录下，保存为`yaml`格式的文件。配置定义方法在解析时的先后关系是`default.yaml->config/->console`。后一个定义会覆盖前一个定义中名称相同的值。

`default.yaml`中设置的是小样本学习中的一些最基础的配置，仅仅依靠`default.yaml`是无法运行程序的，需要用户在`config/`目录下定义已经在LibFewShot中实现了的方法的配置才可以运行。

考虑到小样本方法有一些基本的例如`way, shot`或者`device id`这样的参数是经常需要改动的，每次都修改`yaml`文件的体验很差，LibFewShot支持在命令行中对一些简单的配置进行更改而不需要修改`yaml`文件。同样的，在不同的小样本学习方法的训练和测试过程中，有很多参数是相同的，为了简洁起见，我们将这些相同的参数包装到了一起，放到了`config/headers`目录下，这样就能够通过导入的方式简洁地编写自定义方法的`yaml`文件。

以下是`config/headers`目录下文件的构成。

- `data.yaml`：定义了训练所使用的数据的相关配置。
- `device.yaml`：定义了训练所使用的GPU的相关配置。
- `losses.yaml`：定义了训练所用的损失的相关配置。
- `misc.yaml`：定义了训练过程中一些杂项设置。
- `model.yaml`：定义了模型训练的相关配置。
- `optimizer.yaml`：定义了训练所使用的优化器的相关配置。

## 配置文件的设置

以下介绍配置文件中每部分代表的信息以及如何编写，并且将以DN4方法的配置给出示例。

### 数据设置

+ `data_root`：数据集存放的路径
+ `image_size`：训练中数据将会被resize的大小
+ `use_momery`：是否使用内存加速读取
+ `augment`：是否使用数据增强
+ `augment_times：support_set`使用数据增强的次数。增强多次相当于扩充了`support set`数据。
+ `augment_times_query：query_set`使用数据增强的次数。增强多次相当于扩充了`query set`数据。

```yaml
data_root: /data/miniImageNet--ravi
image_size: 84
use_memory: False
augment: True
augment_times: 1
augment_times_query: 1
```

### 模型设置

+ `backbone`：方法所使用的`backbone`信息。
  + `name`：使用的backbone的名称，需要与LibFewShot中实现的backbone的大小写一致。
  + `kwargs`：`backbone`初始化时用到的参数，必须保持名称与代码中的名称一致。
    + is_flatten：默认False，如果为`True`，则返回flatten后的特征向量。
    + avg_pool：默认False，如果为`True`，则返回`global average pooling`后的特征向量。
    + is_feature：默认False，如果为`True`，则返回`backbone`中每一个`block`的输出。

```yaml
backbone:
	name: Conv64FLeakyReLU
	kwargs:
		is_flatten: False
```

+ `classifier`：方法所使用的方法信息。
  + `name`：使用的方法的名称，需要与LibFewShot中实现的方法的名称一致。
  + `kwargs`：方法初始化时用到的参数，必须保持名称与代码中的名称一致。

```yaml
classifier:
	name: DN4
	kwargs:
		n_k: 3
```

### 训练设置

+ `epoch`：训练的`epoch`数。

+ `test_epoch`：测试的`epoch`数。

+ `parallel_part`：在前向传播时需要并行处理的部分，以列表形式给出这些部分在方法中的变量名称。

+ `pretrain_path`：预训练权重地址，需要指定到文件。训练开始时会检查该设置，如果不为空，将会把目标地址的预训练权重载入到当前训练的`backbone`中。

+ `resume`：如果设置为True，将从默认地址中读取训练状态并继续训练。

+ `way_num`：训练中的`way`的数量。

+ `shot_num`：训练中的`shot`的数量。

+ `query_num`：训练中的`query`的数量。

+ `test_way`：测试中的`way`的数量。如果未指定，将会把`way_num`赋值给`test_way`。

+ `test_shot`：测试中的`shot`的数量。如果未指定，将会把`shot_num`赋值给`test_way`。

+ `test_query`：测试中的`query`的数量。如果未指定，将会把`query_num`赋值给`test_way`。

+ `episode_size`：网络训练一次所使用的任务数量.

+ `batch_size`：`pre-training`的方法在`pre-train`时所使用的`batch size`。在其他种类的方法训练中是无用属性。

+ `train_episode`：训练中每个`epoch`的任务数量。

+ `test_episode`：测试中每个`epoch`的任务数量。

```yaml
epoch: 50
test_epoch: 5

parallel_part:
- emb_func
- dn4_layer

pretrain_path: ~
resume: False

way_num: 5
shot_num: 5
query_num: 15
test_way: ~ 
test_shot: ~
test_query: ~
episode_size: 1
# batch_size只在pre-train中起效
batch_size: 128	
train_episode: 10000
test_episode: 1000
```

### 优化器设置

+ `optimizer`：训练时使用的优化器信息。
  + `name`：优化器名称，当前仅支持`pytorch`提供的所有优化器。
  + `kwargs`：需要传入优化器的参数，名称需要与pytorch优化器所需要的参数名称相同。
  + `other`：当前仅支持单独指定方法中的每一部分所使用的学习率，名称需要与方法中所使用的变量名相同。

```yaml
optimizer:
	name: Adam
	kwargs:
		lr: 0.01
	other:
		emb_func: 0.01
		#演示用，dn4分类时没有可训练参数
		dnf_layer: 0.001
```

+ `lr_scheduler`：训练时使用的学习率调整策略，当前仅支持`pytorch`提供的所有学习率调整策略。
  + `name`：学习率调整策略名称。
  + `kwargs`：除了优化器之外的，需要传入学习率调整策略的参数名称，需要与`pytorch`学习率调整策略所需要的参数名称相同。

```yaml
lr_scheduler:
  name: StepLR
  kwargs:
    gamma: 0.5
    step_size: 10
```

### 损失设置

+ `loss`：训练使用的损失函数信息。
  + `name`：损失函数名称，暂时只支持`pytorch`提供的所有损失函数。
  + `kwargs`：损失函数需要的参数，与`pytorch`中损失函数需要的参数相同

```yaml
loss:
	name: CrossEntropy
	kwargs: ~
```

### 硬件设置

+ `device_ids`：训练可以用到的`gpu`的编号，与`nvidia-smi`命令显示的编号相同。
+ `n_gpu`：训练使用并行训练的`gpu`个数，如果为`1`则不适用并行训练。
+ `deterministic`：是否开启`torch.backend.cudnn.benchmark`以及`torch.backend.cudnn.deterministic`以及是否使训练随机种子确定。
+ `seed`：训练时`numpy`，`torch`，`cuda`使用的种子点。==**移到杂项里面？**==

```yaml
device_ids: 0,1,2,3,4,5,6,7
n_gpu: 4
seed: 0
deterministic: False
```

### 杂项设置

+ log_name：如果为空，即使用自动生成的`classifier.name-data_root-backbone-way_num-shot_num`文件目录。
+ log_level：训练中日志输出等级。
+ log_interval：日志输出间隔的任务数目。
+ result_root：结果存放的根目录
+ save_interval：权重保存的epoch间隔
+ `save_part`：方法中需要保存的部分在方法中的变量名称。这些名称的变量会在模型保存时单独对这些变量保存一次。需要保存的部分在`save_part`下以列表的形式给出。

```yaml
log_name: ~
log_level: info
log_interval: 100
result_root: ./results
save_interval: 10
save_part:
	- emb_func
	- dn4_layer
```
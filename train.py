## main.py文件
import torch
import argparse
import torch.nn as nn
import torchvision

# 新增1:依赖
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 新增2：从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数，后面还会介绍。所以不用考虑太多，照着抄就是了。
#       argparse是python的一个系统库，用来处理命令行调用，如果不熟悉，可以稍微百度一下，很简单！
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# 新增3：DDP backend初始化
#   a.根据local_rank来设定当前使用哪块GPU
# torch.cuda.set_device(local_rank)
torch.cuda.set_device('cuda:'+local_rank)
#   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
dist.init_process_group(backend='nccl')

# 新增4：定义并把模型放置到单独的GPU上，需要在调用`model=DDP(model)`前做哦。
#       如果要加载模型，也必须在这里做哦。
# device = torch.device("cuda", local_rank)
device = torch.device("cuda:"+local_rank)
model = nn.Linear(10, 10).to(device)
# 可能的load模型...

# 新增5：之后才是初始化DDP模型
model = DDP(model, device_ids=[int(local_rank)], output_device=int(local_rank))

my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True)
# 新增1：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！
#       sampler的原理，后面也会介绍。
train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
# 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=batch_size, sampler=train_sampler)


for epoch in range(num_epochs):
    # 新增2：设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        prediction = model(data)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        optimizer.step()
# 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
#    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
# 2. 我只需要在进程0上保存一次就行了，避免多次保存重复的东西。
if dist.get_rank() == 0:
    torch.save(model.module, "saved_model.ckpt")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
class MLP(nn.Module):
    def __init__(self, nIn, num_class):
        super(MLP, self).__init__()
        self.bn0 = nn.BatchNorm2d(54)
        self.conv1 = nn.Conv2d(nIn, 256, kernel_size=1, padding=0, bias=False) 
        self.bn1 = nn.BatchNorm2d(259)
        self.conv2 = nn.Conv2d(259, 128, kernel_size=1, padding=0, bias=False) 
        self.bn2 = nn.BatchNorm2d(131)
        self.conv3 = nn.Conv2d(131, num_class, kernel_size=1, padding=0, bias=False) 
        #self.ReLU = nn.ReLU(inplace=True) 
        self.ReLU = nn.PReLU()
        self._init_weight()
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
   
    def forward(self, x):
        #x:batch,N,channel
        #model_input:batch,channel,N,1
        model_input = x.unsqueeze(2).permute([0,3,1,2])
        #model_input = self.bn0(model_input_pre)
        print('model_input:')
        print(model_input[0,:,0,0])
        print(model_input.shape)
        layer1 = self.conv1(model_input)
        print('layer1')
        print(layer1.shape)
        
        
        #layer1_output = self.ReLU(self.bn1(torch.cat((layer1, model_input[:,-3:,:,:]),1)))
        #layer2 = self.conv2(layer1_output)
        #layer2_output = self.ReLU(self.bn2(torch.cat((layer2, model_input[:,-3:,:,:]),1)))
        #batch,num_class,N
        #last_conv = torch.squeeze(self.conv3(layer2_output),3)
        
        layer1_output = self.ReLU(layer1)
        layer2 = self.conv2(torch.cat((layer1_output, model_input[:,-3:,:,:]),1))
        layer2_output = self.ReLU(layer2)
        last_conv = torch.squeeze(self.conv3(torch.cat((layer2_output, model_input[:,-3:,:,:]),1)),3)
        print('layer1_output:')
        print(layer1_output[0,:,0,0])
        print(layer1_output.shape)
        print('layer2_output:')
        print(layer2_output[0,:,0,0])
        print(layer2_output.shape)
        #batch,num_class,N

        return last_conv


class PointRend(nn.Module):
    def __init__(self, num_classes,N,coarse_size,fine_size,img_size,fine_channels,is_training=True,k=3,belta=0.75):
        super(PointRend, self).__init__()
        self.num_classes = num_classes
        self.N = N
        self.is_training = is_training
        self.coarse_size = coarse_size
        self.img_size = img_size
        self.fine_size = fine_size
        self.k = k
        self.belta = belta
        self.mlp = MLP(fine_channels+3, num_classes)
        if self.is_training:
            self.up1 = nn.Upsample(scale_factor=img_size[0]/coarse_size[0], mode='bilinear')
            self.up2 = nn.Upsample(scale_factor=img_size[0]/fine_size[0], mode='bilinear')
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, fine_grained, coarse_pre):
        if self.is_training:
            #batch,c,h,w
            up_coarse_o = self.up1(coarse_pre)
            up_coarse_softmax = F.softmax(up_coarse_o, dim=1)
            up_fine_o = self.up2(fine_grained)

            #batch,h,w,c
            up_coarse = up_coarse_softmax.permute([0,2,3,1])
            up_coarse_ori_fea = up_coarse_o.permute([0,2,3,1])
            up_fine = up_fine_o.permute([0,2,3,1])
            #选点方法一
            num_over = self.k * self.N
            step1_n = int(self.N * self.belta)
            step2_n = self.N - step1_n
            #torch产生整形随机数作为坐标，组合新的feature map，计算度量距离，计算topk的index，采样剩余index，组合特征向量
            #index = (torch.rand(num_over, 2)*self.img_size[0]).int()
            
            random_1d_index = torch.randperm(self.img_size[0]*self.img_size[1])
            h_index = (random_1d_index/self.img_size[1]).int()
            w_index = (random_1d_index%self.img_size[1]).int()
            all_index = torch.cat((h_index.reshape(-1,1),w_index.reshape(-1,1)),1)
            index = all_index[:num_over,:].cuda()
            pre_list = []
            for i in range(num_over):
                #batch,3
                one_shot = up_coarse[:,index[i][0],index[i][1],:]
                #batch,1,3
                one_shot_ext = one_shot.unsqueeze(1)
                pre_list.append(one_shot_ext)
            #batch,num_over,3
            over_pre = torch.cat(pre_list, 1)
            #计算置信度最高的两个值,top2:n,num_over,2
            top2,_ = torch.topk(over_pre,2,2)
            #batch,num_over
            certain_score = top2[:,:,0] - top2[:,:,1]
            uncertain_score = 1-certain_score
            
            #batch, step1_n 代表over_pre中step1_n个uncertain点
            
            _,uncertain_index = torch.topk(uncertain_score,step1_n,1)
            uncertain_index_reshape = uncertain_index.reshape([-1,1])
            uncertain_index_repeat = uncertain_index_reshape.repeat(1,2)
            uncertain_points_unbatch = torch.gather(index, dim=0, index=uncertain_index_repeat)
            
            #batch,step1_n,2 uncertain点在原图中的坐标
            uncertain_points = uncertain_points_unbatch.reshape([-1,step1_n,2])
            #covrage_points = (torch.rand(step2_n, 2)*self.img_size[0]).int()
        
            covrage_points = all_index[-step2_n:,:].unsqueeze(0).repeat([uncertain_points.shape[0],1,1])
            #batch,N,2 选出的所有点
            all_select_point = torch.cat([uncertain_points.int().cuda(),covrage_points.cuda()],1)

            #组合特征向量
            batch_id = torch.arange(all_select_point.shape[0])
            #1,batch*N
            batch_index = batch_id.reshape([-1,1]).repeat([1,self.N]).reshape([1,-1]).int()
            #2,batch*N
            all_select_point_prtmute = all_select_point.permute([2,0,1]).reshape(2,-1)
            #3,batch*N
            mask_pre_data = torch.cat((batch_index.cuda(),all_select_point_prtmute.cuda()),0)
            val = torch.tensor([1]*mask_pre_data.shape[1]).long()
            #batch,h,w
            mask = torch.sparse.FloatTensor(mask_pre_data.long().cuda(), val.long().cuda(), torch.Size([all_select_point.shape[0],self.img_size[0],self.img_size[1]])).to_dense()
            #mask = torch.where(mask_tmp>1,torch.tensor([1]).int(),mask_tmp)
            select_coarse = torch.masked_select(up_coarse_ori_fea,mask[:,:,:,None].byte()) 
            select_fine = torch.masked_select(up_fine,mask[:,:,:,None].byte())
            
            #batch,N,channel+3
            select_feature = torch.cat([select_fine, select_coarse], -1).reshape([all_select_point.shape[0],self.N, -1])
            #batch,num_class,N

            out = self.mlp(select_feature) 
            pre = F.softmax(out, dim=1)
        
            #debug
            print(pre[0])
            ori_pre = select_coarse.reshape([all_select_point.shape[0],self.N, -1]).permute([0,2,1])
            print(F.softmax(ori_pre, dim=1)[0])    
        #return select_coarse.reshape([all_select_point.shape[0],self.N, -1]).permute([0,2,1]), mask
        return out, mask

if __name__ == '__main__':
    net = PointRend(3,1000,[128,128],[128,128],[256,256],128)
    net = net.cuda()        
    for i in range(5):
      start = time.time()
      coarse_prediction=torch.rand([32, 3, 128, 128]).cuda()
      fine_grained = torch.rand([32, 128, 128, 128]).cuda()
      out,mask = net(up_fine, up_coarse)
      end = time.time()
      print(end-start)
      print(out.shape)
      print(mask.shape)
    torch.save(net, 'model.pkl')
      
 

import argparse
from src.npi_hgnn.dataset_classes import NcRNA_Protein_Subgraph
import torch
import os.path as osp
import os
import time
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from src.npi_hgnn.model_classes import Model_1,Model_2
from src.npi_hgnn.methods import generate_dataset_path,generate_log_path,generate_model_path
def generate_result_path(dataset,nodeVecType,subgraph_type,samplingType):
    path=f'../../data/result/{dataset}'
    if subgraph_type==0:
        path=f'{path}/subgraph3'
    elif subgraph_type==1:
        path=f'{path}/subgraph2'
    elif subgraph_type==2:
        path=f'{path}/subgraph1'
    if nodeVecType==0:
        path=f'{path}/frequency'
    elif nodeVecType==1:
        path=f'{path}/pyfeat'
    elif nodeVecType==2:
        path=f'{path}/no_sequence'
    elif nodeVecType==3:
        path=f'{path}/only_frequency'
    if samplingType==0:
        path=f'{path}/random'
    elif samplingType==1:
        path=f'{path}/fire'
    elif samplingType==2:
        path=f'{path}/reliable'
    elif samplingType==3:
        path=f'{path}/random_reliable'
    return path
def generate_pred_path(dataset,nodeVecType,subgraph_type,samplingType):
    path=f'../../data/pred/{dataset}'
    if subgraph_type==0:
        path=f'{path}/subgraph3'
    elif subgraph_type==1:
        path=f'{path}/subgraph2'
    elif subgraph_type==2:
        path=f'{path}/subgraph1'
    if nodeVecType==0:
        path=f'{path}/frequency'
    elif nodeVecType==1:
        path=f'{path}/pyfeat'
    elif nodeVecType==2:
        path=f'{path}/no_sequence'
    elif nodeVecType==3:
        path=f'{path}/only_frequency'
    if samplingType==0:
        path=f'{path}/random'
    elif samplingType==1:
        path=f'{path}/fire'
    elif samplingType==2:
        path=f'{path}/reliable'
    elif samplingType==3:
        path=f'{path}/random_reliable'
    return path
def parse_args():
    parser = argparse.ArgumentParser(description="train.")
    # NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
    parser.add_argument('--dataset', default="NPHN-Mus", help='dataset name')
    # 0:使用kmer频率和n2v作为节点特征 1:使用pyfeat和n2v作为节点特征 2:只使用n2v作为节点特征
    parser.add_argument('--nodeVecType', default=0,type=int, help='node vector type')
    # 0 rpin、ppin、rrsn组成的异构网络中提取的一阶封闭子图；1使用在rpin二部图上提取的一阶子图，并在其基础上加上ppin和rrsn；2使用rpin上的一阶封闭子图
    parser.add_argument('--subgraph_type', default=1, type=int, help='if use complete subgraph')
    # 0：表示随机采样 1：表示使用fire采样 2：表示使用reliable采样 3、同时使用improved和随机采样
    parser.add_argument('--samplingType', default=3,type=int, help='how num of fold is this')
    parser.add_argument('--fold', default=0,type=int,help='which fold is this')
    parser.add_argument('--epochNumber', default=100, type=int, help='number of training epoch')
    parser.add_argument('--initialLearningRate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--l2WeightDecay', default=0.001, type=float, help='L2 weight')
    parser.add_argument('--batchSize', default=32, type=int, help='batch size')
    parser.add_argument('--num_bases', default=2, type=int, help='Number of bases used for basis-decomposition')
    parser.add_argument('--num_relations', default=3, type=int, help='Number of edges')
    parser.add_argument('--model_code', default=2, type=int, help='model code') # 1 2
    parser.add_argument('--cuda_code', default=0, type=int, help='cuda code')
    parser.add_argument('--droupout_ratio', default=0.5, type=float, help='droupout_ratio')
    parser.add_argument('--gamma', default=0.95, type=float, help='gamma')
    return parser.parse_args()
def write_log(path,value):
    now_time = time.localtime() #获取当前日期和时间
    time_format = '%Y-%m-%d %H:%M:%S' #指定日期和时间格式
    time_put = time.strftime(time_format,now_time) #格式化时间，时间变成YYYY-MM-DD HH:MI:SS
    log_file = open(path,'a') #这里用追加模式，如果文件不存在的话会自动创建
    write_value = '%s %s' %(time_put,value)
    log_file.write(write_value)
    log_file.close()
def write_result(path,value):
    log_file = open(path,'a') #这里用追加模式，如果文件不存在的话会自动创建
    write_value = f'[{value}],\n'
    log_file.write(write_value)
    log_file.close()
def dataset_analysis(dataset):
    dict_label_dataNumber = {}
    for data in dataset:
        label = int(data.y)
        if label not in dict_label_dataNumber:
            dict_label_dataNumber[label] = 1
        else:
            dict_label_dataNumber[label] = dict_label_dataNumber[label] + 1
    print(dict_label_dataNumber)
def train(model,train_loader):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        #assert torch.isnan(loss).sum() == 0, print(loss)
        loss.backward() #计算梯度
        loss_all += data.num_graphs * loss.item()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
        optimizer.step() #更新参数
    return loss_all / len(train_dataset)


def Accuracy_Precision_Sensitivity_Specificity_MCC(model, loader, device,log_path):
    model.eval()
    TP = 0 # TP：被模型预测为正类的正样本
    TN = 0 # TN：被模型预测为负类的负样本
    FP = 0 # FP：被模型预测为正类的负样本
    FN = 0 # FN：被模型预测为负类的正样本
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        for index in range(len(pred)):
            if pred[index] == 1 and data.y[index] == 1:
                TP += 1
            elif pred[index] == 1 and data.y[index] == 0:
                FP += 1
            elif pred[index] == 0 and data.y[index] == 1:
                FN += 1
            else:
                TN += 1
    output = 'TP: %d, FN: %d, TN: %d, FP: %d' % (TP, FN, TN, FP)
    print(output)
    write_log(log_path,output + '\n')
    if (TP + TN + FP + FN) != 0:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    else:
        Accuracy = 0
    if (TP + FP) != 0:
        Precision = (TP) / (TP + FP)
    else:
        Precision = 0
    if (TP + FN) != 0:
        Sensitivity = (TP) / (TP + FN)
    else:
        Sensitivity = 0
    if (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) != 0:
        MCC = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    else:
        MCC = 0
    if (FP + TN) != 0:
        Specificity = TN / (FP + TN)
    else:
        Specificity = 0
    return Accuracy, Precision, Sensitivity, Specificity, MCC
def Accuracy_Precision_Sensitivity_Specificity_MCC_Pred(model, loader, device):
    model.eval()
    TP = 0 # TP：被模型预测为正类的正样本
    TN = 0 # TN：被模型预测为负类的负样本
    FP = 0 # FP：被模型预测为正类的负样本
    FN = 0 # FN：被模型预测为负类的正样本
    pred = []

    for data in loader:
        data = data.to(device)
        pred_label = model(data).max(dim=1)[1]
        for index in range(len(pred_label)):
            if pred_label[index] == 1 and data.y[index] == 1:
                TP += 1
            elif pred_label[index] == 1 and data.y[index] == 0:
                FP += 1
            elif pred_label[index] == 0 and data.y[index] == 1:
                FN += 1
            else:
                TN += 1
        pred_prob=torch.exp(model(data))[:,1].tolist() #预测为正例的概率值
        pred.extend([(value[0], value[1],data.y.tolist()[i], pred_prob[i]) for i, value in enumerate(data.target_link)])
    output = 'TP: %d, FN: %d, TN: %d, FP: %d' % (TP, FN, TN, FP)
    print(output)
    if (TP + TN + FP + FN) != 0:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    else:
        Accuracy = 0
    if (TP + FP) != 0:
        Precision = (TP) / (TP + FP)
    else:
        Precision = 0
    if (TP + FN) != 0:
        Sensitivity = (TP) / (TP + FN)
    else:
        Sensitivity = 0
    if (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) != 0:
        MCC = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    else:
        MCC = 0
    if (FP + TN) != 0:
        Specificity = TN / (FP + TN)
    else:
        Specificity = 0
    return Accuracy, Precision, Sensitivity, Specificity, MCC,pred
if __name__ == "__main__":
    #参数
    args = parse_args()
    print(args)
    #训练集路径
    dataset_train_path=generate_dataset_path(args.dataset,args.fold,args.nodeVecType,args.subgraph_type,args.samplingType,'train')
    #测试集路径
    dataset_test_path=generate_dataset_path(args.dataset,args.fold,args.nodeVecType,args.subgraph_type,args.samplingType,'test')
    #生成负样本随机的测试集
    #dataset_random_test_path=generate_dataset_path(args.dataset,args.fold,args.nodeVecType,args.subgraph_type,args.samplingType,'random_test')
    train_dataset = NcRNA_Protein_Subgraph(dataset_train_path)
    test_dataset = NcRNA_Protein_Subgraph(dataset_test_path)
    #random_test_dataset = NcRNA_Protein_Subgraph(dataset_random_test_path)
    # 打乱数据集
    print('shuffle dataset\n')
    train_dataset = train_dataset.shuffle()
    test_dataset = test_dataset.shuffle()
    #random_test_dataset=random_test_dataset.shuffle()
    #选择CPU或CUDA
    device = torch.device(f"cuda:{args.cuda_code}" if torch.cuda.is_available() else "cpu")
    # 准备日志
    log_saving_path=generate_log_path(args.dataset,args.model_code,args.nodeVecType,args.subgraph_type,args.samplingType)
    print(log_saving_path)
    if not osp.exists(log_saving_path):
        print(f'创建日志文件夹：{log_saving_path}')
        os.makedirs(log_saving_path)
     # 迭代次数
    num_of_epoch = args.epochNumber
    # 学习率
    LR = args.initialLearningRate
    # L2正则化系数
    L2_weight_decay = args.l2WeightDecay
    # 日志基本信息写入
    log_path = log_saving_path + f'/fold_{args.fold}.txt'
    if (os.path.exists(log_path)):
        os.remove(log_path)
    write_log(log_path,f'dataset：{args.dataset}\n')
    write_log(log_path,f'training dataset path: {dataset_train_path}\n')
    write_log(log_path,f'testing dataset path: {dataset_test_path}\n')
    write_log(log_path,f'number of eopch ：{num_of_epoch}\n')
    write_log(log_path,f'learn rate：initial = {LR}，whenever loss increases, multiply by 0.95\n')
    write_log(log_path,f'L2 weight decay = {L2_weight_decay}\n')
    # 记录启起始时间
    start_time = time.time()
    model_saving_path=generate_model_path(args.dataset,args.model_code,args.nodeVecType,args.subgraph_type,args.fold,args.samplingType)
    if not osp.exists(model_saving_path):
        print(f'创建保存模型文件夹：{model_saving_path}')
        os.makedirs(model_saving_path)
    #检查特征维度
    if(train_dataset.num_node_features != test_dataset.num_node_features):
        raise Exception('训练集和测试集的结点特征维度不一致')
    #if(train_dataset.num_node_features != random_test_dataset.num_node_features):
    #    raise Exception('训练集和负样本随机的测试集的结点特征维度不一致')
    #创建模型
    model = globals()[f'Model_{args.model_code}'](train_dataset.num_node_features, args.num_relations, args.num_bases,2).to(device)

    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2_weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)
    # 训练集和测试集
    print(f'number of samples in training dataset：{str(len(train_dataset))}\n')
    print(f'number of samples in testing dataset：{str(len(test_dataset))}\n')
    #print(f'number of samples in random testing dataset：{str(len(random_test_dataset))}\n')
    write_log(log_path,f'number of samples in training dataset：{str(len(train_dataset))}\n')
    write_log(log_path, f'number of samples in testing dataset：{str(len(test_dataset))}\n')
    #write_log(log_path, f'number of samples in random testing dataset：{str(len(random_test_dataset))}\n')
    print('training dataset')
    dataset_analysis(train_dataset)
    print('testing dataset')
    dataset_analysis(test_dataset)
    #print('random testing dataset')
    #dataset_analysis(random_test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batchSize)
    test_loader = DataLoader(test_dataset, batch_size=args.batchSize)
    #random_test_loader = DataLoader(random_test_dataset, batch_size=args.batchSize)
    MCC_max = -1
    epoch_MCC_max = 0
    ACC_MCC_max = 0
    Pre_MCC_max = 0
    Sen_MCC_max = 0
    Spe_MCC_max = 0
    # random_MCC_max = -1
    # random_epoch_MCC_max = 0
    # random_ACC_MCC_max = 0
    # random_Pre_MCC_max = 0
    # random_Sen_MCC_max = 0
    # random_Spe_MCC_max = 0
    # 训练开始
    loss_last = float('inf')
    early_stop=0
    for epoch in range(num_of_epoch):
        loss = train(model,train_loader)
        # loss增大时,降低学习率
        if loss > loss_last:
            scheduler.step()
        loss_last = loss
        # 训练中评价模型，监视训练过程中的模型变化, 并且写入文件
        if (epoch + 1) % 1 == 0 and epoch != num_of_epoch - 1:
            # 用Accuracy, Precision, Sensitivity, MCC评价模型
            Accuracy, Precision, Sensitivity, Specificity, MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model, train_loader, device,log_path)
            output = 'Epoch: {:03d}, training dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(epoch + 1, Accuracy, Precision, Sensitivity, Specificity, MCC)
            print(output)
            write_log(log_path,output + '\n')
            #生成负样本随机的测试集的性能评估
            # random_Accuracy, random_Precision, random_Sensitivity, random_Specificity,random_MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model, random_test_loader, device)
            # output = 'Epoch: {:03d}, random testing dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(epoch + 1, random_Accuracy, random_Precision, random_Sensitivity, random_Specificity, random_MCC)
            # print(output)
            # write_log(log_path,output + '\n')
            # if random_MCC > random_MCC_max:
            #     random_MCC_max = random_MCC
            #     random_epoch_MCC_max = epoch+1
            #     random_ACC_MCC_max = random_Accuracy
            #     random_Pre_MCC_max = random_Precision
            #     random_Sen_MCC_max = random_Sensitivity
            #     random_Spe_MCC_max = random_Specificity
            #测试集性能评估
            Accuracy, Precision, Sensitivity, Specificity, MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model, test_loader, device,log_path)
            output = 'Epoch: {:03d}, testing dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(epoch + 1, Accuracy, Precision, Sensitivity, Specificity, MCC)
            print(output)
            write_log(log_path,output + '\n')
            if MCC > MCC_max:
                MCC_max = MCC
                epoch_MCC_max = epoch+1
                ACC_MCC_max = Accuracy
                Pre_MCC_max = Precision
                Sen_MCC_max = Sensitivity
                Spe_MCC_max = Specificity
                early_stop=0
                # 保存模型
                network_model_path = model_saving_path + f'/{epoch + 1}'
                torch.save(model.state_dict(), network_model_path)
            else:
                early_stop+=1
            if early_stop>20:
                break
    # 训练结束，评价模型，并且把结果写入文件
    Accuracy, Precision, Sensitivity, Specificity, MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model, train_loader, device,log_path)
    output = 'Epoch: {:03d}, training dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(epoch + 1, Accuracy, Precision, Sensitivity, Specificity, MCC)
    print(output)
    write_log(log_path,output + '\n')
    # 随机生成负样本的测试集的性能评估
    # random_Accuracy, random_Precision, random_Sensitivity, random_Specificity, random_MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model, random_test_loader, device)
    # output = 'Epoch: {:03d}, random testing dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(epoch + 1, random_Accuracy, random_Precision, random_Sensitivity, random_Specificity, random_MCC)
    # print(output)
    # write_log(log_path, output + '\n')
    # if random_MCC > random_MCC_max:
    #     random_MCC_max = random_MCC
    #     random_epoch_MCC_max = epoch + 1
    #     random_ACC_MCC_max = random_Accuracy
    #     random_Pre_MCC_max = random_Precision
    #     random_Sen_MCC_max = random_Sensitivity
    #     random_Spe_MCC_max = random_Specificity
    # 测试集性能评估
    Accuracy, Precision, Sensitivity, Specificity, MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model, test_loader, device,log_path)
    output = 'Epoch: {:03d}, testing dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(epoch + 1, Accuracy, Precision, Sensitivity, Specificity, MCC)
    print(output)
    write_log(log_path,output + '\n')
    if MCC > MCC_max:
        MCC_max = MCC
        epoch_MCC_max = args.epochNumber
        ACC_MCC_max = Accuracy
        Pre_MCC_max = Precision
        Sen_MCC_max = Sensitivity
        Spe_MCC_max = Specificity
        # 保存模型
        network_model_path = model_saving_path + f'/{epoch + 1}'
        torch.save(model.state_dict(), network_model_path)
    write_log(log_path,'\n')
    output = f'测试集MCC最大的时候的性能：'
    print(output)
    write_log(log_path,output + '\n')
    output = f'epoch: {epoch_MCC_max}, ACC: {ACC_MCC_max}, Pre: {Pre_MCC_max}, Sen: {Sen_MCC_max}, Spe: {Spe_MCC_max}, MCC: {MCC_max}'
    print(output)
    write_log(log_path,output + '\n')
    # output = f'随机生成负样本的测试集MCC最大的时候的性能：'
    # print(output)
    # write_log(log_path,output + '\n')
    # output = f'epoch: {random_epoch_MCC_max}, ACC: {random_ACC_MCC_max}, Pre: {random_Pre_MCC_max}, Sen: {random_Sen_MCC_max}, Spe: {random_Spe_MCC_max}, MCC: {random_MCC_max}'
    # print(output)
    # write_log(log_path,output + '\n')
    # 完毕
    end_time = time.time()
    print('Time consuming:', end_time - start_time)
    write_log(log_path,'Time consuming:' + str(end_time - start_time) + '\n')
    output = f'{ACC_MCC_max},{Pre_MCC_max},{Sen_MCC_max},{Spe_MCC_max},{MCC_max}'
    write_log(log_path, output + '\n')
    path = generate_result_path(args.dataset,args.nodeVecType,args.subgraph_type,args.samplingType)
    if not osp.exists(path):
        print(f'创建保存结果文件夹：{path}')
        os.makedirs(path)
    write_result( f'{path}/{args.model_code}.txt',output)
    # 保存测试集概率值
    path = generate_pred_path(args.dataset,args.nodeVecType,args.subgraph_type,args.samplingType)
    if not osp.exists(path):
        print(f'创建保存模型文件夹：{path}')
        os.makedirs(path)
    model.load_state_dict(torch.load(model_saving_path + f'/{epoch_MCC_max}'))  # 加载模型
    Accuracy, Precision, Sensitivity, Specificity, MCC,train = Accuracy_Precision_Sensitivity_Specificity_MCC_Pred(model,test_loader,device)
    with open(f'{path}/{args.dataset}_model{args.model_code}.txt','a') as f:
        #f.write(f'RNA\tProtein\tOrigin Lable\tPredict Prob\n')
        for i in train:
            f.write(f'{i[0]}\t{i[1]}\t{i[2]}\t{i[3]}\n')
    print('\nexit\n')
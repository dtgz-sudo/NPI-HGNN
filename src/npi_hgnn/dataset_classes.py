from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
import networkx as nx
from src.npi_hgnn.methods import get_subgraph

class NcRNA_Protein_Subgraph(InMemoryDataset):
    '''
    生成RNA和蛋白质的1跳子图
    '''
    def __init__(self, root,rpin=None, ppin=None, rrsn=None, dict_node_name_vec=None,interaction_list=None,y=None, subgraph_type=0,transform=None, pre_transform=None):
        self.rpin=rpin
        self.ppin = ppin
        self.rrsn = rrsn
        self.dict_node_name_vec=dict_node_name_vec
        self.interaction_list = interaction_list
        self.y=y
        self.subgraph_type=subgraph_type
        self.sum_node = 0.0
        super(NcRNA_Protein_Subgraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        if self.interaction_list != None:
            num_data = len(self.interaction_list)
            print(f'the number of samples:{num_data}')
            data_list = []
            count = 0
            for interaction in self.interaction_list:
                if self.subgraph_type==0:
                    data = self.local_subgraph_generation_with_rphn(interaction,self.y[count])
                elif self.subgraph_type==1:
                    data = self.local_subgraph_generation_with_rpin(interaction, self.y[count])
                else:
                    data = self.local_subgraph_generation_only_rpin(interaction, self.y[count])
                data_list.append(data)
                count = count + 1
                #if count % 100 == 0:
                    #print(f'{count}/{num_data}')
                    #print(f'average node number = {self.sum_node / count}')  # 打印出平均每个封闭子图的节点数
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            print(f'average node number = {self.sum_node / count}')
            torch.save((data, slices), self.processed_paths[0])

    # 下面四个函数用来构建local subgraph
    def local_subgraph_generation_with_rphn(self, interaction,y):
        x = []
        edge_index = [[], []]
        edge_type_list=[] #记录边的类型 rna-protein=0 protein-protein=1 rna-rna=2
        '''
        查找rna的一阶邻居
        '''
        try:
            proteins = set(nx.neighbors(self.rpin, interaction[0]))
        except:
            #print(f'{interaction[0]} not exist {y}')
            proteins=set()
        if self.rrsn is not None:
            try:
                rnas=set(nx.neighbors(self.rrsn,interaction[0]))
            except:
                rnas=set()
        else:
            rnas=set()
        # 构造edge_list
        dict_nodeName_subgraphNodeSerialNumber={}
        subgraph_serial_number=0
        rna_subgraphSerialNumber=subgraph_serial_number
        dict_nodeName_subgraphNodeSerialNumber[interaction[0]] = rna_subgraphSerialNumber
        for protein in proteins:
            if protein==interaction[1]:
                continue
            if protein in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[protein]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[protein] = node2_subgraphSerialNumber
            edge_index[0].append(rna_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(0) #给这条边设置类型为rna-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(rna_subgraphSerialNumber)
            edge_type_list.append(0)  # 给这条边设置类型为protein-rna

        for rna in rnas:
            if rna==interaction[0]:
                continue
            if rna in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[rna]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[rna] = node2_subgraphSerialNumber
            edge_index[0].append(rna_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(2) #给这条边设置类型为rna-rna
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(rna_subgraphSerialNumber)
            edge_type_list.append(2)  # 给这条边设置类型为rna-rna
        '''
        查找protein的一阶邻居
        '''
        try:
            rnas=set(nx.neighbors(self.rpin,interaction[1]))
        except:
            rnas=set()
        try:
            proteins=set(nx.neighbors(self.ppin,interaction[1]))
        except:
            proteins=set()
        # 构造edge_list
        subgraph_serial_number += 1
        protein_subgraphSerialNumber = subgraph_serial_number
        dict_nodeName_subgraphNodeSerialNumber[interaction[1]] = protein_subgraphSerialNumber
        for protein in proteins:
            if protein==interaction[1]:
                continue
            if protein in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[protein]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[protein] = node2_subgraphSerialNumber
            edge_index[0].append(protein_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(1) #给这条边设置类型为protein-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(protein_subgraphSerialNumber)
            edge_type_list.append(1)  # 给这条边设置类型为protein-protein

        for rna in rnas:
            if rna==interaction[0]:
                continue
            if rna in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[rna]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[rna] = node2_subgraphSerialNumber
            edge_index[0].append(protein_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(0) #给这条边设置类型为rna-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(protein_subgraphSerialNumber)
            edge_type_list.append(0)  # 给这条边设置类型为protein-rna
        #构造x
        dict_subgraphNodeSerialNumber_nodeName=dict(zip(dict_nodeName_subgraphNodeSerialNumber.values(),dict_nodeName_subgraphNodeSerialNumber.keys()))
        for i in range(len(dict_subgraphNodeSerialNumber_nodeName)):
            vector = []
            vector.append(0)
            vector.extend(self.dict_node_name_vec[dict_subgraphNodeSerialNumber_nodeName[i]])
            x.append(vector)
        edge_index[0].append(rna_subgraphSerialNumber)
        edge_index[1].append(protein_subgraphSerialNumber)
        edge_type_list.append(0)#给这条边设置类型为rna-protein
        edge_index[0].append(protein_subgraphSerialNumber)
        edge_index[1].append(rna_subgraphSerialNumber)
        edge_type_list.append(0)# 给这条边设置类型为protein-rna
        #print(edge_type_list)
        # y记录这个interaction的真假
        if y == 1:
            y = [1]
        else:
            y = [0]
        self.sum_node += len(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_type_list=torch.tensor(edge_type_list, dtype=torch.long)
        # x为一个封闭子图的特征矩阵 y表示这个封闭子图是正样本还是负样本 edge_index为这个封闭子图的邻接矩阵:[[],[]]
        data = Data(x=x, y=y, edge_index=edge_index,edge_attr=edge_type_list,target_link=interaction)
        return data
    def local_subgraph_generation_with_all(self, interaction,y):
        x = []
        edge_index = [[], []]
        edge_type_list=[] #记录边的类型 rna-protein=0 protein-protein=1 rna-rna=2
        #记录封闭子图所有的rna和蛋白质
        all_rnas=set()
        all_rnas.add(interaction[0])
        all_proteins=set()
        all_proteins.add(interaction[1])
        '''
        查找rna的一阶邻居
        '''
        try:
            proteins = set(nx.neighbors(self.rpin, interaction[0]))
        except:
            #print(f'{interaction[0]} not exist {y}')
            proteins=set()
        if self.rrsn is not None:
            try:
                rnas=set(nx.neighbors(self.rrsn,interaction[0]))
            except:
                rnas=set()
        else:
            rnas=set()
        all_rnas.update(rnas)
        all_proteins.update(proteins)
        # 构造edge_list
        dict_nodeName_subgraphNodeSerialNumber={}
        subgraph_serial_number=0
        rna_subgraphSerialNumber=subgraph_serial_number
        dict_nodeName_subgraphNodeSerialNumber[interaction[0]] = rna_subgraphSerialNumber
        for protein in proteins:
            if protein==interaction[1]:
                continue
            if protein in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[protein]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[protein] = node2_subgraphSerialNumber
            edge_index[0].append(rna_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(0) #给这条边设置类型为rna-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(rna_subgraphSerialNumber)
            edge_type_list.append(0)  # 给这条边设置类型为protein-rna

        for rna in rnas:
            if rna==interaction[0]:
                continue
            if rna in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[rna]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[rna] = node2_subgraphSerialNumber
            edge_index[0].append(rna_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(2) #给这条边设置类型为rna-rna
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(rna_subgraphSerialNumber)
            edge_type_list.append(2)  # 给这条边设置类型为rna-rna
        '''
        查找protein的一阶邻居
        '''
        try:
            rnas=set(nx.neighbors(self.rpin,interaction[1]))
        except:
            rnas=set()
        try:
            proteins=set(nx.neighbors(self.ppin,interaction[1]))
        except:
            proteins=set()
        all_rnas.update(rnas)
        all_proteins.update(proteins)
        # 构造edge_list
        subgraph_serial_number += 1
        protein_subgraphSerialNumber = subgraph_serial_number
        dict_nodeName_subgraphNodeSerialNumber[interaction[1]] = protein_subgraphSerialNumber
        for protein in proteins:
            if protein==interaction[1]:
                continue
            if protein in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[protein]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[protein] = node2_subgraphSerialNumber
            edge_index[0].append(protein_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(1) #给这条边设置类型为protein-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(protein_subgraphSerialNumber)
            edge_type_list.append(1)  # 给这条边设置类型为protein-protein

        for rna in rnas:
            if rna==interaction[0]:
                continue
            if rna in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[rna]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[rna] = node2_subgraphSerialNumber
            edge_index[0].append(protein_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(0) #给这条边设置类型为rna-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(protein_subgraphSerialNumber)
            edge_type_list.append(0)  # 给这条边设置类型为protein-rna
        # 在一阶子图的基础上加入rri网络
        all_rnas.remove(interaction[0])
        rri_subgraph = get_subgraph(all_rnas, self.rrsn.edges)
        for rri in rri_subgraph:
            node1_subgraphSerialNumber = dict_nodeName_subgraphNodeSerialNumber[rri[0]]
            node2_subgraphSerialNumber = dict_nodeName_subgraphNodeSerialNumber[rri[1]]
            edge_index[0].append(node1_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(2)  # 给这条边设置类型为rna-rna
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(node1_subgraphSerialNumber)
            edge_type_list.append(2)  # 给这条边设置类型为rna-rna
        # 在一阶子图的基础上加入ppi网络
        all_proteins.remove(interaction[1])
        ppi_subgraph = get_subgraph(all_proteins, self.ppin.edges)
        for ppi in ppi_subgraph:
            node1_subgraphSerialNumber = dict_nodeName_subgraphNodeSerialNumber[ppi[0]]
            node2_subgraphSerialNumber = dict_nodeName_subgraphNodeSerialNumber[ppi[1]]
            edge_index[0].append(node1_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(1)  # 给这条边设置类型为protein-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(node1_subgraphSerialNumber)
            edge_type_list.append(1)  # 给这条边设置类型为protein-protein
        #构造x
        dict_subgraphNodeSerialNumber_nodeName=dict(zip(dict_nodeName_subgraphNodeSerialNumber.values(),dict_nodeName_subgraphNodeSerialNumber.keys()))
        for i in range(len(dict_subgraphNodeSerialNumber_nodeName)):
            vector = []
            vector.append(0)
            vector.extend(self.dict_node_name_vec[dict_subgraphNodeSerialNumber_nodeName[i]])
            x.append(vector)
        edge_index[0].append(rna_subgraphSerialNumber)
        edge_index[1].append(protein_subgraphSerialNumber)
        edge_type_list.append(0)#给这条边设置类型为rna-protein
        edge_index[0].append(protein_subgraphSerialNumber)
        edge_index[1].append(rna_subgraphSerialNumber)
        edge_type_list.append(0)# 给这条边设置类型为protein-rna
        #print(edge_type_list)
        # y记录这个interaction的真假
        if y == 1:
            y = [1]
        else:
            y = [0]
        self.sum_node += len(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_type_list=torch.tensor(edge_type_list, dtype=torch.long)
        # x为一个封闭子图的特征矩阵 y表示这个封闭子图是正样本还是负样本 edge_index为这个封闭子图的邻接矩阵:[[],[]]
        data = Data(x=x, y=y, edge_index=edge_index,edge_attr=edge_type_list,target_link=interaction)
        return data
    def local_subgraph_generation_with_rpin(self, interaction,y):
        x = []
        edge_index = [[], []]
        edge_type_list=[] #记录边的类型 rna-protein=0 protein-protein=1 rna-rna=2
        #记录封闭子图所有的rna和蛋白质
        all_rnas=set()
        all_rnas.add(interaction[0])
        all_proteins=set()
        all_proteins.add(interaction[1])
        '''
        查找rna在rpin的一阶邻居
        '''
        try:
            proteins = set(nx.neighbors(self.rpin, interaction[0]))
        except:
            #print(f'{interaction[0]} not exist {y}')
            proteins=set()
        all_proteins.update(proteins)
        # 构造edge_list
        dict_nodeName_subgraphNodeSerialNumber={}
        subgraph_serial_number=0
        rna_subgraphSerialNumber=subgraph_serial_number
        dict_nodeName_subgraphNodeSerialNumber[interaction[0]] = rna_subgraphSerialNumber
        for protein in proteins:
            if protein==interaction[1]:
                continue
            if protein in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[protein]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[protein] = node2_subgraphSerialNumber
            edge_index[0].append(rna_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(0) #给这条边设置类型为rna-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(rna_subgraphSerialNumber)
            edge_type_list.append(0)  # 给这条边设置类型为protein-rna
        '''
        查找protein在rpin的一阶邻居
        '''
        try:
            rnas=set(nx.neighbors(self.rpin,interaction[1]))
        except:
            rnas=set()
        all_rnas.update(rnas)
        # 构造edge_list
        subgraph_serial_number += 1
        protein_subgraphSerialNumber = subgraph_serial_number
        dict_nodeName_subgraphNodeSerialNumber[interaction[1]] = protein_subgraphSerialNumber
        for rna in rnas:
            if rna==interaction[0]:
                continue
            if rna in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[rna]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[rna] = node2_subgraphSerialNumber
            edge_index[0].append(protein_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(0) #给这条边设置类型为rna-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(protein_subgraphSerialNumber)
            edge_type_list.append(0)  # 给这条边设置类型为protein-rna
        # 在一阶子图的基础上加入rri网络
        all_rnas.remove(interaction[0])
        rri_subgraph = get_subgraph(all_rnas, self.rrsn.edges)
        for rri in rri_subgraph:
            node1_subgraphSerialNumber = dict_nodeName_subgraphNodeSerialNumber[rri[0]]
            node2_subgraphSerialNumber = dict_nodeName_subgraphNodeSerialNumber[rri[1]]
            edge_index[0].append(node1_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(2)  # 给这条边设置类型为rna-rna
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(node1_subgraphSerialNumber)
            edge_type_list.append(2)  # 给这条边设置类型为rna-rna
        # 在一阶子图的基础上加入ppi网络
        all_proteins.remove(interaction[1])
        ppi_subgraph = get_subgraph(all_proteins, self.ppin.edges)
        for ppi in ppi_subgraph:
            node1_subgraphSerialNumber = dict_nodeName_subgraphNodeSerialNumber[ppi[0]]
            node2_subgraphSerialNumber = dict_nodeName_subgraphNodeSerialNumber[ppi[1]]
            edge_index[0].append(node1_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(1)  # 给这条边设置类型为protein-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(node1_subgraphSerialNumber)
            edge_type_list.append(1)  # 给这条边设置类型为protein-protein
        #构造x
        dict_subgraphNodeSerialNumber_nodeName=dict(zip(dict_nodeName_subgraphNodeSerialNumber.values(),dict_nodeName_subgraphNodeSerialNumber.keys()))
        for i in range(len(dict_subgraphNodeSerialNumber_nodeName)):
            vector = []
            vector.append(0)
            vector.extend(self.dict_node_name_vec[dict_subgraphNodeSerialNumber_nodeName[i]])
            x.append(vector)
        edge_index[0].append(rna_subgraphSerialNumber)
        edge_index[1].append(protein_subgraphSerialNumber)
        edge_type_list.append(0)#给这条边设置类型为rna-protein
        edge_index[0].append(protein_subgraphSerialNumber)
        edge_index[1].append(rna_subgraphSerialNumber)
        edge_type_list.append(0)# 给这条边设置类型为protein-rna
        #print(edge_type_list)
        # y记录这个interaction的真假
        if y == 1:
            y = [1]
        else:
            y = [0]
        self.sum_node += len(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_type_list=torch.tensor(edge_type_list, dtype=torch.long)
        # x为一个封闭子图的特征矩阵 y表示这个封闭子图是正样本还是负样本 edge_index为这个封闭子图的邻接矩阵:[[],[]]
        data = Data(x=x, y=y, edge_index=edge_index,edge_attr=edge_type_list,target_link=interaction)
        return data
    def local_subgraph_generation_only_rpin(self, interaction,y):
        x = []
        edge_index = [[], []]
        edge_type_list=[] #记录边的类型 rna-protein=0 protein-protein=1 rna-rna=2
        '''
        查找rna在rpin的一阶邻居
        '''
        try:
            proteins = set(nx.neighbors(self.rpin, interaction[0]))
        except:
            #print(f'{interaction[0]} not exist {y}')
            proteins=set()
        # 构造edge_list
        dict_nodeName_subgraphNodeSerialNumber={}
        subgraph_serial_number=0
        rna_subgraphSerialNumber=subgraph_serial_number
        dict_nodeName_subgraphNodeSerialNumber[interaction[0]] = rna_subgraphSerialNumber
        for protein in proteins:
            if protein==interaction[1]:
                continue
            if protein in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[protein]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[protein] = node2_subgraphSerialNumber
            edge_index[0].append(rna_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(0) #给这条边设置类型为rna-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(rna_subgraphSerialNumber)
            edge_type_list.append(0)  # 给这条边设置类型为protein-rna
        '''
        查找protein在rpin的一阶邻居
        '''
        try:
            rnas=set(nx.neighbors(self.rpin,interaction[1]))
        except:
            rnas=set()
        # 构造edge_list
        subgraph_serial_number += 1
        protein_subgraphSerialNumber = subgraph_serial_number
        dict_nodeName_subgraphNodeSerialNumber[interaction[1]] = protein_subgraphSerialNumber
        for rna in rnas:
            if rna==interaction[0]:
                continue
            if rna in dict_nodeName_subgraphNodeSerialNumber.keys():
                node2_subgraphSerialNumber=dict_nodeName_subgraphNodeSerialNumber[rna]
            else:
                subgraph_serial_number += 1
                node2_subgraphSerialNumber = subgraph_serial_number
                dict_nodeName_subgraphNodeSerialNumber[rna] = node2_subgraphSerialNumber
            edge_index[0].append(protein_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_type_list.append(0) #给这条边设置类型为rna-protein
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(protein_subgraphSerialNumber)
            edge_type_list.append(0)  # 给这条边设置类型为protein-rna
        #构造x
        dict_subgraphNodeSerialNumber_nodeName=dict(zip(dict_nodeName_subgraphNodeSerialNumber.values(),dict_nodeName_subgraphNodeSerialNumber.keys()))
        for i in range(len(dict_subgraphNodeSerialNumber_nodeName)):
            vector = []
            vector.append(0)
            vector.extend(self.dict_node_name_vec[dict_subgraphNodeSerialNumber_nodeName[i]])
            x.append(vector)
        edge_index[0].append(rna_subgraphSerialNumber)
        edge_index[1].append(protein_subgraphSerialNumber)
        edge_type_list.append(0)#给这条边设置类型为rna-protein
        edge_index[0].append(protein_subgraphSerialNumber)
        edge_index[1].append(rna_subgraphSerialNumber)
        edge_type_list.append(0)# 给这条边设置类型为protein-rna
        #print(edge_type_list)
        # y记录这个interaction的真假
        if y == 1:
            y = [1]
        else:
            y = [0]
        self.sum_node += len(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_type_list=torch.tensor(edge_type_list, dtype=torch.long)
        # x为一个封闭子图的特征矩阵 y表示这个封闭子图是正样本还是负样本 edge_index为这个封闭子图的邻接矩阵:[[],[]]
        data = Data(x=x, y=y, edge_index=edge_index,edge_attr=edge_type_list,target_link=interaction)
        return data


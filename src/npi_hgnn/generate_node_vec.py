# coding:utf-8
import networkx as nx
from src.npi_hgnn.methods import read_all,read_rpin
import argparse
from src.npi_hgnn.methods import generate_n2v,generate_node_vec_with_fre,generate_node_vec_with_pyfeat,generate_node_vec_only_n2v
from src.npi_hgnn.methods import generate_node_vec_only_frequency,generate_node_path,generate_pre_dataset_path
import gc
def parse_args():
	parser = argparse.ArgumentParser(description="Run node2vec.")
	# NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus
	parser.add_argument('--dataset', default="NPHN-Mus", help='dataset name')
	# 0:使用kmer频率和n2v作为节点特征 1:使用pyfeat和n2v作为节点特征 2:只使用n2v作为节点特征
	parser.add_argument('--nodeVecType', default=0,type=int, help='node vector type')
	# 0：表示随机采样 1：表示使用reliable采样 2：表示使用improved采样,3、同时使用improved和随机采样
	parser.add_argument('--samplingType', default=3,type=int, help='how num of fold is this')
	# 0 rpin、ppin、rrsn组成的异构网络中提取的一阶封闭子图；1使用在rpin二部图上提取的一阶子图，并在其基础上加上ppin和rrsn；2使用rpin上的一阶封闭子图
	parser.add_argument('--subgraph_type', default=1, type=int, help='type of subgraph')
	parser.add_argument('--fold', default=0,type=int, help='which fold is this')
	parser.add_argument('--dimensions', type=int, default=64,
	                    help='Number of dimensions. Default is 64.')
	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')
	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')
	parser.add_argument('--window-size', type=int, default=5,
                    	help='Context size for optimization. Default is 5.')
	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')
	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')
	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')
	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')
	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)
	return parser.parse_args()
if __name__ == '__main__':
	print('start generate node feature vector\n')
	args=parse_args()
	node_path=generate_node_path(args.dataset,args.samplingType,args.nodeVecType)
	pre_dataset_path=generate_pre_dataset_path(args.dataset,args.samplingType)
	#读取所有节点
	node_names = []
	with open(f'{pre_dataset_path}/all_node_name') as f:
		lines = f.readlines()
		for line in lines:
			node_names.append(line.strip())
	node_name_vec_dict = dict(zip(node_names, range(len(node_names))))
	graph_path = f'{pre_dataset_path}/dataset_{args.fold}/pos_train_edges'
	if args.subgraph_type==2:
		G, rna_names, protein_names = read_rpin(graph_path) #只使用RPIN网络
	else:
		G, rna_names, protein_names = read_all(graph_path, args.dataset)
	# 添加结点
	G.add_nodes_from(node_names)
	if args.nodeVecType==0:
		print('使用kmer频率和n2v作为节点特征')
		path = f'{node_path}/node2vec/{args.fold}'
		generate_n2v(G, path, args.dimensions, args.walk_length, args.num_walks, args.p, args.q, args.workers)
		generate_node_vec_with_fre(args.fold,args.dataset,node_path)
	elif args.nodeVecType==1:
		print('使用pyfeat和n2v作为节点特征')
		path = f'{node_path}/node2vec/{args.fold}'
		generate_n2v(G, path, args.dimensions, args.walk_length, args.num_walks, args.p, args.q, args.workers)
		generate_node_vec_with_pyfeat(args.fold,args.dataset,node_path)
	elif args.nodeVecType == 2:
		print('只使用n2v作为节点特征')
		path = f'{node_path}/node2vec/{args.fold}'
		generate_n2v(G, path, args.dimensions, args.walk_length, args.num_walks, args.p, args.q, args.workers)
		generate_node_vec_only_n2v(args.fold,args.dataset,node_path)
	else:
		print('只使用kmer频率作为节点特征')
		path = f'{node_path}/node2vec/{args.fold}'
		#generate_n2v(G, path, args.dimensions, args.walk_length, args.num_walks, args.p, args.q, args.workers)
		generate_node_vec_only_frequency(args.fold,args.dataset,node_path)
	del G
	gc.collect()
	print('generate node feature vector end\n')
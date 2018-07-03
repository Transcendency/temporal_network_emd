'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import multiprocessing
import os
from node2vec import Graph

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph(input):
	'''
	Reads the input network in networkx.
	'''
	G = nx.Graph()
	if args.weighted:
		G = nx.read_edgelist(input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def sub_shortest_paths(i):
	#j, i = params
	index = i
	ins = G.nodes()[index]
	shortest_paths_all =[]
	for j in G.nodes()[index:]:
		try:
			for p in nx.all_shortest_paths(G, source=ins, target=j):
				#print p
				shortest_paths_all = shortest_paths_all + p
		except:
			pass

	return shortest_paths_all


def shortest_path_walks(G):
	global path
	p = multiprocessing.Pool()
	path = p.map(sub_shortest_paths, range(len(G.nodes())))
	p.close()
	p.join()
	shpaths = filter(None, path)
	return shpaths	


def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	return

def getwalks(edge_list_t, args):
	walks = []
	for edgelist in edge_list_t:
		nx_G = read_graph(edgelist)
		G = Graph(nx_G, args.directed, args.p, args.q)
		G.preprocess_transition_probs()
		walk = G.simulate_walks(args.num_walks, args.walk_length)
		walks.extend(walk)
	return walks


def fileList(source):
 	ps = []
	for root, dirs, files in os.walk(source, topdown=False):
	   for name in files:
	      ps.append(os.path.join(root, name))
	return sorted(ps)

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	edge_list_t = fileList(args.input)
	walks = getwalks(edge_list_t, args)
	print(np.array(walks).shape)
	# walkss = shortest_path_walks(nx_G)
	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)

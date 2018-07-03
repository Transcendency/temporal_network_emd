import json

with open('./trajectory_network_data/200_500_1000.json', 'r') as f:
	graph_json = json.load(f)

edges = graph_json.get('edges')

for i in range(1000):
    with open('./graph/' + str(i), 'w+') as of:
        of.write('')
        of.close()

for e in edges.iteritems():
    v = e[0].split('_')
    for i in range(e[1]['start'], e[1]['stop'] + 1):
        with open('./graph/' + str(i), 'a') as of:
            # print(v[0] + ' ' + v[1] + '\n')
            of.write(v[0] + ' ' + v[1] + '\n')
            of.close()
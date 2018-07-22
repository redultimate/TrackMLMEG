import sys
sys.path.append('../lib/')

import time
import pickle as pkl

# meg library
# import seed_finder_loop2 as sfl2
import seed_finder_loop3 as sfl3
import read_data as rd

# args: file name, first event, nevent
argvs = sys.argv
start = int(argvs[1])
end = int(argvs[1])+int(argvs[2])
print(start, end)

# path
path_to_data = '../path_to_data'
path_to_tables = '../../trackmlmeg/res'

dh = rd.DataHandler()
train, test, det = dh.get_event_list(path_to_data)

for i in range(start, end):
    print('eventID: ', i)
    event = 'test/event' + '%09d' % i
    #event = 'train_100_events/event' + '%09d' % i

    hits = dh.read_meas(event)
    #hits = dh.read_all(event)
    hits = hits.astype(float)
    hits = hits.loc[:, ['hit_id','x','y', 'volume_id', 'layer_id', 'module_id']].values
    print(hits.shape)

    seeding = sfl3.SeedFinder(10, 50)

    start = time.time()
    hitids, track_parameters = seeding.get_candidates(hits, path_to_tables)
    end = time.time()
    print(str(end-start) + 'sec')

    f = open('../../trackmlmeg/res/hitids_event' + '%09d' % i +'.pkl', 'wb')
    pkl.dump(hitids, f)
    f = open('../../trackmlmeg/res/track_parameters_event' + '%09d' % i +'.pkl', 'wb')
    pkl.dump(track_parameters, f)


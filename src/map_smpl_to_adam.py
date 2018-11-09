import json
import cPickle as cp
import numpy as np
import numpy.linalg as nl

if __name__ == '__main__':
    adam_obj = '/media/posefs1b/Users/hanbyulj/TotalBodyCapture/model/totalmodel/mesh_nofeet.obj'
    adam_v = []
    with open(adam_obj) as f:
        for n in xrange(18540):
            adam_v.append([float(_) for _ in f.readline().split()[1:]])
    adam_v = np.array(adam_v)

    smpl_obj = '/media/posefs1b/Users/hanbyulj/TotalBodyCapture/model/smpl/mesh_smpl.obj'
    smpl_v = []
    with open(smpl_obj) as f:
        for n in xrange(6890):
            smpl_v.append([float(_) for _ in f.readline().split()[1:]])
    smpl_v = np.array(smpl_v)

    with open('../models/neutral_smpl_with_cocoplus_reg.pkl', 'rb') as f:
        smpl_data = cp.load(f)

    cols, rows = smpl_data['cocoplus_regressor'].nonzero()

    map_smpl_adam_v = {}

    for iv in np.unique(rows):
        sv = smpl_v[iv]
        dv = nl.norm(adam_v - sv, axis=1)
        min_v = np.argmin(dv)
        print '{}: {} min distance {}'.format(iv, smpl_v[iv], dv[min_v])
        map_smpl_adam_v[iv] = min_v

    cocoplus_regressor = [[19, 18540, 0]]
    for c, r in zip(cols, rows):
        cocoplus_regressor.append([int(c), int(map_smpl_adam_v[r]), float(smpl_data['cocoplus_regressor'][c, r])])

    with open('/home/donglaix/Documents/Experiments/hand_model/model/adam_cocoplus_regressor.json', 'w') as f:
        json.dump({'cocoplus_regressor': cocoplus_regressor}, f)

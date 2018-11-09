import json
import cPickle as cp

if __name__ == '__main__':
    with open('../models/neutral_smpl_with_cocoplus_reg.pkl', 'rb') as f:
        data = cp.load(f)
    save_data = {}
    cocoplus_regressor = [[19, 6890, 0]]
    cols, rows = data['cocoplus_regressor'].nonzero()
    for c, r in zip(cols, rows):
        cocoplus_regressor.append([int(c), int(r), data['cocoplus_regressor'][c, r]])
    save_data['cocoplus_regressor'] = cocoplus_regressor
    save_data['posedirs'] = data['posedirs'].reshape(6890 * 3, 207).tolist()
    save_data['weights'] = data['weights'].tolist()
    save_data['v_template'] = data['v_template'].reshape(6890 * 3).tolist()
    save_data['shapedirs'] = data['shapedirs'].r.reshape(6890 * 3, 10).tolist()  # this is a chumpy array
    J_regressor = [[24, 6890, 0]]
    cols, rows = data['J_regressor'].nonzero()
    for c, r in zip(cols, rows):
        J_regressor.append([int(c), int(r), data['J_regressor'][c, r]])
    save_data['J_regressor'] = J_regressor

    with open('/home/donglaix/Documents/Experiments/hand_model/model/smpl.json', 'w') as f:
        json.dump(save_data, f)

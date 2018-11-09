from tf_smpl.batch_smpl import SMPL
from tf_smpl.batch_lbs import batch_rodrigues, batch_global_rigid_transformation
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if __name__ == '__main__':
    smpl = SMPL('../models/neutral_smpl_with_cocoplus_reg.pkl')
    beta = tf.zeros([1, 10], dtype=tf.float32)
    npose = np.zeros([1, 72])
    pose = tf.constant(npose, dtype=tf.float32)
    verts, joints, Rs = smpl(beta, pose, get_skin=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    V, J, R = sess.run([verts, joints, Rs])
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(J[0, :, 0], J[0, :, 1], J[0, :, 2], c='r')
    ax.scatter(J[0, 18, 0], J[0, 18, 1], J[0, 18, 2], c='b')
    # ax.scatter(V[0, :, 0], V[0, :, 1], V[0, :, 2], c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()

import numpy as np
import tensorflow as tf
import cv2
import tqdm
from network import Network
import os

# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
IMAGE_SIZE = 128
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 1e-3
BATCH_SIZE = 10
PRETRAIN_EPOCH = 100


def load_data(dir_='../data/npy'):
    x_train = np.load(os.path.join(dir_, 'x_train.npy'))
    x_test = np.load(os.path.join(dir_, 'x_test.npy'))
    return x_train, x_test


def get_points():  # get random points
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])

        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h

        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)


def get_fixed_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[x1:x2, y1:y2] = 1
        mask.append(m)

    return np.array(points), np.array(mask)


def train():
    # x.shape = [n,h,w,c]
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

    # to define which area will be completed, 1 for pixel to be completed
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])

    # x for local discriminator
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])

    # local and global discriminator
    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)

    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # default: load existed model
    # calculate last trained epoch
    # 1st 100 epochs -> mse
    #######################################################################
    if tf.train.get_checkpoint_state('./backup'):
        saver = tf.train.Saver()
        saver.restore(sess, './backup/latest')
    
    x_train, x_test = load_data()
    print (x_train.shape)

    # normalization ->[0-1]
    x_train = x_train/127.5 - 1
    print (x_train[0])
    x_test = x_test / 127.5 - 1
    print(x_test.shape)

    step_num = int(len(x_train) / BATCH_SIZE)

    while True:
        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))

        np.random.shuffle(x_train)

        # current epoch less than 100
        # Completion 
        if sess.run(epoch) <= PRETRAIN_EPOCH:
            g_loss_value = 0
            # process visualization
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                # choose the hole location to be completed
                # max-size is [64x64]
                points_batch, mask_batch = get_points()     
    
                _, g_loss = sess.run([g_train_op, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                g_loss_value += g_loss

            print('Completion loss: {}'.format(g_loss_value))

            np.random.shuffle(x_test) 
            x_batch = x_test[:BATCH_SIZE]
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            # recover to image [0-255]
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))


            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)
            if sess.run(epoch) == PRETRAIN_EPOCH:
                saver.save(sess, './backup/pretrained', write_meta_graph=False)

        # if 1st 100 runs are done, directly jump here
        # Discrimitation
        else:
            g_loss_value = 0
            d_loss_value = 0
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                points_batch, mask_batch = get_fixed_points()
                _, g_loss, completion = sess.run([g_train_op, model.g_loss, model.completion], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                g_loss_value += g_loss

                local_x_batch = []
                local_completion_batch = []
                for i in range(BATCH_SIZE):
                    x1, y1, x2, y2 = points_batch[i]
                    local_x_batch.append(x_batch[i][y1:y2, x1:x2, :])
                    local_completion_batch.append(completion[i][y1:y2, x1:x2, :])
                local_x_batch = np.array(local_x_batch)
                local_completion_batch = np.array(local_completion_batch)

                _, d_loss = sess.run(
                    [d_train_op, model.d_loss], 
                    feed_dict={x: x_batch, mask: mask_batch, local_x: local_x_batch, global_completion: completion, local_completion: local_completion_batch, is_training: True})
                d_loss_value += d_loss

            print('Completion loss: {}'.format(g_loss_value))
            print('Discriminator loss: {}'.format(d_loss_value))

            np.random.shuffle(x_test) 
            x_batch = x_test[:BATCH_SIZE]
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)




if __name__ == '__main__':
    train()
    

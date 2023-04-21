import time
from glob import glob
import numpy as np
import tensorflow as tf
import random
import os
import cv2

def dncnn(input, is_training=True, output_channels=1):
    with tf.compat.v1.variable_scope('block1'):
        output = tf.compat.v1.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 19+1):
        with tf.compat.v1.variable_scope('block%d' % layers):
            output = tf.compat.v1.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.compat.v1.layers.batch_normalization(output, training=is_training))   
    with tf.compat.v1.variable_scope('block17'):
        output = tf.compat.v1.layers.conv2d(output, output_channels, 3, padding='same',use_bias=False)
    return output
'''
@y_true是清晰图片Y，y_out是网络输出O,y_noise噪声图像X，m是正的超参数
@L_contrastive = (1-Y)*||O-X||^2 + Y*max(0, m-||O-Y||^2)

def contrastive_loss(y_true, y_out,y_noise, margin=1):
    y_pred = tf.reduce_mean(tf.square(y_out - y_noise))
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
'''


def contrastive_loss(y_true, y_pred, noise_images, margin=2):
    """
    Triplet loss function to measure the similarity between clean image, noise image and the corresponding denoised image.

    Args:
        y_true: 清晰原图片
        y_pred: 去噪图片
        noise_images: dncnn噪声图片
        margin: 超参数

    Returns:
        The contrastive loss value.
    """
    # 计算正负样本距离
    distance_cp = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3])
    distance_cn = tf.reduce_sum(tf.square(y_true - noise_images), axis=[1, 2, 3])
    distance_pn = tf.reduce_sum(tf.square(y_pred - noise_images), axis=[1, 2, 3])

    # 计算对比损失
    triplet_loss = tf.reduce_mean(tf.maximum((distance_cp - distance_pn + margin), 0) +
                                  tf.maximum((distance_cn - distance_pn + margin), 0))

    return triplet_loss

filepaths = glob('./data/train/original/*.png') #takes all the paths of the png files in the train folder
filepaths = sorted(filepaths)                           #Order the list of files
filepaths_noisy = glob('./data/train/noisy/*.png')
filepaths_noisy = sorted(filepaths_noisy)
ind = list(range(len(filepaths)))

class denoiser(object):
    def __init__(self, sess, input_c_dim=1, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        # build model
        #输入清晰图像
        self.Y_ = tf.compat.v1.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
        #输入模糊图像
        self.X = tf.compat.v1.placeholder(tf.float32, [None, None, None, self.input_c_dim])
        #噪声输出
        self.noise = dncnn(self.X, is_training=self.is_training)
        #输出去噪图像
        self.Y = self.X - self.noise
        '''
        这里的损失函数要改
        '''
        self.loss = contrastive_loss(self.Y_,self.Y,self.noise)

        # self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
        self.dataset = dataset(sess)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, eval_files, noisy_files, summary_writer):
        print("[*] Evaluating...")
        psnr_sum = 0
        
        for i in range(10):
            clean_image = cv2.imread(eval_files[i],cv2.IMREAD_GRAYSCALE)
            #clean_image = cv2.imread(eval_files[i], 0)
            clean_image = clean_image.astype('float32') / 255.0
            clean_image = clean_image[np.newaxis, ..., np.newaxis]
            noisy = cv2.imread(noisy_files[i],cv2.IMREAD_GRAYSCALE)
            #noisy = cv2.imread(noisy_files[i], 0)
            noisy = noisy.astype('float32') / 255.0
            noisy = noisy[np.newaxis, ..., np.newaxis]
            
            output_clean_image = self.sess.run(
                [self.Y],feed_dict={self.Y_: clean_image,
                           self.X: noisy,
                           self.is_training: False})
            psnr = psnr_scaled(clean_image, output_clean_image)
            print("img%d PSNR: %.2f" % (i + 1, psnr))
            psnr_sum += psnr

        avg_psnr = psnr_sum / 10

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)


    def train(self, eval_files, noisy_files, batch_size, ckpt_dir, epoch, lr, eval_every_epoch=1):

        numBatch = int(len(filepaths) * 2)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.compat.v1.summary.scalar('loss', self.loss)
        tf.compat.v1.summary.scalar('lr', self.lr)
        writer = tf.compat.v1.summary.FileWriter('logs', self.sess.graph)
        merged = tf.compat.v1.summary.merge_all()
        clip_all_weights = tf.compat.v1.get_collection("max_norm")        

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_files, noisy_files, summary_writer=writer)  # eval_data value range is 0-255
        for epoch in range(start_epoch, epoch):
            batch_noisy = np.zeros((batch_size,64,64,1),dtype='float32')
            batch_images = np.zeros((batch_size,64,64,1),dtype='float32')
            for batch_id in range(start_step, numBatch):
              try:
                res = self.dataset.get_batch() # If we get an error retrieving a batch of patches we have to reinitialize the dataset
              except KeyboardInterrupt:
                raise
              except:
                self.dataset = dataset(self.sess) # Dataset re init
                res = self.dataset.get_batch()
              if batch_id==0:
                batch_noisy = np.zeros((batch_size,64,64,1),dtype='float32')
                batch_images = np.zeros((batch_size,64,64,1),dtype='float32')
              ind1 = list(range(res.shape[0]//2))
              ind1 = np.multiply(ind1,2)
              for i in range(batch_size):
                random.shuffle(ind1)
                ind2 = random.randint(0,8-1)
                batch_noisy[i] = res[ind1[0],ind2]
                batch_images[i] = res[ind1[0]+1,ind2]
#              for i in range(64):
#                cv2.imshow('raw',batch_images[i])
#                cv2.imshow('noisy',batch_noisy[i])
              _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                                                 feed_dict={self.Y_: batch_images, self.X: batch_noisy, self.lr: lr[epoch],
                                                            self.is_training: True})
              self.sess.run(clip_all_weights)          
              
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                    % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
              iter_num += 1
              writer.add_summary(summary, iter_num)
              
            if np.mod(epoch + 1, eval_every_epoch) == 0: ##Evaluate and save model
                self.evaluate(iter_num, eval_files, noisy_files, summary_writer=writer)
                self.save(iter_num, ckpt_dir)
        print("[*] Training finished.")

    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.compat.v1.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, eval_files, noisy_files, ckpt_dir, save_dir): #, temporal
        """Test DnCNN"""
        # init variables
        tf.compat.v1.global_variables_initializer().run()
        assert len(eval_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == False, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
            
        for i in range(len(eval_files)):
            clean_image = cv2.imread(eval_files[i],cv2.IMREAD_GRAYSCALE)
            clean_image = clean_image.astype('float32') / 255.0
            clean_image = clean_image[np.newaxis, ..., np.newaxis]
            
            noisy = cv2.imread(noisy_files[i],cv2.IMREAD_GRAYSCALE)
            noisy = noisy.astype('float32') / 255.0
            noisy = noisy[np.newaxis, ..., np.newaxis]
          
            output_clean_image = self.sess.run(
                [self.Y],feed_dict={self.Y_: clean_image, self.X: noisy,
                                    self.is_training: False})
            
            out1 = np.asarray(output_clean_image)
               
            psnr = psnr_scaled(clean_image, out1[0,0])
            psnr1 = psnr_scaled(clean_image, noisy)
            
            print("img%d PSNR: %.2f , noisy PSNR: %.2f" % (i + 1, psnr, psnr1))
            psnr_sum += psnr

            cv2.imwrite('./data/denoised/%04d.png'%(i),out1[0,0]*255.0)

        avg_psnr = psnr_sum / len(eval_files)
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

    
class dataset(object):
  def __init__(self,sess):
    self.sess = sess
    seed = time.time()
    random.seed(seed)

    random.shuffle(ind)
    
    filenames = list()
    for i in range(len(filepaths)):
        filenames.append(filepaths_noisy[ind[i]])
        filenames.append(filepaths[ind[i]])

    # Parameters
    num_patches = 8   # number of patches to extract from each image
    patch_size = 64                 # size of the patches
    num_parallel_calls = 1          # number of threads
    batch_size = 32                # size of the batch
    get_patches_fn = lambda image: get_patches(image, num_patches=num_patches, patch_size=patch_size)
    dataset = (
        tf.data.Dataset.from_tensor_slices(filenames)
        .map(im_read, num_parallel_calls=num_parallel_calls)
        .map(get_patches_fn, num_parallel_calls=num_parallel_calls)
        .batch(batch_size)
        .prefetch(batch_size)
    )
    
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    self.iter = iterator.get_next()
  

  def get_batch(self):
        res = self.sess.run(self.iter)
        return res
        
def im_read(filename):
    """Decode the png image from the filename and convert to [0, 1]."""
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image
    
def get_patches(image, num_patches=128, patch_size=64):
    """Get `num_patches` from the image"""
    patches = []
    for i in range(num_patches):
      point1 = random.randint(0,116) # 116 comes from the image source size (180) - the patch dimension (64)
      point2 = random.randint(0,116)
      patch = tf.image.crop_to_bounding_box(image, point1, point2, patch_size, patch_size)
      patches.append(patch)
    patches = tf.stack(patches)
    assert patches.get_shape().dims == [num_patches, patch_size, patch_size, 1]
    return patches
    
def cal_psnr(im1, im2): # PSNR function for 0-255 values
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr
    
def psnr_scaled(im1, im2): # PSNR function for 0-1 values
    mse = ((im1 - im2) ** 2).mean()
    mse = mse * (255 ** 2)
    psnr = 10 * np.log10(255 **2 / mse)
    return psnr

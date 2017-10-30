# import tensorflow as tf
#
#
# # 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# # 加到默认图中.
# #
# # 构造器的返回值代表该常量 op 的返回值.
# matrix1 = tf.constant([[3., 3.]])
#
# # 创建另外一个常量 op, 产生一个 2x1 矩阵.
# matrix2 = tf.constant([[2.],[2.]])
#
# # 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# # 返回值 'product' 代表矩阵乘法的结果.
# product = tf.matmul(matrix1, matrix2)
#
# with tf.Session() as sess:
#   with tf.device("/gpu:0"):
#     matrix1 = tf.constant([[3., 3.]])
#     matrix2 = tf.constant([[2.],[2.]])
#     for j in range(10000):
#         for i in range(10000):
#             product = tf.matmul(matrix1, matrix2)
#             print(product)
#
#
# # # Creates a graph.
# # a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# # b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# # c = tf.matmul(a, b)
# # # Creates a session with log_device_placement set to True.
# # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # # Runs the op.
# # # for j in range(10000):
# # #     for i in range(10000):
# # print(sess.run(c))

from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
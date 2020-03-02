from utils.logger import Logger

log = Logger('./all_log.txt', level='info')  # 保存输出信息
logger = log.logger

CFG = {
    'src_img_path': 'D:/dataSets/MNIST/MNIST_data',
    'tar_img_path': 'D:/dataSets/MNIST/mnist_m' ,
    'test_list': 'D:/dataSets/MNIST/mnist_m/mnist_m_test_labels.txt',
    'test_img_path': 'D:/dataSets/MNIST/mnist_m/mnist_m_test',
    'source_data': 'mnist',
    'target_data': 'mnist_m',
    'checkpoint': 'checkponit',
    'kwargs': {},
    'batch_size': 128,
    'epoch': 100,
    'lr': 1e-3,
    'momentum': .9,
    'log_interval': 10,
    'l2_decay': 1e-3,
    'backbone': 'resnet',
    'img_size': 28,
    'num_workers': 2
}
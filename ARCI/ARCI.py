import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import warnings
import time

theano.config.floatX='float32'
warnings.filterwarnings("ignore")

def train_conv_net(datasets,
                   word_vecs,
                   windows=[3,2],
                   pool_sizes=[2],
                   dim=300,
                   feature_maps=[100, 100],
                   dropout_rate=[0.5],
                   hidden_units=[],
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=50,
                   lr_decay=0.95,
                   conv_non_linear="relu",
                   activations=['relu'],
                   sqr_norm_lim=9, ):

    assert(len(windows)==len(feature_maps))
    assert (len(windows)==len(pool_sizes)+1)
    print('\nbuilding model...')
    index = T.lscalar()
    x1 = T.matrix('x1')
    x2 = T.matrix('x2')
    y = T.ivector('y')
    Words = theano.shared(value=word_vecs, name="Words")
    rng = np.random.RandomState(9999)

    ### define model architecture ###
    img_h = len(datasets[0][0][0])
    filter_shapes = [(feature_maps[0], 1, windows[0], dim)]
    for i in range(1, len(windows)):
        filter_shapes.append((feature_maps[i], 1, windows[i], feature_maps[i - 1]))
    next_layer_input_1 = Words[T.cast(x1.flatten(), dtype="int32")].reshape((x1.shape[0], 1, x1.shape[1], dim))
    next_layer_input_2 = Words[T.cast(x2.flatten(), dtype="int32")].reshape((x1.shape[0], 1, x1.shape[1], dim))
    conv_layers_1 = []
    conv_layers_2 = []
    for i in xrange(len(windows) - 1):
        filter_shape = filter_shapes[i]
        pool_size = (pool_sizes[i], 1)
        conv_layer_1 = LeNetConvPoolLayer(rng, input=next_layer_input_1,
                                          image_shape=(batch_size, 1, img_h, filter_shape[3]),
                                          filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        conv_layer_2 = LeNetConvPoolLayer(rng, input=next_layer_input_2,
                                          image_shape=(batch_size, 1, img_h, filter_shape[3]),
                                          filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        img_h -= windows[i] - 1
        img_h /= pool_sizes[i]
        next_layer_input_1 = T.swapaxes(conv_layer_1.output,1,3)
        next_layer_input_2 = T.swapaxes(conv_layer_2.output,1,3)
        conv_layers_1.append(conv_layer_1)
        conv_layers_2.append(conv_layer_2)
    ###the last convPoolLayer needs different configurations###
    filter_shape = filter_shapes[-1]
    pool_size = (img_h-windows[-1]+1, 1)
    conv_layer_1 = LeNetConvPoolLayer(rng, input=next_layer_input_1,
                                      image_shape=(batch_size, 1, img_h, filter_shape[3]),
                                      filter_shape=filter_shape, poolsize=pool_size)
    conv_layer_2 = LeNetConvPoolLayer(rng, input=next_layer_input_2,
                                      image_shape=(batch_size, 1, img_h, filter_shape[3]),
                                      filter_shape=filter_shape, poolsize=pool_size)
    output_1 = conv_layer_1.output.flatten(2)
    output_2 = conv_layer_2.output.flatten(2)
    conv_layers_1.append(conv_layer_1)
    conv_layers_2.append(conv_layer_2)
    next_layer_input = T.concatenate([output_1, output_2], 1)
    ###MLP with dropout###
    layer_sizes=[feature_maps[-1] * 2]
    for i in hidden_units:
        layer_sizes.append(hidden_units[i])
    layer_sizes.append(2)
    classifier = MLPDropout(rng, input=next_layer_input, layer_sizes=layer_sizes, activations='relu',
                            dropout_rates=dropout_rate)
    ###updates the params with adadelta###
    params = classifier.params
    for conv_layer in conv_layers_1:
        params += conv_layer.params
    for conv_layer in conv_layers_2:
        params += conv_layer.params
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    ###creat minibatches for training set###
    np.random.seed(9999)
    if datasets[0].shape[0] % batch_size > 0:
        data_zipped=zip(datasets[0],datasets[2])
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(data_zipped)
        extra_data = train_set[:extra_data_num]
        train_set=np.append(train_set ,extra_data,axis=0)
    else:
        train_set = datasets[0]
    train_set = np.random.permutation(train_set)
    n_batches = train_set.shape[0] / batch_size
    train_labels = train_set[:,1]
    train_set= np.array(train_set[:,0])
    train_set_x1 = [x[0] for x in train_set]
    train_set_x2 = [x[1] for x in train_set]
    test_set_x1 = [x[0] for x in datasets[1]]
    test_set_x2 = [x[1] for x in datasets[1]]
    test_labels = np.asarray(datasets[3], "int32")
    train_set_x1, train_set_x2, train_labels = shared_dataset((train_set_x1, train_set_x2, train_labels))
    ###theano functions for training and testing###
    train_model = theano.function([index], classifier.errors(y), updates=grad_updates,
                                  givens={
                                      x1: train_set_x1[index * batch_size:(index + 1) * batch_size],
                                      x2: train_set_x2[index * batch_size:(index + 1) * batch_size],
                                      y: train_labels[index * batch_size:(index + 1) * batch_size]},
                                  allow_input_downcast=True)
    test_layer_input_1= Words[T.cast(x1.flatten(),dtype="int32")].reshape((x1.shape[0],1,x1.shape[1],dim))
    test_layer_input_2= Words[T.cast(x2.flatten(),dtype="int32")].reshape((x2.shape[0],1,x2.shape[1],dim))
    for i in xrange(len(conv_layers_1)-1):
        output_1=conv_layers_1[i].predict(test_layer_input_1, len(test_labels))
        output_2=conv_layers_1[i].predict(test_layer_input_2, len(test_labels))
        test_layer_input_1 = T.swapaxes(output_1,1,3)
        test_layer_input_2 = T.swapaxes(output_2,1,3)
    output_1=conv_layers_1[-1].predict(test_layer_input_1, len(test_labels))
    output_2=conv_layers_1[-1].predict(test_layer_input_2, len(test_labels))
    next_layer_input=T.concatenate([output_1.flatten(2), output_2.flatten(2)], 1)
    test_y_pred = classifier.predict(next_layer_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model = theano.function([x1,x2,y], test_error, allow_input_downcast = True)
    ###training###
    print 'training...'
    epoch = 0
    test_accs = []
    train_losses=[]
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_batches)):
                train_losses.append(train_model(minibatch_index))
        else:
            for minibatch_index in xrange(n_batches):
                train_losses.append(train_model(minibatch_index))
        test_error = test_model(test_set_x1,test_set_x2,test_labels)
        train_perf = 1 - np.mean(train_losses)
        test_perf = 1 - test_error
        test_accs.append(test_perf)
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %% , test perf: %.2f %%' % (
            epoch, time.time() - start_time, train_perf * 100., test_perf*100.))
    return max(test_accs)


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x1, data_x2, data_y = data_xy
    shared_x1 = theano.shared(np.asarray(data_x1,
                                         dtype=theano.config.floatX),
                              borrow=borrow)
    shared_x2 = theano.shared(np.asarray(data_x2,
                                         dtype=theano.config.floatX),
                              borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x1, shared_x2, T.cast(shared_y, 'int32')

def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to

if __name__ == "__main__":
    print "loading data...",
    data = cPickle.load(open("data", "rb"))
    word_vecs = data[4]
    datasets = [data[0], data[1], data[2], data[3]]
    execfile("conv_net_classes.py")
    results = []
    acc = train_conv_net(datasets,
                         word_vecs,
                         lr_decay=0.95,
                         windows=[3,2],
                         pool_sizes=[2],
                         hidden_units=[],
                         conv_non_linear="relu",
                         feature_maps=[100,100],
                         shuffle_batch=True,
                         n_epochs=20,
                         sqr_norm_lim=9,
                         batch_size=50,
                         dropout_rate=[0.7])
    print("best acc:" + str(acc))

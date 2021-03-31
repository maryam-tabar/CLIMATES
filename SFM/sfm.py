import os
os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"
os.environ['KERAS_BACKEND'] = 'theano'
import keras
import build
import numpy as np


#Main Run Thread
if __name__=='__main__':
    step = 1
    data_file = './datasets/npp_cluster1.npy'
    hidden_dim  = 256
    freq_dim = 10
    niter = 500
    nsnapshot = 1
    learning_rate = 0.001

    X_train, y_train, X_val, y_val, X_test, y_test, _, _, _ = build.load_data(data_file, step)
    train_len = X_train.shape[1]
    val_len = X_val.shape[1]-X_train.shape[1]
    test_len = X_test.shape[1]-X_val.shape[1]

    model = build.build_model([1, hidden_dim, 1], freq_dim, learning_rate)

    best_error = np.inf
    best_epoch = 0

    for ii in range(int(niter/nsnapshot)):
            model.fit(
                X_train,
                y_train,
                batch_size=64,
                verbose=1,
                nb_epoch=nsnapshot,
                validation_split=0)

    predicted = model.predict(X_val)
    val_error = np.sum((predicted[:,-val_len:,0] - y_val[:,-val_len:,0])**2) / (val_len * predicted.shape[0])

    num_iter = str(nsnapshot * (ii+1))

    if(val_error < best_error):
        best_error = val_error
        best_iter = nsnapshot * (ii+1)
        model.save_weights('./snapshot/weights_cluster1_freq{}_best.hdf5'.format(freq_dim), overwrite = True)


    print 'best iteration ', best_iter
    print 'smallest error ', best_error



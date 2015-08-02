import theano, cPickle, time, os
import theano.tensor as T
import numpy as np
from util import *

# load MNIST data into shared variables
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = \
    np.load('../data/mnist.pkl')

train_y_expand = one_hot(train_y)

train_x, train_y, valid_x, valid_y, test_x, test_y = \
    sharedX(train_x), sharedX(train_y), sharedX(valid_x), \
    sharedX(valid_y), sharedX(test_x),  sharedX(test_y)

train_y_expand = sharedX(train_y_expand)

def exp(__lr_f, __lr_b) :

    max_epochs, batch_size, n_batches = 1000, 20, 2500
    nX, nH1, nH2, nY = 784, 500, 500, 10 # net architecture

    W1 = rand_ortho((nX, nH1), np.sqrt(6./(nX +nH1)));  B1 = zeros((nH1,)) # init params
    W2 = rand_ortho((nH1,nH2), np.sqrt(6./(nH1+nH2)));  B2 = zeros((nH2,))
    W3 = rand_ortho((nH2, nY), np.sqrt(6./(nH2+ nY)));  B3 = zeros((nY,))

    V2 = rand_ortho((nH2,nH1), np.sqrt(6./(nH2+nH1)));  C2 = zeros((nH1,))

    # layer definitions - functions of layers
    F1 = lambda x  : tanh(  T.dot( x,  W1 ) + B1  ) # feedforward
    F2 = lambda h1 : tanh(  T.dot( h1, W2 ) + B2  ) 
    F3 = lambda h2 : sfmx(  T.dot( h2, W3 ) + B3  ) 

    G2 = lambda h2 : tanh(  T.dot( h2, V2 ) + C2  ) # feedback

    i, e = T.lscalar(), T.fscalar() # minibatch index, epochs
    X, Y = T.fmatrix(), T.fvector() # X - data, Y - label 
    Y_exp = T.fmatrix()

    # feedforward
    H1 = F1(X)
    H2 = F2(H1)
    P = F3(H2)
    
    H1_ = G2(H2)  #get activations from the snn
    
    H1_prime = 1 - H1_ ** 2
    H2_prime = 1 - H2 ** 2
    P_prime = P * (1 - P)

    cost = NLL( P, Y ) # cost: Negative Log Likelihood
    cost_derivative = P - Y_exp  #TODO: check if this corresponds to the Negative Log Likelihood

    err  = error( predict(P), Y ) # error

    # gradients 	
    g_V2, g_C2 = T.grad( mse( G2(H2), H1 ), [V2, C2], consider_constant=[H2,H1] )

    delta_3 = cost_derivative * P_prime
    g_W3 = T.dot(H2.T, delta_3)
    g_B3 = delta_3
    
    delta_2 = T.dot(delta_3, W3.T) * H2_prime
    g_W2 = T.dot(H1_.T, delta_2)  #TODO: REPLACE BY H1_
    g_B2 = delta_2

    delta_1 = T.dot(delta_2, W2.T) * H1_prime
    g_W1 = T.dot(X.T, delta_1)
    g_B1 = delta_1

    #g_W3, g_B3 = T.grad( cost, [W3, B3], consider_constant=[H2] )    
    #g_W2, g_B2 = T.grad( cost, [W2, B2], consider_constant=[H1] )
    #g_W1, g_B1 = T.grad( cost, [W1, B1], consider_constant=[X] )

    ###### training ######

    givens_train = { X : train_x[ i*batch_size : (i+1)*batch_size ],  
                     Y : train_y[ i*batch_size : (i+1)*batch_size ],
                     Y_exp : train_y_expand[ i*batch_size : (i+1)*batch_size]  }

    train_inv = theano.function( [i,e], [], givens = givens_train, on_unused_input='ignore', 
        updates=[ (V2, V2 - __lr_b * g_V2),
                  (C2, C2 - __lr_b * g_C2)
                ])

    train_ff_sync = theano.function( [i], [cost, err], givens = givens_train, on_unused_input='ignore', 
        updates=[ (W1 , W1 - (__lr_f/batch_size) * g_W1),
                  (B1 , B1 - (__lr_f/batch_size) * T.sum(g_B1, axis=0)),
                  (W2 , W2 - (__lr_f/batch_size) * g_W2),
                  (B2 , B2 - (__lr_f/batch_size) * T.sum(g_B2, axis=0)),
                  (W3 , W3 - (__lr_f/batch_size) * g_W3),
                  (B3 , B3 - (__lr_f/batch_size) * T.sum(g_B3, axis=0))
                  ])

    # evaluation
    eval_valid = theano.function([], [err], on_unused_input='ignore', givens={ X : valid_x, Y : valid_y }  )
    eval_test  = theano.function([], [err], on_unused_input='ignore', givens={ X : test_x,  Y : test_y  }  )

    print
    print "__lr_f, __lr_b:", __lr_f, __lr_b
    print 'epoch cost train_err valid_err test_err time(sec)'


    # training loop
    t = time.time()
    monitor = { 'train' : [], 'train_desc' : 'cost, err', 
                'valid' : [], 'valid_desc' : 'err',
                'test'  : [], 'test_desc'  : 'err', }

    # training loop
    t = time.time(); monitor = { 'train' : [], 'valid' : [], 'test' : [] }

    for e in range(1,max_epochs+1) :
        for i in range(n_batches) : train_inv(i,e)
        monitor['train'].append(  np.array([ train_ff_sync(i) for i in range(n_batches) ]).mean(axis=0)  )

        if e % 10 == 0 :
            monitor['valid'].append( eval_valid() )
            monitor['test' ].append( eval_test()  )
            print e, monitor['train'][-1][0], monitor['train'][-1][1], monitor['valid'][-1][0], monitor['test'][-1][0], time.time() - t


for i in range(10) : exp(0.01, 0.01)


import tensorflow as tf 
import tensorflow_probability as tfp 
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
tfd = tfp.distributions 

true_mean = np.zeros([2], dtype = np.float32)

true_cor = np.array([[1.0, 0.9], [0.9, 1.0]], dtype = np.float32)

true_var = np.array([4.0, 1.0], dtype = np.float32)

true_cov = np.expand_dims(np.sqrt(true_var), axis = 1).dot(np.expand_dims(np.sqrt(true_var), axis = 0))*true_cor


true_precision = np.linalg.inv(true_cov)

print(true_cov)

print("eigenvalues: ", np.linalg.eigvals(true_cov))

np.random.seed(123)

my_data = np.random.multivariate_normal(mean = true_mean, cov = true_cov, size = 100, check_valid = 'ignore').astype(np.float32)


def log_lik_data_numpy(precision, data):
  # np.linalg.inv is a really inefficient way to get the covariance matrix, but
  # remember we don't care about speed here
  cov = np.linalg.inv(precision)
  rv = scipy.stats.multivariate_normal(true_mean, cov)
  return np.sum(rv.logpdf(data))

print(log_lik_data_numpy(true_precision, my_data))



def log_lik_prior_numpy(precision):
    prior_df = 3
    prior_scale = np.eye(2, dtype = np.float32) / prior_df
    rv = scipy.stats.wishart(df = prior_df, scale = prior_scale)
    return rv.logpdf(precision)

# test case: compute the prior for the true parameters
print(log_lik_prior_numpy(true_precision))

# using tensorflow

with tf.Graph().as_default() as g:
    # case 1: get log probabilities for a vector of iid draws from a single
    # normal distribution
    norm1 = tfd.Normal(loc = 0., scale = 1.0)
    probs1 = norm1.log_prob(tf.constant([1., 0.5, 0.]))

    norm2 = tfd.Normal(loc = [0., 2., 4.], scale = [1., 1., 1.])
    probs2 = norm2.log_prob(tf.constant([1., 0.5, 0.]))
    g.finalize()

with tf.compat.v1.Session(graph = g) as sess:
    print('iid draws from a single normal:', sess.run(probs1))
    print('draws from a batch of normals:', sess.run(probs2))

print(my_data.shape)
print(np.expand_dims(my_data, axis = 1).shape)

print(np.tile(np.expand_dims(my_data, axis = 1), reps = [1,2,1]).shape)

replicated_data = np.tile(np.expand_dims(my_data, axis = 1), reps = [1,2,1])

precisions = np.stack([np.eye(2, dtype = np.float32), true_precision])

print(precisions)
print(precisions.shape)

print(true_precision)

init_precision = tf.expand_dims(tf.eye(2), axis = 0)

print(init_precision.shape)

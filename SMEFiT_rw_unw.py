'''

##======================================##
|| Code for reweighting and unweighting ||
##======================================##

This code applies the reweighting & unweighting procedure based on new (unseen)
experimental data on a number of theoretical predictions.

+--------------+
| Preparations |
+--------------+

In order to run the code, one needs to have the data of an ensemble of Wilson
coefficients sets with corresponding chi2 data. The Wilson coefficients should
have been used as free parameters in a SMEFT theory prediction for the new data
of which the chi2 is determined. In the folder "rw_input_data/", sets of Wilson
coefficients and chi2 data are available.

Needed Python packages:
 * numpy
 * tabulate
 * scipy.stats

+-------------+
| Code set up |
+-------------+

- Load in an ensemble of Wilson coefficients sets. Data files contain:
  i)     the names of the relevant operators for the data ensemble that was used
         in the prior fit
  ii)    the best fit values for the wilson coefficients
  iii)   95% confidence level values
- Load in the chi2 data. The chi2 datafiles contain:
  i)     the chi2
  ii)    number of datapoints of added new data
  iii)   the normalized chi2
- Calculate the weights for each theory prediction according to the
  corresponding chi2.
- Determine the Shannon entropy
- Multiply the coefficients sets by the corresponding weights to obtain the
  reweighted sets
- Apply unweighting to the weights to obtain integer weights
- Take a number copies of the predictions in the prior ensemble according to the
  corresponding integer weights to obtain the unweighted set
  unweighted set.

'''

'''
+----------------------------+
| Load prior and other input |
+----------------------------+
'''

# Import packages and general settings
import numpy as np
import tabulate as tab
import scipy.stats as stats
np.set_printoptions(precision=3)

# Number of replicas in the prior fit
n_reps = 1000
print('\n* Number of replicas :\n ', n_reps)

# Load in the prior distributions for the coefficients and the chi2 data
prior_coeffs_list = []
chi2_list         = []
op_names_list     = []
for rep_number in np.arange(1,n_reps+1) :

    # Loop over the files containing the prior Wilson coefficients
    prior_data_per_rep = open('rw_input_data/wilson_coeffs_NLO_HO_excl_single_top_crossvalON/SMEFT_coeffs_' + str(rep_number) + '.txt')

    op_names       = np.asarray(prior_data_per_rep.readline().split('\t')[:-1])
    coeffs_per_rep = np.asarray(prior_data_per_rep.readline().split()[:], dtype=float)
    conf_levels    = np.asarray(prior_data_per_rep.readline().split()[:], dtype=float)

    prior_data_per_rep.close()

    if rep_number == 1 :
        op_names_list.append(op_names)
    prior_coeffs_list.append(coeffs_per_rep)

    # Loop over the files with the chi2 data
    chi2_per_rep = np.loadtxt('rw_input_data/chi2_incl_1_single_top_1st_set/x2_total_rep_' + str(rep_number) + '.txt', skiprows=1)

    chi2_list.append(chi2_per_rep)

# List of operator names
op_names = np.asarray(op_names_list)[0]

# Obtain the prior distributions and standard deviations
prior_coeffs    = np.asarray(prior_coeffs_list)
prior_means     = np.mean(prior_coeffs, axis=0)
prior_variances = 1/(n_reps-1) * np.sum((prior_coeffs - prior_means)**2, axis=0)
prior_st_devs   = np.sqrt(prior_variances)
print('\n* Number of operators constrained in prior fit :\n ', len(prior_means))

'''
+-------------+
| Reweighting |
+-------------+
'''

# Obtain a 1D array with the chi2 per replica
chi2_array          = np.asarray(chi2_list)
chi2_all_reps       = np.asarray(chi2_array[:,0], dtype=float)
n_datapoints        = np.asarray(chi2_array[:,1], dtype=int)
chi2_norm_all_reps  = np.asarray(chi2_array[:,2], dtype=float)

print('\n* 10 lowest normalized chi2s :\n ', np.sort(chi2_norm_all_reps)[0:10])

# Calculate the weights
unnormalized_weights        = chi2_all_reps**(1/2*(n_datapoints-1)) * np.exp(-1/2*chi2_all_reps)
normalization               = np.sum(unnormalized_weights) / n_reps
nnpdf_weights               = unnormalized_weights / normalization

print('\n* 10 highest weights :\n ', np.sort(nnpdf_weights)[-10:])

# Replace very small weights to prevent infinities/divide-by-0's
zero_weights = np.asarray(np.where(nnpdf_weights < 1.0e-300))[0]
np.put(nnpdf_weights, zero_weights, 1e-300)
print('\n* ' + str(len(zero_weights)) + ' very small weights were replaced by 1.0e-300')

# Check that normalization is satisfied
assert np.round(np.sum(nnpdf_weights)) == n_reps, 'sum of weights should equal number of replicas'
print('\n* sum of weights :\n ', np.sum(nnpdf_weights))

# Determine the Shannon entropy (number of effective replicas)
n_eff = np.exp(1/n_reps * np.sum(nnpdf_weights * np.log(n_reps/nnpdf_weights)))
print('\n* N_eff after reweighting:\n ', n_eff)

# Obtain the reweighted distributions and standard deviations for the coefficients
rw_coeffs    = np.transpose(np.multiply(nnpdf_weights, np.transpose(prior_coeffs)))
rw_means     = np.mean(rw_coeffs, axis=0)
rw_variances = 1/(n_reps-1) * np.sum(np.transpose(nnpdf_weights* np.transpose((prior_coeffs - rw_means)**2)), axis=0)
rw_st_devs   = np.sqrt(rw_variances)

'''
+-----------+
|Unweighting|
+-----------+
'''

# Define probability and cumulants for each replica
probs  = nnpdf_weights / n_reps
probs_cumul = []
for rep_num in np.arange(1, n_reps+1) :
    probs_cumul_rep_num = np.sum(probs[0:rep_num])
    probs_cumul.append(probs_cumul_rep_num)
probs_cumul = np.asarray(probs_cumul)
assert np.round(np.max(probs_cumul), 4) == 1.0000, 'probability cumulants do not add up to 1.0000'

# Calculate the integer weights for unweighting
unw_n_reps       = n_eff
unw_weights_list = []
for rep_num in np.arange(1, n_reps+1) :
    unw_weights_rep_num_list = []
    for unw_rep_num in np.arange(1, unw_n_reps+1) :
        if rep_num == 1 :
            unw_weights_rep_num = np.heaviside(unw_rep_num/unw_n_reps - 0                     , 1.0) * np.heaviside(probs_cumul[rep_num-1] - unw_rep_num/unw_n_reps, 1.0)
        else :
            unw_weights_rep_num = np.heaviside(unw_rep_num/unw_n_reps - probs_cumul[rep_num-2], 1.0) * np.heaviside(probs_cumul[rep_num-1] - unw_rep_num/unw_n_reps, 1.0)
        unw_weights_rep_num_list.append(unw_weights_rep_num)
    unw_weights_list.append(unw_weights_rep_num_list)
unw_weights = np.sum(np.asarray(unw_weights_list, dtype=int), axis=1)

## Check that normalization is satisfied
assert np.round(np.sum(unw_weights)) == np.floor(unw_n_reps), 'integer weights after unweighting do not satisfy normalization'
print('\n* sum of integer weights after unweighting :\n ', np.sum(unw_weights))


## Obtain the unweighted distributions for the coefficients
surv_rep_nums     = np.where(unw_weights != 0 )[0]
surv_prior_coeffs = prior_coeffs[surv_rep_nums]
n_copies          = unw_weights[surv_rep_nums]
unw_coeffs        = np.repeat(surv_prior_coeffs, n_copies, axis=0)
unw_means         = np.mean(unw_coeffs, axis=0)
unw_variances     = 1/(unw_n_reps-1) * np.sum((unw_coeffs - unw_means)**2, axis=0)
unw_st_devs       = np.sqrt(unw_variances)

'''
+-------------------------------+
| Kolmogorov-Smirnov statistics |
+-------------------------------+
'''

ks_stat_list = []
for operator in np.arange(len(op_names)) :
    ks_stat = stats.ks_2samp(prior_coeffs[:,operator], unw_coeffs[:,operator])
    ks_stat_list.append(ks_stat)
ks_stats = np.asarray(ks_stat_list)[:,0]

'''
+-----------------------------+
| Load in posterior for check |
+-----------------------------+
'''

# Load in the poster distributions for the coefficients and the chi2 data
poster_coeffs_list = []
for rep_number in np.arange(1,n_reps+1) :

    # Loop over the files containing the Wilson coefficients
    poster_data_per_rep = open('rw_input_data/wilson_coeffs_NLO_HO_incl_1_single_top_crossvalON/SMEFT_coeffs_' + str(rep_number) + '.txt')

    op_names       = np.asarray(poster_data_per_rep.readline().split('\t')[:-1])
    coeffs_per_rep = np.asarray(poster_data_per_rep.readline().split()[:], dtype=float)
    conf_levels    = np.asarray(poster_data_per_rep.readline().split()[:], dtype=float)

    poster_data_per_rep.close()

    poster_coeffs_list.append(coeffs_per_rep)

# Obtain a 2D array with the coefficients for all replicas
poster_coeffs = np.asarray(poster_coeffs_list)
poster_means  = np.mean(poster_coeffs, axis=0)



# Calculate the standard deviation for the poster distributions
poster_variances = 1/(n_reps-1) * np.sum((poster_coeffs - poster_means)**2, axis=0)
poster_st_devs = np.sqrt(poster_variances)


'''
+--------------------------+
| Print table in  terminal |
+--------------------------+
'''


# Print a table with operators and standard deviations
headers = ['operator',
           'prior std dev',
           'poster st devs',
           'rw std dev',
           'unw std dev'
          ]

table   = np.stack([op_names,
                    prior_st_devs,
                    poster_st_devs,
                    rw_st_devs,
                    unw_st_devs
                 ], axis=1)
print('\n')
print(tab.tabulate(table, headers, tablefmt='github'))

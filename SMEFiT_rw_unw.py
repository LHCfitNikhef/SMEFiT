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
 * matplotlib

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
import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

# Input settings
prior_data      = 'no_single_top'
poster_data     = '1st_t_channel'
chi2_data       = poster_data
n_reps          = 1000
ks_level        = 0.3
reduction_level = 0.3

print('\n* Prior: \n ', prior_data)
print('\n* Posterior:\n ', poster_data)
print('\n* Number of replicas :\n ', n_reps)

# Load in the prior distributions for the coefficients and the chi2 data
prior_coeffs_list = []
chi2_list = []
op_names_list = []

for rep_number in np.arange(1, n_reps+1):

    # Loop over the files containing the prior Wilson coefficients
    prior_data_per_rep = open('rw_input_data/wilson_coeffs/' + prior_data + '/SMEFT_coeffs_' + str(rep_number) + '.txt')

    op_names = np.asarray(prior_data_per_rep.readline().split('\t')[:-1])
    coeffs_per_rep = np.asarray(prior_data_per_rep.readline().split()[:], dtype=float)
    conf_levels = np.asarray(prior_data_per_rep.readline().split()[:], dtype=float)

    prior_data_per_rep.close()

    if rep_number == 1:
        op_names_list.append(op_names)

    prior_coeffs_list.append(coeffs_per_rep)

    # Loop over the files with the chi2 data
    chi2_per_rep = np.loadtxt('rw_input_data/chi2_data/' + chi2_data + '/x2_total_rep_' + str(rep_number) + '.txt', skiprows=1)

    chi2_list.append(chi2_per_rep)

# List of operator names
op_names = np.asarray(op_names_list)[0]

# Obtain the prior distributions and standard deviations
prior_coeffs = np.asarray(prior_coeffs_list)
prior_means = np.mean(prior_coeffs, axis=0)
prior_variances = 1/(n_reps-1) * np.sum((prior_coeffs - prior_means)**2, axis=0)
prior_st_devs = np.sqrt(prior_variances)
print('\n* Number of operators constrained in prior fit :\n ', len(prior_means))

'''
+-------------+
| Reweighting |
+-------------+
'''

# Obtain a 1D array with the chi2 per replica
chi2_array = np.asarray(chi2_list)
chi2_all_reps = np.asarray(chi2_array[:, 0], dtype=float)
n_datapoints = np.asarray(chi2_array[:, 1], dtype=int)
chi2_norm_all_reps = np.asarray(chi2_array[:, 2], dtype=float)

print('\n* 10 lowest normalized chi2s :\n ', np.sort(chi2_norm_all_reps)[0:10])

# Calculate the weights
unnormalized_weights = chi2_all_reps**(1/2*(n_datapoints-1)) * np.exp(-1/2*chi2_all_reps)
normalization = np.sum(unnormalized_weights) / n_reps
nnpdf_weights = unnormalized_weights / normalization

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
rw_coeffs = np.transpose(np.multiply(nnpdf_weights, np.transpose(prior_coeffs)))
rw_means = np.mean(rw_coeffs, axis=0)
rw_variances = 1/(n_reps-1) * \
			   np.sum(np.transpose(nnpdf_weights* np.transpose((prior_coeffs - rw_means)**2)), axis=0)
rw_st_devs = np.sqrt(rw_variances)

'''
+-----------+
|Unweighting|
+-----------+
'''

# Define probability and cumulants for each replica
probs = nnpdf_weights/n_reps
probs_cumul = []

for rep_num in np.arange(1, n_reps+1):
    probs_cumul_rep_num = np.sum(probs[0:rep_num])
    probs_cumul.append(probs_cumul_rep_num)

probs_cumul = np.asarray(probs_cumul)
assert np.round(np.max(probs_cumul), 4) == 1.0000, 'probability cumulants do not add up to 1.0000'

# Calculate the integer weights for unweighting
unw_n_reps = n_eff
unw_weights_list = []
print('\n* Computing the unweighted set ...')
for rep_num in np.arange(1, n_reps+1):
    unw_weights_rep_num_list = []

    for unw_rep_num in np.arange(1, unw_n_reps+1):

        if rep_num == 1:
            unw_weights_rep_num = np.heaviside(unw_rep_num/unw_n_reps - 0, 1.0) \
            					  *np.heaviside(probs_cumul[rep_num-1]-unw_rep_num/unw_n_reps, 1.0)

        else:
            unw_weights_rep_num = np.heaviside(unw_rep_num/unw_n_reps - probs_cumul[rep_num-2], 1.0) \
            					  *np.heaviside(probs_cumul[rep_num-1]-unw_rep_num/unw_n_reps, 1.0)

        unw_weights_rep_num_list.append(unw_weights_rep_num)
    unw_weights_list.append(unw_weights_rep_num_list)
unw_weights = np.sum(np.asarray(unw_weights_list, dtype=int), axis=1)

## Check that normalization is satisfied
assert np.round(np.sum(unw_weights)) == np.floor(unw_n_reps), \
'integer weights after unweighting do not satisfy normalization'
print('\n* sum of integer weights after unweighting :\n ', np.sum(unw_weights))


## Obtain the unweighted distributions for the coefficients
surv_rep_nums = np.where(unw_weights != 0)[0]
surv_prior_coeffs = prior_coeffs[surv_rep_nums]
n_copies = unw_weights[surv_rep_nums]
unw_coeffs = np.repeat(surv_prior_coeffs, n_copies, axis=0)
unw_means = np.mean(unw_coeffs, axis=0)
unw_variances = 1/(unw_n_reps-1) * np.sum((unw_coeffs - unw_means)**2, axis=0)
unw_st_devs = np.sqrt(unw_variances)

'''
+-------------------------------+
| Kolmogorov-Smirnov statistics |
+-------------------------------+
'''

ks_stat_list = []

for operator in np.arange(len(op_names)):
    ks_stat = stats.ks_2samp(prior_coeffs[:, operator], unw_coeffs[:, operator])
    ks_stat_list.append(ks_stat)

ks_stats = np.asarray(ks_stat_list)[:, 0]

'''
+-----------------------------+
| Load in posterior for check |
+-----------------------------+
'''

# Load in the poster distributions for the coefficients and the chi2 data
poster_coeffs_list = []
for rep_number in np.arange(1, n_reps+1):

    # Loop over the files containing the Wilson coefficients
    poster_data_per_rep = open('rw_input_data/wilson_coeffs/' + poster_data + '/SMEFT_coeffs_' + str(rep_number) + '.txt')

    op_names = np.asarray(poster_data_per_rep.readline().split('\t')[:-1])
    coeffs_per_rep = np.asarray(poster_data_per_rep.readline().split()[:], dtype=float)
    conf_levels = np.asarray(poster_data_per_rep.readline().split()[:], dtype=float)

    poster_data_per_rep.close()

    poster_coeffs_list.append(coeffs_per_rep)

# Obtain the posterior distributions
poster_coeffs = np.asarray(poster_coeffs_list)
poster_means = np.mean(poster_coeffs, axis=0)
poster_variances = 1/(n_reps-1) * np.sum((poster_coeffs - poster_means)**2, axis=0)
poster_st_devs = np.sqrt(poster_variances)

# Determine the reduction of the standard deviations
reduction_poster = 1 - poster_st_devs/prior_st_devs
no_reduction_poster = np.where(reduction_poster < 0)
np.put(reduction_poster, no_reduction_poster, 0.0)

reduction_rw = 1 - rw_st_devs/prior_st_devs
no_reduction_rw = np.where(reduction_rw < 0)
np.put(reduction_rw, no_reduction_rw, 0.0)

# Obtain the operators that satisfy the KS level and the sigma reduction level
ops_satisfy_ks = np.where(ks_stats > ks_level)
ops_satisfy_red = np.where(reduction_poster > reduction_level)
constr_op_nums = np.intersect1d(ops_satisfy_ks, ops_satisfy_red)

constr_prior_st_devs = np.take(prior_st_devs, constr_op_nums)
constr_poster_st_devs = np.take(poster_st_devs, constr_op_nums)
constr_rw_st_devs = np.take(rw_st_devs, constr_op_nums)
constr_unw_st_devs = np.take(unw_st_devs, constr_op_nums)
constr_op_names = np.take(op_names, constr_op_nums)
constr_prior_coeffs = np.take(np.transpose(prior_coeffs), constr_op_nums, axis=0)
constr_poster_coeffs = np.take(np.transpose(poster_coeffs), constr_op_nums, axis=0)
constr_rw_coeffs = np.take(np.transpose(rw_coeffs), constr_op_nums, axis=0)
constr_unw_coeffs = np.take(np.transpose(unw_coeffs), constr_op_nums, axis=0)
print('\n* Constrained operators : \n ', constr_op_names)


'''
+--------------+
| Chi2 profile |
+--------------+
'''

alphas = 0.2*np.arange(1,11)
for alpha in alphas :
    rescaled_chi2 = chi2_all_reps/(alpha**2)
    rescaled_unnormalized_weights = rescaled_chi2**(1/2*(n_datapoints-1)) * np.exp(-1/2*rescaled_chi2)
    scaled_normalization = np.sum(rescaled_unnormalized_weights) / n_reps
    rescaled_weights = rescaled_unnormalized_weights / scaled_normalization
    rescaled_chi2_profile = 1/alpha * np.sum(rescaled_weights)
    print(rescaled_chi2_profile)



'''
+--------------------------+
| Print table in  terminal |
+--------------------------+
'''

def print_info_to_terminal() :

    # Print a table with operators and standard deviations
    headers = ['operator',
               'prior std dev',
               'poster st devs',
               'rw std dev',
               'unw std dev',
               'KS stat',
              ]

    terminal_table = np.stack([op_names,
                      prior_st_devs,
                      poster_st_devs,
                      rw_st_devs,
                      unw_st_devs,
                      ks_stats,
                      ], axis=1)

    print('\n')
    print(tab.tabulate(terminal_table, headers, tablefmt='github', floatfmt='.2f'))

    return None

print_info_to_terminal()

'''
+-----------------+
| Saving in files |
+-----------------+
'''

# format the list of operator names
names_list = []
for name in np.arange(len(op_names)) :
    formatted_name = '{:^11}'.format(op_names[name])
    names_list.append(formatted_name)
names_header = ''.join((np.asarray(names_list, dtype=str)))

# save unweighted set in text file
np.savetxt('unw_coeffs.txt', unw_coeffs, fmt='%10.5f', header=names_header, comments='')


'''
+------------------+
| Plotting section |
+------------------+
'''

# Define colors
color_purple        = (0.85, 0.4, 0.55)
color_turqoise      = (0.20, 0.70, 0.60)
color_dark_turqoise = (0.20, 0.50, 0.50)
color_yellow        = (0.95, 0.75, 0.20)
color_green         = (0.10, 0.60, 0.40)
color_green_line    = (0.05, 0.40, 0.10)
color_light_grey    = (0.85, 0.85, 0.85)
color_dark_grey     = (0.00, 0.00, 0.00, 0.60)
color_white         = (0.95,0.95,0.95)

# General settings for plotting
plt.rc('axes', axisbelow=True)
plt.rc('axes', edgecolor=color_dark_grey)
plt.rcParams['xtick.color'] = color_dark_grey
mpl.rcParams['ytick.color'] = color_dark_grey

def plot_two_sigma_bounds() :

    print('\n* Plotting two sigma bounds ...')

    # make figure opbject
    fig, axes = plt.subplots(3,1, sharex=True)
    ax1, ax2, ax3 = axes
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    # needed for bar plots
    bar_width = 0.15
    bar_shift = 0.03
    op_list = np.arange(1, len(op_names)+1)

    # two sigma bounds prior/poster/rw/unw
    ax1.bar(op_list, 2.0*prior_st_devs, label='prior', width=4.1*bar_width, color='black', align='center', alpha=0.7)
    ax1.bar(op_list-bar_width-bar_shift, 2.0*poster_st_devs, label='posterior' , width=bar_width, color=color_purple, align='center' )
    ax1.bar(op_list, 2.0*rw_st_devs, label='reweighted', width=bar_width, color=color_turqoise, align='center')
    ax1.bar(op_list+bar_width+bar_shift, 2.0*unw_st_devs, label='unweighted', width=bar_width, color=color_yellow, align='center')

    # layout settings two sigma bounds
    ax1.grid(True, axis='y', color=color_light_grey)
    ax1.set_yscale('log')
    ax1.set_xticks(op_list)
    ax1.set_ylabel('$2\sigma$ [TeV$^{-2}$]', color=color_dark_grey)
    legend1 = ax1.legend(loc='best', ncol=2, facecolor=color_white)

    # reduction plot
    ax2.axhline(y=reduction_level, color=color_green_line, lw=2, alpha=0.6, linestyle='dotted')
    ax2.bar(op_list, reduction_poster, label='1 - $\\frac{\sigma_{post}}{\sigma_{prior}}$', color=color_purple  , width=1.5*bar_width)
    ax2.bar(op_list, reduction_rw, label='1 - $\\frac{\sigma_{rw}}{\sigma_{prior}}$', color=color_dark_turqoise, width=3.0*bar_width, alpha=0.3)

    # layout reduction plot
    ax2.grid(True, axis='y', color=color_light_grey)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_ylabel('Reduction in $\sigma$', color=color_dark_grey)
    legend2 = ax2.legend(loc='best', ncol=2, facecolor=color_white)

    # plot KS stats
    ax3.axhline(y=ks_level, color=color_green_line, lw=2, alpha=0.6, linestyle='dotted')
    ax3.plot(op_list, ks_stats, 'd', label='$KS$-stat', color=color_dark_turqoise, ms=4, mfc='maroon')
    ax3.vlines(op_list, 0, ks_stats, color=color_dark_turqoise, lw=1, linestyle='dashed')

    # layout KS stats plot
    ax3.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
    ax3.grid(True, axis='y', color=color_light_grey)
    ax3.set_xticklabels(op_names, rotation=90, fontsize=8)
    ax3.set_ylabel('KS-stat', color=color_dark_grey)

    # save figure
    fig.savefig('two_sigma_bounds_KS_reduction_' + poster_data + '.pdf', dpi=1000, bbox_inches='tight')

    return None

def plot_distr_constr_ops() :

    print('\n* Plotting coefficient distributions ...')

    # needed for histogram
    nbins = 30
    constr_op_list = np.arange(1, len(constr_op_names)+1)

    # loop over the constrained operators
    for oper in np.arange(len(constr_op_nums)) :

        # make figure opbject
        fig, ax = plt.subplots()
        fig.tight_layout()

        # make histograms of the coefficients
        hist_range = [-10, 10]

        # prior histogram
        ax.hist(constr_prior_coeffs[oper], bins=nbins, density=True, range=hist_range, histtype='stepfilled', color=color_dark_grey, label='prior', alpha=0.15)
        ax.hist(constr_prior_coeffs[oper], bins=nbins, density=True, range=hist_range, histtype='step', color=color_dark_grey, alpha=0.3)

        # posterior histogram
        ax.hist(constr_poster_coeffs[oper], bins=nbins, density=True, range=hist_range, histtype='stepfilled', color=color_purple, label='posterior', alpha=0.4)

        # unweighted histogram
        ax.hist(constr_unw_coeffs[oper], bins=nbins, density=True, range=hist_range, histtype='stepfilled', color=color_turqoise, lw=4, alpha=0.1)
        ax.hist(constr_unw_coeffs[oper], bins=nbins, density=True, range=hist_range, label='unweighted', histtype='step', lw=2, color=color_turqoise)

        # add legend
        legend = ax.legend(loc='best', title='$' + str(constr_op_names[oper]) + '$', title_fontsize=13, facecolor=color_white)
        legend._legend_box.align = 'left'

        # set x and y labels
        ax.set_xlabel('$c_{' + str(constr_op_names[oper]) + '}$', color=color_dark_grey, fontsize=12)
        ax.set_ylabel('probability density', color=color_dark_grey, fontsize=12)

        # save figure
        fig.savefig('distr_' + str(constr_op_names[oper]) + '_' + poster_data + '.pdf', dpi=1000, bbox_inches='tight')

    return None


# plot_distr_constr_ops()
plot_two_sigma_bounds()
print('\n* Plots are produced and saved')

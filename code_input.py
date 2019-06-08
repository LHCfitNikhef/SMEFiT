'''
+-------------------------------------+
| Input settings for reweighting code |
+-------------------------------------+

The settings for the reweighting procedure can be set in this file.
One needs to choose the following:
(1) the number of replicas used for reweighting
(2) the minimal required error reduction
(3) the minimal required KS-statistic
(4) decide if plots need to be produced
(5) dataset for the prior
(6) dataset for the posterior

Run 'SMEFiT_rw_unw.py' after the options have been set.
'''

# (1) Set the number of replicas
n_reps          = 10000

# (2) Set the minimal required error reduction
reduction_level = 0.3

# (3) Set the minimal required KS-statistic
ks_level        = 0.3

# (4) Set 'on' to produce plots
produce_plots   = 'on'

# (5) Choose a prior from the list
prior_data_list = [
                   'no_single_top',
                   # 'all_datasets',
                   # '1st_t_channel',
                   # 'only_ttbar',
                   # 's_channel',
                   # 't_channel',
                  ]
prior_data = prior_data_list[0]


# (6) Choose a posterior from the list
poster_data_list = [
                    # 'no_single_top',
                    # 'all_datasets',
                    # '1st_t_channel',
                    # 'only_ttbar',
                    # 's_channel',
                    't_channel',
                   ]
poster_data = poster_data_list[0]

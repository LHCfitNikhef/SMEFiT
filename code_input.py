
'''
+----------------------------+
| Input for reweighting code |
+----------------------------+
'''

# Set the number of replicas, the KS criterium and required error reduction
n_reps          = 1000
ks_level        = 0.3
reduction_level = 0.3

# Choose one of the priors
prior_data_list = [
                   'no_single_top',
                   # 'all_datasets',
                   # '1st_t_channel',
                   # 'only_ttbar',
                   # 's_channel',
                   # 't_channel',
                  ]
prior_data = prior_data_list[0]


# Choose the posterior for validation
poster_data_list = [
                    # 'no_single_top',
                    # 'all_datasets',
                    '1st_t_channel',
                    # 'only_ttbar',
                    # 's_channel',
                    # 't_channel',
                   ]
poster_data = poster_data_list[0]

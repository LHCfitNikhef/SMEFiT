# # -*- coding: utf-8 -*-
# import numpy as np
# import pandas as pd
# from collections.abc import Iterable

# import matplotlib.pyplot as py
# from matplotlib import colors as matcolors



# class CoefficientsPlotter:
#     """
#     Plots central values + 95% CL errors, 95% CL bounds,
#     probability distributions, residuals,
#     residual distribution, and energy reach.

#     Also writes a table displaying values for
#     68% CL bounds and central value + 95% errors.

#     Takes into account parameter constraints and displays
#     all non-zero parameters.

#     Uses coeff_groups.yaml YAML file to organize parameters
#     in a meaningful way.

#     Note: coefficients that are known to have disjoint
#     probability distributions (i.e. multiple solutions)
#     are manually separated by including the coefficient name
#     in disjointed_list for disjointed_list2
#     for global and single fit results, respectively.

#     Note: Probability distributions for single parameter
#     fits are included for all coefficients EXCEPT those
#     constrained as a linear combination of two or more parameters
#     (since they have different numbers of posterior samples)

#     Parameters
#     ----------
#         analyzer : Analyzer
#             analysis class object
#     """

#     def __init__(self, analyzer, logo = False):
#         # inherited variables
#         self.path = analyzer.path
#         self.outputname = analyzer.outputname
#         self.fits = analyzer.fits
#         self.config = analyzer.config
#         self.full_coeffs = analyzer.full_coeffs

#         # SMEFiT logo
#         if logo:
#             self.logo = py.imread(f'{self.path}/smefit/analyze/logo.png')
#         else:
#             self.logo = None

#         # TODO: use get_coeff_organised ???
#         #Load coefficient yaml card
#         yaml_file = f'{self.path}/run_cards/analyze/coeff_groups.yaml'
#         with open(yaml_file) as f:
#             coeff_config = yaml.safe_load(f)

#         coeff_list = []
#         for group in coeff_config.keys():
#             for c in coeff_config[group]:
#                 if c=='OPff':
#                     c='Off'
#                 if c in ['OW','OB']:
#                     continue
#                 if np.any(['_c' in k for k in self.fits])\
#                         and np.any([self.config[k]['coefficients'][c]['fixed'] is not False for k in self.fits if c in self.config[k]['coefficients']]):
#                         continue
#                 if c not in coeff_list and\
#                         np.any([c in self.config[k]['coefficients'] for k in self.fits]) and\
#                         np.any([self.config[k]['coefficients'][c]['fixed']  is not True for k in self.fits if c in self.config[k]['coefficients']]):
#                     coeff_list.append(c)

#         self.coeff_list = coeff_list.copy()
#         self.npar = len(self.coeff_list)
#         self.fit_labels = analyzer.fit_labels
#         self.full_labels = analyzer.full_labels

#         # TODO: do we need this, better self.labels???
#         self.labels = coeff_list.copy()
#         self.coeffs = analyzer.coeffs

#         self.disjointed_list = ['Otap','Obp']
#         self.disjointed_list2 = ['Otap','Obp','Opd','OpB']
#         if any(['TOP19' in k for k in self.fits]):
#             self.disjointed_list.append('Otp')

#     @staticmethod
#     def get_conficence_values(temp):
#         """
#         Get confidence level bounds given the distribution
#         """
#         cl_vals = {}
#         cl_vals['low'] = np.nanpercentile(temp,16)
#         cl_vals['high'] = np.nanpercentile(temp,84)
#         cl_vals['low95'] = np.nanpercentile(temp,2.5)
#         cl_vals['high95'] = np.nanpercentile(temp,97.5)

#         return cl_vals

#     @staticmethod
#     def get_cl_erros(cl_vals):
#         error68 = ( cl_vals['high'] - cl_vals['low']) / 2.0
#         error95 = (cl_vals['high95'] - cl_vals['low95']) / 2.0 
#         return error68, error95


#     def set_double_cl(self, full_solution, l):

#         min_val = min(full_solution)
#         max_val = max(full_solution)
#         mid = (max_val + min_val) / 2.0

#         if l=='Obp' or l=='Opd':
#             solution1 = full_solution[full_solution>mid]
#             solution2 = full_solution[full_solution<mid]
#         else:
#             solution1 = full_solution[full_solution<mid]
#             solution2 = full_solution[full_solution>mid]

#         # First solution
#         cl_vals_1 = self.get_conficence_values(solution1)
#         cl_vals_1['mid'] = np.mean(solution1,axis=0)
#         error68, error95 = self.get_cl_erros( cl_vals_1 )
#         first_err = error95

#         # Second solution
#         cl_vals_2 = self.get_conficence_values(solution2)
#         cl_vals_2['mid'] = np.mean(solution2,axis=0)
#         temp = self.get_cl_erros( cl_vals_2 )
#         error68 += temp[0]
#         error95 += temp[1]

#         double_bounds = {'1': cl_vals_1, '2': cl_vals_2, 'errors': [error68, error95], 'second_err': temp[1], 'first_err': first_err}
#         return double_bounds,  1./np.sqrt(error95), cl_vals_1['mid'] / error68  #[ 1./np.sqrt(error68), 1./np.sqrt(error95) ], [ mid / error68 , mid / error95 ]


#     def compute_confidence_level(self,):
#         """Compute 95 % and 68 % confidence levels, energy bounds and residuals"""

#         residuals = {}
#         energy_bounds = {}
#         bounds = {}
#         error_bounds = {}

#         for k in self.fits:

#             name=self.fit_labels[k] #r'${\rm %s}$'%k.replace('_','\_')
#             bounds[name] = {}
#             energy_bounds[name] = np.zeros(self.npar)
#             residuals[name] = np.zeros(self.npar)
#             error_bounds[name] = np.zeros(self.npar)

#             cnt=0
#             for l in self.coeff_list:

#                 if l in self.full_labels[k]:
#                     idx=np.where(np.array(self.full_labels[k])==l)[0][0]
#                 else:
#                     cnt+=1
#                     continue

#                 # Single operator fits
#                 if '-SNS_' in k:
#                     if self.config[k]['coefficients'][l]['fixed'] is not False and self.config[k]['coefficients'][l]['fixed'] is not True:
#                         if isinstance(self.config[k]['coefficients'][l]['value'], Iterable):
#                             mid=0
#                             error68=0
#                             error95=0
#                             for j in range(len(self.config[k]['coefficients'][l]['value'])):
#                                 coeff = self.config[k]['coefficients'][l]['fixed'][j]
#                                 coeff_idx = np.where(np.array(self.labels[k])==coeff)[0][0]
#                                 cl_vals = self.get_conficence_values( np.array(self.coeffs[k][coeff_idx]) )
#                                 mid += (np.mean(temp,axis=0)*self.config[k]['coefficients'][l]['value'][j])/len(self.config[k]['coefficients'][l]['value'])
#                                 temp = self.get_cl_erros( cl_vals )
#                                 error68 += temp[0] ** 2
#                                 error95 += temp[1] ** 2
#                             cl_vals['mid'] = mid
#                             error68 = np.sqrt(error68)
#                             error95 = np.sqrt(error95)
#                         else:
#                             coeff = self.config[k]['coefficients'][l]['fixed']
#                             coeff_idx = np.where(np.array(self.labels[k])==coeff)[0][0]
#                             temp = np.array(self.coeffs[k][coeff_idx])*self.config[k]['coefficients'][l]['value']
#                             cl_vals = self.get_conficence_values(temp) 
#                             cl_vals['mid'] = np.mean(temp,axis=0)
#                             error68, error95 = self.get_cl_erros( cl_vals )
#                     else:
#                         idx2 = np.where(np.array(self.labels[k])==l)[0][0]
#                         # double solution
#                         if '_HO' in k and l in self.disjointed_list2:
#                             full_solution = np.array(self.coeffs[k][idx2])
#                         # single solution
#                         else:
#                             cl_vals = self.get_conficence_values(np.array(self.coeffs[k][idx2]))
#                             cl_vals['mid'] = np.mean(np.array(self.coeffs[k][idx2]),axis=0)
#                             error68, error95 = self.get_cl_erros( cl_vals )
           
#                 # non single operator fits
#                 else:
#                     # double solution
#                     if '_HO' in k and l in self.disjointed_list:
#                         full_solution = np.transpose(np.array(self.full_coeffs[k]))[idx]
#                     # single solution
#                     else:
#                         cl_vals = self.get_conficence_values( np.transpose(np.array(self.full_coeffs[k]))[idx]) 
#                         cl_vals['mid'] = np.mean(np.transpose(np.array(self.full_coeffs[k]))[idx],axis=0)
#                         error68, error95 = self.get_cl_erros( cl_vals )

#                 # double soultion
#                 if '_HO' in k and l in self.disjointed_list:
#                     bounds[name][cnt], energy_bounds[name][cnt], residuals[name][cnt] = self.set_double_cl(full_solution,l)
#                     error_bounds[name][cnt] = bounds[name][cnt]['errors'][1]
#                 # single solution
#                 else:
#                     bounds[name][cnt] = { '1': cl_vals, 'errors' : [error68, error95], 'second_err': 0 }
#                     energy_bounds[name][cnt] = 1./np.sqrt(error95)  #[ 1./np.sqrt(error68), 1./np.sqrt(error95) ]
#                     residuals[name][cnt] = cl_vals['mid'] / error68  # [ cl_vals['mid'] / error68 , cl_vals['mid'] / error95 ]
#                     error_bounds[name][cnt] = error95
         
#                 cnt+=1
#         return bounds, energy_bounds, residuals, error_bounds


#     def plot_coeffs(self, bounds):
#         """Plot central value + 95% CL errors"""

#         nrows,ncols=1,1
#         py.figure(figsize=(nrows*10,ncols*5))

#         ax=py.subplot(111)

#         # X-axis
#         X=np.array([2*i for i in range(self.npar)])
#         # Spacing between fit results
#         val = np.linspace(-0.1*len(self.coeffs),0.1*len(self.coeffs),len(self.coeffs))

#         colors = py.rcParams['axes.prop_cycle'].by_key()['color']
#         i=0
#         for k in self.fits:
#             name = self.fit_labels[k] #r'${\rm %s}$'%k.replace('_','\_')
#             cnt = 0
#             for vals in bounds[name].values():
#                 if cnt==0:
#                     ax.errorbar( X[cnt]+val[i], y = np.array(vals['1']['mid']) , yerr = np.array(vals['errors'][1] - vals['second_err']), \
#                         color=colors[i], fmt='.', elinewidth=2, label=name)
#                 else:
#                     ax.errorbar( X[cnt]+val[i], y=np.array(vals['1']['mid']), yerr = np.array(vals['errors'][1] - vals['second_err']), \
#                         color=colors[i], fmt='.', elinewidth=2)
                
#                 # double soluton
#                 if '_HO' in k and vals['second_err'] != 0:
#                     ax.errorbar(X[cnt]+val[i]+0.5, y=np.array(vals['2']['mid']), yerr= np.array(vals['second_err']), \
#                         color=colors[i], fmt='.', elinewidth=2)
#                 cnt+=1
#             i+=1

#         if self.logo is not None:
#             ax.imshow(self.logo, aspect='auto',transform=ax.transAxes,extent=[.775,.975,.05,.2],zorder=-1, alpha=1)

#         py.plot([i for i in range(-1,200)],np.zeros(201),'k--',alpha=0.7)

#         py.yscale('symlog',linthreshy=1e-1)
#         py.ylim(-200,200)
#         py.yticks([-100,-10,-1,-0.1,0,0.1,1,10,100],[r'$-100$',r'$-10$',r'$-1$',r'$-0.1$',r'$0$',r'$0.1$',r'$1$',r'$10$',r'$100$'])
#         py.ylabel(r'$c_i/\Lambda^2\ ({\rm TeV}^{-2})$',fontsize=18)

#         py.xlim(-1,(self.npar)*2-1)
#         py.tick_params(which='major',direction='in',labelsize=13)
#         my_xticks = [c.replace('O','c') for c in self.coeff_list]
#         py.xticks(X,my_xticks,rotation=90)

#         py.legend(loc=0,frameon=False,prop={'size':13})
#         py.tight_layout()
#         py.savefig(f'{self.path}/reports/{self.outputname}/Coeffs_Central.pdf')


#     def plot_coeffs_bar(self, error95 ):
#         """ Plot 95% CLs for coefficients (bar plot)"""
        
#         py.figure(figsize=(7,5))
#         df = pd.DataFrame.from_dict(error95,orient='index',columns=[c.replace('O','c') for c in self.coeff_list])
#         new_df = df.T

#         ax = new_df.plot(kind='bar',rot=0,width=0.7,figsize=(10,5))

#         if self.logo is not None:
#             ax.imshow(self.logo, aspect='auto',transform=ax.transAxes,extent=[.775,.975,.825,.975],zorder=-1, alpha=1)

#         #Hard cutoff
#         py.plot(np.linspace(-1,2*self.npar+1,2),50*np.ones(2),'k--',alpha=0.7,lw=2)

#         py.xticks(rotation=90)
#         py.tick_params(axis='y',direction='in',labelsize=15)
#         py.yscale('log')
#         py.ylabel(r'$95\%\ {\rm Confidence\ Level\ Bounds}\ (1/{\rm TeV}^2)$',fontsize=11)
#         py.ylim(1e-3,1e3)
#         py.legend(loc=2,frameon=False,prop={'size':11})
#         py.tight_layout()
#         py.savefig(f'{self.path}/reports/{self.outputname}/Coeffs_Bar.pdf')


#     def plot_residuals_bar(self, residual68):
#         """ Plot residuals at 68% CL (bar plot) """
#         py.figure(figsize=(7,5))

#         df = pd.DataFrame.from_dict(residual68,orient='index',columns=[c.replace('O','c') for c in self.coeff_list])
#         new_df = df.T

#         ax = new_df.plot(kind='bar',rot=0,width=0.7,figsize=(10,5))
#         ax.plot([-1,self.npar+1],np.zeros(2),'k--',lw=2)
#         ax.plot([-1,self.npar+1],np.ones(2),'k--',lw=2,alpha=0.3)
#         ax.plot([-1,self.npar+1],-1.*np.ones(2),'k--',lw=2,alpha=0.3)

#         if self.logo is not None:
#             ax.imshow(self.logo, aspect='auto',transform=ax.transAxes,extent=[.775,.975,.825,.975],zorder=-1, alpha=1)

#         py.xticks(rotation=90)
#         py.tick_params(axis='y',direction='in',labelsize=15)
#         py.ylabel(r'${\rm Residuals\ (68\%)}$',fontsize=15)
#         py.ylim(-3,3)
#         py.legend(loc=2,frameon=False,prop={'size':11})
#         py.tight_layout()
#         py.savefig(f'{self.path}/reports/{self.outputname}/Coeffs_Residuals.pdf')

# # -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# from collections.abc import Iterable

import matplotlib.pyplot as py
from matplotlib import colors as matcolors

def coeff_by_group():
    """
    Lists for SMEFT operator classification:
        4H = 4-heavy
        2L2H = 2-light-2-heavy
        2FB = 2-fermion + boson
        B = purely bosonic
    """ # pylint:disable=line-too-long
    return {
        '4H': ['cQQ1','cQQ8','cQt1','cQt8','cQb1','cQb8','ctt1','ctb1','ctb8','cQtQb1','cQtQb8'],
        '2L2H': ['c81qq','c11qq','c83qq','c13qq','c8qt','c1qt','c8ut','c1ut','c8qu','c1qu','c8dt','c1dt','c8qd','c1qd'],
        '2FB': ['ctp','ctG','cbp','ccp','ctap','ctW','ctZ','cbW','cff','cpl1','c3pl1','cpl2','c3pl2','cpl3','c3pl3','cpe','cpmu','cpta','c3pq','c3pQ3','cpqMi','cpQM','cpui','cpdi','cpt','cll'],
        'B': ['cpG','cpGtil','cpB','cpBtil','cpW','cpWtil','cpWB','cpWBtil','cpd','cpD','cWWW','cWWWtil','cW','cB'],
    }

def get_conficence_values(dist):
    """
    Get confidence level bounds given the distribution
    """
    cl_vals = {}
    cl_vals['low'] = np.nanpercentile(dist,16)
    cl_vals['high'] = np.nanpercentile(dist,84)
    cl_vals['low95'] = np.nanpercentile(dist,2.5)
    cl_vals['high95'] = np.nanpercentile(dist,97.5)
    cl_vals['mid'] = np.mean(dist,axis=0)
    return cl_vals

def get_cl_erros(cl_vals):
    error68 = ( cl_vals['high'] - cl_vals['low']) / 2.0
    error95 = (cl_vals['high95'] - cl_vals['low95']) / 2.0
    return error68, error95

def set_double_cl(full_solution, l):

    min_val = min(full_solution)
    max_val = max(full_solution)
    mid = (max_val + min_val) / 2.0

    if l=='Obp' or l=='Opd':
        solution1 = full_solution[full_solution>mid]
        solution2 = full_solution[full_solution<mid]
    else:
        solution1 = full_solution[full_solution<mid]
        solution2 = full_solution[full_solution>mid]

    # First solution
    cl_vals_1 = get_conficence_values(solution1)
    error68, error95 = get_cl_erros( cl_vals_1 )
    first_err = error95

    # Second solution
    cl_vals_2 = get_conficence_values(solution2)
    temp = get_cl_erros( cl_vals_2 )
    error68 += temp[0]
    error95 += temp[1]

    double_bounds = {'1': cl_vals_1, '2': cl_vals_2, 'errors': [error68, error95], 'second_err': temp[1], 'first_err': first_err}
    return double_bounds, cl_vals_1['mid'] / error68
    #[ 1./np.sqrt(error68), 1./np.sqrt(error95) ], [ mid / error68 , mid / error95 ]


class CoefficientsPlotter:
    """
    Plots central values + 95% CL errors, 95% CL bounds and residuals.

    Takes into account parameter constraints and displays
    all non-zero parameters.

    Note: coefficients that are known to have disjoint
    probability distributions (i.e. multiple solutions)
    are manually separated by including the coefficient name
    in disjointed_list for global.

    Note: Probability distributions for single parameter
    fits are included for all coefficients EXCEPT those
    constrained as a linear combination of two or more parameters
    (since they have different numbers of posterior samples)

    Parameters
    ----------
       config : dict
            configuration dictionary
    """ # pylint:disable=line-too-long

    def __init__(self, config ):
        self.report_folder = f"{config['data_path'].parents[1]}/reports/{config['report_name']}"

        coeff_config = coeff_by_group().copy()
        self.coeff_list = []
        for group in coeff_config.values():
            for c in group:
                if c in ['OW','OB']:
                    continue
                if np.any([config[k]['coefficients'][c]['fixed'] is not False for k in config.keys() if c in config[k]['coefficients']]):
                    continue
                if c not in self.coeff_list and np.any([c in config[k]['coefficients'] for k in config.keys() ]):
                    self.coeff_list.append(c)

        self.npar = len(self.coeff_list)

    def compute_confidence_level(self, fit, disjointed_list = None):
        """Compute 95 % and 68 % confidence levels, energy bounds and residuals"""

        bounds = {}
        residuals = []
        error_bounds = []

        for l in self.coeff_list:

            # double soultion
            if l in disjointed_list:
                np.append( bounds, set_double_cl(fit[l], l)[0])
                np.append( residuals, set_double_cl(fit[l],l)[1])
                np.appned( error_bounds, set_double_cl( fit[l],l )[0]['errors'][1] )
            # single solution
            else:
                cl_vals = get_conficence_values(fit[l])
                error68, error95 = get_cl_erros( cl_vals )
                np.append( bounds, { '1': cl_vals, 'errors' : [error68, error95] })
                np.append( residuals, cl_vals['mid'] / error68 )
                np.append(  error_bounds, error95)

        return bounds, residuals, error_bounds


    def plot_coeffs(self, bounds):
        """Plot central value + 95% CL errors"""

        nrows,ncols=1,1
        py.figure(figsize=(nrows*10,ncols*5))

        ax=py.subplot(111)

        # X-axis
        X = 2 * np.array(range(self.npar))
        # Spacing between fit results
        val = np.linspace(-0.1 * self.npar, 0.1 * self.npar, self.npar)
        colors = py.rcParams['axes.prop_cycle'].by_key()['color']

        i=0
        for k in bounds.keys():
            name = r'${\rm %s}$'%k.replace('_','\_')
            cnt = 0
            for vals in bounds[name].values():
                if cnt==0:
                    ax.errorbar( X[cnt]+val[i], y = np.array(vals['1']['mid']) , yerr = np.array(vals['errors'][1] - vals['second_err']), \
                        color=colors[i], fmt='.', elinewidth=2, label=name)
                else:
                    ax.errorbar( X[cnt]+val[i], y=np.array(vals['1']['mid']), yerr = np.array(vals['errors'][1] - vals['second_err']), \
                        color=colors[i], fmt='.', elinewidth=2)

                # double soluton
                if vals['second_err'] != 0.0:
                    ax.errorbar(X[cnt]+val[i]+0.5, y=np.array(vals['2']['mid']), yerr= np.array(vals['second_err']), \
                        color=colors[i], fmt='.', elinewidth=2)
                cnt+=1
            i+=1

        py.plot( list(range(-1,200))np.zeros(201),'k--',alpha=0.7)

        py.yscale('symlog',linthreshy=1e-1)
        py.ylim(-200,200)
        py.yticks([-100,-10,-1,-0.1,0,0.1,1,10,100],[r'$-100$',r'$-10$',r'$-1$',r'$-0.1$',r'$0$',r'$0.1$',r'$1$',r'$10$',r'$100$'])
        py.ylabel(r'$c_i/\Lambda^2\ ({\rm TeV}^{-2})$',fontsize=18)

        py.xlim(-1,(self.npar)*2-1)
        py.tick_params(which='major',direction='in',labelsize=13)
        py.xticks(X,self.coeff_list,rotation=90)

        py.legend(loc=0,frameon=False,prop={'size':13})
        py.tight_layout()
        py.savefig(f'{self.report_folder}/Coeffs_Central.pdf')


    def plot_coeffs_bar(self, error ):
        """ Plot 95% CLs for coefficients (bar plot)"""

        py.figure(figsize=(7,5))
        df = pd.DataFrame.from_dict(error,orient='index',columns=self.coeff_list )
        new_df = df.Txw
        new_df.plot(kind='bar',rot=0,width=0.7,figsize=(10,5))

        #Hard cutoff
        py.plot(np.linspace(-1,2*self.npar+1,2),50*np.ones(2),'k--',alpha=0.7,lw=2)

        py.xticks(rotation=90)
        py.tick_params(axis='y',direction='in',labelsize=15)
        py.yscale('log')
        py.ylabel(r'$95\%\ {\rm Confidence\ Level\ Bounds}\ (1/{\rm TeV}^2)$',fontsize=11)
        py.ylim(1e-3,1e3)
        py.legend(loc=2,frameon=False,prop={'size':11})
        py.tight_layout()
        py.savefig(f'{self.report_folder}/Coeffs_Bar.pdf')


    def plot_residuals_bar(self, residual):
        """ Plot residuals at 68% CL (bar plot) """
        py.figure(figsize=(7,5))

        df = pd.DataFrame.from_dict(residual, orient='index', columns=self.coeff_list)
        new_df = df.T

        ax = new_df.plot(kind='bar',rot=0,width=0.7,figsize=(10,5))
        ax.plot([-1,self.npar+1],np.zeros(2),'k--',lw=2)
        ax.plot([-1,self.npar+1],np.ones(2),'k--',lw=2,alpha=0.3)
        ax.plot([-1,self.npar+1],-1.*np.ones(2),'k--',lw=2,alpha=0.3)

        py.xticks(rotation=90)
        py.tick_params(axis='y',direction='in',labelsize=15)
        py.ylabel(r'${\rm Residuals\ (68\%)}$',fontsize=15)
        py.ylim(-3,3)
        py.legend(loc=2,frameon=False,prop={'size':11})
        py.tight_layout()
        py.savefig(f'{self.report_folder}/Coeffs_Residuals.pdf')

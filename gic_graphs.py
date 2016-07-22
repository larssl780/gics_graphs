#
#  gic_graphs : Python module - create plots of changes in international income distributions
#  
#  this code created by Lars Larsson (lars.larsson@gmail.com), but the code doesn't add anything beyond what's described
#  here: http://go.worldbank.org/NWBUKI3JP0
#  It's also a slight simplification in that it just slices the data by the pure ventiles, it doesn't drill down further into the top ventile as 
#  in the original paper.
#  It's easiest to run the program from within ipython (or just put the below 2 rows into a separate script), like so:
#  import gic_graphs as gg 
#  gg.study_replication()
#  the function produces a replication of the original study, with and without China
#  Easy enough to modify library in order to play around with the data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb
import urllib
import os
def import_data(filename='gic_data.dta'):
    """
    Tries to read local dta-file, if file not found, download from the worldbank site
    """
    if not os.path.isfile(filename):
        print 'DTA file not found, downloading from worldbank'
        testfile = urllib.URLopener()
        testfile.retrieve("http://siteresources.worldbank.org/INTRES/Resources/469232-1107449512766/LM_WPID_web.dta", "gic_data.dta")
    df = pd.read_stata(filename)
    
    return df


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ 
    Author: user Alleo on Stackoverflow (http://stackoverflow.com/a/29677616/1933152)
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def project(df, columns):
    return df.ix[:, columns]
def extract_ventiles(data, from_year=1988, to_year=2008):
    """
    Does the main work

    """
    data_from = data.query('bin_year==%d'%from_year)
    data_from = project(data_from, ['RRinc', 'pop'])
    data_from.dropna(inplace=True)
    data_to = data.query('bin_year==%d'%to_year)
    data_to = project(data_to, ['RRinc', 'pop'])
    data_to.dropna(inplace=True)
    
    bins_from = weighted_quantile(data_from['RRinc'].values,np.arange(0, 1.01, 0.05),sample_weight=data_from['pop'].values )
    bins_to = weighted_quantile(data_to['RRinc'].values,np.arange(0, 1.01, 0.05),sample_weight=data_to['pop'].values )
   
    # then for each bin, work out the average income:
    incs_from = data_from['RRinc'].values
    pops_from = data_from['pop'].values
    incs_to = data_to['RRinc'].values
   
    pops_to = data_to['pop'].values
    pop_buckets_from = {}
    inc_buckets_from = {}
    pop_buckets_to = {}
    inc_buckets_to = {}
    quants = range(5,101, 5)
    wavg_inc_from = 0
    pop_from = 0
    for i in range(len(incs_from)):
        inc = incs_from[i]
        idx = np.searchsorted(bins_from, inc)
        real_idx = max(idx-1, 0)
        
        quant = quants[real_idx]
        
        pop = pops_from[i]
        pop_from += pop
        wavg_inc_from += pop * inc
        if quant in pop_buckets_from:
            pop_buckets_from[quant] += pop
        else:
            pop_buckets_from[quant] = pop
        if quant in inc_buckets_from:
            inc_buckets_from[quant] += pop * inc
        else:
            inc_buckets_from[quant] = pop * inc

    wavg_inc_from /= pop_from
    w_incs_from = {}

    for q, inc in inc_buckets_from.iteritems():
        w_incs_from[q] = inc / pop_buckets_from[q]

    wavg_inc_to = 0
    pop_to = 0
    for i in range(len(incs_to)):
        inc = incs_to[i]

        idx = np.searchsorted(bins_to, inc)
        real_idx = max(idx-1, 0)
        
        quant = quants[real_idx]
        pop = pops_to[i]
        wavg_inc_to += pop * inc
        pop_to += pop
        if quant in pop_buckets_to:
            pop_buckets_to[quant] += pop
        else:
            pop_buckets_to[quant] = pop
        if quant in inc_buckets_to:
            inc_buckets_to[quant] += pop * inc
        else:
            inc_buckets_to[quant] = pop * inc

    wavg_inc_to /= pop_to

    tot_growth = wavg_inc_to / wavg_inc_from
    gpa = pow(tot_growth, 1.0/20) -1
    print 'Total growth: %f pc (%f pc pa)'%(100 * (tot_growth - 1), 100 * gpa)
    w_incs_to = {}

    for q, inc in inc_buckets_to.iteritems():
        w_incs_to[q] = inc / pop_buckets_to[q]

    # print w_incs_to
    cum_growths = {}

    for q, inc_to in w_incs_to.iteritems():
        cum_growths[q] = 100 *(inc_to / w_incs_from[q]-1)
    rv =  pd.DataFrame({'ventile':cum_growths.keys(), 'growth':cum_growths.values()}, columns=['ventile', 'growth'])
    rv.sort('ventile', inplace=True)    
    return rv
def study_replication(from_year=1988, to_year=2008):
    """
    mysample_only is what's done in the original article (gets you the elephant graph, if exclude_china = False)
    """
    data = import_data()

    orig_data = data.query('mysample==1')
    orig_study = extract_ventiles(orig_data, from_year=from_year, to_year=to_year)
    ex_china_data = data.query("(country != 'China')&(mysample==1)")
    ex_china_study = extract_ventiles(ex_china_data, from_year=from_year, to_year=to_year)
    
    plt.plot(orig_study['ventile'], orig_study['growth'], label='Original Study' )
    plt.plot(ex_china_study['ventile'], ex_china_study['growth'], label='Ex China' )
    plt.plot(ex_china_study['ventile'].values, [0]*len(ex_china_study), label='0 growth' )
    plt.legend(loc='best')
    plt.title('Cumulative Real Income Growth %d - %d for different ventiles'%(from_year,to_year))
    plt.xlabel('%d income ventile'%from_year)
    plt.ylabel('Population-weighted avg growth in PPP 2005 USD income')
    plt.show()




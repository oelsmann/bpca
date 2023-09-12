#    GPLv3 License

#    BPCA: Bayesian Principal Component Analysis
#    Copyright (C) 2023  Julius Oelsmann

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

#    main bpca class

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from theano.tensor import *
from theano import tensor
from arviz import *
import arviz as az
import pickle
import xarray as xr
import pymc3 as pm
import pandas as pd
import numpy as np
import copy

class bpca_model(pm.model.Model):
    """ PyMC model for Bayesian Principal Component analysis.
    
    Parameters
    ----------
    observed: xarray.DataArray of time*space dimensions
    
    name: str, default: True,
        model name
    
    model_trend :  bool, default: False
        estimate trend or not        
        
    number_of_pcs: int,
        number of modelled pcs
    
    sigma: float,
        variance prior 
        
    sigma_offset: float,
        offset-variance prior 
        
    sigma_random_walk: float,
        random walk innovations prior 
        
    sigma_random_walk_factor: float,
        factor by which sigma_random_walk is divided for subsequent PCs;
        sets some prior contraints, such thus the first EOFs capture most of the variance.
        e.g. if sigma_random_walk_factor is 2 and sigma_random_walk is 2,
        then sigma_random_walk(PC1) = 2; sigma_random_walk(PC2) = 1; sigma_random_walk(PC3) = 0.5 ...
            
    sigma_eofs: float,
        EOF variance prior 
            
    trend_factor_sigma: float,
        trend-variance prior 
        
    trend_factor_nu: float,
        trend-shape prior parameter for Student's T distribution
    
    trend_distr: str,
        'student_t' or 'normal'
    
    estimate_point_variance :  bool, default: False,
        estimate variance (sigma) at every point
        
    estimate_cluster_sigma :  bool, default: False,
        estimate variance (sigma) for different clusters      
    
    trend_pattern :  None or np.array() of size space_dim, default: None,
        prior of trend pattern, initialize trends
    
    initialize_trend_pattern :  bool, default: False,
        enable trend initialization
    
    estimate_offsets :  bool, default: True,    
        estimate constant offsets (at every location),        
        
    estimate_cluster_sigma :  bool, default: False,    
        estimate variance for different cluster (in space)
        
    estimate_sigma_eof : bool, default: True,
        estimate hierarchical sigma for eof pattern
        
    cluster_index :  None or np.array(), default: None,    
        array of cluster indices (0,1,2,...) indicating the clusters for which sigma should be computed
        
    """
                                
    def __init__(self, 
                 observed=None,
                 name='',
                 model_trend=True,
                 number_of_pcs = 3 ,
                 sigma=0.4, 
                 sigma_offset =0.2, 
                 sigma_random_walk=0.1,
                 sigma_random_walk_factor=2,
                 sigma_eofs=0.2,
                 trend_factor_sigma=0.001, 
                 trend_factor_nu=2.1303,
                 trend_distr='student_t',
                 estimate_point_variance=False,
                 estimate_cluster_sigma=False,
                 trend_pattern=None,
                 initialize_trend_pattern=False,
                 estimate_offsets=False,
                 estimate_sigma_eof=True,
                 cluster_index=None,
                 **kwargs):

        super().__init__(name)

        Y = pd.DataFrame(observed.values[:,:]) # time vs space
        time_dim,space_dim=Y.shape

        with pm.Model() as model:
            if estimate_offsets:
                offset = pm.Normal("offset", 0,sigma=sigma_offset,shape = space_dim)                                
            else:
                offset = np.empty(space_dim)*0
            sigma_eofs = [sigma_eofs]*number_of_pcs
            if estimate_sigma_eof:
                sigma_eofs  = pm.HalfNormal('sigma_eof',sigma=np.asarray(sigma_eofs),shape = number_of_pcs)                
            if estimate_point_variance:
                if estimate_cluster_sigma:
                    number_of_cluster= len(np.unique(cluster_index))
                    print('estimate different sigma for different clusters')
                    sigma_hier = pm.HalfNormal('sigma_hier',sigma=sigma,shape = number_of_cluster)
                    coeff_mat = np.zeros([len(cluster_index),number_of_cluster])
                    for i in range(number_of_cluster):
                        coeff_mat[:,i] = (cluster_index==i)*1
                    sigma = pm.Deterministic("sigma", det_dot(sigma_hier, coeff_mat).flatten())
                    
                else:
                    sigma=pm.HalfNormal('sigma',sigma=sigma,shape = space_dim)
            else:                
                sigma=pm.HalfNormal('sigma',sigma=sigma)
            PCS_EOFs_mult= 0 
            for i in range(number_of_pcs):
                    
                PCS_EOFs_mult=pm.math.matrix_dot(pm.GaussianRandomWalk("PC"+str(i), mu=0,sd=sigma_random_walk, shape=time_dim)[:,np.newaxis],
                                                        pm.Normal("W"+str(i), 0,sigma=sigma_eofs[i],shape = space_dim)[np.newaxis,:])+ PCS_EOFs_mult
                    
                sigma_random_walk=sigma_random_walk/sigma_random_walk_factor    
                         
            if model_trend:
                
                mu_trend=0                 
                if initialize_trend_pattern:
                    mu_trend=trend_pattern
                    
                if trend_distr=='student_t':
                    trend_pattern = pm.StudentT("trend_g", mu=mu_trend,sigma=trend_factor_sigma,nu=trend_factor_nu,shape = space_dim)
                else:
                    trend_pattern = pm.Normal("trend_g", mu_trend,sigma=trend_factor_sigma,shape = space_dim)                    
                shift=-6 # so time series is centered to 2014
                trend_series = np.linspace(-time_dim/2 + shift,time_dim-1-time_dim/2 + shift,time_dim)
                trend = pm.math.matrix_dot(trend_series[:,np.newaxis],trend_pattern[np.newaxis,:])
                
                PCS_EOFs_mult=PCS_EOFs_mult + trend
            
            mu = pm.Deterministic("Estimates",   PCS_EOFs_mult + 
                                          offset[np.newaxis,:])

            Y_obs = pm.Normal('Observations', mu=mu, sigma=sigma, observed=Y)    
            
            
          
            
class bpca():
    """Bayesian Principal Component analysis.
    
    Parameters
    ----------
    dataset: xarray.DataArray of time*space dimensions
    
    run_settings: Run settings, i.e. input arguments for pm.sample()
    
    model_settings: Dictionary of model settings
    
    name: str model name, used to save the model
    
    """
    
    
    def __init__(self,
                 dataset,
                 run_settings={'n_samples':4000,'compress':True,'sample_settings':{'tune':2000,'cores':8,
                     'target_accept':0.9,'return_inferencedata':True,'check_convergence':False,'adjust_pca_symmetry':False}},
                 model_settings={},
                 normalization_settings={},
                 name='bpca_model'
                ):
        
        self.dataset=self._normalize_data(dataset)      
        self.estimated_dataset=None
        self.estimated_dataset_map=None  
        self.estimated_dataset_map_pattern = []         
        self.run_settings = run_settings
        self.model_settings = model_settings
        self.normalization_settings = normalization_settings        
        self.name = name
        self.model = bpca_model(observed =  self.dataset,**self.model_settings)

        self.trace = None
        self.random = None
        self.compressed=False
        self.chain_stats = {}
        self.convergence_stats = {}
        self.random=[]   
        self.initial_values={}
        #self.=_normalize_data(dataset)


    def _normalize_data(self,dataset):
        """
        normalize and adjust the dataset
        
        """
        
        return dataset
    
    def get_pca_correlation_sorting_indices(self,data_comb):
        """
        get indices to sort PCs and scale their variances
        
        """
        var_indices   = []
        chain_indices = []
        signs         = []
        #stds_         = []
        
        corr_ = data_comb.corr()
        # always use chain 0 as reference
        variable = 'PC'
        for i in np.arange(self.model_settings['number_of_pcs']):
            for j in np.arange(self.run_settings['sample_settings']['cores']):
                corr_sub = corr_[variable+str(i)+'c0']
                corr_sub2 = corr_sub[np.asarray(['c'+str(j) in val for val in corr_sub.index])]  
                var_index = int(corr_sub2[abs(corr_sub2)==np.max(abs(corr_sub2))].index[0][2:3])
                var_indices.append(var_index)
                chain_indices.append(j)
                signs.append(int(np.sign(corr_sub2[abs(corr_sub2)==np.max(abs(corr_sub2))])))
                # tbd
                #std_orig = float(self.trace.posterior[[variable+str(i)][0]].mean(dim='draw').std())
                #std_match = float(self.trace.posterior[[variable+str(var_index)][j]].mean(dim='draw').std())
                #stds_.append(std_orig/std_match)
                
        return var_indices,chain_indices,signs#,stds_        
        
    def get_pca_correlation(self):
        variable = 'PC'
        series_w = self.trace.posterior[[variable+str(i) for i in np.arange(self.model_settings['number_of_pcs'])]].mean(dim='draw')
        name=[]
        data_ =[]
        for i in np.arange(self.model_settings['number_of_pcs']):
            for j in np.arange(self.run_settings['sample_settings']['cores']):
                data_.append(series_w[variable+str(i)][j].values)
                name.append(variable+str(i)+'c'+str(j))
        data_comb = pd.DataFrame(data_,index=name).T
        return data_comb
    
    def adjust_pca_symmetry(self):
        """
        
        """
        data_comb = self.get_pca_correlation()
        var_indices,chain_indices,signs = self.get_pca_correlation_sorting_indices(data_comb)

        vars_ = ['PC','W']
        relevant_data = copy.deepcopy(self.trace.posterior[[variable+str(i) for i in np.arange(self.model_settings['number_of_pcs']) for variable in vars_]])
        if 'sigma_eof' in self.trace.posterior:
            relevant_data['sigma_eof'] = copy.deepcopy(self.trace.posterior['sigma_eof'])
        
        # normalize pattern and pcs
        for i in np.arange(self.model_settings['number_of_pcs']):
            for j in np.arange(self.run_settings['sample_settings']['cores']):
                std_ = float(relevant_data['W'+str(i)][j].mean(dim='draw').std())
                relevant_data['W'+str(i)][j] = relevant_data['W'+str(i)][j]/std_
                relevant_data['PC'+str(i)][j] = relevant_data['PC'+str(i)][j]*std_
                if 'sigma_eof' in self.trace.posterior:
                    relevant_data['sigma_eof'][j,:,i] = relevant_data['sigma_eof'][j,:,i]/std_
        # update trace
        for variable in vars_:
            ii = 0
            for i in np.arange(self.model_settings['number_of_pcs']):
                for j in np.arange(self.run_settings['sample_settings']['cores']):
                    self.trace.posterior[variable+str(i)][j] =signs[ii]*relevant_data[variable+str(var_indices[ii])][chain_indices[ii]].values
                    print('replaced variable '+variable+str(i)+' (chain '+str(j)+' ) with variable '+variable+str(var_indices[ii])+' (chain '+str(chain_indices[ii])+')')
                    ii=ii+1 
                    
        if 'sigma_eof' in self.trace.posterior:
            ii = 0
            variable = 'sigma_eof'
            for i in np.arange(self.model_settings['number_of_pcs']):
                for j in np.arange(self.run_settings['sample_settings']['cores']):
                    self.trace.posterior['sigma_eof'][j,:,i] =relevant_data['sigma_eof'][chain_indices[ii],:,var_indices[ii]].values
                    print('replaced variable '+variable+str(i)+' (chain '+str(j)+' ) with variable '+variable+str(var_indices[ii])+' (chain '+str(chain_indices[ii])+')')
                    ii=ii+1             
        
    def get_chain_statistics(self):
        """
        
        """
        all_={}

        for i in range(len(self.trace.posterior.chain)):
            chain_single=self.trace.sel(chain=[i])    
            all_[str(i)]=chain_single
        self.chain_stats=pm.compare(all_)   
        
    def check_convergence(self,check_main_components=True):
        data_ = []
        summary_out = az.summary(self.trace)
        for var in list(self.trace.posterior.keys()):
            select = [var in val for val in summary_out.index.values]
            data_.append(summary_out[select].mean().T)
        summary_out_combined =pd.concat(data_,axis=1,keys=list(self.trace.posterior.keys()))
        self.convergence_stats = summary_out_combined 
        if check_main_components:
            convergence_stats = summary_out_combined.loc[['r_hat','ess_mean']]
            convergence_stats.loc['ess_mean']=summary_out_combined.loc['ess_mean']/self.run_settings['n_samples']
            self.convergence_stats = convergence_stats[['PC0','W0','trend_g','sigma']] 
        
    def recombine_datasets(self,chain=0,kind='mean',draw=4,with_offset=False):
        """reconstruct dataset with PCs

        Parameters:

        chain: int
            chain to select

        kind: str, select averaged MCMC: 'mean',
            select a random section: 'random',
            compute the mean of random estimates: 'random_mean'           
            use interpolated map: 'maps',

        draw: int, select draw

        """


        data = 0
        data_std = 0
        dataset=copy.deepcopy(self.dataset)
        dataset_std=copy.deepcopy(self.dataset).rename(dataset.name+'_std') 
        chain_z = chain
        if kind == 'mean':
            mean_trace = self.trace['mean'].posterior
            std_trace = self.trace['std'].posterior
        elif kind =='random':
            mean_trace = self.random['mean'].sel({'draw':draw})
            std_trace = self.random['std'].sel({'draw':draw})
        elif kind =='random_mean':
            mean_trace = self.random['mean'].mean(dim='draw')
            std_trace = self.random['std'].mean(dim='draw')      
        if kind == 'maps':
            mean_trace = self.estimated_dataset_map_pattern[0]
            std_trace = self.estimated_dataset_map_pattern[1]   
            mean_trace_rand = self.trace['mean'].posterior
            std_trace_rand = self.trace['std'].posterior       

            time_ = self.dataset.time
            dataset = xr.concat([mean_trace['W0']]*len(time_),dim='time').rename(self.dataset.name)*np.nan
            dataset_std = xr.concat([mean_trace['W0']]*len(time_),dim='time').rename(dataset.name+'_std')*np.nan
            dataset['time']=time_
            chain_z =...
            if self.model_settings['model_trend']:
                # split data by trend and varibaility results
                dataset_trend_series     =copy.deepcopy(dataset).rename('trend_series')
                dataset_trend_series_std =copy.deepcopy(dataset).rename('trend_series_std')
                dataset_trend     = copy.deepcopy(dataset)[0,:].rename('trend')
                dataset_trend_std = copy.deepcopy(dataset)[0,:].rename('trend_std')
                dataset_pcs       = copy.deepcopy(dataset).rename('pcs')
                dataset_pcs_std   = copy.deepcopy(dataset).rename('pcs_std')   

        for pcs_id in range(self.model_settings['number_of_pcs']):
            if kind == 'maps':
                # use random w
                w_=mean_trace_rand['PC'+str(pcs_id)][chain,:].values
                w_std=std_trace_rand['PC'+str(pcs_id)][chain,:].values  
            else:
                w_=mean_trace['PC'+str(pcs_id)][chain,:].values
                w_std=std_trace['PC'+str(pcs_id)][chain,:].values  
            z_ = mean_trace['W'+str(pcs_id)][chain_z,:].values
            z_std = std_trace['W'+str(pcs_id)][chain_z,:].values

            z_p = z_std/z_
            w_p = w_std/w_
            combined_rel_error = np.sqrt(np.repeat(w_p[:,np.newaxis],dataset.shape[1],axis=1)**2+np.repeat(z_p[np.newaxis,:],dataset.shape[0],axis=0)**2)

            data = data+np.matmul(w_[:,np.newaxis],z_[np.newaxis,:])


            data_std = data_std+abs(np.matmul(w_[:,np.newaxis],z_[np.newaxis,:])*combined_rel_error)  
            if with_offset:
                data_std = np.sqrt(data_std**2 + std_trace['offset'][chain,:].values**2)

        if self.model_settings['model_trend']:
            if kind =='maps':
                trend_map = mean_trace['W99'].values/1000.
                trend_map_std = std_trace['W99'].values/1000.            
            else:

                trend_map = mean_trace['trend_g'][chain,:].values
                trend_map_std = std_trace['trend_g'][chain,:].values        

            time_dim = len(dataset.time)
            shift=-6
            trend_series = np.linspace(-time_dim/2 + shift,time_dim-1-time_dim/2 + shift,time_dim)
            combined_rel_error_trend = np.matmul(abs(trend_series)[:,np.newaxis],trend_map_std[np.newaxis,:])
            trend_field = np.matmul(trend_series[:,np.newaxis],trend_map[np.newaxis,:])

            # overwrite fields
            if kind == 'maps':
                dataset_pcs[:,:]       = copy.deepcopy(data)
                dataset_pcs_std[:,:]   = copy.deepcopy(data_std)   

            data = data + trend_field
            data_std = data_std + combined_rel_error_trend
        dataset[:,:]=data
        #dataset_std[:,:]=np.sqrt(data_std+std_trace['offset'][chain,:].values**2)
        dataset_std[:,:]=data_std

        if kind == 'maps':
            if self.model_settings['model_trend']:
                dataset_trend_series[:,:]     = trend_field
                dataset_trend_series_std[:,:] = combined_rel_error_trend
                dataset_trend[:]     = trend_map
                dataset_trend_std[:] = trend_map_std
                self.estimated_dataset_map   = xr.merge([dataset,dataset_std,
                                                         dataset_trend_series,dataset_trend_series_std,
                                                         dataset_trend,dataset_trend_std,
                                                         dataset_pcs,dataset_pcs_std]) 
            else:
                self.estimated_dataset_map   = xr.merge([dataset,dataset_std]) 
                self.estimated_dataset_map.attrs={'chain':chain}
        else:
            self.estimated_dataset   = xr.merge([dataset,dataset_std])   
            self.estimated_dataset.attrs={'chain':chain}
    
    @property
    def explained_variance(self):
        """
        
        """
        if self.estimated_dataset == None:
            self.recombine_datasets()
            
        return 1-((self.estimated_dataset[self.dataset.name]-self.dataset).var(dim='time')/self.dataset.var(dim='time'))
        
        
    
    def compress(self,random_sample_size = 20,number_of_rands = 5):
        
        """compress trace
        compute mean and std-dev along draw dimension
        1. derive statistics options
        3. mean, std of trace
        4. set compressed to True
        """
        
        self.get_chain_statistics() # set and safe statistics

        factor_ = int(self.trace.posterior.draw.max().values/number_of_rands)

        # make some random averages over some intervals
        RANDOM={}
        for op in ['mean','std']:
            for chain in self.trace.posterior['chain'].values:
                RAND_sub = []
                start_ = []
                for random_i in range(number_of_rands):
                    select_arr = np.arange(random_sample_size)+factor_*random_i
                    RAND_sub.append(getattr(self.trace.posterior.sel({'draw':select_arr}),op)(dim='draw'))
                    start_.append(factor_*random_i)
                RANDOM[op] = xr.concat(RAND_sub,dim='draw')
                RANDOM[op].attrs={'Random: draw indices': str(start_),'Random: draw size' : random_sample_size}
        
        self.random = RANDOM
        
        COMPRESSED_TRACE={} # compress trace
        for op in ['mean','std']:
            elements=[]
            for element in ['posterior','sample_stats','log_likelihood']:
                elements.append(getattr(getattr(self.trace,element),op)(dim='draw'))

            COMPRESSED_TRACE[op]=az.InferenceData(posterior=elements[0],sample_stats=elements[1],
                                    log_likelihood=elements[2])            

        self.trace=COMPRESSED_TRACE

        self.compressed = True
        print('successfully compressed trace')  
        
        
    def run(self):
        """
        run model
        """
        with self.model:
            self.trace = pm.sample(self.run_settings['n_samples'],**self.run_settings['sample_settings'])            

        if self.run_settings['adjust_pca_symmetry']:
            self.adjust_pca_symmetry()
        
        if self.run_settings['check_convergence']:   
            self.check_convergence()
            
        if self.run_settings['compress']:
            self.compress()    
    

    def save(self,save_dir='',kind='bpca'):
        """
        save object
        """
        if self.name == '' or save_dir=='':
            raise Exception('Define Object.name and save_dir before saving!')
        else:
            if kind == 'bpca':
                with open(save_dir+self.name+'.bpca', 'wb') as ilame_file:
                    pickle.dump(self, ilame_file, pickle.HIGHEST_PROTOCOL)
            else:
                self.trace.to_netcdf(save_dir+self.name)

    def load(save_dir='',kind='bpca'):
        """
        load object
        """
        if save_dir=='':
            raise Exception('Define filename before loading!')
        else:
            if kind == 'bpca':
                with open(save_dir+'.bpca', 'rb') as ilame_file:
                    self = pickle.load(ilame_file)  
            else:
                self=xr.open_dataset(save_dir)
            return self

def file_reader(file,variable='auto',resample='D'):
    
    """ read different file types
    
    'txt','nectdf (.nc)','tenv3', 'txyz2'
    
    Parameters
    ----------
    
    file : str
        location of file
    variable : str
        variable to read from file 
        (default: 'auto', selects the first availabe variable/column)
        
    """

    ending = file.split(".",1) 

    if len(ending)==1:
        ending ='txt'
    else:
        ending=ending[1]
    if ending =='txt' or ending =='':
        if variable=='auto':
            variable='Height'
        data=pd.read_csv(file,delim_whitespace=True)
        data['Year']=pd.to_datetime(data['Year']-1970.,unit='Y')
        data.set_index('Year',inplace=True)
        data=data.resample(resample).mean()[variable] 
    elif ending =='tenv3':
        if variable=='auto':
            variable='____up(m)'
        data=pd.read_csv(file,delim_whitespace=True,header=None)
        data['Year']=pd.to_datetime(data[2]-1970.,unit='Y')
        data.set_index('Year',inplace=True)
        data=data.resample(resample).mean()[variable]   
    elif ending =='txyz2':
        if variable=='auto':
            variable=3
        file = 'discotimes/examples/HOB2'
        data=pd.read_csv(file,delim_whitespace=True,header=None)
        data['Year']=pd.to_datetime(data[2]-1970.,unit='Y')
        data.set_index('Year',inplace=True)
        data=data.resample(resample).mean()[variable]
    elif ending =='nc':
        data = xr.open_dataset(file)
    else:
        raise Exception('File of type *.'+ending+' not implemented')
    return data    

def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?
    
    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)
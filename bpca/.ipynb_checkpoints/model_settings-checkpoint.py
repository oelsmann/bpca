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

# Define the model settings here


def set_settings(external_settings={}):
    """
    Make/Change default model settings 
    
    Parameters
    ----------
    external_settings: dict,
        can contain dicts: 'model_settings', 'run_settings', 'initial_run_settings'
    """
    
    settings={}
    settings['model_settings']   = model_settings(external_settings=external_settings)
    settings['run_settings']   = run_settings(external_settings=external_settings)        
    settings['normalization_settings']   = normalization_settings(external_settings=external_settings)  
    
    return settings

def run_settings(external_settings={}):
    specs={'n_samples':4000,'compress':True,'adjust_pca_symmetry':True,'check_convergence':True,
           'sample_settings':{'tune':2000,'cores':4,
                      'target_accept':0.9,'return_inferencedata':True}}

    if 'run_settings' in external_settings:
        for item in external_settings['run_settings']:
            specs[item]=external_settings['run_settings'][item]  
    return specs 

def normalization_settings(external_settings={}):
    
    specs={}
    if 'normalization_settings' in external_settings:
        for item in external_settings['normalization_settings']:
            specs[item]=external_settings['normalization_settings'][item]  
    return specs          

def model_settings(external_settings={}):
    
    specs = {'number_of_pcs' :3,'model_trend':True,'sigma_random_walk':0.001,
                      'estimate_point_variance':False,'estimate_cluster_sigma':False,'trend_pattern':None,
                      'initialize_trend_pattern':False,'estimate_offsets':False,
                      'trend_factor_sigma':0.01,'trend_factor_nu':2.1,
                      'trend_distr':'normal','cluster_index':None,'sigma':0.4,'sigma_offset':0.1,
                      'sigma_eofs':0.15,'sigma_random_walk_factor':0.04}  
    if 'model_settings' in external_settings:
        for item in external_settings['model_settings']:
            specs[item]=external_settings['model_settings'][item]  
    return specs
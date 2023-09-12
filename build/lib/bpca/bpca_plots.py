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


from matplotlib import pyplot as plt
from bpca.utils import *


def plot_synthetic_data_maps(data_set_synt,synthetic_data_settings,coastline,
                             validation=False,map_data=False,bpca_object=None,save=False,plt_dir='',indices=None):
    """
    plot different stages of the results of the synthetic experiments
    
    
    """
    my_cmap = cmap('BlueWhiteOrangeRed_c')
    mask_gps=(data_set_synt['ID']==0).values
    mask_sattg =~mask_gps
    
    if validation:
        if map_data: 
            fig,axs = plt.subplots(4,4,figsize=(16,16),gridspec_kw={'width_ratios':[2,2,1.4,1.4]})
        else:
            fig,axs = plt.subplots(3,4,figsize=(16,12),gridspec_kw={'width_ratios':[2,2,1.4,1.4]})                 
    else:
        fig,axs = plt.subplots(1,4,figsize=(16,4),gridspec_kw={'width_ratios':[2,2,1.4,1.4]})
                               
    axs = axs.flatten()                       
    x = np.linspace(0,synthetic_data_settings['max_lev'],101)
    y = np.linspace(0,synthetic_data_settings['max_lev'],101)

    dx = (x[1]-x[0])/2.
    dy = (y[1]-y[0])/2.
    extent = [x[0]-dx, x[-1]+dx, y[0]-dy, y[-1]+dy]

    y = y[:,np.newaxis]
    field_1,field_2,field_1_eq,field_2_eq=make_field(x,y)

    cs = axs[0].imshow(field_1+field_2 +field_2_eq,cmap=my_cmap,vmin=-3,vmax=3,origin='lower',extent=extent)
    cbar = fig.colorbar(cs, ax=axs[0])
    axs[0].set_title('A',loc='left',fontweight='bold')
    axs[0].set_title(r'Trends, $\hat{g}, g$ ',loc='center')
    
    cbar.set_label('mm/year')
    
    def coast_(x):
        return x*-1/np.cos(-20) + 36
    coastline = coast_(x)

    max_b = 6
    min_b = -max_b

    axs[0].plot(x,coastline,color='grey')
    axs[0].set_xlim(0,20)
    axs[0].set_ylim(0,20)
    axs[0].set_aspect(1)
    axs[0].scatter(data_set_synt['lat'][mask_gps],data_set_synt['lon'][mask_gps],
                     c=data_set_synt['trend_with_noise'][mask_gps],cmap=my_cmap,vmin=-3,vmax=3,s=18,edgecolor='k')

    axs[0].scatter(data_set_synt['lat'][mask_sattg],data_set_synt['lon'][mask_sattg],
                     c=data_set_synt['trend_with_noise'][mask_sattg],
                     cmap=my_cmap,vmin=-3,vmax=3,s=100,edgecolor='k')


    cs2 = axs[1].imshow(field_1_eq,cmap=cmap('BlueWhiteOrangeRed_c',start=min_b,ende=max_b),vmax=max_b,vmin=min_b,origin='lower',extent=extent)
    fig.colorbar(cs2, ax=axs[1])
    axs[1].set_title('B',loc='left',fontweight='bold')
    axs[1].set_title(r'1. EOF, $\hat{W}, W$',loc='center')

    axs[1].set_xlim(0,20)
    axs[1].set_ylim(0,20)
    axs[1].plot(x,coastline,color='grey')
    axs[1].set_aspect(1)

    axs[1].scatter(data_set_synt['lat'][mask_gps],data_set_synt['lon'][mask_gps],
                     c=data_set_synt['eof_with_noise'][mask_gps],
                     cmap=cmap('BlueWhiteOrangeRed_c',start=min_b,ende=max_b),vmax=max_b,vmin=min_b,s=18,edgecolor='k')
    
    axs[1].scatter(data_set_synt['lat'][mask_sattg],data_set_synt['lon'][mask_sattg],
                     c=data_set_synt['eof_with_noise'][mask_sattg],
                     cmap=cmap('BlueWhiteOrangeRed_c',start=min_b,ende=max_b),vmax=max_b,vmin=min_b,s=100,edgecolor='k')

    axs[2].plot(data_set_synt.time,data_set_synt.pc_timeseries.values)
    axs[2].set_title('C',loc='left',fontweight='bold')
    axs[2].set_title('1. PC',loc='center')

    axs[3].plot(data_set_synt.time,data_set_synt['missing_data_gps'],label='miss. data % [GNSS]')
    axs[3].plot(data_set_synt.time,data_set_synt['missing_data_sattg'],label='miss. data % [SATTG]')

    axs[3].set_title('D',loc='left',fontweight='bold')
    axs[3].set_title('Missing data',loc='center')
    axs[3].legend(frameon=False)
    add_='plot'
    if validation:
        chain=0
        trend_fit = bpca_object.trace['mean'].posterior['trend_g'][chain,:]*1000.

        axs[4].plot(x,coastline,color='grey')

        axs[4].set_xlim(0,20)
        axs[4].set_ylim(0,20)
        axs[4].set_aspect(1)

        axs[4].scatter(data_set_synt.lat.values,data_set_synt.lon.values,c=trend_fit.T,cmap=my_cmap,vmin=-3,vmax=3,s=18,edgecolor='k')
        axs[4].scatter(data_set_synt.lat.values[mask_sattg],data_set_synt.lon.values[mask_sattg],c=trend_fit.T[mask_sattg],cmap=my_cmap,vmin=-3,vmax=3,s=100,edgecolor='k')

        #cs = axs[1,0].imshow(cmap=my_cmap,vmin=-3,vmax=3,origin='lower',extent=extent)

        cbar = fig.colorbar(cs, ax=axs[4])
        axs[4].set_title('E',loc='left',fontweight='bold')
        axs[4].set_title(r'Trends, $g_{estimated}$ ',loc='center')
        #data_set_synt.lon.values
        cbar.set_label('mm/year')


        axs[5].set_xlim(0,20)
        axs[5].set_ylim(0,20)
        axs[5].plot(x,coastline,color='grey')
        axs[5].set_aspect(1)
        fig.colorbar(cs2, ax=axs[5])
        z_zero = bpca_object.trace['mean'].posterior['W0'][chain,:]

        
        # compute scaling factor to plot the estimates on the same scale (PCs, and EOFs)

        std_scale = float(data_set_synt['eof_with_noise'].std()/bpca_object.trace['mean'].posterior['W0'][chain,:].std())*np.sign(np.corrcoef(data_set_synt.pc_timeseries.values,(bpca_object.trace['mean'].posterior['PC0'][chain,:]*1000.).values)[0,1])
        std_scale = float(bpca_object.trace['mean'].posterior['PC0'][chain,:].std()*1000./data_set_synt['pc_timeseries'].std())*np.sign(np.corrcoef(data_set_synt.pc_timeseries.values,(bpca_object.trace['mean'].posterior['PC0'][chain,:]*1000.).values)[0,1])

        axs[5].scatter(data_set_synt.lat.values,data_set_synt.lon.values,c=bpca_object.trace['mean'].posterior['W0'][chain,:]*std_scale,
                         cmap=cmap('BlueWhiteOrangeRed_c',start=min_b,ende=max_b),vmax=max_b,vmin=min_b,s=18,edgecolor='k')

        axs[5].scatter(data_set_synt.lat.values[mask_sattg],data_set_synt.lon.values[mask_sattg],c=bpca_object.trace['mean'].posterior['W0'][chain,:][mask_sattg]*std_scale,
                         cmap=cmap('BlueWhiteOrangeRed_c',start=min_b,ende=max_b),vmax=max_b,vmin=min_b,s=100,edgecolor='k')
        #fig.colorbar(cs2, ax=axs[1,1])
        axs[5].set_title('F',loc='left',fontweight='bold')
        axs[5].set_title(r'1. EOF, $W_{estimated}$ ',loc='center')
        
        mean_est = bpca_object.trace['mean'].posterior['PC0'][chain,:]*1000.
        std_est  = bpca_object.trace['std'].posterior['PC0'][chain,:]*1000.
        axs[6].plot(bpca_object.dataset.time,mean_est/std_scale,label='1. PC (estimate)')        
        axs[6].fill_between(bpca_object.dataset.time.values, (mean_est - std_est)/std_scale, (mean_est + std_est)/std_scale, alpha=0.2,label='1 sigma')
        axs[6].set_title('G',loc='left',fontweight='bold')
        axs[6].set_title(r'1. PC, $1. PC_{estimated}$',loc='center')
        axs[6].plot(data_set_synt.time,data_set_synt.pc_timeseries.values,color='red',label='1. PC')
        axs[6].legend()
        #axs[7].plot(miss_gnss,label='miss. data % [GNSS]')
        #axs[7].plot(miss_sattg,label='miss. data % [SATTG]')

        #axs[7].set_title('D',loc='left',fontweight='bold')
        #axs[7].legend(frameon=False)  
        
        plt_change = [['K',10],
                      ['L',11],
                      ['I',8],
                      ['J',9]]
            
        if validation and not map_data:
            plt_change = [['I',8],
                          ['J',9],
                          ['I',8],
                          ['J',9]]
            fig.delaxes(axs[10])
            fig.delaxes(axs[11])
            
        trend_diff_with_noise = data_set_synt['trend_with_noise'].values-trend_fit.values
        combined_error = np.sqrt(data_set_synt['trend_un']**2+(bpca_object.trace['std'].posterior['trend_g'][chain]*1000.).values**2)
        
        axs[plt_change[0][1]].hist(trend_diff_with_noise[:-synthetic_data_settings['number_sattg']]/combined_error[:-synthetic_data_settings['number_sattg']],alpha=0.5,color='blue',density=True,label='GPS')
        axs[plt_change[0][1]].hist(trend_diff_with_noise[mask_sattg]/combined_error[mask_sattg],alpha=0.5,color='red',density=True,label='SATTG')
        axs[plt_change[0][1]].axvline(x = -1,color='grey',linestyle='--')
        axs[plt_change[0][1]].axvline(x = 1,color='grey',linestyle='--')   
        axs[plt_change[0][1]].set_title(plt_change[0][0],loc='left',fontweight='bold')
        axs[plt_change[0][1]].set_title(r'$Sig. ratio, g$ ',loc='center')
        axs[plt_change[0][1]].set_xlabel(r'$ (g-g_{estimated})/\sqrt{\sigma_g^2 + \sigma_{g,estimated}^2}$')
        axs[plt_change[0][1]].legend()
        
        #std_scale = float(bpca_object.trace['mean'].posterior['PC0'][chain,:].std()*1000./data_set_synt['pc_timeseries'].std())*np.sign(np.corrcoef(data_set_synt.pc_timeseries.values,(bpca_object.trace['mean'].posterior['PC0'][chain,:]*1000.).values)[0,1])
        eof_diff_with_noise = data_set_synt['eof_with_noise'].values-(bpca_object.trace['mean'].posterior['W0'][chain,:]*std_scale).values #std_scale
        combined_error_eof = np.sqrt((data_set_synt['eof_un']**2+(bpca_object.trace['std'].posterior['W0'][chain,:]*std_scale).values**2))        
        
        axs[plt_change[1][1]].hist(eof_diff_with_noise[:-synthetic_data_settings['number_sattg']]/combined_error_eof[:-synthetic_data_settings['number_sattg']],alpha=0.5,color='blue',density=True,label='GPS')
        axs[plt_change[1][1]].hist(eof_diff_with_noise[mask_sattg]/combined_error_eof[mask_sattg],alpha=0.5,color='red',density=True,label='SATTG')
        axs[plt_change[1][1]].axvline(x = -1,color='grey',linestyle='--')
        axs[plt_change[1][1]].axvline(x = 1,color='grey',linestyle='--')
        axs[plt_change[1][1]].set_title(plt_change[1][0],loc='left',fontweight='bold')
        axs[plt_change[1][1]].set_title(r'$Sig. ratio, W$ ',loc='center')
        axs[plt_change[1][1]].set_xlabel(r'$ (W-W_{estimated})/\sqrt{\sigma_W^2 + \sigma_{W,estimated}^2}$')
        axs[plt_change[1][1]].legend()        
        
        std_sigma = (data_set_synt['data_with_noise']-data_set_synt['data']).std(dim='time')
        axs[7].hist(std_sigma[:-synthetic_data_settings['number_sattg']],alpha=0.5,color='blue',density=True,label=r'$\sigma_{GPS}$')
        axs[7].hist(std_sigma[mask_sattg],alpha=0.5,color='red',density=True,label=r'$\sigma_{SATTG}$')
        
        mu = bpca_object.trace['mean'].posterior['sigma_hier'][chain,0]*1000.
        axs[7].axvline(x = mu,color='blue',label='estimate')
        
        mu = bpca_object.trace['mean'].posterior['sigma_hier'][chain,1]*1000.
        axs[7].axvline(x = mu,color='red',label='estimate')
        
        #axs[7].set_title('C',loc='left',fontweight='bold')
        axs[7].set_title('H',loc='left',fontweight='bold')
        axs[7].set_title(r'$\sigma$',loc='center')
        axs[7].set_xlabel('mm',fontweight='bold')
        axs[7].legend()   
        
        
            
        
        add_=add_+'_validation_'
        
        if map_data:
            max_b = 3
            min_b = -max_b         

            sig = abs(bpca_object.estimated_dataset_map_pattern[0]['W99'])<2*abs(bpca_object.estimated_dataset_map_pattern[1]['W99'])

            axs[8].scatter(bpca_object.estimated_dataset_map_pattern[0]['lat'],
                           bpca_object.estimated_dataset_map_pattern[0]['lon'],
                           c=bpca_object.estimated_dataset_map_pattern[0]['W99'],cmap=cmap('BlueWhiteOrangeRed_c',start=-3,ende=3),vmin=-3,vmax=3)





            # compute scaling factor to plot the estimates on the same scale (PCs, and EOFs)

            std_scale = float(data_set_synt['eof_with_noise'].std()/bpca_object.trace['mean'].posterior['W0'][chain,:].std())*np.sign(np.corrcoef(data_set_synt.pc_timeseries.values,(bpca_object.trace['mean'].posterior['PC0'][chain,:]*1000.).values)[0,1])
            std_scale = float(bpca_object.trace['mean'].posterior['PC0'][chain,:].std()*1000./data_set_synt['pc_timeseries'].std())*np.sign(np.corrcoef(data_set_synt.pc_timeseries.values,(bpca_object.trace['mean'].posterior['PC0'][chain,:]*1000.).values)[0,1])

            axs[8].scatter(data_set_synt.lat.values,data_set_synt.lon.values,c=trend_fit.T,cmap=my_cmap,vmin=-3,vmax=3,s=18,edgecolor='k')
            axs[8].scatter(data_set_synt.lat.values[mask_sattg],data_set_synt.lon.values[mask_sattg],c=trend_fit.T[mask_sattg],cmap=my_cmap,vmin=-3,vmax=3,s=100,edgecolor='k')

            axs[8].scatter(bpca_object.estimated_dataset_map_pattern[0]['lat'][sig],
                           bpca_object.estimated_dataset_map_pattern[0]['lon'][sig],
                           c=bpca_object.estimated_dataset_map_pattern[0]['W99'][sig],s=0.1,edgecolor='k')   

            #fig.colorbar(cs2, ax=axs[1,1])
            axs[8].set_title('I',loc='left',fontweight='bold')    
            axs[8].set_title(r'Trends, $g_{estimated},g_{estimated, 2D}$ ',loc='center')
            axs[8].set_xlim(0,20)
            axs[8].set_ylim(0,20)
            axs[8].plot(x,coastline,color='grey')
            axs[8].set_aspect(1)
            cbar = fig.colorbar(cs, ax=axs[8]) 
            cbar.set_label('mm/year')            
  
            max_b = 6
            min_b = -max_b

            axs[9].scatter(bpca_object.estimated_dataset_map_pattern[0]['lat'],
                           bpca_object.estimated_dataset_map_pattern[0]['lon'],
                           c=bpca_object.estimated_dataset_map_pattern[0]['W0']*std_scale,
                           cmap=cmap('BlueWhiteOrangeRed_c',start=min_b,ende=max_b),vmin=-6,vmax=6)
            sig = abs(bpca_object.estimated_dataset_map_pattern[0]['W0']*std_scale)<2*abs(bpca_object.estimated_dataset_map_pattern[1]['W0']*std_scale)


            axs[9].set_xlim(0,20)
            axs[9].set_ylim(0,20)
            axs[9].set_aspect(1)

            axs[9].scatter(data_set_synt.lat.values,data_set_synt.lon.values,c=bpca_object.trace['mean'].posterior['W0'][chain,:]*std_scale,
                             cmap=cmap('BlueWhiteOrangeRed_c',start=min_b,ende=max_b),vmax=max_b,vmin=min_b,s=18,edgecolor='k')

            axs[9].scatter(data_set_synt.lat.values[mask_sattg],data_set_synt.lon.values[mask_sattg],
                           c=bpca_object.trace['mean'].posterior['W0'][chain,:][mask_sattg]*std_scale,
                             cmap=cmap('BlueWhiteOrangeRed_c',start=min_b,ende=max_b),vmax=max_b,vmin=min_b,s=100,edgecolor='k')

            axs[9].scatter(bpca_object.estimated_dataset_map_pattern[0]['lat'][sig],
                           bpca_object.estimated_dataset_map_pattern[0]['lon'][sig],
                           c=bpca_object.estimated_dataset_map_pattern[0]['W0'][sig],s=0.1,edgecolor='k')         

            #cs = axs[1,0].imshow(cmap=my_cmap,vmin=-3,vmax=3,origin='lower',extent=extent)
            fig.colorbar(cs2, ax=axs[9])
            axs[9].set_title('J',loc='left',fontweight='bold')
            axs[9].set_title(r'1. EOF, $W_{estimated},W_{estimated, 2D}$ ',loc='center')

            x_new = np.linspace(-2.,22.,24*4+1)
            y = np.linspace(-2.,22.,24*4+1)
            y = y[:,np.newaxis]

            field_1,field_2,field_1_eq,field_2_eq=make_field(x_new,y)

            cs4 = axs[12].scatter(bpca_object.estimated_dataset_map_pattern[0]['lat'],
                   bpca_object.estimated_dataset_map_pattern[0]['lon'],
                   c=bpca_object.estimated_dataset_map_pattern[0]['W99']-(field_1+field_2 +field_2_eq).T.flatten(),cmap=cmap('BlueWhiteOrangeRed_c',start=-1,ende=1),vmin=-1,vmax=1)#

            sig = abs((field_1+field_2 +field_2_eq).T.flatten() - bpca_object.estimated_dataset_map_pattern[0]['W99'])<2*bpca_object.estimated_dataset_map_pattern[1]['W99']

            #sig = abs((field_1+field_2 +field_2_eq).T.flatten() - test_data.estimated_dataset_map_pattern[0]['W99'])<test_data.estimated_dataset_map_pattern[1]['W99']

            axs[12].scatter(bpca_object.estimated_dataset_map_pattern[0]['lat'][sig],
                       bpca_object.estimated_dataset_map_pattern[0]['lon'][sig],
                       c=bpca_object.estimated_dataset_map_pattern[0]['W99'][sig],s=0.1,edgecolor='k')    


            axs[12].set_title('M',loc='left',fontweight='bold')      
            axs[13].set_title(r'$\Delta W_{2D}: W_{estimated,2D}-\hat{W}_{2D}$ ',loc='center')
            axs[15].set_title(r'$\Delta W_{2D}$ ',loc='center')
            axs[12].set_xlim(0,20)
            axs[12].set_ylim(0,20)
            axs[12].plot(x,coastline,color='grey')
            axs[12].set_aspect(1)     
            cbar = fig.colorbar(cs4, ax=axs[12]) 
            cbar.set_label('mm/year')

            cs4 = axs[13].scatter(bpca_object.estimated_dataset_map_pattern[0]['lat'],
                   bpca_object.estimated_dataset_map_pattern[0]['lon'],
                   c=(bpca_object.estimated_dataset_map_pattern[0]['W0']*std_scale-(field_1_eq).T.flatten()),cmap=cmap('BlueWhiteOrangeRed_c',start=-1,ende=1),vmin=-6,vmax=6)#

            sig = abs((field_1_eq).T.flatten() - bpca_object.estimated_dataset_map_pattern[0]['W0']*std_scale)<2*bpca_object.estimated_dataset_map_pattern[1]['W0']*std_scale

            #sig = abs((field_1+field_2 +field_2_eq).T.flatten() - test_data.estimated_dataset_map_pattern[0]['W99'])<test_data.estimated_dataset_map_pattern[1]['W99']

            axs[13].scatter(bpca_object.estimated_dataset_map_pattern[0]['lat'][sig],
                       bpca_object.estimated_dataset_map_pattern[0]['lon'][sig],
                       c=bpca_object.estimated_dataset_map_pattern[0]['W0'][sig],s=0.1,edgecolor='k')    

            axs[12].set_title(r'$\Delta g_{2D}: g_{estimated,2D}-\hat{g}_{2D}$ ',loc='center')
            axs[14].set_title(r'$\Delta g_{2D}$ ',loc='center')
            axs[13].set_title('N',loc='left',fontweight='bold')        
            axs[13].set_xlim(0,20)
            axs[13].set_ylim(0,20)
            axs[13].plot(x,coastline,color='grey')
            axs[13].set_aspect(1)     
            fig.colorbar(cs4, ax=axs[13]) 


            axs[7].hist(std_sigma[:-synthetic_data_settings['number_sattg']],alpha=0.5,color='blue',density=True,label='GPS')
            axs[7].hist(std_sigma[mask_sattg],alpha=0.5,color='red',density=True,label='SATTG')

            axs[15].hist((data_set_synt['eof']-bpca_object.estimated_dataset_map_pattern[0]['W0'][indices].values)*std_scale,alpha=0.5,color='blue',density=True,label=r'$\hat{W} - W_{estimated}$')
            axs[15].hist((data_set_synt['eof_with_noise']-bpca_object.estimated_dataset_map_pattern[0]['W0'][indices].values)*std_scale,alpha=0.5,color='red',density=True,label='$W - W_{estimated}$')
            axs[15].legend()    
            axs[15].set_title('P',loc='left',fontweight='bold') 

            axs[14].hist(data_set_synt['trend']-bpca_object.estimated_dataset_map_pattern[0]['W99'][indices].values,alpha=0.5,color='blue',density=True,label=r'$\hat{g} - g_{estimated}$')
            axs[14].hist(data_set_synt['trend_with_noise']-bpca_object.estimated_dataset_map_pattern[0]['W99'][indices].values,alpha=0.5,color='red',density=True,label=r'$g - g_{estimated}$')
            axs[14].set_title('O',loc='left',fontweight='bold') 
            axs[14].legend()  
            add_=add_+'_with_maps_'
    plt.tight_layout()
    if save:
        plt.savefig(plt_dir + add_+bpca_object.name +'.pdf',bbox_inches='tight')
        plt.savefig(plt_dir + add_+bpca_object.name +'.png',bbox_inches='tight',dpi=600)
    plt.show()
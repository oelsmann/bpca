# Quick Start

-----
<a id='top'></a>

## Content

<div style="height:10px;"></div>

- [// Why Bayesian Principal Component Analysis?](#why)
- [// Creating Synthetic data](#synth)
- [// Setting up the data/model](#setup)
- [// Exp1: Running the model](#running)
- [// Model diagnostics](#diagnostics)
- [// Plotting and Visualization](#plotting-and-visualization)
- [// Exp2: Omitting individual stations](#exp_running)
- [// References](#References)

-----

<a id='why'></a>
### Why Bayesian Principal Component Analysis?

BPCA can be used to estimate the principal components for discontinuous/incomplete data. A full empirical probability distribution of the parameters is estimated [2], in contrast to maximum likelihood estimation or traditional PCA [3].


#### Model definition

We estimate height changes $\mathbf{U}(\mathbf{x},\mathbf{t})$ at every station in space $\mathbf{x}$ and time $\mathbf{t}$ as described by the following model: 

\begin{equation*}
 \mathbf{U}(\mathbf{x},\mathbf{t}) =  \mathbf{g}(\mathbf{x})\mathbf{t} + \sum_{k=1}^{k_{n}} \mathbf{W_k}(\mathbf{x}) \mathbf{p_k}(\mathbf{t})  + \mathbf{\epsilon}(\mathbf{x})\label{eq1}
\end{equation*}

Here, $\mathbf{p_k}(\mathbf{t})$ are $\mathbf{t}$-dimensional latent variables, or principal components, which are mapped onto the observations by the transformation matrix $\mathbf{W_k}(\mathbf{x})$. $\mathbf{W_k}(\mathbf{x})$ represent the spatial pattern of the common modes of variability (i.e., the EOF pattern) while the principal components $\mathbf{p_k}\mathbf{t})$ modulate the evolution of these pattern in time. The $\mathbf{x}$-dimensional vector $\mathbf{g}$ accounts for constant linear trends in the time series; $\mathbf{\epsilon}(\mathbf{x})$ describes technique-dependent Gaussian noise. 

For each parameter we define prior distributions. We assign Gaussian distributions to $\mathbf{g}$ and $\mathbf{W_k}(\mathbf{x})$, and a halfnormal distribution for the estimated variance $\mathbf{\epsilon}(\mathbf{x})$:

\begin{equation*}
  P(\mathbf{g}) \sim \mathcal{N}(\mathbf{\mu_{g}},\,\mathbf{\sigma_g^{2}})\label{eq2} \\
\end{equation*}
\begin{equation*}
  P(\mathbf{W}) \sim \mathcal{N}(\mathbf{\mu_{W}},\,\mathbf{\sigma_W^{2}})\label{eq3} \\
\end{equation*}
\begin{equation*}
  P(\mathbf{\epsilon}) \sim Halfnormal(\mathbf{\sigma_\epsilon^{2}})\label{eq4} \\
\end{equation*}

The technique-dependent variance $\mathbf{\epsilon}(\mathbf{x})$ is estimated individually for the two different techniques (GNSS and SATTG), considering that noise amplitudes differ by one order of magnitude. The principal components are modelled as Gaussian Random Walks, to simulate smoothly varying behaviour of the VLM. With this constrain we avoid that spurious high frequency signals are absorbed by the PCs:

\begin{equation*}
  \mathbf{p_k}(\mathbf{t}) = \mathbf{p_k}(\mathbf{t}-1) + \mathbf{h_k}(\mathbf{t}), P(\mathbf{h_k}) \sim \mathcal{N}(\mathbf{\mu_{h_k}},\mathbf{\sigma_{h_k}^2})
\end{equation*}

#### Model fitting

The model parameter are simultaneously estimated within a Bayesian Framework. The model implementation, the distribution setups and the sampling algorithms are all based on PyMC3, an extensive python-library to setup Bayesian models (see https://docs.pymc.io/). More information on the exact implementation is given in the paper.

[back to top ](#top)

----
## Creating synthetic data
<a id='synth'></a>
We create here synthetic data that mimic two classes of observations, GPS and differences of altimetry and tide gauge data. Here, GPS data has lower noise than the tide gauge data.


```python
from bpca.utils import *
from bpca.bpca_plots import *

my_cmap = cmap('BlueWhiteOrangeRed_c')

random_seed = 100
random.seed(random_seed)
np.random.seed(random_seed)

# all units are in mm here

synthetic_data_settings={'number_gnss': 500,
                            'number_sattg' : 30,
                            'sigma_gnss' : 5,
                            'sigma_sattg' : 15,
                            'std_deviation_trends' : 0.5,
                            'std_deviation_pcs' : 0.5,
                            'max_lev' : 20,
                            'time_series_length':26,
                            'miss_gnss_factor':20,
                            'miss_sattg_factor':2,
                            'miss_data_background_factor':5,
                            'random_seed':random_seed,
                            'relaxation_coeff':1}

data_set_synt,coastline = create_synthetic_data(**synthetic_data_settings)
```

This function creates some trend $\mathbf{\hat{g(x)}}$ and eof pattern $\mathbf{\hat{W(x)}}$, which are perturbed by normally distributed perturbations to obtain the trends $\mathbf{g(x)}$ and eof pattern $\mathbf{W(x)}$, which will present the input observations to the model.


```python
data_set_synt
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt, dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:             (time: 26, x: 530)
Coordinates:
    lon                 (x) float64 2.13 8.218 6.741 18.46 ... 14.69 7.39 16.16
    lat                 (x) float64 10.87 5.567 8.49 ... 8.696 11.68 8.096
    ID                  (x) float64 0.0 0.0 0.0 0.0 0.0 ... 1.0 1.0 1.0 1.0 1.0
  * time                (time) datetime64[ns] 1995-12-31 ... 2020-12-31
Dimensions without coordinates: x
Data variables:
    data                (time, x) float64 5.368 8.306 41.04 ... 17.89 7.989
    data_with_noise     (time, x) float64 12.01 3.411 31.81 ... -31.21 -27.83
    trend               (x) float64 0.6127 0.2164 0.2775 ... -0.04241 -0.963
    trend_with_noise    (x) float64 0.6757 -0.1703 -0.5353 ... 0.13 -1.343
    eof                 (x) float64 -2.556 -0.4312 -4.757 ... 4.082 3.761 3.804
    eof_with_noise      (x) float64 -2.627 -0.7315 -4.453 ... 4.07 3.64 3.414
    pc_timeseries       (time) float64 -6.931 -6.931 -6.931 ... 3.365 4.055 4.7
    time_series         (time) float64 0.0 1.0 2.0 3.0 ... 22.0 23.0 24.0 25.0
    missing_data_gps    (time) float64 0.3 0.2341 0.1899 0.1602 ... 0.1 0.1 0.1
    missing_data_sattg  (time) float64 0.12 0.1181 0.1164 ... 0.1018 0.1016</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-0f29709d-02a3-4128-86d1-93fd5cb804cf' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-0f29709d-02a3-4128-86d1-93fd5cb804cf' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 26</li><li><span>x</span>: 530</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-42983c80-af26-4bef-b5e2-abfe02ed87ec' class='xr-section-summary-in' type='checkbox'  checked><label for='section-42983c80-af26-4bef-b5e2-abfe02ed87ec' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>lon</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.13 8.218 6.741 ... 7.39 16.16</div><input id='attrs-5f5b8230-1675-4fb9-a61b-e67a82ff8e96' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5f5b8230-1675-4fb9-a61b-e67a82ff8e96' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b3d0efb9-f657-48de-ba7c-7ce1d286cc0e' class='xr-var-data-in' type='checkbox'><label for='data-b3d0efb9-f657-48de-ba7c-7ce1d286cc0e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([2.13020231e+00, 8.21799310e+00, 6.74071776e+00, 1.84618315e+01,
       1.66036389e+01, 1.64643029e+01, 8.35755280e+00, 7.03856257e+00,
       2.56565155e+00, 1.57334274e+01, 2.96332971e+00, 3.78155152e-01,
       1.92928091e+00, 4.64760736e+00, 1.21813013e+01, 1.74633426e+01,
       4.54328410e+00, 1.92788663e+00, 4.30865453e+00, 3.50111591e-02,
       6.07085670e+00, 1.10218871e+01, 9.60328452e+00, 3.96746246e-01,
       2.36835745e+00, 4.13995346e+00, 1.15061309e+01, 9.26670498e+00,
       1.48689913e+01, 5.05130483e+00, 1.37969674e+01, 6.38177472e+00,
       9.83477369e+00, 7.04776433e+00, 1.70793250e+01, 5.06567286e+00,
       5.67511802e+00, 3.81545604e+00, 1.04392482e+01, 9.62716890e+00,
       1.15048486e+01, 1.11782073e+01, 1.74510151e+01, 2.74294358e-01,
       3.32786602e-01, 1.88876768e+01, 1.13626910e+01, 1.27599491e+01,
       2.54798047e+00, 1.35697196e+01, 1.46842605e+01, 7.10186554e+00,
       1.95553438e+01, 7.31514463e+00, 1.83639882e+01, 1.09084517e+01,
       1.74145964e+01, 2.59660929e+00, 9.57074717e+00, 1.69429901e+01,
       1.85745606e+00, 1.17420176e+01, 8.90182504e+00, 1.63910559e+01,
       3.50383569e+00, 1.42924340e+01, 1.60635020e+00, 2.46333378e+00,
       3.15083607e+00, 5.26613091e-01, 5.51702431e+00, 7.31356718e+00,
       8.96602427e+00, 2.86960492e+00, 6.85544133e+00, 8.24171458e+00,
       9.71465422e+00, 1.27815342e+01, 8.08344603e-01, 1.33402622e+01,
...
       2.26593882e+00, 1.20707398e+01, 3.41475745e+00, 1.40553788e+01,
       1.22468902e+01, 8.58216322e+00, 3.87329293e+00, 5.60771147e+00,
       4.73899109e+00, 5.64048555e+00, 6.33916639e+00, 4.56189676e+00,
       1.08241830e+01, 1.75222183e+01, 4.72018211e-01, 7.80895944e+00,
       1.20002094e+01, 1.21326708e+01, 6.11869592e+00, 1.91134725e+01,
       4.85170185e+00, 1.04319759e+01, 7.48520632e+00, 4.57115036e+00,
       2.57074464e+00, 1.36933357e+01, 1.08505505e+01, 8.96445150e+00,
       1.56768218e+01, 7.28678496e+00, 8.51253182e+00, 3.16411870e-01,
       4.88880648e+00, 7.42534033e+00, 1.18755795e+00, 1.68383895e+01,
       6.30165045e+00, 1.78249891e+01, 1.14723401e+01, 1.73872587e+01,
       7.80326765e-01, 6.62202976e+00, 6.40753419e+00, 7.34726520e+00,
       5.15301955e+00, 1.59169192e+01, 1.54506710e+01, 6.21811286e+00,
       1.34595707e+01, 1.72293436e+01, 4.00236551e+00, 8.72940496e+00,
       2.83912924e+00, 1.23354293e+01, 6.62909985e+00, 6.64627990e+00,
       5.03779606e+00, 1.44593545e+01, 5.68979684e+00, 1.10223000e+01,
       1.41243581e+00, 5.73001520e+00, 1.19561822e+01, 8.39311577e-01,
       1.54508131e+01, 1.45537062e+00, 7.35581475e+00, 1.36534265e+01,
       1.49186595e+01, 6.94943994e+00, 5.13596235e+00, 3.48992616e+00,
       1.19043762e+01, 1.24593201e+01, 1.84523407e+01, 1.46898532e+01,
       7.39040781e+00, 1.61610704e+01])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lat</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>10.87 5.567 8.49 ... 11.68 8.096</div><input id='attrs-099b4ce3-cbc9-4314-a5d2-346f3ddab3d7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-099b4ce3-cbc9-4314-a5d2-346f3ddab3d7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e95b2173-587c-4c84-8d49-3dcf5daed691' class='xr-var-data-in' type='checkbox'><label for='data-e95b2173-587c-4c84-8d49-3dcf5daed691' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1.08680988e+01, 5.56738770e+00, 8.49035181e+00, 9.43771238e-02,
       2.43138242e+00, 2.73413179e+00, 4.18404244e+00, 3.70656439e+00,
       2.16753781e+00, 4.39394985e+00, 3.43882025e+00, 5.48147494e+00,
       6.72223900e+00, 3.50820907e+00, 7.45664093e+00, 1.13770147e-01,
       5.04852707e+00, 3.05099425e-01, 1.19768675e+01, 2.10295371e+00,
       7.63886890e+00, 7.29521132e-01, 1.19883978e+00, 1.15380300e+01,
       1.26036787e+01, 4.08782641e-01, 4.20053155e+00, 5.01390458e+00,
       5.71791381e+00, 7.19015688e+00, 7.09591223e+00, 6.80380431e+00,
       3.56161979e+00, 4.75388417e+00, 8.97245649e-01, 1.01086286e+01,
       7.52504909e+00, 1.18561080e+01, 2.85200629e+00, 7.75532561e+00,
       7.26376008e+00, 4.08690554e+00, 5.53530123e+00, 4.93071762e+00,
       3.47216003e+00, 6.80770446e+00, 1.84111207e+00, 1.76920346e+00,
       7.90071864e+00, 6.71192883e+00, 6.26132883e+00, 1.08080915e+01,
       5.93587502e+00, 2.21575802e+00, 6.25280596e+00, 9.13958260e+00,
       5.08515036e+00, 1.28220252e+01, 4.00247214e+00, 6.18000697e+00,
       1.25064752e+01, 3.33388262e+00, 4.63562730e-01, 3.21489097e+00,
       4.21956837e+00, 7.21050502e+00, 5.43661698e+00, 9.21203242e+00,
       1.00071179e+01, 1.05191187e+01, 2.79804624e-02, 7.89400573e+00,
       8.05760663e+00, 7.08596600e+00, 1.00122864e+01, 1.80865576e+00,
       5.47125840e+00, 5.30892827e-01, 7.99973793e-01, 5.66280719e+00,
...
       7.17865184e+00, 4.78361884e+00, 3.88248893e+00, 3.79281336e+00,
       1.19949319e+00, 7.96311160e+00, 9.14975422e+00, 3.46590817e+00,
       1.11032045e+01, 7.61325298e-01, 1.59825298e-02, 2.62295629e-01,
       2.94209699e+00, 5.31247837e+00, 1.20989017e+00, 5.15731262e+00,
       4.58122666e+00, 4.03708973e+00, 4.28729011e+00, 1.82495001e+00,
       1.00868007e+01, 7.28312226e+00, 2.80146515e+00, 5.35250204e+00,
       5.47592156e+00, 6.77244936e+00, 1.43207589e+00, 9.97193751e+00,
       6.12328428e+00, 8.63637987e+00, 1.09636712e+01, 1.18455782e+01,
       2.15868764e+00, 5.75629868e+00, 5.29834680e+00, 6.58583554e+00,
       3.07878566e+00, 6.11472555e+00, 8.37907626e+00, 2.88445043e+00,
       1.04242061e+01, 1.14642175e+01, 3.70165513e+00, 1.53900426e+00,
       4.62959529e+00, 4.47476944e+00, 1.58631287e+00, 1.05810628e+01,
       9.19834484e+00, 7.65996817e+00, 1.30576607e+01, 1.11286406e+01,
       1.35323565e+01, 9.65708678e+00, 1.19857375e+01, 1.19787266e+01,
       1.26351200e+01, 8.79035102e+00, 1.23690502e+01, 1.01929513e+01,
       1.41145645e+01, 1.23526378e+01, 9.81185073e+00, 1.43484462e+01,
       8.38575458e+00, 1.40970436e+01, 1.16891782e+01, 9.11923577e+00,
       8.60291692e+00, 1.18550124e+01, 1.25950601e+01, 1.32667780e+01,
       9.83299183e+00, 9.60652917e+00, 7.16088497e+00, 8.69628864e+00,
       1.16750614e+01, 8.09591129e+00])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>ID</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.0 0.0 0.0 ... 1.0 1.0 1.0 1.0</div><input id='attrs-44a71d87-1394-4cdc-bd15-46043d0e51d6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-44a71d87-1394-4cdc-bd15-46043d0e51d6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-47e9ef3b-8d81-4a72-8be6-f32f0ed8a246' class='xr-var-data-in' type='checkbox'><label for='data-47e9ef3b-8d81-4a72-8be6-f32f0ed8a246' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>1995-12-31 ... 2020-12-31</div><input id='attrs-90feced5-ddc2-428d-823a-ba5edfe826d2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-90feced5-ddc2-428d-823a-ba5edfe826d2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dbbd5a60-591c-4c20-89f4-328d6eb9dd5a' class='xr-var-data-in' type='checkbox'><label for='data-dbbd5a60-591c-4c20-89f4-328d6eb9dd5a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;1995-12-31T00:00:00.000000000&#x27;, &#x27;1996-12-31T00:00:00.000000000&#x27;,
       &#x27;1997-12-31T00:00:00.000000000&#x27;, &#x27;1998-12-31T00:00:00.000000000&#x27;,
       &#x27;1999-12-31T00:00:00.000000000&#x27;, &#x27;2000-12-31T00:00:00.000000000&#x27;,
       &#x27;2001-12-31T00:00:00.000000000&#x27;, &#x27;2002-12-31T00:00:00.000000000&#x27;,
       &#x27;2003-12-31T00:00:00.000000000&#x27;, &#x27;2004-12-31T00:00:00.000000000&#x27;,
       &#x27;2005-12-31T00:00:00.000000000&#x27;, &#x27;2006-12-31T00:00:00.000000000&#x27;,
       &#x27;2007-12-31T00:00:00.000000000&#x27;, &#x27;2008-12-31T00:00:00.000000000&#x27;,
       &#x27;2009-12-31T00:00:00.000000000&#x27;, &#x27;2010-12-31T00:00:00.000000000&#x27;,
       &#x27;2011-12-31T00:00:00.000000000&#x27;, &#x27;2012-12-31T00:00:00.000000000&#x27;,
       &#x27;2013-12-31T00:00:00.000000000&#x27;, &#x27;2014-12-31T00:00:00.000000000&#x27;,
       &#x27;2015-12-31T00:00:00.000000000&#x27;, &#x27;2016-12-31T00:00:00.000000000&#x27;,
       &#x27;2017-12-31T00:00:00.000000000&#x27;, &#x27;2018-12-31T00:00:00.000000000&#x27;,
       &#x27;2019-12-31T00:00:00.000000000&#x27;, &#x27;2020-12-31T00:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-7d2e17a9-33ba-4117-abae-88c54952fc8d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7d2e17a9-33ba-4117-abae-88c54952fc8d' class='xr-section-summary' >Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>data</span></div><div class='xr-var-dims'>(time, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>5.368 8.306 41.04 ... 17.89 7.989</div><input id='attrs-0d05f8a5-d43f-4231-b78b-4f19ce714c79' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0d05f8a5-d43f-4231-b78b-4f19ce714c79' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fae8ae6f-9b2b-4236-bbb1-de3396193643' class='xr-var-data-in' type='checkbox'><label for='data-fae8ae6f-9b2b-4236-bbb1-de3396193643' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[  5.3676007 ,   8.3057017 ,  41.03603669, ...,  -0.21411381,
        -27.6971336 ,   1.85423184],
       [  6.04328981,   8.13540771,  40.50073292, ...,  -1.68758928,
        -27.56718284,   0.5109851 ],
       [  6.71897892,   7.96511371,  39.96542915, ...,  -3.16106476,
        -27.43723209,  -0.83226164],
       ...,
       [ -6.13477573,  -3.1423462 , -17.12404276, ...,   7.80006066,
         12.76618465,   6.11583122],
       [ -7.27120262,  -3.81729758, -20.73154148, ...,   9.13450125,
         15.40722828,   7.12834067],
       [ -8.29063481,  -4.45966704, -24.14069169, ...,  10.28765584,
         17.88614954,   7.98875669]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>data_with_noise</span></div><div class='xr-var-dims'>(time, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>12.01 3.411 31.81 ... -31.21 -27.83</div><input id='attrs-1f721ad7-8f1e-4e2c-90db-e5fbdf3f3754' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1f721ad7-8f1e-4e2c-90db-e5fbdf3f3754' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-75c2b434-d5ea-492e-a8ad-0ed58d9268b7' class='xr-var-data-in' type='checkbox'><label for='data-75c2b434-d5ea-492e-a8ad-0ed58d9268b7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 12.01345567,   3.41103284,  31.80997102, ...,  -2.28717546,
        -55.32244545,  26.90272769],
       [  5.6743888 ,  12.04226064,          nan, ..., -14.34677572,
        -48.05197476,  10.00792972],
       [  3.02503984,  12.00345781,  38.85781047, ...,          nan,
        -19.28611819,   0.85757531],
       ...,
       [         nan,   4.65502509,          nan, ..., -50.84530457,
          3.7753648 ,  45.40572421],
       [ -6.19821402,  -7.73436989, -23.719421  , ...,  -7.45028508,
          4.02985954,  -2.49048107],
       [ -3.66839706,  -0.60996792, -21.64468166, ...,  -4.75995224,
        -31.21122348, -27.82789679]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>trend</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.6127 0.2164 ... -0.04241 -0.963</div><input id='attrs-449c6fa7-b277-4bbd-8450-6dffa147c57d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-449c6fa7-b277-4bbd-8450-6dffa147c57d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-aba58a15-9744-44e7-bfc9-85fea0a4cc34' class='xr-var-data-in' type='checkbox'><label for='data-aba58a15-9744-44e7-bfc9-85fea0a4cc34' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.61272387,  0.21644217,  0.27746108, -0.95166823, -0.88734113,
       -0.88265929,  0.24800453,  0.5361113 ,  1.58718199, -0.85140156,
        1.4605837 ,  1.7737732 ,  1.31822719,  1.07588299, -0.49165133,
       -0.89046702,  1.00093811,  1.66887848,  0.25578109,  2.11075284,
        0.46442609, -0.19516863,  0.04777863,  0.65528307,  0.35070958,
        1.18490725, -0.31055786,  0.0458384 , -0.78703068,  0.69144593,
       -0.68958334,  0.48177792, -0.01539067,  0.48933674, -0.88551879,
        0.36698557,  0.54481658,  0.31596508, -0.11001536, -0.13255376,
       -0.39298609, -0.2562015 , -1.0336186 ,  1.86739681,  2.00329774,
       -1.17391023, -0.25212397, -0.45561828,  1.03107689, -0.65416915,
       -0.77694826,  0.04684814, -1.1990892 ,  0.50578718, -1.12254642,
       -0.36867665, -1.02035145,  0.30767484,  0.0222312 , -1.00326073,
        0.40057999, -0.33005673,  0.16998531, -0.88760417,  1.29186784,
       -0.75007629,  1.56048911,  0.83485913,  0.62535797,  0.82670378,
        0.85647706,  0.23124516, -0.04508269,  1.09929166,  0.1370153 ,
        0.31256421, -0.05060766, -0.44733281,  1.92991619, -0.60307641,
        0.05956649,  0.50902894,  1.75881782, -0.28513732,  0.40697933,
       -0.3397172 , -0.49283797, -0.71781013,  0.401944  ,  0.00765703,
       -1.00224047, -0.40602513, -0.9231814 ,  0.5795944 ,  0.84817428,
       -0.39086397,  0.87433995,  0.49204025,  1.46073332, -0.24568399,
...
       -0.70270131,  0.27146795, -0.18877285,  0.3898387 , -1.1281988 ,
       -0.05070505, -0.09500471,  0.67101248,  2.09985455,  1.51188792,
        0.04617078,  0.47711809, -1.01044623, -0.79722229,  0.37196949,
       -0.90719251, -0.35400749,  0.90023065,  1.52837186, -1.03503344,
        0.03465863,  1.04001493,  1.19001019, -0.40903064,  1.33445129,
       -0.65441202, -0.37945423,  0.02034508,  0.6469017 ,  0.85973167,
        0.30130148,  0.8591654 ,  0.6767813 ,  1.081895  , -0.17674168,
       -1.03449316,  2.01432777,  0.31561392, -0.39404995, -0.40200377,
        0.71021985, -1.04730711,  0.39739695, -0.23617885,  0.4631399 ,
        0.96902927,  1.36841584, -0.67058583, -0.16781981, -0.1429772 ,
       -0.88087509,  0.17999689, -0.13437462,  0.60873645,  1.05225493,
        0.35761712,  1.65655979, -1.00172785,  0.71583914, -1.07710746,
       -0.42112665, -0.96294452,  0.8203664 ,  0.05723091,  0.67303816,
        0.49806804,  0.90045627, -0.87132797, -0.76318723,  0.17397335,
       -0.68771528, -1.05600061,  0.17693968, -0.16820191,  0.2133988 ,
       -0.56242921,  0.02042636,  0.01908682,  0.12719729, -0.79505778,
        0.08772428, -0.41430595,  0.23069991,  0.08508995, -0.51962098,
        0.22970792, -0.89558075,  0.23059882, -0.0394057 , -0.70890539,
       -0.84247063, -0.00515985,  0.12164918,  0.19553705, -0.51376652,
       -0.57638007, -1.14857678, -0.81902202, -0.04240954, -0.96301205])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>trend_with_noise</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.6757 -0.1703 ... 0.13 -1.343</div><input id='attrs-5a41f0e0-4518-4bdc-b136-df9248e51af3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5a41f0e0-4518-4bdc-b136-df9248e51af3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6cf216e5-16db-4b86-b1c5-6719f1e0dd44' class='xr-var-data-in' type='checkbox'><label for='data-6cf216e5-16db-4b86-b1c5-6719f1e0dd44' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 6.75689110e-01, -1.70293996e-01, -5.35303771e-01, -1.37646965e+00,
       -3.94609420e-01, -7.06590589e-01,  1.19926149e+00,  3.45095033e-01,
        1.34375649e+00, -8.37887959e-01,  1.64352452e+00,  1.69075889e+00,
        1.58842833e+00,  9.64083880e-01, -3.60800809e-01, -1.32987987e+00,
        1.11262887e+00,  1.80684943e+00,  2.88105714e-01,  2.23640093e+00,
        3.44962362e-01, -8.24615634e-01, -1.20500246e+00,  4.85697667e-01,
        5.92455928e-01,  1.01871802e+00,  8.65671162e-02,  7.75045077e-02,
       -5.09779425e-01,  3.10494644e-01, -1.22669435e+00, -1.43175233e-01,
        4.36043180e-01,  2.85179132e-01, -1.05239742e+00, -1.76061669e-01,
        1.08451872e+00, -1.00999601e-01, -5.63362053e-01,  1.07018394e-01,
       -1.80764333e-01,  2.42140261e-01, -1.26289978e+00,  1.90977694e+00,
        1.62740336e+00, -1.70403470e+00,  5.31463101e-02, -2.52482764e-01,
        1.06580432e+00,  3.08195607e-01, -4.99859114e-02,  1.08996723e+00,
       -1.36859672e+00,  3.21658289e-02, -1.17348221e+00, -6.51161615e-01,
       -1.16061916e+00,  1.39346731e-01, -5.24825003e-01, -1.27076313e+00,
        4.26281774e-01, -5.60133707e-01, -1.33975383e+00, -1.28934362e+00,
        1.79963346e+00, -1.17126623e+00,  1.86198186e+00,  8.24295696e-01,
        1.15516780e+00,  1.12349698e+00,  1.28755790e+00,  3.81995941e-01,
       -2.85842733e-01,  1.16035834e+00,  3.70396568e-01,  4.58601317e-01,
        5.44643972e-01, -1.02958081e+00,  9.73996948e-01, -3.14693443e-01,
...
        4.93546245e-01, -2.40581538e-01,  1.40833456e+00, -8.20908729e-01,
        5.06902867e-01,  9.53902952e-02,  1.20818113e+00,  1.28039177e+00,
       -6.20130853e-01, -1.45461882e-01,  3.22968267e-01,  1.38283697e+00,
        7.30820036e-01, -1.53450762e+00,  2.13678647e+00,  1.37168974e-01,
       -8.78069779e-01,  1.10319003e-02,  8.79643336e-01, -2.56030861e-01,
        4.02330384e-01,  2.40531319e-01,  1.85516439e-01,  1.23041443e+00,
        8.48062355e-01, -6.70344336e-01,  3.33933863e-01, -7.67381068e-01,
       -8.44234278e-01, -5.07058597e-01, -4.33913841e-01,  9.45107722e-01,
        1.33241804e+00,  3.59998913e-01,  1.74739069e+00, -2.91203683e-01,
        1.08800549e+00, -7.34286475e-01, -7.68162649e-01, -1.20030852e+00,
        1.44948551e+00,  2.58210341e-01,  8.54057620e-01,  1.36126623e+00,
        4.49254057e-01, -1.07794465e+00, -1.48601428e+00, -8.82406599e-02,
       -1.36005015e+00, -2.29614435e+00,  9.53946136e-02, -4.25595282e-01,
       -5.18059124e-02, -1.14173954e+00,  1.38175535e+00, -1.24869529e-01,
        1.50912511e+00, -7.42974636e-01,  5.34150154e-01, -4.19492459e-02,
        5.24519344e-01,  1.57394524e-01, -7.25770531e-01, -9.27121234e-01,
       -7.29333124e-01,  3.15254031e-01, -4.36296085e-01, -3.77539192e-01,
       -1.02291456e+00,  5.19576594e-03, -2.75412553e-01,  6.35655197e-01,
       -1.92784892e-01, -1.28704899e+00, -8.49647927e-01, -1.47347548e+00,
        1.29950751e-01, -1.34324674e+00])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>eof</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.556 -0.4312 ... 3.761 3.804</div><input id='attrs-a57167f3-75da-4e66-8947-126e0753696a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a57167f3-75da-4e66-8947-126e0753696a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7358e993-719d-4e87-a228-384eb40ac8a7' class='xr-var-data-in' type='checkbox'><label for='data-7358e993-719d-4e87-a228-384eb40ac8a7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-2.55618778e+00, -4.31235385e-01, -4.75677612e+00, -1.37389185e-03,
       -2.20455815e-01, -3.77320859e-01, -1.22518573e-02, -3.66493487e-04,
       -9.57164950e-12, -2.62625573e+00, -6.87056154e-08, -1.74401699e-05,
       -2.02680948e-02, -2.73726041e-06, -1.76971434e+00, -4.20848769e-04,
       -1.95067686e-03, -7.35477380e-19,  3.94551856e+00, -6.28824936e-15,
       -2.98379488e+00, -2.16964743e-07, -1.79649453e-07, -2.12945081e+00,
        3.34298540e+00, -1.50834942e-15, -2.70523466e-01, -2.92254058e-01,
       -3.51049490e+00, -1.01243604e+00, -6.09336764e-01, -1.21619031e+00,
       -8.91340723e-03, -1.67458243e-02, -4.69852749e-03, -1.64245863e+00,
       -2.26340049e+00,  3.39778670e+00, -1.52503800e-03, -4.26968413e+00,
       -3.54243264e+00, -1.59066481e-01, -1.15256420e+00, -9.61620714e-07,
       -2.28860111e-10,  3.10286953e+00, -9.31072302e-05, -5.30051625e-04,
       -6.45863226e-01, -2.55538343e+00, -2.50562991e+00,  4.32857452e+00,
        2.17141452e+00, -5.18720404e-07,  2.16770042e+00,  3.81212595e+00,
       -2.35028458e+00,  3.51824654e+00, -2.69793878e-02,  4.18660384e-01,
        2.78180931e+00, -3.53887107e-02, -5.13734250e-10, -8.45810045e-01,
       -1.13223888e-05,  7.42976128e-01, -1.13722268e-04, -3.19580664e+00,
       -3.98093172e+00, -3.52450584e+00, -3.91836220e-15, -4.64552888e+00,
       -4.15997975e+00, -1.48419890e-01,  1.11785773e+00, -3.50178941e-07,
       -9.76981076e-01, -1.98718953e-06, -1.50496464e-18, -3.88179399e+00,
...
       -1.04810011e-01, -1.17218695e+00, -1.82213365e-06, -6.59201553e-01,
       -2.07842453e-05, -4.64433253e+00, -4.25552976e+00, -1.24920230e-05,
        2.33710292e+00, -1.33734991e-12, -3.56729722e-14, -1.60144218e-15,
       -3.49251161e-03, -1.71090866e+00, -1.71851290e-17, -1.17866102e-01,
       -8.07884790e-01, -3.02096210e-01, -9.78442940e-04, -3.06596199e-01,
       -2.07738446e+00, -4.50760377e+00, -1.35886294e-05, -5.99227775e-03,
       -5.98162635e-04, -2.14874229e+00, -6.04884772e-06,  4.16773209e+00,
       -1.59056169e+00, -4.25155682e+00,  4.49057089e+00, -1.23977216e+00,
       -2.35530284e-09, -3.50889498e-01, -3.06530194e-05,  1.75892353e+00,
       -6.53612088e-06,  1.26413841e+00,  1.47212567e+00, -7.99125159e-01,
       -3.59784067e+00,  4.46057968e+00, -1.36924926e-04, -1.22923363e-08,
       -9.34450365e-04, -2.84061719e+00, -7.72198332e-03,  2.59387880e+00,
        4.23330038e+00,  3.54913174e+00,  2.83915235e+00,  4.02843171e+00,
        2.48792912e+00,  4.29934702e+00,  3.57850191e+00,  3.58283091e+00,
        3.14491807e+00,  4.11611603e+00,  3.32947731e+00,  4.28451262e+00,
        2.06415680e+00,  3.34058618e+00,  4.30530439e+00,  1.90033511e+00,
        3.95018700e+00,  2.07661534e+00,  3.75325666e+00,  4.21475537e+00,
        4.04506054e+00,  3.65770329e+00,  3.17319403e+00,  2.68469508e+00,
        4.30547176e+00,  4.29560020e+00,  3.21644535e+00,  4.08175393e+00,
        3.76111998e+00,  3.80429947e+00])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>eof_with_noise</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.627 -0.7315 ... 3.64 3.414</div><input id='attrs-89e9ac41-8b64-48bb-b83b-016a41a07369' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-89e9ac41-8b64-48bb-b83b-016a41a07369' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ff2b8930-e041-4093-ad55-da40e190a3e8' class='xr-var-data-in' type='checkbox'><label for='data-ff2b8930-e041-4093-ad55-da40e190a3e8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-2.62652642e+00, -7.31463088e-01, -4.45291648e+00,  2.79345778e-01,
        6.74763535e-02,  7.74501936e-01, -6.88344400e-01,  2.00686806e-01,
        4.49520669e-01, -2.83154379e+00, -4.69148246e-01,  7.25008171e-01,
       -4.43408865e-01,  1.20161119e+00, -1.30744207e+00,  5.65906368e-01,
       -2.98503854e-01,  7.56910968e-01,  4.16610972e+00,  8.64795813e-01,
       -3.04915859e+00,  3.52756718e-02, -4.29398734e-01, -2.17298922e+00,
        3.85389069e+00,  8.94975572e-01, -4.91379514e-01, -3.33824472e-01,
       -3.10025152e+00, -1.28098408e+00, -4.62843972e-01, -1.24276696e+00,
       -4.86177509e-01, -6.64862101e-01,  1.39916453e-01, -1.44539794e+00,
       -1.78468006e+00,  2.98039995e+00, -3.62653721e-01, -4.52957908e+00,
       -3.21212747e+00, -3.35320206e-01, -1.40884516e+00,  8.91747748e-01,
       -1.60719366e-02,  3.60107118e+00,  1.28087301e-01,  8.37262208e-01,
       -1.16628524e+00, -3.00914049e+00, -1.88483875e+00,  4.46356324e+00,
        2.53670085e+00,  1.64460856e-01,  2.36121889e+00,  4.43743934e+00,
       -2.22051307e+00,  3.93616954e+00, -1.30438641e+00, -1.90877596e-02,
        2.15203312e+00,  2.49106197e-01,  1.81475691e-01, -9.68275873e-01,
       -4.45324916e-01,  8.56706556e-01, -6.79550823e-01, -2.63536333e+00,
       -4.38707106e+00, -4.20725578e+00, -1.00530093e-01, -4.74259802e+00,
       -3.77924930e+00,  1.12697078e-01,  1.33616833e+00, -2.31911032e-01,
       -1.19217938e+00, -2.19731167e-01,  4.22268788e-01, -4.05510364e+00,
...
       -8.42864565e-02, -1.10444298e+00, -8.47642169e-01, -5.70140416e-01,
       -2.26600655e-02, -3.61833333e+00, -4.55188085e+00, -8.26193823e-02,
        2.23693139e+00, -3.58437694e-01,  8.33855897e-03, -2.36682437e-01,
        5.55755441e-01, -1.46787205e+00, -5.13850788e-01, -9.99189174e-01,
       -7.55701009e-01, -7.19520414e-01,  5.22188271e-01, -7.54575039e-01,
       -2.75684253e+00, -3.84239497e+00,  4.64642966e-01, -2.82944850e-02,
        4.74851026e-01, -2.46495502e+00, -5.30581256e-01,  4.35356185e+00,
       -1.54258052e+00, -3.55269139e+00,  3.45953900e+00, -8.27313517e-01,
        5.89724174e-02, -1.11823161e-01, -2.58291342e-01,  1.50226576e+00,
        4.74587016e-01,  1.62352521e+00,  2.40283882e+00, -7.86203047e-01,
       -2.93983062e+00,  4.13772911e+00,  7.07425123e-01,  1.98537215e-01,
       -1.81526701e-01, -3.10560124e+00,  4.70609916e-01,  1.85437870e+00,
        3.57214931e+00,  2.22523738e+00,  2.57193881e+00,  4.02278078e+00,
        2.10972970e+00,  3.92089209e+00,  4.15837444e+00,  3.93603400e+00,
        2.80787817e+00,  4.16912562e+00,  3.69982914e+00,  4.00524046e+00,
        1.46220461e+00,  2.76123853e+00,  3.75764970e+00,  1.69435227e+00,
        4.08318944e+00,  1.62804440e+00,  2.74198607e+00,  4.03623679e+00,
        2.94848640e+00,  3.41379562e+00,  3.53671592e+00,  3.82957950e+00,
        4.64792823e+00,  4.30849756e+00,  2.09877874e+00,  4.06986404e+00,
        3.63964105e+00,  3.41449217e+00])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>pc_timeseries</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-6.931 -6.931 -6.931 ... 4.055 4.7</div><input id='attrs-24f312df-dc11-4ee2-92cb-a9fafef3d028' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-24f312df-dc11-4ee2-92cb-a9fafef3d028' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bb65ff31-7fd2-4fb1-89b0-b4b0e0c1cbd7' class='xr-var-data-in' type='checkbox'><label for='data-bb65ff31-7fd2-4fb1-89b0-b4b0e0c1cbd7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-6.93147181, -6.93147181, -6.93147181, -6.93147181, -6.93147181,
       -6.93147181, -6.93147181, -6.93147181, -6.93147181, -6.93147181,
       -6.93147181, -6.93147181, -6.93147181, -6.93147181, -6.93147181,
       -5.10825624, -3.56674944, -2.23143551, -1.05360516,  0.        ,
        0.9531018 ,  1.82321557,  2.62364264,  3.36472237,  4.05465108,
        4.70003629])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>time_series</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 1.0 2.0 3.0 ... 23.0 24.0 25.0</div><input id='attrs-0869da18-3e71-4e6a-82dc-6d9f7922869a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0869da18-3e71-4e6a-82dc-6d9f7922869a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cca3a411-b072-4f41-8cd3-01e5e4470614' class='xr-var-data-in' type='checkbox'><label for='data-cca3a411-b072-4f41-8cd3-01e5e4470614' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
       13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>missing_data_gps</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3 0.2341 0.1899 ... 0.1 0.1 0.1</div><input id='attrs-4ab1fe59-8484-4d39-a57e-2b541f07c598' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4ab1fe59-8484-4d39-a57e-2b541f07c598' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-71e29772-6b50-4caa-9bca-4e9d2854ca45' class='xr-var-data-in' type='checkbox'><label for='data-71e29772-6b50-4caa-9bca-4e9d2854ca45' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.3       , 0.23406401, 0.18986579, 0.16023884, 0.1403793 ,
       0.12706706, 0.11814359, 0.11216201, 0.10815244, 0.10546474,
       0.10366313, 0.10245547, 0.10164595, 0.10110331, 0.10073957,
       0.10049575, 0.10033231, 0.10022276, 0.10014932, 0.10010009,
       0.10006709, 0.10004497, 0.10003015, 0.10002021, 0.10001355,
       0.10000908])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>missing_data_sattg</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.12 0.1181 ... 0.1018 0.1016</div><input id='attrs-c76e761c-b084-4758-b47b-511f9b757bfe' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c76e761c-b084-4758-b47b-511f9b757bfe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d438e261-34b5-4ef2-8330-2ea867d87cdd' class='xr-var-data-in' type='checkbox'><label for='data-d438e261-34b5-4ef2-8330-2ea867d87cdd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.12      , 0.11809675, 0.11637462, 0.11481636, 0.1134064 ,
       0.11213061, 0.11097623, 0.10993171, 0.10898658, 0.10813139,
       0.10735759, 0.10665742, 0.10602388, 0.10545064, 0.10493194,
       0.1044626 , 0.10403793, 0.10365367, 0.10330598, 0.10299137,
       0.10270671, 0.10244913, 0.10221606, 0.10200518, 0.10181436,
       0.1016417 ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-44f089fe-c35b-4a53-b4a4-12e24ff9581a' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-44f089fe-c35b-4a53-b4a4-12e24ff9581a' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




```python
plot_synthetic_data_maps(data_set_synt,synthetic_data_settings,coastline)
```


![png](bpca_tutorial_files/bpca_tutorial_7_0.png)


[back to top ](#top)

-----
## Setting up the data/model
<a id='setup'></a>

To set up the model import the bpca class and the standard model settings and have a look into the model settings. 


```python
from bpca.bpca import bpca as bpca
from bpca.model_settings import set_settings
import xarray as xr
```


```python
settings = set_settings()
settings
```




    {'model_settings': {'number_of_pcs': 3,
      'model_trend': True,
      'sigma_random_walk': 0.001,
      'estimate_point_variance': False,
      'estimate_cluster_sigma': False,
      'trend_pattern': None,
      'initialize_trend_pattern': False,
      'estimate_offsets': False,
      'trend_factor_sigma': 0.01,
      'trend_factor_nu': 2.1,
      'trend_distr': 'normal',
      'cluster_index': None,
      'sigma': 0.4,
      'sigma_offset': 0.1,
      'sigma_eofs': 0.15,
      'sigma_random_walk_factor': 0.04},
     'run_settings': {'n_samples': 4000,
      'compress': True,
      'sample_settings': {'tune': 2000,
       'cores': 4,
       'target_accept': 0.9,
       'return_inferencedata': True}},
     'normalization_settings': {}}



The dictionary contains information required for the priors ('model_settings'), the sampling ('run_settings'), as well as some information for data-normalization ('normalization_settings'). 

### Settings
Description of different seetings to choose from:
#### Model settings
* 'number_of_pcs': Number of maximum PCs to use
* 'model_trend': Estimate trend (or not)
* 'sigma_random_walk': standard deviation of gaussian random walk innovations (of PCs)
* 'estimate_point_variance': Estimate variance (i.e. white noise) for every station
* 'estimate_cluster_sigma': Estimate variance (i.e. white noise) for different cluster
* 'estimate_offsets': Estimate offsets for every station (or not)
* 'estimate_sigma_eof': Estimate hierarchical sigma for eof pattern
* 'trend_pattern': Initialize trends with np.array() of size space_dim
* 'initialize_trend_pattern': Initialize trend (or not)
* 'trend_factor_sigma': Prior stddev: trends
* 'trend_factor_nu': Prior shape factor for studen't distr.: trends
* 'trend_distr': Apply 'students_t' or 'normal' distribution for trends
* 'cluster_index': Define different clusters for which different variances are computed. Use np.array, with cluster indices (0,1,2, ...)
* 'sigma': Prior stddev: sigma, float or np.array
* 'sigma_offset': Prior stddev: offsets, float or np.array
* 'sigma_eofs': Prior stddev: EOFs, float or np.array
* 'sigma_random_walk_factor': factor by which sigma_random_walk is divided for subsequent PCs,
    sets some prior contraints, such thus the first EOFs capture most of the variance.
    e.g. if sigma_random_walk_factor is 2 and sigma_random_walk is 2,
    then sigma_random_walk(PC1) = 2; sigma_random_walk(PC2) = 1; sigma_random_walk(PC3) = 0.5 ...

#### Run settings

(https://docs.pymc.io/api/inference.html)

*  'n_samples': number of samples
*  'adjust_pca_symmetry' : match pcs from different chains (tbd before compress)
*  'check_convergence' : compute convergence statistics (e.g., rhat, ess) (tbd before compress)
*  'compress': compress chain (compute mean and std of all variables)

#### Sampling settings

*   'tune': 2000, number of tuning steps
*   'cores': 4, number of chains
*   'target_accept': 0.9, acceptance rate
*   'return_inferencedata': True

Control the number of cores ('cores') used or the number of sampling iterations ('n_samples'). You can also control settings of the NUTS sampler (https://docs.pymc.io/api/inference.html)

#### Model initialization


```python
name='test_bpca_model'
settings['model_settings']['number_of_pcs']=1
settings['model_settings']['trend_factor_sigma']=0.001
settings['model_settings']['sigma']=0.003

settings['model_settings']['trend_distr']='normal'
settings['model_settings']['sigma_random_walk']=0.001
settings['model_settings']['sigma']=0.4
settings['model_settings']['sigma_random_walk_factor']=0.04
settings['model_settings']['sigma_eofs']=1.15
```

[back to top ](#top)

-----
## Exp1: Running the model
<a id='running'></a>

Now run the model. Here we set only 2 chains and 500 iterations. As shown in the main text, different variance parameters are estimated for different techniques, because GNSS VLM data is much more precise than the ones derived from 'SAT minus TG'. We estimate different white noise parameter for the different cluster:


```python
settings['model_settings']['cluster_index'] = data_set_synt['ID'].values
settings['model_settings']['estimate_cluster_sigma']=True
settings['model_settings']['estimate_sigma_eof']=True
settings['model_settings']['estimate_point_variance']=True
settings['run_settings']['n_samples'] =1500
settings['run_settings']['compress'] =False
settings['run_settings']['sample_settings'] ={'tune': 2000,
                                                  'cores': 2,
                                                  'target_accept': 0.9,
                                                  'return_inferencedata': True}
settings['run_settings']['check_convergence'] =False
settings['run_settings']['adjust_pca_symmetry'] =False
```

Initialize the model


```python
bpca_object = bpca(data_set_synt['data_with_noise']/1000.,run_settings=settings['run_settings'],
                  model_settings = settings['model_settings'],name=name)
```

    estimate different sigma for different clusters


    /home/oelsmann/.conda/envs/vlad_py37/lib/python3.6/site-packages/pymc3/model.py:1668: ImputationWarning: Data in Observations contains missing values and will be automatically imputed from the sampling distribution.
      warnings.warn(impute_message, ImputationWarning)



```python
bpca_object.run()
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [Observations_missing, trend_g, W0, PC0, sigma_hier, sigma_eof]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='5000' class='' max='5000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [5000/5000 02:31<00:00 Sampling 2 chains, 0 divergences]
</div>



    Sampling 2 chains for 2_000 tune and 500 draw iterations (4_000 + 1_000 draws total) took 151 seconds.
    The rhat statistic is larger than 1.05 for some parameters. This indicates slight problems during sampling.
    The estimated number of effective samples is smaller than 200 for some parameters.


[back to top ](#top)

-----
## Model output and diagnostics
<a id='output'></a>


The output can be accessed here:


```python
bpca_object.trace
```




    Inference data with groups:
    	> posterior
    	> log_likelihood
    	> sample_stats
    	> observed_data




```python
bpca_object.trace.posterior
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt, dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:                     (Estimates_dim_0: 26, Estimates_dim_1: 530, Observations_missing_dim_0: 934, PC0_dim_0: 26, W0_dim_0: 530, chain: 2, draw: 1500, sigma_dim_0: 530, sigma_eof_dim_0: 1, sigma_hier_dim_0: 2, trend_g_dim_0: 530)
Coordinates:
  * chain                       (chain) int64 0 1
  * draw                        (draw) int64 0 1 2 3 4 ... 1496 1497 1498 1499
  * PC0_dim_0                   (PC0_dim_0) int64 0 1 2 3 4 5 ... 21 22 23 24 25
  * W0_dim_0                    (W0_dim_0) int64 0 1 2 3 4 ... 526 527 528 529
  * trend_g_dim_0               (trend_g_dim_0) int64 0 1 2 3 ... 527 528 529
  * Observations_missing_dim_0  (Observations_missing_dim_0) int64 0 1 ... 933
  * sigma_eof_dim_0             (sigma_eof_dim_0) int64 0
  * sigma_hier_dim_0            (sigma_hier_dim_0) int64 0 1
  * sigma_dim_0                 (sigma_dim_0) int64 0 1 2 3 ... 526 527 528 529
  * Estimates_dim_0             (Estimates_dim_0) int64 0 1 2 3 ... 22 23 24 25
  * Estimates_dim_1             (Estimates_dim_1) int64 0 1 2 3 ... 527 528 529
Data variables:
    PC0                         (chain, draw, PC0_dim_0) float64 -0.02674 ......
    W0                          (chain, draw, W0_dim_0) float64 -1.264 ... 0.14
    trend_g                     (chain, draw, trend_g_dim_0) float64 0.000873...
    Observations_missing        (chain, draw, Observations_missing_dim_0) float64 ...
    sigma_eof                   (chain, draw, sigma_eof_dim_0) float64 0.9998...
    sigma_hier                  (chain, draw, sigma_hier_dim_0) float64 0.004...
    sigma                       (chain, draw, sigma_dim_0) float64 0.00495 .....
    Estimates                   (chain, draw, Estimates_dim_0, Estimates_dim_1) float64 ...
Attributes:
    created_at:                 2023-09-11T13:31:37.588499
    arviz_version:              0.8.3
    inference_library:          pymc3
    inference_library_version:  3.9.3
    sampling_time:              203.96838188171387
    tuning_steps:               2000</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-48ece187-968d-4441-99be-eb0f4e2c4e21' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-48ece187-968d-4441-99be-eb0f4e2c4e21' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>Estimates_dim_0</span>: 26</li><li><span class='xr-has-index'>Estimates_dim_1</span>: 530</li><li><span class='xr-has-index'>Observations_missing_dim_0</span>: 934</li><li><span class='xr-has-index'>PC0_dim_0</span>: 26</li><li><span class='xr-has-index'>W0_dim_0</span>: 530</li><li><span class='xr-has-index'>chain</span>: 2</li><li><span class='xr-has-index'>draw</span>: 1500</li><li><span class='xr-has-index'>sigma_dim_0</span>: 530</li><li><span class='xr-has-index'>sigma_eof_dim_0</span>: 1</li><li><span class='xr-has-index'>sigma_hier_dim_0</span>: 2</li><li><span class='xr-has-index'>trend_g_dim_0</span>: 530</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-6d7cae57-8701-47f1-8498-19db709cd997' class='xr-section-summary-in' type='checkbox'  checked><label for='section-6d7cae57-8701-47f1-8498-19db709cd997' class='xr-section-summary' >Coordinates: <span>(11)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1</div><input id='attrs-89827614-0f48-456d-8a05-b6474e965725' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-89827614-0f48-456d-8a05-b6474e965725' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2924f50a-1df7-4028-b24b-8c34352bf960' class='xr-var-data-in' type='checkbox'><label for='data-2924f50a-1df7-4028-b24b-8c34352bf960' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1496 1497 1498 1499</div><input id='attrs-597f8e7e-3cf5-410c-9c0e-7695ee1b88cc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-597f8e7e-3cf5-410c-9c0e-7695ee1b88cc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e3f9dbc9-3593-4bc3-a0e8-cef25894a253' class='xr-var-data-in' type='checkbox'><label for='data-e3f9dbc9-3593-4bc3-a0e8-cef25894a253' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1497, 1498, 1499])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>PC0_dim_0</span></div><div class='xr-var-dims'>(PC0_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 ... 20 21 22 23 24 25</div><input id='attrs-299c2743-98a7-4817-9eac-37d056593ca2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-299c2743-98a7-4817-9eac-37d056593ca2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bbe79d74-923f-434a-a585-fe62b4858524' class='xr-var-data-in' type='checkbox'><label for='data-bbe79d74-923f-434a-a585-fe62b4858524' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>W0_dim_0</span></div><div class='xr-var-dims'>(W0_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 525 526 527 528 529</div><input id='attrs-e6ec776b-ca8a-4feb-9fbe-fd4754ce3a2a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e6ec776b-ca8a-4feb-9fbe-fd4754ce3a2a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-47cb5471-3857-4295-ae4d-b749603f1d4f' class='xr-var-data-in' type='checkbox'><label for='data-47cb5471-3857-4295-ae4d-b749603f1d4f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 527, 528, 529])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>trend_g_dim_0</span></div><div class='xr-var-dims'>(trend_g_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 525 526 527 528 529</div><input id='attrs-fa4253f8-22b1-4adc-9305-c47c24b28bb5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fa4253f8-22b1-4adc-9305-c47c24b28bb5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-94f86d59-2901-42da-8a12-bf2d00d96e9a' class='xr-var-data-in' type='checkbox'><label for='data-94f86d59-2901-42da-8a12-bf2d00d96e9a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 527, 528, 529])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>Observations_missing_dim_0</span></div><div class='xr-var-dims'>(Observations_missing_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 929 930 931 932 933</div><input id='attrs-c95985c2-8dc7-44a5-8742-3735ab299016' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c95985c2-8dc7-44a5-8742-3735ab299016' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-921ea082-1729-43a3-ba1e-864fee45e9eb' class='xr-var-data-in' type='checkbox'><label for='data-921ea082-1729-43a3-ba1e-864fee45e9eb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 931, 932, 933])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sigma_eof_dim_0</span></div><div class='xr-var-dims'>(sigma_eof_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-4b5bd0ce-f12d-41f9-96ce-fee03700636c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4b5bd0ce-f12d-41f9-96ce-fee03700636c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-520b8f4e-b5ab-4e85-8f8f-ac97c1a29753' class='xr-var-data-in' type='checkbox'><label for='data-520b8f4e-b5ab-4e85-8f8f-ac97c1a29753' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sigma_hier_dim_0</span></div><div class='xr-var-dims'>(sigma_hier_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1</div><input id='attrs-fdd081db-640e-4a57-b322-40c7a6bc6a0f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fdd081db-640e-4a57-b322-40c7a6bc6a0f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-948f2c3d-b911-4238-bc35-d88fc557b677' class='xr-var-data-in' type='checkbox'><label for='data-948f2c3d-b911-4238-bc35-d88fc557b677' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sigma_dim_0</span></div><div class='xr-var-dims'>(sigma_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 525 526 527 528 529</div><input id='attrs-2fa95030-0946-41f8-bf5d-4871525d0f2e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2fa95030-0946-41f8-bf5d-4871525d0f2e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9cb18702-ba02-4c59-9509-cc2ed6e83053' class='xr-var-data-in' type='checkbox'><label for='data-9cb18702-ba02-4c59-9509-cc2ed6e83053' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 527, 528, 529])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>Estimates_dim_0</span></div><div class='xr-var-dims'>(Estimates_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 ... 20 21 22 23 24 25</div><input id='attrs-6697e012-72d4-45a9-9a04-64d3099dbacf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6697e012-72d4-45a9-9a04-64d3099dbacf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5ae28886-9c7e-4b2a-af19-53fd31d3f7f3' class='xr-var-data-in' type='checkbox'><label for='data-5ae28886-9c7e-4b2a-af19-53fd31d3f7f3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>Estimates_dim_1</span></div><div class='xr-var-dims'>(Estimates_dim_1)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 525 526 527 528 529</div><input id='attrs-b2995738-551e-4b03-9b95-0c4d48e6c0dd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b2995738-551e-4b03-9b95-0c4d48e6c0dd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4514e6c3-8755-46a4-ad15-c52b6bf92ed7' class='xr-var-data-in' type='checkbox'><label for='data-4514e6c3-8755-46a4-ad15-c52b6bf92ed7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 527, 528, 529])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-760bda5d-bcdb-4fd6-88ae-baa45e3c048f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-760bda5d-bcdb-4fd6-88ae-baa45e3c048f' class='xr-section-summary' >Data variables: <span>(8)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>PC0</span></div><div class='xr-var-dims'>(chain, draw, PC0_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.02674 -0.02592 ... 0.008556</div><input id='attrs-a08f72dd-2a43-44fd-b415-bc4445dcf310' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a08f72dd-2a43-44fd-b415-bc4445dcf310' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-938ec0d2-b8f9-4796-b178-dc83d17c2d67' class='xr-var-data-in' type='checkbox'><label for='data-938ec0d2-b8f9-4796-b178-dc83d17c2d67' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-0.02674423, -0.02591511, -0.02672051, ...,  0.00762448,
          0.00944682,  0.01048541],
        [-0.02672521, -0.02647441, -0.02650396, ...,  0.00784468,
          0.00905247,  0.00996677],
        [-0.02730713, -0.02728981, -0.02763627, ...,  0.00789347,
          0.00931016,  0.01021963],
        ...,
        [-0.03190407, -0.03170055, -0.03230527, ...,  0.00918379,
          0.0108271 ,  0.01175244],
        [-0.0326619 , -0.0324817 , -0.0326568 , ...,  0.00933149,
          0.0114557 ,  0.01282426],
        [-0.03299979, -0.03226048, -0.03310242, ...,  0.00921068,
          0.01142205,  0.012637  ]],

       [[-0.03247955, -0.03239562, -0.03251576, ...,  0.00984484,
          0.01129494,  0.0127952 ],
        [-0.03226753, -0.03171121, -0.03213043, ...,  0.00879323,
          0.01134809,  0.01235291],
        [-0.03208159, -0.03185297, -0.03240444, ...,  0.00961992,
          0.01104943,  0.0127394 ],
        ...,
        [-0.02216232, -0.02194368, -0.02236579, ...,  0.00632998,
          0.00769231,  0.0084293 ],
        [-0.02200454, -0.02178133, -0.02240436, ...,  0.00616867,
          0.00780683,  0.00827944],
        [-0.02182458, -0.02205034, -0.02221257, ...,  0.00611793,
          0.0078641 ,  0.00855621]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>W0</span></div><div class='xr-var-dims'>(chain, draw, W0_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.264 -0.313 -2.787 ... 1.59 0.14</div><input id='attrs-6760299b-43c8-4338-b152-f17b1bf68c67' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6760299b-43c8-4338-b152-f17b1bf68c67' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-083dcdeb-cdaf-4a2a-ab9a-54882f42f6ac' class='xr-var-data-in' type='checkbox'><label for='data-083dcdeb-cdaf-4a2a-ab9a-54882f42f6ac' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-1.26351553, -0.31303704, -2.78686154, ...,  1.33235328,
          1.28205652,  0.16731504],
        [-1.02821341, -0.32438367, -2.94224043, ...,  0.78996322,
          1.25297015,  0.07196427],
        [-1.07694767, -0.3732432 , -2.64327582, ...,  1.18011456,
          1.46334976,  0.48314411],
        ...,
        [-1.13711158, -0.35349797, -2.32846525, ...,  0.81888676,
          1.40507102,  0.06813752],
        [-0.9320351 , -0.18904337, -2.25849703, ...,  1.14642113,
          0.67917302,  0.31061015],
        [-0.89148894, -0.23944457, -2.44585511, ...,  1.08543593,
          0.78290575,  0.20161691]],

       [[-1.06517889, -0.21410144, -2.32769019, ...,  0.86492675,
          1.24123217,  0.68738738],
        [-1.14104628, -0.19865618, -2.3645876 , ...,  1.1289626 ,
          1.3197226 ,  0.28781119],
        [-1.09947011, -0.15636152, -2.33856994, ...,  0.88757977,
          1.0345876 ,  0.14057748],
        ...,
        [-1.40074809, -0.32174343, -3.23407909, ...,  1.25810742,
          0.84134704,  0.176341  ],
        [-1.3154524 , -0.57474559, -3.54193007, ...,  1.37518133,
          0.86508934,  0.02156708],
        [-1.32444829, -0.80142524, -3.50940383, ...,  1.29920712,
          1.5901911 ,  0.14000946]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>trend_g</span></div><div class='xr-var-dims'>(chain, draw, trend_g_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0008739 -0.0004804 ... -5.466e-06</div><input id='attrs-0eee37f2-0525-4204-be0a-8a09967a7e73' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0eee37f2-0525-4204-be0a-8a09967a7e73' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-96b53241-4241-4b0e-911a-58a29e5dd708' class='xr-var-data-in' type='checkbox'><label for='data-96b53241-4241-4b0e-911a-58a29e5dd708' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 8.73946400e-04, -4.80381944e-04, -5.76131060e-04, ...,
         -1.69493673e-03,  1.04685346e-03, -1.30007752e-03],
        [ 3.03505436e-04, -3.84587870e-04, -3.81305667e-04, ...,
         -5.51864751e-04,  1.35528151e-03, -1.15575325e-03],
        [ 5.22119916e-04, -2.96951113e-04, -6.71396688e-04, ...,
         -9.59113304e-04,  6.76647692e-04, -1.20130053e-03],
        ...,
        [ 9.07891301e-04, -1.67201317e-04, -5.05551682e-04, ...,
         -3.50695836e-04,  2.75382356e-04, -4.79223374e-04],
        [ 4.64320468e-04, -5.38979367e-04, -6.22286741e-04, ...,
         -1.19135072e-03,  2.01430435e-03, -1.26956082e-03],
        [ 6.53351416e-04, -5.05253010e-04, -2.55401338e-04, ...,
         -1.13492898e-03,  1.73804687e-03, -8.67615943e-04]],

       [[ 7.11504826e-04, -5.19597170e-04, -5.57121424e-04, ...,
         -5.76593309e-04,  8.06681184e-04, -1.86007190e-03],
        [ 8.85339120e-04, -5.89420031e-04, -5.65232580e-04, ...,
         -1.29551176e-03,  5.28705077e-04, -1.12230169e-03],
        [ 9.51348623e-04, -5.67517168e-04, -5.81773153e-04, ...,
         -6.86329129e-04,  1.43262701e-03, -5.57452238e-04],
        ...,
        [ 5.91484288e-04, -5.02698417e-04, -7.97472818e-04, ...,
         -2.23879302e-04,  2.10152107e-03, -5.51980719e-04],
        [ 3.52353208e-04, -5.57276144e-05, -3.82070882e-04, ...,
         -3.49612769e-04,  2.02784133e-03, -3.33282488e-04],
        [ 4.62333887e-04,  4.08930795e-04, -4.70005209e-04, ...,
         -3.62574607e-04,  1.42737151e-03, -5.46634280e-06]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Observations_missing</span></div><div class='xr-var-dims'>(chain, draw, Observations_missing_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.009511 0.04829 ... 0.003904</div><input id='attrs-112b79aa-d1b5-4671-863c-2f5c663dc72e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-112b79aa-d1b5-4671-863c-2f5c663dc72e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4c332395-59e4-4539-8032-999626c3f114' class='xr-var-data-in' type='checkbox'><label for='data-4c332395-59e4-4539-8032-999626c3f114' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-0.00951081,  0.04828794,  0.03036967, ...,  0.02300513,
          0.00833734,  0.01242169],
        [-0.01554617,  0.03755364,  0.0229121 , ...,  0.0320556 ,
          0.00375788,  0.00324694],
        [-0.0140127 ,  0.04197579,  0.01770211, ...,  0.02847786,
          0.01059209,  0.02693105],
        ...,
        [-0.01366105,  0.03656668,  0.01647841, ...,  0.01968191,
          0.00426877,  0.02011401],
        [-0.01666328,  0.04065036,  0.01380438, ...,  0.0350773 ,
          0.01269423,  0.0146602 ],
        [-0.01490565,  0.03898793,  0.01663605, ...,  0.03095715,
          0.01199888,  0.00926896]],

       [[-0.0139898 ,  0.04150058,  0.01063147, ...,  0.02269037,
          0.00966303,  0.01387948],
        [-0.02855615,  0.04602389,  0.02359412, ...,  0.03542703,
          0.01192539,  0.05095396],
        [-0.02298941,  0.04070087,  0.01919425, ...,  0.02211439,
          0.00605031,  0.04363543],
        ...,
        [-0.0136649 ,  0.0386167 ,  0.0228399 , ...,  0.03067863,
          0.01579005,  0.00751601],
        [-0.0091984 ,  0.03371852,  0.01726148, ...,  0.0202399 ,
          0.00678894,  0.04101906],
        [-0.00664495,  0.03894545,  0.01615318, ...,  0.02445694,
          0.01711462,  0.00390408]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_eof</span></div><div class='xr-var-dims'>(chain, draw, sigma_eof_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.9998 1.01 0.9825 ... 1.263 1.281</div><input id='attrs-97f8df13-3bd7-459d-b77e-5a37f7398603' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-97f8df13-3bd7-459d-b77e-5a37f7398603' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-01fd687a-8f51-4af4-b9c6-fbd084f34ac9' class='xr-var-data-in' type='checkbox'><label for='data-01fd687a-8f51-4af4-b9c6-fbd084f34ac9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[0.99983095],
        [1.00983559],
        [0.98248072],
        ...,
        [0.87188005],
        [0.84800167],
        [0.85549626]],

       [[0.89770176],
        [0.86742841],
        [0.89857151],
        ...,
        [1.28013194],
        [1.26278401],
        [1.28108311]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_hier</span></div><div class='xr-var-dims'>(chain, draw, sigma_hier_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.00495 0.01489 ... 0.01428</div><input id='attrs-f4661a86-8226-48f5-a0ce-14dedc0ebe88' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f4661a86-8226-48f5-a0ce-14dedc0ebe88' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0819e161-3cdc-4e88-b1e6-89d1bb190964' class='xr-var-data-in' type='checkbox'><label for='data-0819e161-3cdc-4e88-b1e6-89d1bb190964' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[0.00494977, 0.01488726],
        [0.00489711, 0.01534856],
        [0.00508328, 0.01449655],
        ...,
        [0.00494076, 0.01550083],
        [0.00497638, 0.014353  ],
        [0.00496866, 0.01423519]],

       [[0.00499702, 0.01481715],
        [0.00505099, 0.01417006],
        [0.0050178 , 0.01504514],
        ...,
        [0.00494677, 0.01399244],
        [0.00496555, 0.01447377],
        [0.0050336 , 0.01427776]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma</span></div><div class='xr-var-dims'>(chain, draw, sigma_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.00495 0.00495 ... 0.01428 0.01428</div><input id='attrs-a108d2b0-8c37-421a-a7ce-c45ed9558e3f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a108d2b0-8c37-421a-a7ce-c45ed9558e3f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-100c2cd3-2dc1-4be4-a845-fcc654cca044' class='xr-var-data-in' type='checkbox'><label for='data-100c2cd3-2dc1-4be4-a845-fcc654cca044' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[0.00494977, 0.00494977, 0.00494977, ..., 0.01488726,
         0.01488726, 0.01488726],
        [0.00489711, 0.00489711, 0.00489711, ..., 0.01534856,
         0.01534856, 0.01534856],
        [0.00508328, 0.00508328, 0.00508328, ..., 0.01449655,
         0.01449655, 0.01449655],
        ...,
        [0.00494076, 0.00494076, 0.00494076, ..., 0.01550083,
         0.01550083, 0.01550083],
        [0.00497638, 0.00497638, 0.00497638, ..., 0.014353  ,
         0.014353  , 0.014353  ],
        [0.00496866, 0.00496866, 0.00496866, ..., 0.01423519,
         0.01423519, 0.01423519]],

       [[0.00499702, 0.00499702, 0.00499702, ..., 0.01481715,
         0.01481715, 0.01481715],
        [0.00505099, 0.00505099, 0.00505099, ..., 0.01417006,
         0.01417006, 0.01417006],
        [0.0050178 , 0.0050178 , 0.0050178 , ..., 0.01504514,
         0.01504514, 0.01504514],
        ...,
        [0.00494677, 0.00494677, 0.00494677, ..., 0.01399244,
         0.01399244, 0.01399244],
        [0.00496555, 0.00496555, 0.00496555, ..., 0.01447377,
         0.01447377, 0.01447377],
        [0.0050336 , 0.0050336 , 0.0050336 , ..., 0.01427776,
         0.01427776, 0.01427776]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Estimates</span></div><div class='xr-var-dims'>(chain, draw, Estimates_dim_0, Estimates_dim_1)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.01719 0.0175 ... 0.02217 0.001165</div><input id='attrs-3a18b63b-ef04-41b7-a4d4-55a9213f8918' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3a18b63b-ef04-41b7-a4d4-55a9213f8918' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6972594a-059a-4a23-93fe-a228155b7d4c' class='xr-var-data-in' type='checkbox'><label for='data-6972594a-059a-4a23-93fe-a228155b7d4c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[[ 0.01718677,  0.01749919,  0.08547897, ..., -0.00342897,
          -0.05417784,  0.02022676],
         [ 0.0170131 ,  0.01675926,  0.08259217, ..., -0.00401921,
          -0.05206799,  0.01906541],
         [ 0.01890469,  0.016531  ,  0.08426058, ..., -0.00678723,
          -0.05205371,  0.01763058],
         ...,
         [-0.00613786, -0.00430827, -0.0235529 , ...,  0.00337876,
           0.01396243, -0.00392462],
         [-0.00756647, -0.00535911, -0.02920762, ...,  0.00411181,
           0.01734562, -0.00491979],
         [-0.0080048 , -0.00616461, -0.03267817, ...,  0.00380065,
           0.01972401, -0.0060461 ]],

        [[ 0.02171261,  0.01597639,  0.08587679, ..., -0.0106265 ,
          -0.05923623,  0.02003605],
         [ 0.02175825,  0.01551045,  0.08475759, ..., -0.01098025,
          -0.05756672,  0.01889835],
         [ 0.02209214,  0.01513545,  0.08446323, ..., -0.01155546,
          -0.05624846,  0.01774047],
...
         [-0.00670518, -0.00376833, -0.02337728, ...,  0.00708459,
           0.01344781, -0.00120009],
         [-0.00850775, -0.00476558, -0.02956161, ...,  0.00898775,
           0.01689282, -0.00149804],
         [-0.0087771 , -0.00509294, -0.03161764, ...,  0.00928806,
           0.01932951, -0.00182113]],

        [[ 0.02012118,  0.00972108,  0.08552136, ..., -0.02146573,
          -0.06182531, -0.00295179],
         [ 0.02088253,  0.01031095,  0.08584365, ..., -0.02212162,
          -0.06075695, -0.00298886],
         [ 0.02155972,  0.01084989,  0.08594296, ..., -0.02269496,
          -0.05958754, -0.00301704],
         ...,
         [-0.00625354, -0.00326734, -0.02335029, ...,  0.00649815,
           0.01543816,  0.0008347 ],
         [-0.00810393, -0.00425784, -0.02994834, ...,  0.00840423,
           0.01964228,  0.00107372],
         [-0.00855826, -0.00440358, -0.03284723, ...,  0.00894084,
           0.02217024,  0.00116515]]]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-64fb0405-3474-44e0-b795-e68f150fde2b' class='xr-section-summary-in' type='checkbox'  checked><label for='section-64fb0405-3474-44e0-b795-e68f150fde2b' class='xr-section-summary' >Attributes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2023-09-11T13:31:37.588499</dd><dt><span>arviz_version :</span></dt><dd>0.8.3</dd><dt><span>inference_library :</span></dt><dd>pymc3</dd><dt><span>inference_library_version :</span></dt><dd>3.9.3</dd><dt><span>sampling_time :</span></dt><dd>203.96838188171387</dd><dt><span>tuning_steps :</span></dt><dd>2000</dd></dl></div></li></ul></div></div>



Since we run different independent chains, and the principal components might be subject to label switching and symmetry, we manually match the PCs of the individual chains with the following function (see also discussion here: https://discourse.pymc.io/t/unique-solution-for-probabilistic-pca/1324/2):


```python
bpca_object.adjust_pca_symmetry()
# Note sigma_eof is not sorted here
```

    replaced variable PC0 (chain 0 ) with variable PC0 (chain 0)
    replaced variable PC0 (chain 1 ) with variable PC0 (chain 1)
    replaced variable W0 (chain 0 ) with variable W0 (chain 0)
    replaced variable W0 (chain 1 ) with variable W0 (chain 1)
    replaced variable sigma_eof0 (chain 0 ) with variable sigma_eof0 (chain 0)
    replaced variable sigma_eof0 (chain 1 ) with variable sigma_eof0 (chain 1)


Now check the convergence statistics (representing the mean statistics)


```python
bpca_object.check_convergence()
bpca_object.convergence_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC0</th>
      <th>W0</th>
      <th>trend_g</th>
      <th>sigma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>r_hat</th>
      <td>1.040000</td>
      <td>1.005962</td>
      <td>1.000000</td>
      <td>1.000094</td>
    </tr>
    <tr>
      <th>ess_mean</th>
      <td>0.066564</td>
      <td>0.709587</td>
      <td>1.826898</td>
      <td>2.041592</td>
    </tr>
  </tbody>
</table>
</div>



Check some of the markov chains (here PC0 is shown for every time step):


```python
bpca_object.trace.posterior['PC0'][1,:,:].plot()
```




    <matplotlib.collections.QuadMesh at 0x7f36f6bf6080>




![png](bpca_tutorial_files/bpca_tutorial_33_1.png)


Finally we can compress the trace


```python
bpca_object.compress()
```

    successfully compressed trace


... which returns a dictionary with mean and std estimates of the trace (removing the draw dimension)


```python
bpca_object.trace
```




    {'mean': Inference data with groups:
     	> posterior
     	> log_likelihood
     	> sample_stats,
     'std': Inference data with groups:
     	> posterior
     	> log_likelihood
     	> sample_stats}



After compressing the markov chains, for every the mean and standard-deviations of the parameters are provided.


```python
bpca_object.trace['mean'].posterior
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt, dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:                     (Estimates_dim_0: 26, Estimates_dim_1: 530, Observations_missing_dim_0: 1574, PC0_dim_0: 26, W0_dim_0: 530, chain: 2, sigma_dim_0: 530, sigma_eof_dim_0: 1, sigma_hier_dim_0: 2, trend_g_dim_0: 530)
Coordinates:
  * chain                       (chain) int64 0 1
  * PC0_dim_0                   (PC0_dim_0) int64 0 1 2 3 4 5 ... 21 22 23 24 25
  * W0_dim_0                    (W0_dim_0) int64 0 1 2 3 4 ... 526 527 528 529
  * trend_g_dim_0               (trend_g_dim_0) int64 0 1 2 3 ... 527 528 529
  * Observations_missing_dim_0  (Observations_missing_dim_0) int64 0 1 ... 1573
  * sigma_eof_dim_0             (sigma_eof_dim_0) int64 0
  * sigma_hier_dim_0            (sigma_hier_dim_0) int64 0 1
  * sigma_dim_0                 (sigma_dim_0) int64 0 1 2 3 ... 526 527 528 529
  * Estimates_dim_0             (Estimates_dim_0) int64 0 1 2 3 ... 22 23 24 25
  * Estimates_dim_1             (Estimates_dim_1) int64 0 1 2 3 ... 527 528 529
Data variables:
    PC0                         (chain, PC0_dim_0) float64 0.01404 ... -0.00941
    W0                          (chain, W0_dim_0) float64 1.387 ... -0.5798
    trend_g                     (chain, trend_g_dim_0) float64 0.0005844 ... ...
    Observations_missing        (chain, Observations_missing_dim_0) float64 0...
    sigma_eof                   (chain, sigma_eof_dim_0) float64 1.336 1.579
    sigma_hier                  (chain, sigma_hier_dim_0) float64 0.004982 .....
    sigma                       (chain, sigma_dim_0) float64 0.004982 ... 0.0...
    Estimates                   (chain, Estimates_dim_0, Estimates_dim_1) float64 ...</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-d74e05a1-731d-49ea-b1b9-443b3b595e49' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-d74e05a1-731d-49ea-b1b9-443b3b595e49' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>Estimates_dim_0</span>: 26</li><li><span class='xr-has-index'>Estimates_dim_1</span>: 530</li><li><span class='xr-has-index'>Observations_missing_dim_0</span>: 1574</li><li><span class='xr-has-index'>PC0_dim_0</span>: 26</li><li><span class='xr-has-index'>W0_dim_0</span>: 530</li><li><span class='xr-has-index'>chain</span>: 2</li><li><span class='xr-has-index'>sigma_dim_0</span>: 530</li><li><span class='xr-has-index'>sigma_eof_dim_0</span>: 1</li><li><span class='xr-has-index'>sigma_hier_dim_0</span>: 2</li><li><span class='xr-has-index'>trend_g_dim_0</span>: 530</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-f489535b-293c-48fe-bb94-8277f44b8d43' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f489535b-293c-48fe-bb94-8277f44b8d43' class='xr-section-summary' >Coordinates: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1</div><input id='attrs-8d3c78ef-a1e1-4eb8-a9fc-27464591d8ec' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8d3c78ef-a1e1-4eb8-a9fc-27464591d8ec' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6244f296-1755-44a7-9ac3-761a8746b849' class='xr-var-data-in' type='checkbox'><label for='data-6244f296-1755-44a7-9ac3-761a8746b849' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>PC0_dim_0</span></div><div class='xr-var-dims'>(PC0_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 ... 20 21 22 23 24 25</div><input id='attrs-880c7624-e119-4ae2-923f-c92fc1d003ca' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-880c7624-e119-4ae2-923f-c92fc1d003ca' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-429469f4-1e19-4f08-8962-4ba0ac055249' class='xr-var-data-in' type='checkbox'><label for='data-429469f4-1e19-4f08-8962-4ba0ac055249' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>W0_dim_0</span></div><div class='xr-var-dims'>(W0_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 525 526 527 528 529</div><input id='attrs-03c1d6c1-997e-47ec-bf2d-10281c1d1a1a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-03c1d6c1-997e-47ec-bf2d-10281c1d1a1a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d2ef1747-843f-477e-a07e-99e26c542222' class='xr-var-data-in' type='checkbox'><label for='data-d2ef1747-843f-477e-a07e-99e26c542222' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 527, 528, 529])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>trend_g_dim_0</span></div><div class='xr-var-dims'>(trend_g_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 525 526 527 528 529</div><input id='attrs-57234c40-e7e4-43a0-ac14-ec2d204d302a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-57234c40-e7e4-43a0-ac14-ec2d204d302a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d3fd3aa2-114f-4923-850f-4348eca2b227' class='xr-var-data-in' type='checkbox'><label for='data-d3fd3aa2-114f-4923-850f-4348eca2b227' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 527, 528, 529])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>Observations_missing_dim_0</span></div><div class='xr-var-dims'>(Observations_missing_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1570 1571 1572 1573</div><input id='attrs-3c786cc2-1378-4b0e-8e3f-83b30d98b5dd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3c786cc2-1378-4b0e-8e3f-83b30d98b5dd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4ebff0a9-d6b0-4e9b-84a9-5eb58fa04c46' class='xr-var-data-in' type='checkbox'><label for='data-4ebff0a9-d6b0-4e9b-84a9-5eb58fa04c46' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1571, 1572, 1573])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sigma_eof_dim_0</span></div><div class='xr-var-dims'>(sigma_eof_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-34307cba-45eb-4f06-9f8f-0296f73da71a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-34307cba-45eb-4f06-9f8f-0296f73da71a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3354363b-07a8-4506-8470-668bb69a7075' class='xr-var-data-in' type='checkbox'><label for='data-3354363b-07a8-4506-8470-668bb69a7075' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sigma_hier_dim_0</span></div><div class='xr-var-dims'>(sigma_hier_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1</div><input id='attrs-14ed3a5a-ab12-429c-9381-966027d3c63e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-14ed3a5a-ab12-429c-9381-966027d3c63e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0444622a-3f93-46c6-8513-6f7cf0496ea4' class='xr-var-data-in' type='checkbox'><label for='data-0444622a-3f93-46c6-8513-6f7cf0496ea4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sigma_dim_0</span></div><div class='xr-var-dims'>(sigma_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 525 526 527 528 529</div><input id='attrs-17d4be08-44bd-4aa0-a27a-fb1932652440' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-17d4be08-44bd-4aa0-a27a-fb1932652440' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-88229a7d-9e14-4988-83f0-96b79dd0deb2' class='xr-var-data-in' type='checkbox'><label for='data-88229a7d-9e14-4988-83f0-96b79dd0deb2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 527, 528, 529])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>Estimates_dim_0</span></div><div class='xr-var-dims'>(Estimates_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 ... 20 21 22 23 24 25</div><input id='attrs-c7ab7487-206d-4417-b99d-60f6a55c004e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c7ab7487-206d-4417-b99d-60f6a55c004e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-68596ab4-dd55-4de9-b91a-d02170363f84' class='xr-var-data-in' type='checkbox'><label for='data-68596ab4-dd55-4de9-b91a-d02170363f84' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>Estimates_dim_1</span></div><div class='xr-var-dims'>(Estimates_dim_1)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 525 526 527 528 529</div><input id='attrs-3b1a79a7-32ee-497b-9331-88918a0d4b09' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3b1a79a7-32ee-497b-9331-88918a0d4b09' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7ab3aaa5-63d5-4b5a-8889-43033c22e2ee' class='xr-var-data-in' type='checkbox'><label for='data-7ab3aaa5-63d5-4b5a-8889-43033c22e2ee' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 527, 528, 529])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ae354c83-4245-4700-afc7-df98b446e596' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ae354c83-4245-4700-afc7-df98b446e596' class='xr-section-summary' >Data variables: <span>(8)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>PC0</span></div><div class='xr-var-dims'>(chain, PC0_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.01404 0.01363 ... -0.00941</div><input id='attrs-4856a79b-5cb8-4884-9123-f0f87ac2e1a7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4856a79b-5cb8-4884-9123-f0f87ac2e1a7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-40dc92b3-1a2c-4f84-b539-5c93aa5479db' class='xr-var-data-in' type='checkbox'><label for='data-40dc92b3-1a2c-4f84-b539-5c93aa5479db' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1.40447401e-02,  1.36325772e-02,  1.39120810e-02,
         1.40296587e-02,  1.44083221e-02,  1.45549972e-02,
         1.42998361e-02,  1.45896903e-02,  1.47965421e-02,
         1.44528409e-02,  1.50045573e-02,  1.48537693e-02,
         1.47093533e-02,  1.52431040e-02,  1.49104803e-02,
         1.07470149e-02,  8.19920989e-03,  4.92103149e-03,
         2.36892933e-03,  1.08881875e-04, -2.27764269e-03,
        -4.20876339e-03, -5.86084263e-03, -7.10760952e-03,
        -8.68345658e-03, -9.63654981e-03],
       [ 1.37152434e-02,  1.33137978e-02,  1.35876119e-02,
         1.37245200e-02,  1.40748593e-02,  1.42428798e-02,
         1.39121306e-02,  1.42021636e-02,  1.43954893e-02,
         1.40675175e-02,  1.46256817e-02,  1.44446651e-02,
         1.43104388e-02,  1.48382086e-02,  1.45603579e-02,
         1.04729194e-02,  7.95849752e-03,  4.76773573e-03,
         2.28024040e-03,  8.64467486e-05, -2.29366976e-03,
        -4.10979764e-03, -5.70478320e-03, -6.95979098e-03,
        -8.48766635e-03, -9.40959601e-03]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>W0</span></div><div class='xr-var-dims'>(chain, W0_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.387 0.07302 ... -0.6685 -0.5798</div><input id='attrs-8d701c9d-b46d-42f9-b603-7c7c74577310' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8d701c9d-b46d-42f9-b603-7c7c74577310' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-14fe3931-2500-41b7-b303-22859f645ee1' class='xr-var-data-in' type='checkbox'><label for='data-14fe3931-2500-41b7-b303-22859f645ee1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1.38719103,  0.07301608,  2.23268867, ..., -0.98527741,
        -0.57252443, -0.66198787],
       [ 1.35341226,  0.09446948,  2.21098776, ..., -1.00480816,
        -0.66853238, -0.57975537]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>trend_g</span></div><div class='xr-var-dims'>(chain, trend_g_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0005844 -0.00039 ... -0.000417</div><input id='attrs-c1780040-e9f9-4edb-8f37-3caf13c03d51' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c1780040-e9f9-4edb-8f37-3caf13c03d51' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2335f918-adfd-4d0f-a3fe-367aa8b2ff23' class='xr-var-data-in' type='checkbox'><label for='data-2335f918-adfd-4d0f-a3fe-367aa8b2ff23' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 5.84386687e-04, -3.89968371e-04, -4.58365657e-04, ...,
        -1.02032297e-05,  1.06844563e-03, -4.72081412e-04],
       [ 5.57975983e-04, -3.87338510e-04, -4.61285274e-04, ...,
        -1.30500305e-05,  9.48567786e-04, -4.16963614e-04]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Observations_missing</span></div><div class='xr-var-dims'>(chain, Observations_missing_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0009188 0.04034 ... 0.0266</div><input id='attrs-1cfa49cd-2932-448f-87e5-99677a41c549' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1cfa49cd-2932-448f-87e5-99677a41c549' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-74bf53b7-1a04-40f5-b11b-86898e53bc18' class='xr-var-data-in' type='checkbox'><label for='data-74bf53b7-1a04-40f5-b11b-86898e53bc18' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.00091876,  0.04033943, -0.02912118, ...,  0.00264853,
         0.01510604,  0.02000862],
       [ 0.00113503,  0.04120877, -0.0283649 , ...,  0.00291899,
         0.01476329,  0.02659982]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_eof</span></div><div class='xr-var-dims'>(chain, sigma_eof_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.336 1.579</div><input id='attrs-b3044a18-0d93-41de-8e2a-9309bd1ac303' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b3044a18-0d93-41de-8e2a-9309bd1ac303' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-48c79f82-5183-4679-b025-cfa644b9853e' class='xr-var-data-in' type='checkbox'><label for='data-48c79f82-5183-4679-b025-cfa644b9853e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1.33572638],
       [1.57886351]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_hier</span></div><div class='xr-var-dims'>(chain, sigma_hier_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.004982 0.02486 0.004976 0.02471</div><input id='attrs-fe1eaa87-fed5-49c6-af2c-d2e990a8435a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fe1eaa87-fed5-49c6-af2c-d2e990a8435a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fa1f7025-0b1e-4db8-96d2-e807dfe70bd7' class='xr-var-data-in' type='checkbox'><label for='data-fa1f7025-0b1e-4db8-96d2-e807dfe70bd7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00498231, 0.02486083],
       [0.00497604, 0.02471289]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma</span></div><div class='xr-var-dims'>(chain, sigma_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.004982 0.004982 ... 0.02471</div><input id='attrs-8f6d37f6-1c4a-4227-9613-fd27f9a3a8fc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8f6d37f6-1c4a-4227-9613-fd27f9a3a8fc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-18a18477-f101-400f-8df3-c649721e0292' class='xr-var-data-in' type='checkbox'><label for='data-18a18477-f101-400f-8df3-c649721e0292' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00498231, 0.00498231, 0.00498231, ..., 0.02486083, 0.02486083,
        0.02486083],
       [0.00497604, 0.00497604, 0.00497604, ..., 0.02471289, 0.02471289,
        0.02471289]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Estimates</span></div><div class='xr-var-dims'>(chain, Estimates_dim_0, Estimates_dim_1)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.007458 0.008395 ... 0.002872</div><input id='attrs-fd965718-7581-4dad-ba5b-95383942b1a6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fd965718-7581-4dad-ba5b-95383942b1a6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-12dd0da1-c738-41d5-9178-6bc4cdd1b138' class='xr-var-data-in' type='checkbox'><label for='data-12dd0da1-c738-41d5-9178-6bc4cdd1b138' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 7.45804381e-03,  8.39499728e-03,  3.85607846e-02, ...,
         -1.28185447e-02, -2.82463142e-02,  3.90662788e-04],
        [ 7.49241983e-03,  7.97437734e-03,  3.72210512e-02, ...,
         -1.24595214e-02, -2.69488051e-02,  2.05582073e-04],
        [ 8.44501996e-03,  7.60313653e-03,  3.73532707e-02, ...,
         -1.27359221e-02, -2.60083566e-02, -4.41790008e-04],
        ...,
        [-7.02540945e-03, -2.04932133e-03, -1.68895273e-02, ...,
          6.55630737e-03,  8.26262642e-03,  2.42893561e-03],
        [-8.53452348e-03, -2.56163317e-03, -2.07139444e-02, ...,
          7.99501543e-03,  1.02280210e-02,  2.91246725e-03],
        [-9.19834993e-03, -3.01522241e-03, -2.31760268e-02, ...,
          8.86907519e-03,  1.18327093e-02,  3.00114137e-03]],

       [[ 7.65893052e-03,  8.57729462e-03,  3.85457266e-02, ...,
         -1.33894236e-02, -2.69654615e-02,  5.85339874e-05],
        [ 7.68311321e-03,  8.16009465e-03,  3.72067795e-02, ...,
         -1.29985475e-02, -2.57367676e-02, -1.20370560e-04],
        [ 8.60316795e-03,  7.78518325e-03,  3.73550714e-02, ...,
         -1.32555063e-02, -2.49791141e-02, -6.95743687e-04],
        ...,
        [-7.03571311e-03, -2.16138253e-03, -1.69533667e-02, ...,
          6.83952171e-03,  8.31871956e-03,  2.32239677e-03],
        [-8.50458721e-03, -2.69783443e-03, -2.07388175e-02, ...,
          8.34181772e-03,  1.02621859e-02,  2.78124966e-03],
        [-9.18175164e-03, -3.16169567e-03, -2.32025677e-02, ...,
          9.25015950e-03,  1.18125669e-02,  2.87215443e-03]]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-b507f543-6b24-43cf-812b-7b95ccab71a7' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-b507f543-6b24-43cf-812b-7b95ccab71a7' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



Recombine estimated trends and pcs and apply uncertainty propagation:


```python
bpca_object.recombine_datasets()
```


```python
bpca_object.estimated_dataset
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt, dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:              (time: 26, x: 530)
Coordinates:
    lon                  (x) float64 2.13 8.218 6.741 18.46 ... 14.69 7.39 16.16
    lat                  (x) float64 10.87 5.567 8.49 ... 8.696 11.68 8.096
    ID                   (x) float64 0.0 0.0 0.0 0.0 0.0 ... 1.0 1.0 1.0 1.0 1.0
  * time                 (time) datetime64[ns] 1995-12-31 ... 2020-12-31
Dimensions without coordinates: x
Data variables:
    data_with_noise      (time, x) float64 0.008921 0.008888 ... 0.004385
    data_with_noise_std  (time, x) float64 0.008511 0.007558 ... 0.00741 0.00825
Attributes:
    chain:    0</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-399eb72c-250d-4324-ab88-790c57d07679' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-399eb72c-250d-4324-ab88-790c57d07679' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 26</li><li><span>x</span>: 530</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-5aac66da-fd5d-4871-9ea5-0537994bcee1' class='xr-section-summary-in' type='checkbox'  checked><label for='section-5aac66da-fd5d-4871-9ea5-0537994bcee1' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>lon</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.13 8.218 6.741 ... 7.39 16.16</div><input id='attrs-ce3021d4-24c8-415d-9625-42036c0281f6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ce3021d4-24c8-415d-9625-42036c0281f6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-63c465c9-e472-4448-8668-1c3665f083c7' class='xr-var-data-in' type='checkbox'><label for='data-63c465c9-e472-4448-8668-1c3665f083c7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([2.13020231e+00, 8.21799310e+00, 6.74071776e+00, 1.84618315e+01,
       1.66036389e+01, 1.64643029e+01, 8.35755280e+00, 7.03856257e+00,
       2.56565155e+00, 1.57334274e+01, 2.96332971e+00, 3.78155152e-01,
       1.92928091e+00, 4.64760736e+00, 1.21813013e+01, 1.74633426e+01,
       4.54328410e+00, 1.92788663e+00, 4.30865453e+00, 3.50111591e-02,
       6.07085670e+00, 1.10218871e+01, 9.60328452e+00, 3.96746246e-01,
       2.36835745e+00, 4.13995346e+00, 1.15061309e+01, 9.26670498e+00,
       1.48689913e+01, 5.05130483e+00, 1.37969674e+01, 6.38177472e+00,
       9.83477369e+00, 7.04776433e+00, 1.70793250e+01, 5.06567286e+00,
       5.67511802e+00, 3.81545604e+00, 1.04392482e+01, 9.62716890e+00,
       1.15048486e+01, 1.11782073e+01, 1.74510151e+01, 2.74294358e-01,
       3.32786602e-01, 1.88876768e+01, 1.13626910e+01, 1.27599491e+01,
       2.54798047e+00, 1.35697196e+01, 1.46842605e+01, 7.10186554e+00,
       1.95553438e+01, 7.31514463e+00, 1.83639882e+01, 1.09084517e+01,
       1.74145964e+01, 2.59660929e+00, 9.57074717e+00, 1.69429901e+01,
       1.85745606e+00, 1.17420176e+01, 8.90182504e+00, 1.63910559e+01,
       3.50383569e+00, 1.42924340e+01, 1.60635020e+00, 2.46333378e+00,
       3.15083607e+00, 5.26613091e-01, 5.51702431e+00, 7.31356718e+00,
       8.96602427e+00, 2.86960492e+00, 6.85544133e+00, 8.24171458e+00,
       9.71465422e+00, 1.27815342e+01, 8.08344603e-01, 1.33402622e+01,
...
       2.26593882e+00, 1.20707398e+01, 3.41475745e+00, 1.40553788e+01,
       1.22468902e+01, 8.58216322e+00, 3.87329293e+00, 5.60771147e+00,
       4.73899109e+00, 5.64048555e+00, 6.33916639e+00, 4.56189676e+00,
       1.08241830e+01, 1.75222183e+01, 4.72018211e-01, 7.80895944e+00,
       1.20002094e+01, 1.21326708e+01, 6.11869592e+00, 1.91134725e+01,
       4.85170185e+00, 1.04319759e+01, 7.48520632e+00, 4.57115036e+00,
       2.57074464e+00, 1.36933357e+01, 1.08505505e+01, 8.96445150e+00,
       1.56768218e+01, 7.28678496e+00, 8.51253182e+00, 3.16411870e-01,
       4.88880648e+00, 7.42534033e+00, 1.18755795e+00, 1.68383895e+01,
       6.30165045e+00, 1.78249891e+01, 1.14723401e+01, 1.73872587e+01,
       7.80326765e-01, 6.62202976e+00, 6.40753419e+00, 7.34726520e+00,
       5.15301955e+00, 1.59169192e+01, 1.54506710e+01, 6.21811286e+00,
       1.34595707e+01, 1.72293436e+01, 4.00236551e+00, 8.72940496e+00,
       2.83912924e+00, 1.23354293e+01, 6.62909985e+00, 6.64627990e+00,
       5.03779606e+00, 1.44593545e+01, 5.68979684e+00, 1.10223000e+01,
       1.41243581e+00, 5.73001520e+00, 1.19561822e+01, 8.39311577e-01,
       1.54508131e+01, 1.45537062e+00, 7.35581475e+00, 1.36534265e+01,
       1.49186595e+01, 6.94943994e+00, 5.13596235e+00, 3.48992616e+00,
       1.19043762e+01, 1.24593201e+01, 1.84523407e+01, 1.46898532e+01,
       7.39040781e+00, 1.61610704e+01])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lat</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>10.87 5.567 8.49 ... 11.68 8.096</div><input id='attrs-ea7eedeb-7348-48a2-83e1-ec7388bf3fab' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ea7eedeb-7348-48a2-83e1-ec7388bf3fab' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0d715505-5ee3-4e12-a817-81e8cd4fe46c' class='xr-var-data-in' type='checkbox'><label for='data-0d715505-5ee3-4e12-a817-81e8cd4fe46c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1.08680988e+01, 5.56738770e+00, 8.49035181e+00, 9.43771238e-02,
       2.43138242e+00, 2.73413179e+00, 4.18404244e+00, 3.70656439e+00,
       2.16753781e+00, 4.39394985e+00, 3.43882025e+00, 5.48147494e+00,
       6.72223900e+00, 3.50820907e+00, 7.45664093e+00, 1.13770147e-01,
       5.04852707e+00, 3.05099425e-01, 1.19768675e+01, 2.10295371e+00,
       7.63886890e+00, 7.29521132e-01, 1.19883978e+00, 1.15380300e+01,
       1.26036787e+01, 4.08782641e-01, 4.20053155e+00, 5.01390458e+00,
       5.71791381e+00, 7.19015688e+00, 7.09591223e+00, 6.80380431e+00,
       3.56161979e+00, 4.75388417e+00, 8.97245649e-01, 1.01086286e+01,
       7.52504909e+00, 1.18561080e+01, 2.85200629e+00, 7.75532561e+00,
       7.26376008e+00, 4.08690554e+00, 5.53530123e+00, 4.93071762e+00,
       3.47216003e+00, 6.80770446e+00, 1.84111207e+00, 1.76920346e+00,
       7.90071864e+00, 6.71192883e+00, 6.26132883e+00, 1.08080915e+01,
       5.93587502e+00, 2.21575802e+00, 6.25280596e+00, 9.13958260e+00,
       5.08515036e+00, 1.28220252e+01, 4.00247214e+00, 6.18000697e+00,
       1.25064752e+01, 3.33388262e+00, 4.63562730e-01, 3.21489097e+00,
       4.21956837e+00, 7.21050502e+00, 5.43661698e+00, 9.21203242e+00,
       1.00071179e+01, 1.05191187e+01, 2.79804624e-02, 7.89400573e+00,
       8.05760663e+00, 7.08596600e+00, 1.00122864e+01, 1.80865576e+00,
       5.47125840e+00, 5.30892827e-01, 7.99973793e-01, 5.66280719e+00,
...
       7.17865184e+00, 4.78361884e+00, 3.88248893e+00, 3.79281336e+00,
       1.19949319e+00, 7.96311160e+00, 9.14975422e+00, 3.46590817e+00,
       1.11032045e+01, 7.61325298e-01, 1.59825298e-02, 2.62295629e-01,
       2.94209699e+00, 5.31247837e+00, 1.20989017e+00, 5.15731262e+00,
       4.58122666e+00, 4.03708973e+00, 4.28729011e+00, 1.82495001e+00,
       1.00868007e+01, 7.28312226e+00, 2.80146515e+00, 5.35250204e+00,
       5.47592156e+00, 6.77244936e+00, 1.43207589e+00, 9.97193751e+00,
       6.12328428e+00, 8.63637987e+00, 1.09636712e+01, 1.18455782e+01,
       2.15868764e+00, 5.75629868e+00, 5.29834680e+00, 6.58583554e+00,
       3.07878566e+00, 6.11472555e+00, 8.37907626e+00, 2.88445043e+00,
       1.04242061e+01, 1.14642175e+01, 3.70165513e+00, 1.53900426e+00,
       4.62959529e+00, 4.47476944e+00, 1.58631287e+00, 1.05810628e+01,
       9.19834484e+00, 7.65996817e+00, 1.30576607e+01, 1.11286406e+01,
       1.35323565e+01, 9.65708678e+00, 1.19857375e+01, 1.19787266e+01,
       1.26351200e+01, 8.79035102e+00, 1.23690502e+01, 1.01929513e+01,
       1.41145645e+01, 1.23526378e+01, 9.81185073e+00, 1.43484462e+01,
       8.38575458e+00, 1.40970436e+01, 1.16891782e+01, 9.11923577e+00,
       8.60291692e+00, 1.18550124e+01, 1.25950601e+01, 1.32667780e+01,
       9.83299183e+00, 9.60652917e+00, 7.16088497e+00, 8.69628864e+00,
       1.16750614e+01, 8.09591129e+00])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>ID</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.0 0.0 0.0 ... 1.0 1.0 1.0 1.0</div><input id='attrs-f133539a-b548-4ab6-bb8b-e973b2630e56' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f133539a-b548-4ab6-bb8b-e973b2630e56' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3db7cd4b-21fc-456a-a6f0-5e3930c640b8' class='xr-var-data-in' type='checkbox'><label for='data-3db7cd4b-21fc-456a-a6f0-5e3930c640b8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>1995-12-31 ... 2020-12-31</div><input id='attrs-918c0748-c998-4902-91f6-c42257122bf5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-918c0748-c998-4902-91f6-c42257122bf5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-13a9419a-04f8-46b7-9736-2c98241d8d79' class='xr-var-data-in' type='checkbox'><label for='data-13a9419a-04f8-46b7-9736-2c98241d8d79' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;1995-12-31T00:00:00.000000000&#x27;, &#x27;1996-12-31T00:00:00.000000000&#x27;,
       &#x27;1997-12-31T00:00:00.000000000&#x27;, &#x27;1998-12-31T00:00:00.000000000&#x27;,
       &#x27;1999-12-31T00:00:00.000000000&#x27;, &#x27;2000-12-31T00:00:00.000000000&#x27;,
       &#x27;2001-12-31T00:00:00.000000000&#x27;, &#x27;2002-12-31T00:00:00.000000000&#x27;,
       &#x27;2003-12-31T00:00:00.000000000&#x27;, &#x27;2004-12-31T00:00:00.000000000&#x27;,
       &#x27;2005-12-31T00:00:00.000000000&#x27;, &#x27;2006-12-31T00:00:00.000000000&#x27;,
       &#x27;2007-12-31T00:00:00.000000000&#x27;, &#x27;2008-12-31T00:00:00.000000000&#x27;,
       &#x27;2009-12-31T00:00:00.000000000&#x27;, &#x27;2010-12-31T00:00:00.000000000&#x27;,
       &#x27;2011-12-31T00:00:00.000000000&#x27;, &#x27;2012-12-31T00:00:00.000000000&#x27;,
       &#x27;2013-12-31T00:00:00.000000000&#x27;, &#x27;2014-12-31T00:00:00.000000000&#x27;,
       &#x27;2015-12-31T00:00:00.000000000&#x27;, &#x27;2016-12-31T00:00:00.000000000&#x27;,
       &#x27;2017-12-31T00:00:00.000000000&#x27;, &#x27;2018-12-31T00:00:00.000000000&#x27;,
       &#x27;2019-12-31T00:00:00.000000000&#x27;, &#x27;2020-12-31T00:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a82161c6-4a46-4c1f-832b-a237a87cc739' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a82161c6-4a46-4c1f-832b-a237a87cc739' class='xr-section-summary' >Data variables: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>data_with_noise</span></div><div class='xr-var-dims'>(time, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.008921 0.008888 ... 0.004385</div><input id='attrs-5a052ed7-0e21-43d1-bd65-109c2f696fa4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5a052ed7-0e21-43d1-bd65-109c2f696fa4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-62454920-c614-413f-8e9a-08ab2cd2f267' class='xr-var-data-in' type='checkbox'><label for='data-62454920-c614-413f-8e9a-08ab2cd2f267' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.00892078,  0.00888827,  0.03996949, ..., -0.00787381,
        -0.03045105,  0.00097955],
       [ 0.00882445,  0.00843491,  0.03843791, ..., -0.00781881,
        -0.02906119,  0.00067246],
       [ 0.01003117,  0.00803202,  0.03890382, ..., -0.00901876,
        -0.02841449, -0.00052687],
       ...,
       [-0.00739882, -0.002085  , -0.01687767, ...,  0.00697461,
         0.00955934,  0.00357404],
       [-0.00927248, -0.00260717, -0.0211338 , ...,  0.00874132,
         0.01196283,  0.00448395],
       [-0.00967241, -0.00307228, -0.02313079, ...,  0.00908872,
         0.01352583,  0.00438474]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>data_with_noise_std</span></div><div class='xr-var-dims'>(time, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.008511 0.007558 ... 0.00825</div><input id='attrs-f2916065-2d9f-49bc-bb3e-4e3ec6cf7db2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f2916065-2d9f-49bc-bb3e-4e3ec6cf7db2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bc200905-5e06-421a-b9f2-ae65d519d850' class='xr-var-data-in' type='checkbox'><label for='data-bc200905-5e06-421a-b9f2-ae65d519d850' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00851057, 0.00755839, 0.0109858 , ..., 0.01726056, 0.01627797,
        0.01792407],
       [0.00812001, 0.00722025, 0.01047943, ..., 0.01648164, 0.01554183,
        0.01711797],
       [0.00800141, 0.00709057, 0.01034743, ..., 0.01619092, 0.01524083,
        0.01680894],
       ...,
       [0.00287405, 0.00247435, 0.00377845, ..., 0.00566723, 0.00525685,
        0.00586217],
       [0.00360367, 0.00309678, 0.0047405 , ..., 0.00709605, 0.00658025,
        0.00733834],
       [0.00401618, 0.00348342, 0.00526254, ..., 0.00796804, 0.00740969,
        0.00825005]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-7a031172-24e7-40ae-bddd-93540d2d1c51' class='xr-section-summary-in' type='checkbox'  checked><label for='section-7a031172-24e7-40ae-bddd-93540d2d1c51' class='xr-section-summary' >Attributes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>chain :</span></dt><dd>0</dd></dl></div></li></ul></div></div>



This property shows how much of the variance of the observed data is explained:


```python
bpca_object.explained_variance.mean()
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt, dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;data_with_noise&#x27; ()&gt;
array(0.61372726)</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'data_with_noise'</div></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-be4c1c32-7963-48b0-8adb-2f08b23c5d87' class='xr-array-in' type='checkbox' checked><label for='section-be4c1c32-7963-48b0-8adb-2f08b23c5d87' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.6137</span></div><div class='xr-array-data'><pre>array(0.61372726)</pre></div></div></li><li class='xr-section-item'><input id='section-851bc66a-4f5a-4de5-89e9-31b56d0f3a22' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-851bc66a-4f5a-4de5-89e9-31b56d0f3a22' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-aba3f467-c97a-462f-9bdf-50f3712a058b' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-aba3f467-c97a-462f-9bdf-50f3712a058b' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



Save and load the model as follows:


```python
bpca_object.save(save_dir = 'bpca/examples/model_name')
```


```python
bpca_object = bpca.load(save_dir = 'bpca/examples/model_name')
```

[back to top ](#top)

## Plotting and Visualization
<a id='plotting-and-visualization'></a>

### Individual time series


```python
fig,axs = plt.subplots(1,4,figsize=(16,4),sharey=True)

i=0
for index,title in zip([40,100,510,525],['A','B','C','D']):
    mean_est = bpca_object.estimated_dataset['data_with_noise'][:,index]
    std_est = bpca_object.estimated_dataset['data_with_noise_std'][:,index]
    mean_est.plot(ax=axs[i],color='white',linewidth=3)
    mean_est.plot(ax=axs[i],label='model estimate',color='blue')
    axs[i].fill_between(bpca_object.dataset.time.values, mean_est - std_est, mean_est + std_est, alpha=0.2,label='1 sigma')
    (data_set_synt['data'][:,index]/1000.).plot(ax=axs[i],label='true training data (without noise)',color='orange')
    bpca_object.dataset[:,index].plot(ax=axs[i],label='training data (with white noise =\n model input)',color='red')
    axs[i].set_title(label=title,loc='left',fontweight='bold')
    axs[i].set_title(('GPS' if data_set_synt['ID'][index] == 0 else 'SATTG')+': #'+str(index)) 
    
    axs[i].set_xlabel('m')
    axs[i].set_ylabel('')
    i=i+1
    
axs[0].legend()    
plt.tight_layout()
plt.savefig('time_series_exp1.png',dpi=600,bbox_inches='tight')
plt.savefig('time_series_exp1.pdf',bbox_inches='tight')
```


![png](bpca_tutorial_files/bpca_tutorial_51_0.png)


Shown are <font color='red'>the actual training data with white noise</font>, <font color='orange'>the training data without white noise</font>, <font color='blue'>the model mean</font> estimate, as well as some random samples from the markov chain and the recomputed uncertainty of the final estimate.

### Spatial estimates 

First compute uncertainties of $\mathbf{g(x)}$ and eof pattern $\mathbf{W(x)}$ analytically.


```python
data_set_synt = compute_analytical_uncertainties(data_set_synt)
```


```python
plot_synthetic_data_maps(data_set_synt,synthetic_data_settings,coastline,
                    validation=True,bpca_object=bpca_object)
```


![png](bpca_tutorial_files/bpca_tutorial_56_0.png)


As explained in the paper, we interpolate the point estimates using another Bayesian approach, which can be found here: https://github.com/oelsmann/TransTessellate2D. The 2D fields can be loaded into the bpca object to derive the final reconstruction.

These 2D files should be two netcdfs, where one file represents the mean estimate and the other the uncertainty. The data should span the region of interest (with arbitrary resolution). The files should contain the spatial pattern W0, W1, Wx, and the trend pattern (called W99).


```python
mean_files
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt, dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:  (x: 9409)
Coordinates:
    lon      (x) float64 -2.0 -1.75 -1.5 -1.25 -1.0 ... 21.25 21.5 21.75 22.0
    lat      (x) float64 -2.0 -2.0 -2.0 -2.0 -2.0 ... 22.0 22.0 22.0 22.0 22.0
Dimensions without coordinates: x
Data variables:
    W0       (x) float64 0.3387 0.3305 0.3261 0.3225 ... 0.3726 0.3918 0.416
    W99      (x) float64 2.301 2.238 2.187 2.14 ... -1.727 -1.758 -1.794 -1.851</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-5f8257c4-7e6f-4f2d-bb79-2f185419723c' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-5f8257c4-7e6f-4f2d-bb79-2f185419723c' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span>x</span>: 9409</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-3d1cf1e6-cffa-4cd7-98ec-07038e00c237' class='xr-section-summary-in' type='checkbox'  checked><label for='section-3d1cf1e6-cffa-4cd7-98ec-07038e00c237' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>lon</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.0 -1.75 -1.5 ... 21.5 21.75 22.0</div><input id='attrs-96e0d831-6587-4c72-82d8-9d38b3aa8268' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-96e0d831-6587-4c72-82d8-9d38b3aa8268' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-80eda75b-0a64-4910-91b4-5c025b8fee30' class='xr-var-data-in' type='checkbox'><label for='data-80eda75b-0a64-4910-91b4-5c025b8fee30' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-2.  , -1.75, -1.5 , ..., 21.5 , 21.75, 22.  ])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lat</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.0 -2.0 -2.0 ... 22.0 22.0 22.0</div><input id='attrs-506d3090-7e64-4778-8534-e304366ae9c8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-506d3090-7e64-4778-8534-e304366ae9c8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a3cb970e-c014-48de-a64e-47656a32a680' class='xr-var-data-in' type='checkbox'><label for='data-a3cb970e-c014-48de-a64e-47656a32a680' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-2., -2., -2., ..., 22., 22., 22.])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2fc37609-32ab-4a58-8463-5618f615079b' class='xr-section-summary-in' type='checkbox'  checked><label for='section-2fc37609-32ab-4a58-8463-5618f615079b' class='xr-section-summary' >Data variables: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>W0</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3387 0.3305 ... 0.3918 0.416</div><input id='attrs-02121eb4-7e15-46df-baa2-2886b91a48b9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-02121eb4-7e15-46df-baa2-2886b91a48b9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0652e7f8-0da0-477a-a937-4c4bb1e209f0' class='xr-var-data-in' type='checkbox'><label for='data-0652e7f8-0da0-477a-a937-4c4bb1e209f0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.338675, 0.330495, 0.326142, ..., 0.372644, 0.391777, 0.416032])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>W99</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.301 2.238 2.187 ... -1.794 -1.851</div><input id='attrs-4781b2d3-8889-4806-9127-b2235b1deac0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4781b2d3-8889-4806-9127-b2235b1deac0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4146a899-ebdf-40e2-9ad9-997592f9bec7' class='xr-var-data-in' type='checkbox'><label for='data-4146a899-ebdf-40e2-9ad9-997592f9bec7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 2.301447,  2.237631,  2.186807, ..., -1.758045, -1.793877,
       -1.850701])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-b9211e9e-dcc9-4e20-b09b-7bd4935c3a70' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-b9211e9e-dcc9-4e20-b09b-7bd4935c3a70' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




```python
bpca_object.estimated_dataset_map_pattern=[mean_files,std_files]
bpca_object.recombine_datasets(kind='maps')
```

Match 2D fields and point locations for validation.


```python
lon1 = data_set_synt.x.lon.values
lat1 = data_set_synt.x.lat.values

lon2 = bpca_object.estimated_dataset_map_pattern[0].lon.values
lat2 = bpca_object.estimated_dataset_map_pattern[0].lat.values

indices = np.argmin((np.repeat(lon1[:,np.newaxis],len(lon2),axis=1)-np.repeat(lon2[np.newaxis,:],len(lon1),axis=0))**2+(np.repeat(lat1[:,np.newaxis],len(lat2),axis=1)-np.repeat(lat2[np.newaxis,:],len(lat1),axis=0))**2,axis=1)
```


```python
plot_synthetic_data_maps(data_set_synt,synthetic_data_settings,coastline,
                    validation=True,map_data=True,bpca_object=bpca_objectt,indices=indices)
```


![png](bpca_tutorial_files/bpca_tutorial_62_0.png)


[back to top ](#top)

----
## Exp2: omitting individual stations
<a id='exp_running'></a>

Omit some stations from the synthetic dataset (note that lon and lat are reversed).


```python
plot_synthetic_data_maps(data_set_synt.sel({'x':(data_set_synt.lon>13) | (data_set_synt.lon<10)}),synthetic_data_settings,coastline)
```


![png](bpca_tutorial_files/bpca_tutorial_66_0.png)



```python
data_set_synt['data_with_noise'].sel({'x':(data_set_synt.lon>13) | (data_set_synt.lon<10)})
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt, dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;data_with_noise&#x27; (time: 26, x: 458)&gt;
array([[ 23.04892112,  11.32336058,  78.88208563, ..., -12.40985097,
        -69.79189362,  24.67772429],
       [ 16.70985425,  19.95458837,  91.54017576, ..., -20.23500132,
        -65.3776309 ,  14.00354682],
       [ 14.06050529,  19.91578555,          nan, ..., -12.72176553,
        -48.06613667,   7.97603548],
       ...,
       [ -0.55901971,   3.4149369 , -18.624443  , ..., -29.91613046,
          9.59332226,  22.7263143 ],
       [ -5.44058197,  -9.16038391,          nan, ...,  -4.07008733,
         10.99079468,  -7.11781562],
       [ -2.63565213,  -2.19435957, -30.8661067 , ...,  -2.71946949,
         -9.04685145, -23.40893421]])
Coordinates:
    lon      (x) float64 2.13 8.218 6.741 18.46 16.6 ... 18.45 14.69 7.39 16.16
    lat      (x) float64 10.87 5.567 8.49 0.09438 ... 7.161 8.696 11.68 8.096
    ID       (x) float64 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ... 1.0 1.0 1.0 1.0 1.0 1.0
  * time     (time) datetime64[ns] 1995-12-31 1996-12-31 ... 2020-12-31
Dimensions without coordinates: x</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'data_with_noise'</div><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 26</li><li><span>x</span>: 458</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-c47220f5-f00b-4405-98b7-974be27ac890' class='xr-array-in' type='checkbox' checked><label for='section-c47220f5-f00b-4405-98b7-974be27ac890' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>23.05 11.32 78.88 24.39 8.08 nan ... 21.69 -12.25 -2.719 -9.047 -23.41</span></div><div class='xr-array-data'><pre>array([[ 23.04892112,  11.32336058,  78.88208563, ..., -12.40985097,
        -69.79189362,  24.67772429],
       [ 16.70985425,  19.95458837,  91.54017576, ..., -20.23500132,
        -65.3776309 ,  14.00354682],
       [ 14.06050529,  19.91578555,          nan, ..., -12.72176553,
        -48.06613667,   7.97603548],
       ...,
       [ -0.55901971,   3.4149369 , -18.624443  , ..., -29.91613046,
          9.59332226,  22.7263143 ],
       [ -5.44058197,  -9.16038391,          nan, ...,  -4.07008733,
         10.99079468,  -7.11781562],
       [ -2.63565213,  -2.19435957, -30.8661067 , ...,  -2.71946949,
         -9.04685145, -23.40893421]])</pre></div></div></li><li class='xr-section-item'><input id='section-a233db4a-9055-4065-ab7b-05e7ba897942' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a233db4a-9055-4065-ab7b-05e7ba897942' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>lon</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.13 8.218 6.741 ... 7.39 16.16</div><input id='attrs-09a5efe4-c8d6-4f19-bad0-631e7dc360f5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-09a5efe4-c8d6-4f19-bad0-631e7dc360f5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bb02b92f-08d4-4fff-a183-9e2583711adb' class='xr-var-data-in' type='checkbox'><label for='data-bb02b92f-08d4-4fff-a183-9e2583711adb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([2.13020231e+00, 8.21799310e+00, 6.74071776e+00, 1.84618315e+01,
       1.66036389e+01, 1.64643029e+01, 8.35755280e+00, 7.03856257e+00,
       2.56565155e+00, 1.57334274e+01, 2.96332971e+00, 3.78155152e-01,
       1.92928091e+00, 4.64760736e+00, 1.74633426e+01, 4.54328410e+00,
       1.92788663e+00, 4.30865453e+00, 3.50111591e-02, 6.07085670e+00,
       9.60328452e+00, 3.96746246e-01, 2.36835745e+00, 4.13995346e+00,
       9.26670498e+00, 1.48689913e+01, 5.05130483e+00, 1.37969674e+01,
       6.38177472e+00, 9.83477369e+00, 7.04776433e+00, 1.70793250e+01,
       5.06567286e+00, 5.67511802e+00, 3.81545604e+00, 9.62716890e+00,
       1.74510151e+01, 2.74294358e-01, 3.32786602e-01, 1.88876768e+01,
       2.54798047e+00, 1.35697196e+01, 1.46842605e+01, 7.10186554e+00,
       1.95553438e+01, 7.31514463e+00, 1.83639882e+01, 1.74145964e+01,
       2.59660929e+00, 9.57074717e+00, 1.69429901e+01, 1.85745606e+00,
       8.90182504e+00, 1.63910559e+01, 3.50383569e+00, 1.42924340e+01,
       1.60635020e+00, 2.46333378e+00, 3.15083607e+00, 5.26613091e-01,
       5.51702431e+00, 7.31356718e+00, 8.96602427e+00, 2.86960492e+00,
       6.85544133e+00, 8.24171458e+00, 9.71465422e+00, 8.08344603e-01,
       1.33402622e+01, 6.48329784e+00, 7.30042054e+00, 1.57653457e+00,
       3.11114410e+00, 1.45973961e+01, 6.12249608e+00, 7.94425831e+00,
       1.69289733e+01, 1.59035910e+01, 6.03255190e+00, 5.51846644e+00,
...
       7.39963327e-02, 2.89502860e+00, 9.59711390e+00, 6.80688995e+00,
       1.67889109e+01, 1.58386507e+01, 5.30295782e+00, 1.57075982e+01,
       5.54859764e+00, 1.12878989e+00, 1.93570903e+01, 9.56186257e+00,
       2.92928175e+00, 2.26593882e+00, 3.41475745e+00, 1.40553788e+01,
       8.58216322e+00, 3.87329293e+00, 5.60771147e+00, 4.73899109e+00,
       5.64048555e+00, 6.33916639e+00, 4.56189676e+00, 1.75222183e+01,
       4.72018211e-01, 7.80895944e+00, 6.11869592e+00, 1.91134725e+01,
       4.85170185e+00, 7.48520632e+00, 4.57115036e+00, 2.57074464e+00,
       1.36933357e+01, 8.96445150e+00, 1.56768218e+01, 7.28678496e+00,
       8.51253182e+00, 3.16411870e-01, 4.88880648e+00, 7.42534033e+00,
       1.18755795e+00, 1.68383895e+01, 6.30165045e+00, 1.78249891e+01,
       1.73872587e+01, 7.80326765e-01, 6.62202976e+00, 6.40753419e+00,
       7.34726520e+00, 5.15301955e+00, 1.59169192e+01, 1.54506710e+01,
       6.21811286e+00, 1.34595707e+01, 1.72293436e+01, 4.00236551e+00,
       8.72940496e+00, 2.83912924e+00, 6.62909985e+00, 6.64627990e+00,
       5.03779606e+00, 1.44593545e+01, 5.68979684e+00, 1.41243581e+00,
       5.73001520e+00, 8.39311577e-01, 1.54508131e+01, 1.45537062e+00,
       7.35581475e+00, 1.36534265e+01, 1.49186595e+01, 6.94943994e+00,
       5.13596235e+00, 3.48992616e+00, 1.84523407e+01, 1.46898532e+01,
       7.39040781e+00, 1.61610704e+01])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lat</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>10.87 5.567 8.49 ... 11.68 8.096</div><input id='attrs-e31f06eb-37f8-42af-99c4-80d9bdd9ac43' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e31f06eb-37f8-42af-99c4-80d9bdd9ac43' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8e028942-25f7-42e6-b66b-7b1939b86f5d' class='xr-var-data-in' type='checkbox'><label for='data-8e028942-25f7-42e6-b66b-7b1939b86f5d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([10.86809884,  5.5673877 ,  8.49035181,  0.09437712,  2.43138242,
        2.73413179,  4.18404244,  3.70656439,  2.16753781,  4.39394985,
        3.43882025,  5.48147494,  6.722239  ,  3.50820907,  0.11377015,
        5.04852707,  0.30509942, 11.97686754,  2.10295371,  7.6388689 ,
        1.19883978, 11.53802999, 12.60367873,  0.40878264,  5.01390458,
        5.71791381,  7.19015688,  7.09591223,  6.80380431,  3.56161979,
        4.75388417,  0.89724565, 10.10862859,  7.52504909, 11.85610802,
        7.75532561,  5.53530123,  4.93071762,  3.47216003,  6.80770446,
        7.90071864,  6.71192883,  6.26132883, 10.80809151,  5.93587502,
        2.21575802,  6.25280596,  5.08515036, 12.82202517,  4.00247214,
        6.18000697, 12.50647516,  0.46356273,  3.21489097,  4.21956837,
        7.21050502,  5.43661698,  9.21203242, 10.00711793, 10.51911872,
        0.02798046,  7.89400573,  8.05760663,  7.085966  , 10.01228639,
        1.80865576,  5.4712584 ,  0.79997379,  5.66280719, 11.6468834 ,
        2.20096662,  3.4629982 , 11.6538763 ,  3.79347058,  8.21541346,
        9.73782965,  6.19179636,  7.19356205,  6.42663864,  4.1641448 ,
        9.83685821,  7.50878495,  6.8747907 ,  2.66057379,  9.12078115,
        3.1947246 , 10.40321374,  4.36544515,  2.69837445,  7.74345256,
        5.0166804 ,  1.59146878,  3.24493837,  8.11762645,  8.34181471,
        8.49694476,  1.40398228,  6.03504827,  0.71253993,  5.88660883,
...
        6.92872271, 10.32483758,  7.89871676,  5.84946334,  2.41312652,
        8.62368389,  5.43903121,  5.47452552,  8.99824657,  6.7822504 ,
        4.78237243,  4.10322152,  3.90422895,  7.80659151,  1.97046846,
       13.14941515,  3.91257498, 10.51357521,  6.21618208,  9.30225858,
        2.39349568,  2.2893719 ,  0.80016652,  5.77996655,  7.32067564,
        1.47486172,  9.7603383 ,  7.42156145,  1.9452624 ,  6.25769611,
        1.08594742,  3.55397469,  7.4246201 ,  7.17865184,  3.88248893,
        3.79281336,  7.9631116 ,  9.14975422,  3.46590817, 11.10320449,
        0.7613253 ,  0.01598253,  0.26229563,  5.31247837,  1.20989017,
        5.15731262,  4.28729011,  1.82495001, 10.0868007 ,  2.80146515,
        5.35250204,  5.47592156,  6.77244936,  9.97193751,  6.12328428,
        8.63637987, 10.9636712 , 11.84557824,  2.15868764,  5.75629868,
        5.2983468 ,  6.58583554,  3.07878566,  6.11472555,  2.88445043,
       10.42420612, 11.46421754,  3.70165513,  1.53900426,  4.62959529,
        4.47476944,  1.58631287, 10.58106281,  9.19834484,  7.65996817,
       13.05766066, 11.12864065, 13.53235651, 11.98573749, 11.97872662,
       12.63512002,  8.79035102, 12.3690502 , 14.11456451, 12.35263781,
       14.34844623,  8.38575458, 14.09704358, 11.68917818,  9.11923577,
        8.60291692, 11.85501245, 12.59506012, 13.26677796,  7.16088497,
        8.69628864, 11.67506137,  8.09591129])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>ID</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.0 0.0 0.0 ... 1.0 1.0 1.0 1.0</div><input id='attrs-1c0a1d57-18f1-4317-a982-aaaf872427be' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1c0a1d57-18f1-4317-a982-aaaf872427be' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-52b6155b-1fcd-4fcb-b3a2-a82022112fbc' class='xr-var-data-in' type='checkbox'><label for='data-52b6155b-1fcd-4fcb-b3a2-a82022112fbc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>1995-12-31 ... 2020-12-31</div><input id='attrs-bd1fe228-d7a9-4dde-9b91-08d5f67d925d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bd1fe228-d7a9-4dde-9b91-08d5f67d925d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9dde690a-9a3c-48a9-8e4e-6672b7882a67' class='xr-var-data-in' type='checkbox'><label for='data-9dde690a-9a3c-48a9-8e4e-6672b7882a67' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;1995-12-31T00:00:00.000000000&#x27;, &#x27;1996-12-31T00:00:00.000000000&#x27;,
       &#x27;1997-12-31T00:00:00.000000000&#x27;, &#x27;1998-12-31T00:00:00.000000000&#x27;,
       &#x27;1999-12-31T00:00:00.000000000&#x27;, &#x27;2000-12-31T00:00:00.000000000&#x27;,
       &#x27;2001-12-31T00:00:00.000000000&#x27;, &#x27;2002-12-31T00:00:00.000000000&#x27;,
       &#x27;2003-12-31T00:00:00.000000000&#x27;, &#x27;2004-12-31T00:00:00.000000000&#x27;,
       &#x27;2005-12-31T00:00:00.000000000&#x27;, &#x27;2006-12-31T00:00:00.000000000&#x27;,
       &#x27;2007-12-31T00:00:00.000000000&#x27;, &#x27;2008-12-31T00:00:00.000000000&#x27;,
       &#x27;2009-12-31T00:00:00.000000000&#x27;, &#x27;2010-12-31T00:00:00.000000000&#x27;,
       &#x27;2011-12-31T00:00:00.000000000&#x27;, &#x27;2012-12-31T00:00:00.000000000&#x27;,
       &#x27;2013-12-31T00:00:00.000000000&#x27;, &#x27;2014-12-31T00:00:00.000000000&#x27;,
       &#x27;2015-12-31T00:00:00.000000000&#x27;, &#x27;2016-12-31T00:00:00.000000000&#x27;,
       &#x27;2017-12-31T00:00:00.000000000&#x27;, &#x27;2018-12-31T00:00:00.000000000&#x27;,
       &#x27;2019-12-31T00:00:00.000000000&#x27;, &#x27;2020-12-31T00:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f2c8c1f6-85ec-46e9-ad4e-48174795585b' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-f2c8c1f6-85ec-46e9-ad4e-48174795585b' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



Rerun the model.


```python
settings['model_settings']['cluster_index'] = data_set_synt['ID'].sel({'x':(data_set_synt.lon>13) | (data_set_synt.lon<10)}).values

bpca_object_exp2 = bpca(data_set_synt['data_with_noise'].sel({'x':(data_set_synt.lon>13) | (data_set_synt.lon<10)})/1000.,run_settings=settings['run_settings'],
                  model_settings = settings['model_settings'],name=name+'exp2')
```

    estimate different sigma for different clusters



```python
bpca_object_exp2.run()
bpca_object_exp2.adjust_pca_symmetry()
bpca_object_exp2.check_convergence()
bpca_object_exp2.compress()
bpca_object_exp2.recombine_datasets()
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [Observations_missing, trend_g, W0, PC0, sigma_hier, sigma_eof]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='7000' class='' max='7000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [7000/7000 02:34<00:00 Sampling 2 chains, 0 divergences]
</div>



    Sampling 2 chains for 2_000 tune and 1_500 draw iterations (4_000 + 3_000 draws total) took 155 seconds.
    The rhat statistic is larger than 1.4 for some parameters. The sampler did not converge.
    The estimated number of effective samples is smaller than 200 for some parameters.


    replaced variable PC0 (chain 0 ) with variable PC0 (chain 0)
    replaced variable PC0 (chain 1 ) with variable PC0 (chain 1)
    replaced variable W0 (chain 0 ) with variable W0 (chain 0)
    replaced variable W0 (chain 1 ) with variable W0 (chain 1)
    replaced variable sigma_eof0 (chain 0 ) with variable sigma_eof0 (chain 0)
    replaced variable sigma_eof0 (chain 1 ) with variable sigma_eof0 (chain 1)
    successfully compressed trace



```python
bpca_object_exp2.trace['mean'].posterior
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt, dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:                     (Estimates_dim_0: 26, Estimates_dim_1: 458, Observations_missing_dim_0: 777, PC0_dim_0: 26, W0_dim_0: 458, chain: 2, sigma_dim_0: 458, sigma_eof_dim_0: 1, sigma_hier_dim_0: 2, trend_g_dim_0: 458)
Coordinates:
  * chain                       (chain) int64 0 1
  * PC0_dim_0                   (PC0_dim_0) int64 0 1 2 3 4 5 ... 21 22 23 24 25
  * W0_dim_0                    (W0_dim_0) int64 0 1 2 3 4 ... 454 455 456 457
  * trend_g_dim_0               (trend_g_dim_0) int64 0 1 2 3 ... 455 456 457
  * Observations_missing_dim_0  (Observations_missing_dim_0) int64 0 1 ... 776
  * sigma_eof_dim_0             (sigma_eof_dim_0) int64 0
  * sigma_hier_dim_0            (sigma_hier_dim_0) int64 0 1
  * sigma_dim_0                 (sigma_dim_0) int64 0 1 2 3 ... 454 455 456 457
  * Estimates_dim_0             (Estimates_dim_0) int64 0 1 2 3 ... 22 23 24 25
  * Estimates_dim_1             (Estimates_dim_1) int64 0 1 2 3 ... 455 456 457
Data variables:
    PC0                         (chain, PC0_dim_0) float64 -0.02552 ... 0.01002
    W0                          (chain, W0_dim_0) float64 -1.233 ... 0.269
    trend_g                     (chain, trend_g_dim_0) float64 0.0006577 ... ...
    Observations_missing        (chain, Observations_missing_dim_0) float64 -...
    sigma_eof                   (chain, sigma_eof_dim_0) float64 1.017 1.017
    sigma_hier                  (chain, sigma_hier_dim_0) float64 0.004969 .....
    sigma                       (chain, sigma_dim_0) float64 0.004969 ... 0.0...
    Estimates                   (chain, Estimates_dim_0, Estimates_dim_1) float64 ...</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-c159bea9-8152-4927-94c9-d0642d436d65' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-c159bea9-8152-4927-94c9-d0642d436d65' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>Estimates_dim_0</span>: 26</li><li><span class='xr-has-index'>Estimates_dim_1</span>: 458</li><li><span class='xr-has-index'>Observations_missing_dim_0</span>: 777</li><li><span class='xr-has-index'>PC0_dim_0</span>: 26</li><li><span class='xr-has-index'>W0_dim_0</span>: 458</li><li><span class='xr-has-index'>chain</span>: 2</li><li><span class='xr-has-index'>sigma_dim_0</span>: 458</li><li><span class='xr-has-index'>sigma_eof_dim_0</span>: 1</li><li><span class='xr-has-index'>sigma_hier_dim_0</span>: 2</li><li><span class='xr-has-index'>trend_g_dim_0</span>: 458</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-ed6da047-2061-4613-9506-aed1427f7fbc' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ed6da047-2061-4613-9506-aed1427f7fbc' class='xr-section-summary' >Coordinates: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1</div><input id='attrs-dbb72d38-b29e-4f58-947a-a44a02a63e8e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dbb72d38-b29e-4f58-947a-a44a02a63e8e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4d8def3b-8bfb-4ae6-b157-c89489ff583f' class='xr-var-data-in' type='checkbox'><label for='data-4d8def3b-8bfb-4ae6-b157-c89489ff583f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>PC0_dim_0</span></div><div class='xr-var-dims'>(PC0_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 ... 20 21 22 23 24 25</div><input id='attrs-bf420e28-2198-4167-970d-ca6932310be6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bf420e28-2198-4167-970d-ca6932310be6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6a2b9782-ade8-4e89-8cc2-3c16e1d0200b' class='xr-var-data-in' type='checkbox'><label for='data-6a2b9782-ade8-4e89-8cc2-3c16e1d0200b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>W0_dim_0</span></div><div class='xr-var-dims'>(W0_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 453 454 455 456 457</div><input id='attrs-0557f894-fc14-4781-a723-93248a570481' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0557f894-fc14-4781-a723-93248a570481' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7d2c5471-be9e-4a29-992f-7221822061c9' class='xr-var-data-in' type='checkbox'><label for='data-7d2c5471-be9e-4a29-992f-7221822061c9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 455, 456, 457])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>trend_g_dim_0</span></div><div class='xr-var-dims'>(trend_g_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 453 454 455 456 457</div><input id='attrs-945f0e49-c4d3-4a0b-b24f-1b16cae7e0da' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-945f0e49-c4d3-4a0b-b24f-1b16cae7e0da' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7037081d-0599-48a2-83e0-8dd15162181c' class='xr-var-data-in' type='checkbox'><label for='data-7037081d-0599-48a2-83e0-8dd15162181c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 455, 456, 457])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>Observations_missing_dim_0</span></div><div class='xr-var-dims'>(Observations_missing_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 772 773 774 775 776</div><input id='attrs-999cd0fa-4079-4b37-b61b-eb6e5419a3d7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-999cd0fa-4079-4b37-b61b-eb6e5419a3d7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4f2c4fba-39c1-454d-960e-3827b0c9cdb0' class='xr-var-data-in' type='checkbox'><label for='data-4f2c4fba-39c1-454d-960e-3827b0c9cdb0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 774, 775, 776])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sigma_eof_dim_0</span></div><div class='xr-var-dims'>(sigma_eof_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-c60a16f8-4c40-49cd-a71c-94b48d470dc7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c60a16f8-4c40-49cd-a71c-94b48d470dc7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c5a1b096-afa1-4965-be23-3eb58cebb003' class='xr-var-data-in' type='checkbox'><label for='data-c5a1b096-afa1-4965-be23-3eb58cebb003' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sigma_hier_dim_0</span></div><div class='xr-var-dims'>(sigma_hier_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1</div><input id='attrs-9b9bf53e-3cda-4def-8a19-3e177f11fe3d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9b9bf53e-3cda-4def-8a19-3e177f11fe3d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2d119e27-ff39-49f6-8e57-905970400677' class='xr-var-data-in' type='checkbox'><label for='data-2d119e27-ff39-49f6-8e57-905970400677' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>sigma_dim_0</span></div><div class='xr-var-dims'>(sigma_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 453 454 455 456 457</div><input id='attrs-bce35426-116b-49db-b495-868d9b98b494' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bce35426-116b-49db-b495-868d9b98b494' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-16e0efa3-af90-4088-be82-f7c39795333f' class='xr-var-data-in' type='checkbox'><label for='data-16e0efa3-af90-4088-be82-f7c39795333f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 455, 456, 457])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>Estimates_dim_0</span></div><div class='xr-var-dims'>(Estimates_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 ... 20 21 22 23 24 25</div><input id='attrs-1c538ac7-67ab-41f6-a9e1-43b115af67c7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1c538ac7-67ab-41f6-a9e1-43b115af67c7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8eb885fd-accf-44c0-9f8c-a87d37ef93c2' class='xr-var-data-in' type='checkbox'><label for='data-8eb885fd-accf-44c0-9f8c-a87d37ef93c2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>Estimates_dim_1</span></div><div class='xr-var-dims'>(Estimates_dim_1)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 453 454 455 456 457</div><input id='attrs-389acc9c-cf30-45a4-a1fd-96f7136b7db2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-389acc9c-cf30-45a4-a1fd-96f7136b7db2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f6cbfcdd-55d0-4ece-a1ed-d73be079d6a2' class='xr-var-data-in' type='checkbox'><label for='data-f6cbfcdd-55d0-4ece-a1ed-d73be079d6a2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 455, 456, 457])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e9ade7b6-3fa9-4abe-9956-78a816782f12' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e9ade7b6-3fa9-4abe-9956-78a816782f12' class='xr-section-summary' >Data variables: <span>(8)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>PC0</span></div><div class='xr-var-dims'>(chain, PC0_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.02552 -0.02542 ... 0.01002</div><input id='attrs-f117b6f0-5396-4697-817f-db9b0c9d45bf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f117b6f0-5396-4697-817f-db9b0c9d45bf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b099e872-e1f5-4cb6-930b-1c3b737b485f' class='xr-var-data-in' type='checkbox'><label for='data-b099e872-e1f5-4cb6-930b-1c3b737b485f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-0.02552211, -0.02542106, -0.02596134, -0.0258404 , -0.02653824,
        -0.02628028, -0.02609131, -0.02633008, -0.02680358, -0.02650839,
        -0.02705182, -0.02715123, -0.02709503, -0.02748515, -0.02722926,
        -0.01629168, -0.01087152, -0.00617306, -0.00273857, -0.00019928,
         0.00268129,  0.00443443,  0.00626299,  0.0074423 ,  0.00911788,
         0.00989139],
       [-0.02582659, -0.0257419 , -0.02630276, -0.02615961, -0.02686721,
        -0.02659819, -0.0264276 , -0.02662815, -0.02711433, -0.02681127,
        -0.02735031, -0.02744409, -0.02738698, -0.02778621, -0.02751915,
        -0.0164614 , -0.01098545, -0.00623364, -0.00277858, -0.00019563,
         0.0027018 ,  0.00448248,  0.00633166,  0.00754054,  0.00924292,
         0.01001999]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>W0</span></div><div class='xr-var-dims'>(chain, W0_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.233 -0.3487 -2.9 ... 1.3 0.269</div><input id='attrs-270347a9-731f-4cd5-b2f1-2d2099370e8e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-270347a9-731f-4cd5-b2f1-2d2099370e8e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-51ad0823-5a7a-43de-8766-c0a0fae8513f' class='xr-var-data-in' type='checkbox'><label for='data-51ad0823-5a7a-43de-8766-c0a0fae8513f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.23287141e+00, -3.48680277e-01, -2.89955860e+00,
         1.60392369e-01,  1.06966600e-01,  2.17131163e-01,
        -3.55383487e-01,  1.95003865e-01, -6.76990503e-03,
        -6.31977167e-01, -4.02300427e-01,  5.45787800e-01,
        -2.12214136e-01,  8.04839080e-01,  3.17474026e-01,
        -1.55691488e-01,  7.08789733e-01,  1.05108634e+00,
         6.43320118e-01, -2.13568332e+00, -2.32622984e-01,
        -4.91653296e-01,  5.68091681e-01,  7.37986241e-01,
         2.75454895e-01, -6.97441898e-01, -6.54195288e-01,
        -3.00921410e-01, -1.06610665e+00, -3.55571191e-01,
        -3.72578927e-01,  2.69099643e-01, -1.15427411e+00,
        -1.23157674e+00,  1.78270105e-01, -2.81466664e+00,
        -4.62952584e-01,  3.72195515e-01, -8.97525980e-03,
         5.46695310e-01, -7.17112872e-01, -1.38028940e+00,
        -4.00144754e-01,  2.22746248e+00,  4.54834000e-01,
        -5.17071367e-02,  3.58016836e-01, -2.62048446e-01,
         6.72458488e-01, -8.24155601e-01, -1.77307668e-01,
        -6.83173113e-02,  7.24243155e-03, -2.26729147e-01,
        -2.15606014e-01,  5.96346175e-02, -4.47873165e-01,
        -6.99623984e-01, -1.88415628e+00, -1.30243451e+00,
...
         3.52134190e-01, -7.50125026e-02, -3.58286526e-03,
        -7.10807270e-02, -2.02251460e-01, -2.07826518e-01,
        -8.48056594e-01,  5.78695230e-01, -3.15152324e-01,
        -1.63656621e+00,  1.93997892e-01,  1.11584427e-01,
         1.66227646e-01, -1.04575310e+00,  2.50408216e+00,
        -4.11397308e-01, -2.05351020e+00,  2.09136763e+00,
         4.15950631e-02, -1.10601733e-01, -1.72049079e-01,
        -1.55025410e-01,  1.83482525e-01,  2.70761182e-01,
         2.97009473e-01, -1.72941343e-01, -6.34310354e-01,
         2.03249068e+00,  4.00259220e-01,  3.08225345e-01,
         1.85958195e-02, -9.99520139e-01, -6.85903443e-02,
         3.79224237e-01,  1.53254717e+00, -5.92006456e-01,
         3.18849240e-01,  2.66893809e+00, -3.82713003e-01,
         2.21172122e+00,  1.93495887e+00,  1.13100624e+00,
         1.19180610e+00,  1.71477894e+00,  1.45422700e-01,
         2.57556225e-01, -3.90850354e-01,  1.04150852e+00,
         2.26157093e-01,  6.08573438e-01,  1.23764186e+00,
         1.44146444e-01,  1.30145760e+00,  1.30378756e+00,
         1.41435750e+00, -6.42141784e-01,  1.23287056e+00,
         1.30026454e+00,  2.68974653e-01]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>trend_g</span></div><div class='xr-var-dims'>(chain, trend_g_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0006577 -0.0003802 ... -0.0008522</div><input id='attrs-c0b5c8f7-8d42-4c36-9891-5f9b166cee86' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c0b5c8f7-8d42-4c36-9891-5f9b166cee86' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d734f6a8-df9e-4bf1-b0f4-a95da4c1606f' class='xr-var-data-in' type='checkbox'><label for='data-d734f6a8-df9e-4bf1-b0f4-a95da4c1606f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 6.57674747e-04, -3.80152926e-04, -6.56579662e-04,
        -1.32500057e-03, -3.73190399e-04,  3.91482204e-04,
         9.60995120e-04,  1.04423345e-04,  1.76192560e-03,
        -1.28290278e-03,  1.98266434e-03,  1.56568923e-03,
         1.52908960e-03,  1.07365137e-03, -1.11375401e-03,
         1.12090149e-03,  1.42050119e-03,  5.47164134e-04,
         2.31544209e-03,  6.80942164e-04, -1.32948461e-03,
         2.06320232e-04,  7.79779220e-04,  7.87415818e-04,
        -9.75853389e-04, -9.93019074e-04,  7.31763004e-05,
        -9.68472825e-04,  5.26942728e-04,  3.50284797e-04,
        -1.43173242e-04, -1.37202356e-03,  7.67932915e-05,
         1.24163898e-03,  3.58765603e-04, -2.52883357e-04,
        -1.09393610e-03,  2.36707086e-03,  1.56942825e-03,
        -1.73902787e-03,  1.11215826e-03,  2.30135318e-04,
        -1.93885109e-04,  1.29135485e-03, -1.55345002e-03,
         3.45356146e-04, -1.27342714e-03, -1.35630309e-03,
         1.86648878e-04, -8.09052175e-04, -1.58303885e-03,
         1.17824478e-04, -1.18235406e-03, -1.15386491e-03,
         1.57451671e-03, -6.66494856e-04,  1.69660341e-03,
         2.18771257e-04,  3.94845449e-04,  8.33926798e-04,
...
        -2.08046803e-04, -2.87883717e-04,  4.21432811e-04,
         1.29219369e-03, -1.43275218e-03,  1.86753602e-03,
         4.77852316e-04,  5.31267908e-04, -3.72327049e-04,
         1.43306282e-04,  3.02609275e-04,  7.52416179e-04,
         1.37838268e-03, -8.18640278e-04,  1.48423789e-04,
        -1.00935419e-03, -1.13116959e-03, -4.17694900e-04,
         6.62871152e-04,  1.72286031e-03,  4.61613220e-04,
         1.87496694e-03, -5.61025412e-04,  1.12895774e-03,
        -8.47261057e-04, -1.07084932e-03,  1.52862338e-03,
        -6.84715520e-05,  9.17961077e-04,  8.98014456e-04,
         9.94013704e-05, -6.10031518e-04, -7.33934955e-04,
         2.38902773e-04, -1.02078517e-03, -1.74564029e-03,
         2.31884342e-04, -7.29060875e-04,  9.02267481e-04,
         1.06292374e-03,  4.57995888e-04,  8.84121220e-04,
         5.62976017e-04,  2.69113943e-05, -2.66042817e-04,
         6.66982269e-04, -9.89704958e-04, -3.66327473e-04,
        -1.86945874e-04,  1.18506365e-03,  7.22683413e-05,
        -8.10807216e-04,  2.38711381e-04, -2.02769087e-04,
         7.69086665e-04,  1.69766468e-04, -7.37395123e-04,
         1.17449690e-03, -8.52196169e-04]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Observations_missing</span></div><div class='xr-var-dims'>(chain, Observations_missing_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.01274 0.04041 ... 0.01527</div><input id='attrs-3b73a28d-0fc9-4e39-89b8-9d4829eea4fe' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3b73a28d-0fc9-4e39-89b8-9d4829eea4fe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fe3c224c-1df3-4868-a3a0-c0d20dd2a847' class='xr-var-data-in' type='checkbox'><label for='data-fe3c224c-1df3-4868-a3a0-c0d20dd2a847' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-0.01274013,  0.04040889, -0.03358657, ...,  0.0255106 ,
         0.00842984,  0.01495069],
       [-0.01301428,  0.04049093, -0.03347883, ...,  0.02561845,
         0.00835608,  0.01527353]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_eof</span></div><div class='xr-var-dims'>(chain, sigma_eof_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.017 1.017</div><input id='attrs-8201a113-9131-4a22-94f3-b2c21db4a9b1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8201a113-9131-4a22-94f3-b2c21db4a9b1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eb475021-593c-4333-b12d-4acdc066cd13' class='xr-var-data-in' type='checkbox'><label for='data-eb475021-593c-4333-b12d-4acdc066cd13' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1.01658019],
       [1.0172251 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_hier</span></div><div class='xr-var-dims'>(chain, sigma_hier_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.004969 0.01504 0.004967 0.01506</div><input id='attrs-b5f4845d-5929-47ce-a95c-9844f1ff8f2c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b5f4845d-5929-47ce-a95c-9844f1ff8f2c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c8d69078-ea07-459f-bd2d-1c5ac56d8cb2' class='xr-var-data-in' type='checkbox'><label for='data-c8d69078-ea07-459f-bd2d-1c5ac56d8cb2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00496943, 0.01504201],
       [0.00496684, 0.0150554 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma</span></div><div class='xr-var-dims'>(chain, sigma_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.004969 0.004969 ... 0.01506</div><input id='attrs-55c7ab36-9991-46b9-9b1f-75c8070bf993' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-55c7ab36-9991-46b9-9b1f-75c8070bf993' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a719c384-b3ce-427b-9139-b0c9d2372b1a' class='xr-var-data-in' type='checkbox'><label for='data-a719c384-b3ce-427b-9139-b0c9d2372b1a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
        0.00496943, 0.00496943, 0.00496943, 0.00496943, 0.00496943,
...
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.00496684, 0.00496684,
        0.00496684, 0.00496684, 0.00496684, 0.0150554 , 0.0150554 ,
        0.0150554 , 0.0150554 , 0.0150554 , 0.0150554 , 0.0150554 ,
        0.0150554 , 0.0150554 , 0.0150554 , 0.0150554 , 0.0150554 ,
        0.0150554 , 0.0150554 , 0.0150554 , 0.0150554 , 0.0150554 ,
        0.0150554 , 0.0150554 , 0.0150554 , 0.0150554 , 0.0150554 ,
        0.0150554 , 0.0150554 , 0.0150554 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Estimates</span></div><div class='xr-var-dims'>(chain, Estimates_dim_0, Estimates_dim_1)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.01871 0.01605 ... -0.002448</div><input id='attrs-d39f6f56-40d4-41b9-a063-5552f4fe7a1f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d39f6f56-40d4-41b9-a063-5552f4fe7a1f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2fcfcffe-29c3-4b37-b9be-867bdb00b433' class='xr-var-data-in' type='checkbox'><label for='data-2fcfcffe-29c3-4b37-b9be-867bdb00b433' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 0.01870635,  0.01605429,  0.08586604, ..., -0.01744097,
         -0.05546656,  0.00887586],
        [ 0.01924005,  0.0156388 ,  0.08491993, ..., -0.01804235,
         -0.05416228,  0.00809648],
        [ 0.02055423,  0.0154459 ,  0.08580988, ..., -0.0194395 ,
         -0.05368419,  0.00714423],
        ...,
        [-0.00646523, -0.00409384, -0.02402438, ...,  0.00621155,
          0.01436375, -0.001337  ],
        [-0.00785668, -0.00505471, -0.02949748, ...,  0.00753664,
          0.01771131, -0.00171795],
        [-0.00814425, -0.00570439, -0.03237908, ...,  0.00775787,
          0.01989313, -0.00233328]],

       [[ 0.01871641,  0.01610491,  0.08557541, ..., -0.01724639,
         -0.05530758,  0.00931357],
        [ 0.01926138,  0.01569299,  0.08470361, ..., -0.01787155,
         -0.05402031,  0.00849302],
        [ 0.02057852,  0.01550548,  0.08566054, ..., -0.0192917 ,
         -0.05355687,  0.00748354],
        ...,
        [-0.00646525, -0.00411248, -0.02400614, ...,  0.00617348,
          0.0143316 , -0.00140494],
        [-0.00786102, -0.00507935, -0.02948564, ...,  0.00749867,
          0.01767808, -0.00180171],
        [-0.00814833, -0.0057251 , -0.03233376, ...,  0.00769852,
          0.01984408, -0.00244797]]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ea834e65-01ad-477f-8780-869a38b2c4ad' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-ea834e65-01ad-477f-8780-869a38b2c4ad' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



Recompute 2D fields and add to the dataset.


```python
bpca_object_exp2.estimated_dataset_map_pattern=[mean_files,std_files]
bpca_object_exp2.recombine_datasets(chain=chain,kind='maps')
```


```python
data_set_synt2=data_set_synt.sel({'x':(data_set_synt.lon>13) | (data_set_synt.lon<10)})

lon1 = data_set_synt2.x.lon.values
lat1 = data_set_synt2.x.lat.values

lon2 = bpca_object_exp2.estimated_dataset_map_pattern[0].lon.values
lat2 = bpca_object_exp2.estimated_dataset_map_pattern[0].lat.values
indices2 = np.argmin((np.repeat(lon1[:,np.newaxis],len(lon2),axis=1)-np.repeat(lon2[np.newaxis,:],len(lon1),axis=0))**2+(np.repeat(lat1[:,np.newaxis],len(lat2),axis=1)-np.repeat(lat2[np.newaxis,:],len(lat1),axis=0))**2,axis=1)
```


```python
plot_synthetic_data_maps(data_set_synt2,synthetic_data_settings,coastline,
                    validation=True,bpca_object=bpca_object_exp2,map_data=True,indices=indices2)
```


![png](bpca_tutorial_files/bpca_tutorial_75_0.png)


[back to top ](#top)
<hr>

## References
<a id='References'></a>


[1] Oelsmann, J., Marcos M., Passaro, M., Sánchez, L., Dettmering D., Dangendorf S., Seitz F. Vertical land motion reconstruction unveils non-linear effects on relative sea level changes from 1900-2150. Nat Geosciences, in review, 2023

[2] Oelsmann, J., Passaro, M., Sánchez, L. et al. Bayesian modelling of piecewise trends and discontinuities to improve the estimation of coastal vertical land motion. J Geod 96, 62 (2022). https://doi.org/10.1007/s00190-022-01645-6

[3] Wudong, L., Jiang, W.-P., Li, Z., Chen, H., Chen, Q., Wang, J., Zhu, G.: Extracting common mode errors of regional gnss position time series in the presence of missing data by variational bayesian principal component analysis. Sensors 20, 2298 (2020). https://doi.org/10.3390/s20082298

[4] Gruszczynski, M., Klos, A., Bogusz, J.: A Filtering of Incomplete GNSS Position Time Series with Probabilistic Principal Component Analysis. Pure and Applied Geophysics 175(5), 1841{1867 (2018). https://doi.org/10.1007/s00024-018-1856-3.


## Data sources

Blewitt G, Kreemer C, Hammond WC, Gazeaux J (2016) Midas robust trend estimator for accurate gps station velocities without step detection. Journal of Geophysical Research: Solid Earth 121(3):2054–2068, DOI 10.1002/2015JB01255    
    
Caron L, Ivins ER, Larour E, Adhikari S, Nilsson J, Blewitt G (2018) Gia model statistics for grace hydrology, cryosphere, and ocean science. Geophysical Research Letters 45(5):2203– 2212, DOI 10.1002/2017GL076644    
    
Frederikse T, Landerer F, Caron L, Adhikari S, Parkes D, Humphrey V, Dangendorf S, Hogarth P, Zanna L, Cheng L, Wu YH (2020) The causes of sea-level rise since 1900. Nature 584:393–397, DOI 10.1038/s41586-020-2591-3
    
Holgate SJ, Matthews A, Woodworth PL, Rickards LJ, Tamisiea ME, Bradshaw E, Fo-den  PR,  Gordon  KM,  Jevrejeva  S,  Pugh  J  (2013)  New  Data  Systems  and  Products at the  Permanent  Service  for  Mean  Sea  Level.  Journal  of  Coastal  Research  pp  493–504,  DOI  10.2112/JCOASTRES-D-12-00175.1,  URLhttps://doi.org/10.2112/JCOASTRES-D-12-00175.1    
    

[back to top ](#top)

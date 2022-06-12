# Analysis
import xarray as xr
import numpy as np
import pandas as pd

# General
from io import StringIO
from urllib.request import urlopen

# my stuff

from helpful_utilities.general import lowpass_butter

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'inline')

#Stats models
import statsmodels.formula.api as smf

# Functions:

def read_data_file(url):
    """
    Read in url as a data file.

    Parameters
    ----------
    url (string): url to convert to data file

    Returns
    -------
    decoded_html (string): url data in easily changeable form
    """
    with urlopen(url) as response:
        html_response = response.read()
        encoding = response.headers.get_content_charset('utf-8')
        decoded_html = html_response.decode(encoding)

    return decoded_html

def create_time_array(input_data, header=1):
    """
    Converts data into more easily navigable dataset by month and year.

    Parameters
    ----------
    input_data (string): unmodified temperature change data
    header (int): indicates what line to start at in order to cut off irrelevant heading text

    Returns
    -------
    da (xarray): sorted temperature change data
    """
    year = []
    data = []

    lines = input_data.split('\n')
    for line in lines[header:]:
        line_split = line.split()
        
        if len(line_split) == 13:

            year.append(int(line_split[0]))
            data.append(np.array(line_split[1:]).astype(float))
    
    start_date = '1/%04i' % year[0]
    end_date = '1/%04i' % (year[-1]+1)
    
    time = (pd.date_range(start=start_date, end=end_date, freq='M') +
            pd.DateOffset(days=-15))
 
    data = np.array(data).flatten()
    data[data == -99.99] = np.nan
    data[data == -9999] = np.nan
    da = xr.DataArray(data, dims='time', coords={'time': time})

    return da

def construct_lowpass_gmt_array(input_gmt):
    """
    Converts gmt into more easily navigable dataset by month and year.
    Also removes irrelevant gmt data and applies low-pass butterworth filter.

    Parameters
    ----------
    input_gmt (string): unmodified global mean temperature data

    Returns
    -------
    da_gmt_lp (xarray): sorted global mean temperature data
    """
    
    # reads gmt data as a file
    string_data = StringIO(input_gmt)
    
    # creates pandas dataframe from gmt data 
    gmt_data = pd.read_csv(string_data, comment='%', header=None, delim_whitespace=True).loc[:, :2]
    gmt_data.columns = ['year', 'month', 'GMTA']
    
    # drops the second half of dataframe, which infers sea ice T from water temperature
    stop_idx = np.where(gmt_data['year'] == gmt_data['year'][0])[0][12] - 1
    gmt_data = gmt_data.loc[:stop_idx, :]
    
    # labels start and end dates of each year
    if gmt_data.loc[:, 'month'].values[-1] == 12:
        gmt_time = pd.date_range(start='%04i-%02i' % (gmt_data.loc[0, 'year'], gmt_data.loc[0, 'month']), 
                                 end='%04i-%02i' % (gmt_data.loc[:, 'year'].values[-1] + 1, 
                                                1),
                                 freq='M') + pd.DateOffset(days=-15)
    else:
        gmt_time = pd.date_range(start='%04i-%02i' % (gmt_data.loc[0, 'year'], gmt_data.loc[0, 'month']), 
                                 end='%04i-%02i' % (gmt_data.loc[:, 'year'].values[-1], 
                                                    gmt_data.loc[:, 'year'].values[-1] + 1),
                                 freq='M') + pd.DateOffset(days=-15)

    da_gmt = xr.DataArray(gmt_data['GMTA'], dims='time', coords={'time': gmt_time})
    
    # applies lowpass butterworth filter to gmt values to flatten extraneous values
    gmt_lowpass = lowpass_butter(12, 1/10, 3, da_gmt.values)
    da_gmt_lp = da_gmt.copy(data=gmt_lowpass)

    return da_gmt_lp


def interpolate_to_daily(da):
    """
    Linearly estimates unknown daily mean temperatures based on monthly temperature data.

    Parameters
    ----------
    da (xarray): time array of temperature data

    Returns
    -------
    interpolated_da (xarray): 
    """
    interpolated_da = da.resample(time='1D').interpolate('linear')

    return interpolated_da


def only_relevant_times(da_day):
    """
    Shortens time scale and alters time labels to include only relevant dates.

    Parameters
    ----------
    da_day (xarray): daily-interpolated temperature data

    Returns
    -------
    labeled_time_array (xarray): daily temperature data with shortened time scale
    """
    labeled_time_array = da_day.sel({'time': (np.isin(da_day['time'], da_sst_ts['time']))})
    
    return labeled_time_array


def gram_schmidt(X):
    """
    Applies Gram-Schmidt process to distinguish correlation between different temperature events.

    Parameters
    ----------
    X (numpy array): temperature data for multiple temperature events

    Returns
    -------
    x (numpy array): orthogonalized temperature data
    """
    x = X[0]
    n = len(X)
    
    for i in range(n-1):
        x = x - (np.dot(x, X[i+1])/np.dot(X[i+1], X[i+1])*X[i+1])
    return x


def apply_quant_reg(formula, df):
    """
    Estimates quantile regression model for temperature data.

    Parameters
    ----------
    formula (string): formula specifying the regression model
    df (pandas dataframe): seasonal cycle for each temperature event and global mean temperature
    
    Returns
    -------
    df_qr (QuantReg): estimated quantile regression model for temperature data
    """
    df_qr = smf.quantreg(formula, df)

    return df_qr


def fit_quant_reg_model(mod):
    """
    Fits regression model for each quantile.

    Parameters
    ----------
    mod (QuantReg): quantile regression model

    Returns
    -------
    fit (RegressionModel): fitted regression model
    fits (list): fitted regression model for each quantile
    """
    fits = []
    for q in qs:
        fit = mod.fit(q=q, max_iter=10000)
        fits.append(fit)
    
    print(type(fit))
    return [fit, fits]


def get_predictors(fits, df, predictor_names):
    """
    Estimates predictors from quantile regression model.

    Parameters
    ----------
    fits (list): fitted regression model for each quantile
    df (pandas dataframe): seasonal cycle for each temperature event and global mean temperature
    predictor_names (list): names of (non-constant) predictors that should be estimated

    Returns
    -------
    beta0 (RegressionModel): fitted regression model
    beta_enso (list): enso predictor fitted from regression model
    beta_dmi (list): dmi predictor fitted from regression model
    beta_gmt (list): gmt predictor fitted from regression model
    yhat (list): temperatures predicted from regression model
    yhat_no_gmt (list): temperatures predicted from regression model with gmt predictor minimized
    """
    beta0 = []
    beta_enso = []
    beta_dmi = []
    beta_gmt = [] 
    yhat = []
    yhat_no_gmt = []
    
    for f in fits:
        beta0.append(f.params.Intercept)
        beta_enso.append(f.params['enso'])
        beta_dmi.append(f.params['dmi'])
        beta_gmt.append(f.params['gmt'])
        yhat.append(f.predict())
        
        # minimizes gmt parameter and appends all other predictor parameters to yhat_no_gmt
        yhat_no_gmt_element = f.params.Intercept + f.params['gmt']*df['gmt'].min()
        i = 0
        while i < len(predictor_names):
            yhat_no_gmt_element += f.params[predictor_names[i]]*df[predictor_names[i]]
            i += 1
            
        yhat_no_gmt.append(yhat_no_gmt_element)
            
    return [beta0, beta_enso, beta_dmi, beta_gmt, yhat, yhat_no_gmt]
    

def create_quantile_time_array(data):
    """
    Creates quantile- and time-dimension array.

    Parameters
    ----------
    data (list): predicted temperatures from quantile regression model

    Returns
    -------
    quantile_array (xarray): predicted temperature data with quantile and time dimensions
    """

    quantile_array = xr.DataArray(np.array(data),
                       dims=('quantile', 'time'), 
                       coords={'quantile': qs, 'time': da_sst_ts.time})
    return quantile_array


def get_legendre_poly_bases(qs):
    """
    Applies Gram-Schmidt process to Legendre polynomials to obtain Legendre bases.

    Parameters
    ----------
    qs (list): quantiles of interest

    Returns
    -------
    legendre_bases (numpy array): vector stack of legendre bases
    """
    P0 = np.ones((len(qs)))
    P1 = 2*qs - 1
    P2 = 0.5*(3*P1**2 - 1)
    P3 = 0.5*(5*P1**3 - 3*P1)
    P4 = 1/8*(35*P1**4 - 30*P1**2 + 3)
    P5 = 1/8*(63*P1**5 - 70*P1**3 + 15*P1)

    # Limited correlation remains between P1 and P3 due to limited sampling
    # Orthogonalize using Gram-Schmidt for better interpretability
    
    P3_orth = gram_schmidt([P3, P1])
    P4_orth = gram_schmidt([P4, P2])
    P5_orth = gram_schmidt([P5, P1, P3])

    legendre_bases = np.vstack((P0, P1, P2, P3_orth, P4_orth, P5_orth))
    return legendre_bases

def plot_coefficients(qs, q2, s, beta_data, ylabel):
    """
    Plots sensitivity coefficients of each percentile for each temperature event predictor.

    Parameters
    ----------
    qs (list): quantiles of interest
    q2 (int): text positioning argument
    s (string): text label
    beta_data (list): sensitivity of each percentile to temperature event predictor
    ylabel (string): y-axis label

    Returns
    -------
    None
    """ 
    fig, ax = plt.subplots(1, figsize=(5,3))
    
    
    fontsize = 16
    labelsize = 14
    
    ax.plot(qs, beta_data, '-sk')
    ax.set_ylabel(ylabel, fontsize = fontsize)
    ax.tick_params(labelsize = labelsize)
    ax.set_xlabel('Quantile', fontsize = fontsize)
    ax.text(0.02, q2, s, transform = ax.transAxes, fontsize=fontsize)

    
def plot_legendre_poly(bases):
    """
    Plots Legendre basis functions to summarize trends across percentiles.

    Parameters
    ----------
    bases (numpy array): vector stack of Legendre bases 

    Returns
    -------
    None
    """ 
    fig, ax = plt.subplots(1, figsize=(5,3))
    fontsize = 16
    labelsize = 14
    
    for i in range(4):
        ax.plot(qs, bases[i, :].T, label='L %i' % i)
    ax.tick_params(labelsize=labelsize)
    ax.set_xlabel('Quantile', fontsize=fontsize)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='lower right', ncol=4, fontsize=12)
    ax.text(0.02, 0.85, '(e)', transform=ax.transAxes, fontsize=fontsize)


# Actions and Annotations:


da_sst_ts = xr.open_dataarray('/home/data/NOAA_OISSTv2/indian_ocean_ts.nc')
da_sst_ts = da_sst_ts.sel({'time': da_sst_ts['time.year'] <= 2021})

dmi_f = read_data_file('https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data')
enso_f = read_data_file('https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data')
gmt_f = read_data_file("http://berkeleyearth.lbl.gov/auto/Global/Land_and_Ocean_complete.txt")

da_dmi = create_time_array(dmi_f)
da_enso = create_time_array(enso_f)
da_gmt_lp = construct_lowpass_gmt_array(gmt_f)

da_gmt_day = interpolate_to_daily(da_gmt_lp)
da_dmi_day = interpolate_to_daily(da_dmi)
da_enso_day = interpolate_to_daily(da_enso)

da_gmt_day_sub = only_relevant_times(da_gmt_day)
da_enso_day_sub = only_relevant_times(da_enso_day)
da_dmi_day_sub = only_relevant_times(da_dmi_day)

X = np.array([da_enso_day_sub, da_gmt_day_sub])
Y = np.array([da_dmi_day_sub, da_gmt_day_sub, da_enso_day_sub])

da_enso_day_orth = gram_schmidt(X)
da_dmi_day_orth = gram_schmidt(Y)

doy = da_sst_ts['time.dayofyear']
seasonal_cycle = pd.DataFrame({'sst': da_sst_ts,
                   's01': np.sin(2*np.pi*doy/365.25),
                   's02': np.cos(2*np.pi*doy/365.25),
                   's03': np.sin(4*np.pi*doy/365.25),
                   's04': np.cos(4*np.pi*doy/365.25),
                   's05': np.sin(6*np.pi*doy/365.25),
                   's06': np.cos(6*np.pi*doy/365.25),
                   'enso': da_enso_day_orth,
                   'dmi': da_dmi_day_orth,
                   'gmt': da_gmt_day_sub})

mod = apply_quant_reg('sst ~ s01 + s02 + s03 + s04 + s05 + s06 + enso + dmi + gmt', seasonal_cycle)

qs = np.arange(0.05, 1, 0.05) # define quantiles globally since they're used later on
qs = np.round(qs, 2)

fit = fit_quant_reg_model(mod)[0]
fits_list = fit_quant_reg_model(mod)[1]

predictor_names = ['s01', 's02', 's03', 's04', 's05', 's06', 'enso', 'dmi']

predictors = get_predictors(fits_list, seasonal_cycle, predictor_names)

beta0 = predictors[0]
beta_enso = predictors[1]
beta_dmi = predictors[2]
beta_gmt = predictors[3]
yhat = predictors[4]
yhat_no_gmt = predictors[5]

da_yhat = create_quantile_time_array(yhat)
da_yhat_noCC = create_quantile_time_array(yhat_no_gmt)

bases = get_legendre_poly_bases(qs)

# plot SST:
fig = plt.figure(figsize=(20, 8), constrained_layout=True)
fontsize = 16    
labelsize = 14
widths = [1, 1, 1, 1]
heights = [1, 0.6]
spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                        height_ratios=heights)
ax_ts = fig.add_subplot(spec[0, :])
this_ts = da_sst_ts.sel(time=slice('2014-09', '2017-09'))

da_yhat.sel({'time': slice('2014-09', '2017-09'), 'quantile': 0.9}).plot(ax=ax_ts, color='tab:red', 
                                                                         label='90th percentile')
da_yhat.sel({'time': slice('2014-09', '2017-09'), 'quantile': 0.5}).plot(ax=ax_ts, color='tab:orange', 
                                                                         label='50th percentile')
da_yhat.sel({'time': slice('2014-09', '2017-09'), 'quantile': 0.1}).plot(ax=ax_ts, color='tab:blue', 
                                                                         label='10th percentile')

da_yhat_noCC.sel({'time': slice('2014-09', '2017-09'), 'quantile': 0.9}).plot(ax=ax_ts, 
                                                                              color='tab:red', ls='--')
da_yhat_noCC.sel({'time': slice('2014-09', '2017-09'), 'quantile': 0.5}).plot(ax=ax_ts, 
                                                                              color='tab:orange', ls='--')
da_yhat_noCC.sel({'time': slice('2014-09', '2017-09'), 'quantile': 0.1}).plot(ax=ax_ts, 
                                                                              color='tab:blue', ls='--')
lower = da_yhat.sel({'time': slice('2014-09', '2017-09'), 'quantile': 0.9})
upper = np.max((lower, this_ts), axis=0)
ax_ts.fill_between(this_ts.time, lower, upper, color='tab:red')

lower = da_yhat_noCC.sel({'time': slice('2014-09', '2017-09'), 'quantile': 0.9})
upper = np.max((lower, this_ts), axis=0)
ax_ts.fill_between(this_ts.time, lower, upper, color='tab:red', alpha=0.5)
this_ts.plot(ax=ax_ts, color='k', label='Observations')
ax_ts.tick_params(labelsize=labelsize)
ax_ts.set_xlabel('')
ax_ts.set_ylabel('SST ($^\circ$C)', fontsize=fontsize)
ax_ts.legend(fontsize=fontsize, loc='upper right')
ax_ts.text(0.01, 0.05, 'Dashed = GMTA of 1981', transform=ax_ts.transAxes, fontsize=fontsize)
ax_ts.text(0.01, 0.9, '(a)', transform=ax_ts.transAxes, fontsize=fontsize)

plot_coefficients(qs, 0.85, '(b)', beta_enso, r'$\beta_\mathrm{Nino34}$')
plot_coefficients(qs, 0.02, '(c)', beta_dmi, r'$\beta_\mathrm{DMI}$')
plot_coefficients(qs, 0.02, '(d)', beta_gmt, r'$\beta_\mathrm{GMTA}$')

plot_legendre_poly(bases)
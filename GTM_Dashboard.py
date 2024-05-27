# =============================================================================
# libs list
# =============================================================================
# main ones
import polars as pl
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# others
import pandas as pd
import itertools
import math
# import random
import matplotlib



# =============================================================================
# Dashboard Config Setting & Header
# =============================================================================
st.set_page_config(
    page_title= 'GTM Dashboard'
    , page_icon= ':chart_with_upwards_trend:'
    , layout= 'wide'
    # ,initial_sidebar_state="collapsed"
    
)

st.title(':chart_with_upwards_trend: GTM Dashboard')
st.write('The Main Go-to-Market Dashboard for a New Deposit Service in Digital Banking')
# st.write('Implemented by Polars for Maximum Performance ( 10-100 faster than Pandas ) on a Single Cluster.')
# st.write('Keywords: Dashboard, GTM Strategy, Python (Polars, Streamlit, Dash, Plotly)')
# st.write(""" For the sake of demonstration, all the calculations are done on the spot when any change is requested.
#          Note: Although it is implemented in Polars, depolyment is on a public server; so from time to time ETL could take seconds ( using Pandas would be minutes).""")
# st.write("""You can zoom in/out on plots or sort data tables by each columns.""")
st.markdown('<style> div.block-container{padding-top:1rem;}</style>', unsafe_allow_html = True)


# =============================================================================
# Functions & Decorators & Constant Variables
# =============================================================================
def check_metric( func ):
    
    def wrapper( df , KPI_name , kpi_type = '€', delta_type = 'normal' ):
            
        if df.is_empty():
            # st.write('Empty')
            title = f"{freq} {KPI_name}"
            return 'No Record' , 'No prev. stat' , [ title , f'0 {kpi_type}', 'No prev. stat', delta_type]
        
        # st.write('not')
        return func(df , KPI_name , kpi_type, delta_type )
        
    return wrapper
    
    
@check_metric
def metric_rep ( df , KPI_name , kpi_type = '€', delta_type = 'normal' ):
    
    val , pct = df.row(-1)
    
    val_copy , pct_copy = val , pct
    
    
    if kpi_type =='%':
        val *= 100
        
    len_val = len(f'{val:.0f}')
    m = (len_val - 1)// 3 
    r = 2 - ( (len_val -1 )%  3)
    
    if kpi_type =='#' and m == 0 :
        r = 0     

    suffix= [ '' , 'K' , 'M' ,'B'][m]
    val *= ( K_frac ** m )
    
    
    
    
    len_pct = len(f'{abs(pct):.0f}') if pct!= None else "PP"
    m_pct = (len_pct - 1)// 3 
    r_pct = 2 - ( (len_pct -1 )%  3)

    suffix_pct= [ '' , 'K' , 'M' ,'B'][m_pct]
    pct *= ( K_frac ** m_pct )

    
    title = f"{freq} {KPI_name}"
    val = f"{val:03.{r}f}{suffix}{kpi_type}"
    pct = f"{pct:+.{r_pct}f}{suffix_pct} % vs. prev. {period_names[freq_index]}"\
        if pct != None else "No prev. stat"
    delta_type = delta_type if pct!= None else "off"
    
    return  val_copy , pct_copy, [title , val, pct, delta_type]

K_frac = 10**(-3)
M_frac = 10**(-6)
B_frac = 10**(-9)

period_cols= [  'quarter' , 'month' , 'week' , 'date' ]
period_names= [  'Quarter' , 'Month' , 'Week' , 'Day' ]

mperiod_cols= [ 'm'+x  for x in period_cols ]

period_prefixes = [  'Q' , 'M' , 'W', 'D' ]
period_truncs = [  '1q' , '1mo' , '1w', '1d' ]

period_labels = [  'Quarterly' , 'Monthly' , 'Weekly' , 'Daily' ]
period_index = { x: i for i ,x in enumerate(period_labels)}



# =============================================================================
# 
# Data Prepration
# 
# =============================================================================


if 'trxns_pdf' not in st.session_state:
    st.session_state.trxns_pdf = pl.scan_csv('deposit_trxns.csv')

    st.session_state.trxns_pdf = st.session_state.trxns_pdf.with_columns( 
        pl.col(
            [ 'acc_maturity_datetime' 
             , 'trxn_datetime' 
             , 'acc_opening_datetime'
        ]).str.to_datetime()
    )

    # st.session_state.trxns_pdf = st.session_state.trxns_pdf.sort(pl.col('trxn_datetime'))
    

    #
    st.session_state.trxns_pdf = st.session_state.trxns_pdf.with_columns( 
        pl.col('trxn_type').replace(
            {
                'open' : 'gain'
                ,'re-open' : 'gain'
                ,'cross-open' : 'gain'
                ,'renew' : 'gain'
                ,'top_up' : 'gain'
                ,'mature' : 'lose'
                ,'break' : 'lose'
                ,'withdraw' : 'lose'
            }
        ).alias('trxn_group')
    )


    # 

    st.session_state.trxns_pdf = st.session_state.trxns_pdf.with_columns( 
        pl.when( 
            pl.col('trxn_type') == 'renew'
            
        ).then( 
            pl.col('balance')
        ).otherwise(
            pl.col('tpv')
        ).alias('atpv') # artificial tpv, since renewal's tpv is 0
            
        , pl.when( 
            pl.col('trxn_type') == 'open'
        ).then( 
            1
        ).when( 
            pl.col('trxn_type') == 're-open'
        ).then( 
            1
        ).when( 
            pl.col('trxn_type') == 'cross-open'
        ).then( 
            1
        ).when( 
            pl.col('trxn_type') == 'break'
        ).then( 
            -1
        ).when( 
            pl.col('trxn_type') == 'mature'
        ).then( 
            -1
        ).otherwise(
            0
        ).alias('account_status') # artificial account value ( open and close! )
    )

            
    st.session_state.trxns_pdf = st.session_state.trxns_pdf.with_columns( 
        pl.col('account_status').cum_sum().over('user_id').alias('account_count')
    )
    
    st.session_state.trxns_pdf = st.session_state.trxns_pdf.with_columns( 
        pl.when( 
            ( pl.col('account_count') == 0 ) & ( pl.col('account_status') == -1 )
        ).then( 
            -1
        ).when( 
            ( pl.col('account_count') == 1 ) & ( pl.col('account_status') == 1 )
        ).then( 
            1
        ).otherwise(
            0
        ).alias('user_status') # artificial user value ( join or die! )
    )
    
    
    st.session_state.trxns_pdf = st.session_state.trxns_pdf.collect().lazy()

if 'rgsns_pdf' not in st.session_state:
    st.session_state.rgsns_pdf = pl.scan_csv('deposit_rgsns.csv')
    
    st.session_state.rgsns_pdf = st.session_state.rgsns_pdf.with_columns( 
        pl.col(
            [ 
             'trxn_datetime' 
             , 'signup_datetime'
        ]).str.to_datetime()
    )
    
    openings = st.session_state.trxns_pdf.filter(
        pl.col('trxn_type') == 'open' 
    ).select(
        pl.col( ['user_id' , 'trxn_id' ,'trxn_datetime' ] )    
    ).with_columns(pl.lit('activate').alias('trxn_type') )
        
    st.session_state.rgsns_pdf = pl.concat(
        [ st.session_state.rgsns_pdf , openings]
        , how = 'diagonal'
    ).sort(
        ['user_id' ,'trxn_datetime']
    ).with_columns( 
        pl.col(
            [
                'country','channel' 
                 , 'signup_datetime','customer_acquisition_cost'
            ]
       ).forward_fill() 
    )
    # st.session_state.rgsns_pdf.
    
    # st.session_state.rgsns_pdf = st.session_state.rgsns_pdf.sort(pl.col('trxn_datetime'))
    

    
if 'lcns_pdf' not in st.session_state:
    st.session_state.lcns_pdf = pl.scan_csv('deposit_lcns.csv')


    st.session_state.lcns_pdf = st.session_state.lcns_pdf.select( pl.all().sort_by('country' ) )
    st.session_state.lcns_pdf = st.session_state.lcns_pdf.with_columns( country_category = pl.lit('small'))

    st.session_state.lcns_pdf = st.session_state.lcns_pdf.with_columns(
    pl.when( pl.col('population') > 20 )
                .then(pl.lit('XBig'))
            .when( pl.col('population') > 10 )
                .then(pl.lit('Big'))
            .when( pl.col('population') > 5 )
                .then(pl.lit('Med'))
            .otherwise(pl.lit('Small'))
        .alias('country_category')
    ).rename( {'population' :'market_size'})
    st.session_state.lcns_pdf = st.session_state.lcns_pdf.with_columns( 
        pl.col('market_size').mul(M_frac).alias('market_size') 
        )
 

#

# if 'time_frame' not in st.session_state:
#     st.session_state.time_frame = st.session_state.trxns_pdf.select(
#         pl.col( ['date','week' ,'month' , 'quarter' , 'year'] ).unique()
#     )
    

# first_day = st.session_state.trxns_pdf.select('date').min().collect().to_series().min()
# last_day = st.session_state.trxns_pdf.select( 'date').max().collect().to_series().max()
# duration = (last_day - first_day).days
# 
# st.dataframe( st.session_state.rgsns_pdf.collect() )


# =============================================================================
# Dashboard Inputs
# =============================================================================

with st.expander("Filters:"):
    
    cols = st.columns(4)
    
    
    # date_container = cols[1].container()
    # all_dates = cols[1].checkbox("Whole Period",value=1 )
    
    # if 'first_day_filter' not in st.session_state:
    #     st.session_state['first_day_filter'] = first_day
    
    # if 'last_day_filter' not in st.session_state:
    #     st.session_state['last_day_filter'] = last_day
    
    # # first_day_filter , last_day_filter = first_day , last_day
    
    # if all_dates:
    #     date_container.date_input(
    #         'Analysis Interval'
    #         , value= (first_day , last_day)
    #         , min_value= first_day
    #         ,  max_value= last_day
    #         , disabled= True)
    #     st.session_state['first_day_filter'] = first_day
    #     st.session_state['last_day_filter'] = last_day

    # else:
    #     dates = date_container.date_input(
    #         'Analysis Interval'
    #         , value= (first_day , last_day)
    #         , min_value= first_day
    #         ,  max_value= last_day
    #     )
    #     if len(dates) == 1:
    #         st.session_state.first_day_filter = dates[0]
    #     else:
    #         st.session_state.first_day_filter , st.session_state.last_day_filter = dates
    
    # analysis_duration = (st.session_state.last_day_filter - st.session_state.first_day_filter).days + 1
    # freq_condition = [187 , 63, 15 , 1]
    # freq_condition = sum([ (analysis_duration//x) > 0 for x in freq_condition]) * -1
    
    freq_condition = 0
    freq = cols[0].selectbox('Frequency', period_labels[freq_condition:], index= 1  )
    freq_index = period_index[freq]
    # filtered_period_cols = [ period_cols[  freq_index ] ]
    calculation_list = [ freq ]
    
    # st.session_state.last_day_filter = cols[1].date_input(
    #     'Analysis Closing Date'
    #     , value=  last_day
    #     , min_value= first_day
    #     ,  max_value= last_day
    # )
    

    country_list = st.session_state.lcns_pdf.select('country').collect().to_pandas()['country'].tolist()
    location_container = cols[-3].container()
    all_countries = cols[-3].checkbox("All Locations",value=1 )
    selected_locations = []
     
    if all_countries:
        location_container.multiselect("Locations:",
             ['All'],['All'] , disabled= True)
        selected_locations = country_list
    else:
        selected_locations =  location_container.multiselect("Locations:",
            country_list)
        
    service_list = st.session_state.trxns_pdf.select('service').unique().collect().to_pandas().dropna()['service'].sort_values().tolist()
    service_container = cols[-2].container()
    all_services = cols[-2].checkbox("All Services",value=1 )
    selected_services = []
    if all_services:
        service_container.multiselect("Services:",
             ['All'],['All'] , disabled= True)
        selected_services = service_list
    
    else:
        selected_services =  service_container.multiselect("Services:",
            service_list)
        
    channel_list = st.session_state.rgsns_pdf.select('channel').unique().collect().to_pandas().dropna()['channel'].sort_values().tolist()
    channel_container = cols[-1].container()
    all_channels = cols[-1].checkbox("All Channels",value=1 )
    selected_channels = []
    if all_channels:
        channel_container.multiselect("Channels:",
             ['All'],['All'] , disabled= True)
        selected_channels = channel_list
    
    else:
        selected_channels =  channel_container.multiselect("Channels:",
            channel_list)
# =============================================================================
# Appling Filters
# =============================================================================

filtered_trxns = st.session_state.trxns_pdf

filtered_rgsns = st.session_state.rgsns_pdf



# filtered_trxns = filtered_trxns.filter( pl.col('trxn_datetime') >= st.session_state.first_day_filter )
# filtered_trxns = filtered_trxns.filter( pl.col('trxn_datetime') <= st.session_state.last_day_filter )

# not working with states cuase new feature extreaction is with state
if len(selected_locations) > 0:
    filtered_trxns = filtered_trxns.filter( pl.col('country').is_in( selected_locations))
    filtered_rgsns = filtered_rgsns.filter( pl.col('country').is_in( selected_locations))

# null is added to keep signups in the data
if len(selected_services) > 0:
    filtered_trxns = filtered_trxns.filter(
        pl.col('service').is_in( selected_services ) \
            | pl.col('service').is_null()
            )
if len(selected_channels) > 0:
    filtered_trxns = filtered_trxns.filter( pl.col('channel').is_in( selected_channels))
    filtered_rgsns = filtered_rgsns.filter( pl.col('channel').is_in( selected_channels))


    
channels = sorted ( filtered_rgsns.select(pl.col('channel') ).unique().collect().to_series().to_list() )
services = sorted ( filtered_trxns.select(pl.col('service') ).unique().collect().to_series().to_list() )

color_map = dict()
channel_colors = { x: 'blue' for x in channels }
color_map.update(channel_colors)
gains = ['activate','verify', 'signup','renew', 'open' ,'cross-open','re-open' , 'ongoing']
gain_colors = { x: 'green' for x in gains  }
color_map.update(gain_colors)
losses = ['reject' , 'mature', 'break' ,'dropped']
loss_colors = { x: 'red' for x in losses  }
color_map.update(loss_colors)
services_colors = { x: 'purple' for x in services + ['others']  }
color_map.update(services_colors)




# =============================================================================
# Feature Extraction - acquisition KPIs
# =============================================================================



# if 'acq_kpis' not in st.session_state:


acq_kpis = {}

acq_cr_kpis = {}

signups = filtered_rgsns#.filter( pl.col('trxn_type') != 'signup'  )
cols = [   'country'  ,'channel']

for  label in  calculation_list :
    
    index = period_index[label]
    trunc = period_truncs[index]
    date_column = 'date'
    
    signups = signups.with_columns(
        pl.col(
            'signup_datetime'
        ).dt.truncate(
            trunc 
        ).dt.strftime(
            '%y-%m-%d'
        ).alias(date_column)
    )
    
    combs = itertools.chain.from_iterable(
        itertools.combinations(cols, r) 
        for r in range( len(cols)+1 , -1 , -1)
    )


    kpis = [ 
        signups.group_by( pl.col( [ *x , 'trxn_type', date_column  ] )).agg( 
            pl.col('user_id').count().alias( 'U')
            , pl.col('customer_acquisition_cost').mean().alias('A_CAC')
            , (pl.col('trxn_datetime') - pl.col('signup_datetime')).dt.total_hours().mean().alias('A_TAT')
        ) for x in combs 
    ] 
    kpis = pl.concat(kpis , how= 'diagonal')
    kpis = kpis.with_columns(pl.col(cols).fill_null('all')).sort( pl.col( date_column ) )

    
    kpis = kpis.with_columns(
        pl.col([  *cols , 'trxn_type' , date_column ])
        , pl.col("U").cum_sum().over( [ * cols , 'trxn_type'] ).alias("cum_U")
    )
    
    
    cr_kpis = kpis.collect().pivot(
        index = ['country' , 'channel' ,'date' ]  
        , columns = 'trxn_type' 
        , values = 'U' 
    ).sort( pl.col( date_column ) ).fill_null(0)

    cr_kpis = cr_kpis.with_columns(
        ( pl.col('verify') / pl.col('signup') ).alias('CR_vs')
        , ( pl.col('activate') / pl.col('verify') ).alias('CR_av')
        , ( pl.col('activate') / pl.col('signup') ).alias('CR_as')
    )
    
    cr_kpis = cr_kpis.with_columns(
        pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
        .pct_change().over(['country' , 'channel']).mul(100)\
        .name.prefix("pct_change_")
    )
    

    kpis = kpis.with_columns(
        pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
        .pct_change().over( [* cols , 'trxn_type'] ).mul(100)\
        .name.prefix("pct_change_") 
    )
        
    acq_cr_kpis[ period_labels[index] ] = cr_kpis    
    acq_kpis[ period_labels[index] ] = kpis#.sort( date_column ).collect()

    
       
    

# =============================================================================
# Feature Extraction - transaction KPIs
# =============================================================================

# if 'trxn_kpis' not in st.session_state:

trxn_kpis = {}

cols = [   'country' , 'service' , 'trxn_group' ,'trxn_type'   ]
trxns = filtered_trxns

for  label in  calculation_list :
    
    index = period_index[label]
    trunc = period_truncs[index]
    date_column = 'date'

    trxns = trxns.with_columns(
        pl.col(
            'trxn_datetime'
        ).dt.truncate(
            trunc 
        ).dt.strftime(
            '%y-%m-%d'
        ).alias(date_column)
    )
    
    combs = itertools.chain.from_iterable(
        itertools.combinations(cols, r) 
        for r in range( len(cols)+1 , -1 , -1)
    )

    kpis = [ 
        trxns.group_by(  pl.col( [*x , date_column] ) ).agg( 
            pl.col('user_id').n_unique().alias( 'U')
            , pl.col('trxn_id').count().alias(  'T')
            , 
            pl.col('tpv').sum().alias(  'TPV')
            , pl.col('atpv').sum().alias(  'ATPV')

            , pl.col('user_status').sum().alias( 'AU')
            , pl.col('account_status').sum().alias( 'AA')
            , ( pl.col('user_status') > 0 ).sum().alias( 'U_ACTN')
            , ( pl.col('user_status') < 0 ).sum().alias( 'U_DCTN')
            , ( pl.col('account_status') > 0 ).sum().alias( 'A_ACTN')
            , ( pl.col('account_status') < 0 ).sum().alias( 'A_DCTN')
        ) for x in combs 
    ] 
    kpis = pl.concat(kpis , how= 'diagonal')
    kpis = kpis.with_columns(pl.col(cols).fill_null('all')).sort( pl.col(  date_column ) ) 
    
    kpis = kpis.with_columns(
        pl.col([ 'date', 'service' ,'trxn_type' , 'country'  ])
        , pl.col("TPV").cum_sum().over(cols).alias("DUM")
        , pl.col('AU').cum_sum().over(cols).alias("ACTIVE_C")
        , pl.col("AA").cum_sum().over(cols).alias("ACTIVE_A")
    )
    
    kpis = kpis.with_columns(
        ( pl.col(  'DUM') / pl.col(  'ACTIVE_C') ).fill_nan(0).alias(  'DUMpC')
        # , ( pl.col(  'T') / pl.col(  'U') ).alias(  'TpU')
        # , ( pl.col(  'cum_T') / pl.col(  'cum_U') ).alias(  'cum_TpU')
        # , ( pl.col(  'TPV') / pl.col(  'U') ).alias(  'TPVpU')
        # , ( pl.col(  'TPV') / pl.col(  'T') ).alias(  'TPVpT')
        # , ( pl.col(  'DUM') / pl.col(  'U') ).alias(  'DUMpC')
        # , ( pl.col(  'DUM') / pl.col(  'T') ).alias(  'DUMpT')
        
    )
    
    kpis = kpis.with_columns(
        pl.when( pl.col('DUMpC').is_infinite() ).then(0).otherwise(pl.col('DUMpC')).alias(  'DUMpC')
    )
    
    kpis = kpis.with_columns(
    pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
    .pct_change().over(cols).mul(100).round(1)\
    .name.prefix("pct_change_") )
    
    trxn_kpis[ period_labels[index] ] = kpis#.sort(by = 'date').collect()

# =============================================================================
# Feature Extraction - Due KPIs
# =============================================================================

# if 'due_kpis' not in st.session_state:


due_kpis = {}

cols = [ 'country' , 'service' , 'trxn_group' ,'trxn_type' ]
dues = filtered_trxns

for  label in  calculation_list :
    
    index = period_index[label]
    trunc = period_truncs[index]
    
    date_column = 'date'
    due_date_column = 'mdate'
    
    dues = dues.with_columns(
        pl.col(
            'trxn_datetime'
        ).dt.truncate(
            trunc 
        ).dt.strftime(
            '%y-%m-%d'
        ).alias(date_column)
            
        , pl.col(
            'acc_maturity_datetime'
        ).dt.truncate(
            trunc 
        ).dt.strftime(
            '%y-%m-%d'
        ).alias(due_date_column)
    )
    
    combs = itertools.chain.from_iterable(
        itertools.combinations(cols, r) 
        for r in range( len(cols)+1 , -1 , -1)
    )

    kpis = [ 
        dues.group_by(  pl.col( [ *x , due_date_column, date_column] ) ).agg( 
            pl.col('user_id').n_unique().alias( 'U')
            , pl.col('trxn_id').n_unique().alias(  'T')
            , pl.col('atpv').sum().alias(  'ATPV')
        ) for x in combs 
    ] 
    kpis = pl.concat(kpis , how= 'diagonal')
    kpis = kpis.with_columns(pl.col(cols).fill_null('all')).sort( pl.col( [due_date_column , date_column ]) )
   
    
    
    
    
    kpis = kpis.with_columns(
        pl.col([ due_date_column, date_column, *cols ])
        , pl.col("ATPV").cum_sum().over([ due_date_column ,  *cols ]).alias("DD")#.mul(-1)
    )
    
    # kpis = kpis.with_columns(
    #     ( pl.col(  'T') / pl.col(  'U') ).alias(  'TpU')
    #     , ( pl.col(  'DD') / pl.col(  'U') ).alias(  'DDpU')
    #     , ( pl.col(  'DD') / pl.col(  'T') ).alias(  'DDpT')
    # )
    
    kpis = kpis.filter( 
        pl.col(due_date_column) != pl.col(date_column)
    ).unique(
        subset = [due_date_column , *cols ] 
        , keep = "last"
    ).sort( pl.col( [ due_date_column , date_column ]) )
        
    kpis = kpis.with_columns(
        pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
        .pct_change().over( cols ).mul(100)\
        .name.prefix("pct_change_") 
    )
    
    
    
    due_kpis[ period_labels[index] ] = kpis#.sort(by = ['mdate', 'date']).collect()


# =============================================================================
# Let's Go
# =============================================================================
# if 'trxns' not in st.session_state:


trxns = trxn_kpis[freq].collect()
dues = due_kpis[freq].collect()
acqs = acq_kpis[freq].collect()
acqs_cr = acq_cr_kpis[freq]


if trxns.is_empty() or dues.is_empty():
    st.write('No Transaction Occured During This Interval.')
    st.write('Please Select Another Interval.')
elif acqs.is_empty():
    st.write('No Acquisition Occured During This Interval.')
    st.write('Please Select Another Interval.')
else:
    # st.dataframe(acqs)

    tabs_name = [
        "Main Metrics"
        , "Health Metrics"
        , "Acquisition Metrics"
        
        , "Market Expansion Plan"
    ]
    tabs = { k : v for k , v in zip( tabs_name, st.tabs(tabs_name) ) }

# =============================================================================
# North Star Metrics
# =============================================================================
    with tabs["Main Metrics"]:

        # st.header('Main Metrics')

        title_cols = st.columns(2)
        metric_cols = st.columns(4)
        fig_cols = st.columns(2)
        
        title_cols[0].subheader('North Star Metrics')

        final_trxns = trxns.filter( pl.col('date') == pl.col('date').max() )
        final_acqs = acqs.filter( pl.col('date') == pl.col('date').max() )
        final_acqs_cr = acqs_cr.filter( pl.col('date') == pl.col('date').max() )
        
        first_due = dues.filter( 
            pl.col('mdate') > pl.col('date').max()
        ).unique( subset=[ 'country' , 'service' , 'trxn_group' ,'trxn_type' , 'date'] , keep="first")    
        

        final_trxn_all = final_trxns.filter( 
            (pl.col('country') == 'all' )
            & ( pl.col('service') == 'all' )
            & ( pl.col('trxn_group') == 'all' )
            & ( pl.col('trxn_type') == 'all' )
        )#.sort('date')
        
        

        DUM , DUM_chng, rep_data = metric_rep(
            final_trxn_all.select( pl.col(['DUM', 'pct_change_DUM']) )
            , 'Deposits under Management'
        )
        metric_cols[0].metric(
            * rep_data
        )
        
        
        
        NACC , NACC_chng, rep_data = metric_rep( 
            final_trxn_all.select( pl.col(['A_ACTN', 'pct_change_A_ACTN']) )
            , 'Account Activation'
            , kpi_type='#'
        )
        metric_cols[1].metric(
        * rep_data
        )
        
        
        trxn_over_time_by_service = trxns.filter( 
            (pl.col('country') == 'all' )
            & ( pl.col('trxn_type') == 'all' )
            & ( pl.col('trxn_group') == 'all' )
        )
        

        # time_axis = trxn_over_time_by_service.select( pl.col('date').unique() ).to_series().to_list()
        # time_axis_sorted = sorted(time_axis) 
        
        fig = px.ecdf( 
            trxn_over_time_by_service.to_pandas()#.sort_values('date')
            , x= 'date' , y = "TPV" 
            , color = 'service' 
            , ecdfnorm = None
            , markers = True
            , color_discrete_map= { 
                'all' : 'green' ,'Fixed14' : 'purple' 
                , 'Flexible9' :'orange' , 'Locked14' : 'blue'
            }
            , title= freq + ' Deposits under Management')
    
        fig.for_each_yaxis(lambda y: y.update(title = 'Deposits under Management (€)'))
        
        
        fig_cols[0].plotly_chart(fig,use_container_width=True)
        

    # =============================================================================
    # Counter Metrics
    # =============================================================================

        title_cols[1].subheader('Counter Metrics')

            
        first_due_all = first_due.filter( 
            (pl.col('country') == 'all')
            & ( pl.col('service') == 'all')
            & (pl.col('trxn_group') == 'all')
            & (pl.col('trxn_type') == 'all')
        )
        
        DD ,DD_chng, rep_data = metric_rep( 
            first_due_all.select( pl.col(['DD', 'pct_change_DD']) )
            , 'Due Deposits'
            , delta_type = 'inverse'
        )
        
        
        metric_cols[2].metric(
            * rep_data
        )
        
        # 
        
        
        DACC ,DACC_chng, rep_data = metric_rep( 
            final_trxn_all.select( pl.col(['A_DCTN', 'pct_change_A_DCTN']))
            , 'Account Deactivation'
            , kpi_type='#'
            , delta_type = 'inverse'
        )
        
        metric_cols[3].metric(
            * rep_data
        )
        
        trxn_over_time_by_type = trxns.filter( 
            (pl.col('country') == 'all' )
            & ( pl.col('trxn_type').is_in( [ 'break', 'withdraw' , 'mature', 'renew'] ) )
            & ( pl.col('trxn_group') == 'all' )
            & ( pl.col('service') == 'all' )
        ).with_columns( (pl.col('ATPV').abs()).alias('pATPV') )
        
        due_all = dues.filter( 
            (pl.col('country') == 'all')
            & ( pl.col('service') == 'all')
            & (pl.col('trxn_group') == 'all')
            & (pl.col('trxn_type') == 'all')
        ).to_pandas()
        
        
        fig = px.bar( 
            trxn_over_time_by_type.to_pandas()
            , x = 'date' 
            , y = "pATPV" 
            , color = 'trxn_type' 
        
            , category_orders = { 
                'trxn_type' : ['mature' ,'break' , 'withdraw', 'renew'] 
                # , 'date' : time_axis_sorted
            }
            , color_discrete_map= { 
                'mature': 'blue' 
                , 'break':'red' 
                , 'withdraw' :'orange'
                , 'renew' : 'green'
            }  
            , labels= { 'pATPV': 'Due Deposits & Realized Transactions (€)'}
            , title = '{} Due Deposits & Realized Transactions'.format(freq)    
                            
        )
        
        
        fig.add_trace( go.Scatter(
                x = due_all.mdate
                , y = due_all.DD
                , name = 'Due Deposits'
                , mode= 'lines+markers'
                , line=dict(color='grey')
                , stackgroup='one'
                
                
            ) )
        
        fig_cols[1].plotly_chart(fig,use_container_width=True)

# =============================================================================
# Acquisition
# =============================================================================
    
    with tabs["Acquisition Metrics"]:

        
        # st.header( "Acquisition Metrics" )

        # title_cols = st.columns(2)
        acq_cols = st.columns([3.5,1,1,1])

        x_map = dict()
        level_0 = channels
        level_mapper = { x: 0 for x in  level_0 }
        x_map.update(level_mapper)
        level_1 = ['verify','reject']
        level_mapper = { x: 1 for x in  level_1 }
        x_map.update(level_mapper)
        level_2 = ['activate' , 'dropped']
        level_mapper = { x: 2 for x in  level_2 }
        x_map.update(level_mapper)
        level_3 =  [ *services , 'others']
        level_mapper =  { x: 3 for x in level_3 }
        x_map.update(level_mapper)
        
        
        extra_cols = []
        
        acquisition_flow = filtered_rgsns.filter(
            pl.col('trxn_type' ) != 'signup'
        ).select( 
            pl.col([* extra_cols  ,'user_id' , 'channel','trxn_type', 'trxn_datetime' ])
        )
        
        srvice_flow = filtered_trxns.filter(
            pl.col('trxn_type' ) == 'open'
        ).select( 
            pl.col([* extra_cols  ,'user_id' , 'trxn_datetime'])
            , pl.col('service').alias('trxn_type')
        )
        
        flow = pl.concat(
            [ acquisition_flow , srvice_flow ]
            , how = 'diagonal'
        ).with_columns(
            pl.col('trxn_type').shift(1).over([* extra_cols, 'user_id' ]).alias('source')
            , pl.col( 'trxn_type' ).alias('target')
        ).with_columns(
            pl.when( 
                pl.col('source').is_null() 
            ).then( pl.col('channel') ).otherwise(pl.col('source')).alias('source')
        ).group_by([* extra_cols  , 'source','target']).len().collect().to_pandas()   
        
        unmatched = flow.groupby('target').len.sum() - flow.groupby('source').len.sum()
        
        unmatched = unmatched[unmatched!= 0].dropna().rename('len').reset_index().rename(columns={'index':'source'})
        unmatched.loc[ unmatched.source.isin(level_1), 'target'] = 'dropped'
        unmatched.loc[ unmatched.source.isin(level_2), 'target'] = 'others'
        unmatched = unmatched.dropna()

        flow = pd.concat( [flow , unmatched] )
        
        flow = flow.sort_values(['source' , 'target'])
        
        nodes = flow.melt( 
            id_vars = extra_cols  , value_vars = ['source','target'] , value_name = 'node' 
        ).drop(columns = 'variable').drop_duplicates( ignore_index =True).reset_index() 
        
        flow = flow.reset_index().merge(
            nodes
            , how = 'left'
            , left_on = [* extra_cols  , 'source']
            , right_on = [* extra_cols  ,'node']
            , suffixes = ('' , '_source')
        )

        flow = flow.merge(
            nodes
            , how = 'left'
            , left_on = [* extra_cols  , 'target']
            , right_on = [* extra_cols  ,'node']
            , suffixes = ('' , '_target')
        )
        
        nodes['color'] = nodes.node.map(color_map)
        flow['color'] = flow.target.map(color_map)
        
        nodes['x'] = nodes.node.map(x_map)
        nodes['x'] = ( (nodes.x +1) / (nodes.x.max() + 2) ).clip(0.05 , 0.95)
        
        nodes['y'] = nodes.groupby('x').cumcount()
        nodes['y'] = nodes.groupby('x').y.transform( lambda x: ( x + 1 ) / ( x.max() + 2 )  ).fillna(0.5).clip(0.05 , 0.95)
        
        fig = go.Figure(
            go.Sankey( 
                arrangement='snap',
                node = dict(
                    pad = 20,
                    label = nodes.node,
                    color = nodes.color , 
                    x = nodes.x,
                    y = nodes.y,
                ),
                
                link = dict(
                    source = flow.index_source,  
                    target = flow.index_target,
                    value = flow.len
                    ,color= 'rgba' + flow.color.apply( matplotlib.colors.to_rgb).apply(lambda x: tuple([*x , 0.5])).astype(str)
                )
            )
        )
        
        fig.update_layout(title_text=f'Acquisition Flow', font_size=10)
        
        acq_cols[0].plotly_chart(fig , use_container_width=True)



    # =============================================================================
    # 
    # =============================================================================
        
        acq_cols[1].title("")
        acq_cols[1].title("")
        acq_cols[2].title("")
        acq_cols[2].title("")
        acq_cols[3].title("")
        acq_cols[3].title("")
        
        
        acq_all = final_acqs.filter( 
            ( pl.col('country') == 'all' ) 
            & ( pl.col('channel') == 'all' ) 
        )
        
        acq_ql_all = acq_all.filter(
            pl.col('trxn_type') == 'signup'
        )
        

        
        qualified_leads ,qualified_leads_chng, ql_rep = metric_rep(
            acq_ql_all.select(['U' , 'pct_change_U'] )
            , 'New Signups'
            , kpi_type = '#'
        )
        
        A_CAC , A_CAC_chng, cac_rep = metric_rep(
            acq_ql_all.select(['A_CAC', 'pct_change_A_CAC'])
            , 'avg. Customer Acq. Cost'
            , delta_type= 'inverse'
        )
        
        acq_vr_all = acq_all.filter(
            pl.col('trxn_type') == 'verify'
        )
        acq_op_all = acq_all.filter(
            pl.col('trxn_type') == 'activate'
        )
    

        verified_clients ,  verified_clients_chng, vc_rep = metric_rep( 
            acq_vr_all.select(['U' , 'pct_change_U' ])
            , 'New Verified Signups'
            , kpi_type = '#'
        )
        A_TAT , A_TAT_chng, tat_rep = metric_rep(
            acq_vr_all.select(['A_TAT', 'pct_change_A_TAT' ] )
            , 'avg. Turn Around Time'
            , kpi_type = 'Hours'
            , delta_type= 'inverse'
        )
        
        acq_cr_all = final_acqs_cr.filter( 
            ( pl.col('country') == 'all' ) 
            & ( pl.col('channel') == 'all' ) 
        )
        conversion_rate, conversion_rate_chng, crc_rep = metric_rep( 
            acq_cr_all.select([ 'CR_as', 'pct_change_CR_as'])
            , 'Conversion Rate'
            , kpi_type = '%'
        )
        
        
        # new_deposits, new_deposits_chng, nd_rep = metric_rep( 
        #     health_open.select('TPV' , 'pct_change_TPV' )
        #     , 'New Deposits'
        # )

        new_clients, new_clients_chng, nc_rep = metric_rep( 
            acq_op_all.select('U' , 'pct_change_U' )
            , 'New Clients'
            , kpi_type='#'
            
        )

        # acq_cols[1].title("")
        # acq_cols[1].title("")
        acq_cols[1].metric( * ql_rep )
        acq_cols[1].metric( * cac_rep )
        
        
        # acq_cols[2].title("")
        # acq_cols[2].title("")
        acq_cols[2].metric( * vc_rep )
        acq_cols[2].metric( * tat_rep )
        
        # acq_cols[3].title("")
        # acq_cols[3].title("")
        acq_cols[3].metric( * nc_rep )
        acq_cols[3].metric( * crc_rep )
        
        # acq_cols[2].metric('Expected Revenue', 1,1)
        
        # acq_cols[3].metric('New Deposits', 1,1)
    # metric_cols[3].metric('New Cross-Sell Accounts', 1,1)


    # =============================================================================
    # 
    # =============================================================================


        acq_cols = st.columns([3.5 ,3])
        
        final_segmented_acqs = final_acqs.filter( 
            (pl.col('country') != 'all')
            & (pl.col('channel') != 'all')
            & (pl.col('trxn_type').is_in( ['verify' ,'reject' ]) )
        ).with_columns( pl.col('trxn_type').replace( { 'verify' : 'Verified' , 'reject' : 'Rejected'}))

        # data = data.with_columns( pl.col('U')* K_frac) 
        
        
        fig = px.treemap(
            final_segmented_acqs.to_pandas()
            , path= [px.Constant('Europe'),'country' ,'channel' , 'trxn_type']
            , values = "U" 
            , color = 'A_CAC'
            
            , color_continuous_scale='rdylgn_r'
            ,labels=dict(
                
                market_size="Market Size (M#)"
                , A_CAC="Avg. CAC (€)"
                , U="New Users"
                , DUMpC="Depoists per Client (€)"
                , country = 'Country'
            )

            
        , title= 'Current {} Acquisitions by Segment'.format(period_names[freq_index])
        )
        # fig.data[0].customdata = data.to_pandas()['A_CAC'].tolist()

        fig.data[0].texttemplate = "%{label}<br>%{value:.0f} #<br>%{percentRoot}"
        
        
        # fig.update_traces(root_color="grey")
        fig.update_traces(legendgrouptitle_text="AA" )
        
        acq_cols[0].plotly_chart(fig,use_container_width=True)
        
        
        
    # =============================================================================
    #     
    # =============================================================================
        final_acqs_by_channel = final_acqs.filter(
            ( pl.col('trxn_type') != 'reject' )
            & ( pl.col('channel') != 'all' )
            & ( pl.col('country') == 'all' )
        ).sort('U' , descending = True).to_pandas()
        
        channel_list = final_acqs_by_channel.channel.unique()
        count = len(channel_list)
        
        rows = math.ceil( count / 3 )
        cols =  math.ceil(count / rows)
        
        fig = make_subplots( 
            rows = rows
            , cols = cols
            , specs = [[{"type": "funnelarea"} for _ in range(0, cols)] for _ in range(0, rows) ]
            , horizontal_spacing = 0.01 
            , vertical_spacing= 0.01
        )
        
        for i , x in enumerate( channel_list ):
            
            query = final_acqs_by_channel.query( ''' channel == @x ''')
            vals = query.U
            pcts = (vals / vals.max() * 100)
            fig.add_trace(
                go.Funnelarea(
                    values= vals
                    , text = pcts
                    , labels = query.trxn_type 
                    , title = x
                    , textinfo = "value+text"
                    , texttemplate= '%{value} # <br>%{text:.0f}%'

                    , marker= dict(colors=[  'orange', 'gold'  , 'green' ])
                    
                ) 
                , col = i % cols + 1
                , row = i // cols + 1
                
            )
        fig.update_layout(
            legend=dict(
                orientation = "h",
                yanchor = "top",
                y = 1.15 ,
                xanchor="left",
                # x = 1.2
            )
        ) 
        fig.update_layout(title = f'{period_labels[freq_index]} Acquisitions Funnel by Channel')
            
        acq_cols[1].plotly_chart(fig , use_container_width= True )

# =============================================================================
# Health Metrics
# =============================================================================

    with tabs["Health Metrics"]:

        # st.subheader( "Health Metrics" )
        
        health_cols = st.columns([3.5,1,1,1])


        # #####################    


        DUM_all = final_trxns.filter( 
            (pl.col('country') != 'all')
            & (pl.col('trxn_type') == 'all')
            & ( pl.col('trxn_group') == 'all' )
            & (pl.col('service') == 'all')
        ).select(['country', 'DUM' ]).to_pandas().set_index('country').sort_index()
        
        DD_all = first_due.filter( 
            (pl.col('country') != 'all')
            & (pl.col('trxn_type') == 'all')
            & ( pl.col('trxn_group') == 'all' )
            & (pl.col('service') == 'all')
        )
        DD_all = DD_all.filter( 
            pl.col('mdate') > pl.col('date').max()
        ).filter( 
            pl.col('mdate') == pl.col('mdate').min()
        ).select(['country', 'DD' ]).to_pandas().set_index('country').sort_index()  
        
        DD_to_DUM =  ( DD_all.squeeze() / DUM_all.squeeze() *100 ).rename('% DD/DUM').round(2).reset_index()

        
        fig = px.choropleth(  
            DD_to_DUM 
            , scope='europe'
            , locations="country"
            , color = '% DD/DUM'
            , color_continuous_scale="RdYlGn_r"
            , locationmode= 'country names'
            , title = 'Comming {} Due Deposits to Deposits under Management Ratio by Country'.format(period_names[freq_index])
        )
        fig.update_geos(fitbounds="locations", visible=True, bgcolor= 'rgba(0,0,0,0)')
        # fig.update(layout_coloraxis_showscale=False)
        # fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))

        health_cols[0].plotly_chart(fig,use_container_width=True)


        DUM = final_trxns.filter( 
            (pl.col('country') != 'all')
            & (pl.col('trxn_type') == 'all')
            & ( pl.col('trxn_group') == 'all' )
            & (pl.col('service') != 'all')
        ) 
        
        DUM = DUM.melt( id_vars= [ 'country' ,'service' ] , value_vars= ['DUM'] )
        

        
        DD = first_due.filter( 
            (pl.col('country') != 'all')
            & (pl.col('trxn_type') == 'all')
            & ( pl.col('trxn_group') == 'all' )
            & (pl.col('service') != 'all')
        )
        
        DD = DD.filter( 
            pl.col('mdate') > pl.col('date').max()
        ).unique( subset=[ 'country' , 'service' , 'date'] , keep="first")    
        

        DD = DD.melt( id_vars= [ 'country' ,'service'] , value_vars= ['DD'] )
        
        DD_DUM = pl.concat([ DD , DUM])
        
        DD_DUM = DD_DUM.filter( pl.col('value') != 0 )
        
        DD_DUM = DD_DUM.with_columns(
            pl.when( pl.col('variable') == 'DUM').then( pl.col('value').log10())\
                .otherwise(pl.col('value').log10().mul(-1) ).alias('signed_value')
                )
        DD_DUM = DD_DUM.with_columns( pl.col('value') * M_frac)
        
        fig = px.treemap(
            DD_DUM.to_pandas(), path= [px.Constant('Europe'),'country'  , 'service' ,'variable']
            , values = "value" 
            , color = 'signed_value'
            , color_continuous_scale='rdylgn'
            , color_continuous_midpoint= 0
            
            , title = '{} Deposits under Management & Due Deposits by Segment'.format(period_labels[freq_index])
            
            
        )    
        
        fig.update(layout_coloraxis_showscale=False)
        fig.data[0].texttemplate = "%{label}<br>%{value:.2f} M€<br>%{percentRoot}"
        
        
        st.plotly_chart(fig,use_container_width=True)

        # # #####################


        health_data = final_trxns.filter(
            (pl.col('service') == 'all')
            & (pl.col('country') == 'all')
        )
        # health_data.with_columns( pl.col('trxn_type').replace([ 'break' , 'mature' , 'withdraw'] , 'churn') )
        health_all = health_data.filter(
            ( pl.col('trxn_group') == 'all' ) 
            & ( pl.col('trxn_type') == 'all')
        ) 
        
        health_churn = health_data.filter(
            ( pl.col('trxn_group') == 'lose' ) 
            & ( pl.col('trxn_type') == 'all')
        ) 
        health_renew = health_data.filter(
            ( pl.col('trxn_group') == 'all' ) 
            & ( pl.col('trxn_type') == 'renew')
        ) 
        

        active_clients , active_clients_chng, ac_rep = metric_rep(
            health_all.select(['ACTIVE_C','pct_change_ACTIVE_C'])
            , 'Active Clients'
            , kpi_type= '#'
        )
        churned_clients , churned_clients_chng , cc_rep = metric_rep(
            health_churn.select(['U_DCTN','pct_change_U_DCTN'])
            , 'Churned Clients'
            , kpi_type= '#'
            , delta_type= 'inverse'
        )
        
        renewed_clients , renewed_clients_chng , rc_rep = metric_rep(
            health_renew.select(['U','pct_change_U'])
            , 'Retained Clients'
            , kpi_type= '#'
        )
        
        active_accounts , active_account_chng, aa_rep = metric_rep(
            health_all.select(['ACTIVE_A','pct_change_ACTIVE_A'])
            , 'Active Accounts'
            , kpi_type= '#'
        )
            
        churned_accounts , churned_account_chng, ca_rep = metric_rep(
            health_churn.select(['A_DCTN','pct_change_A_DCTN'])
            , 'Closed Accounts'
            , kpi_type= '#'
            , delta_type= 'inverse'
        )
            
        renewed_accounts , renewed_account_chng, ra_rep = metric_rep(
            health_renew.select(['T','pct_change_T'])
            , 'Renewed Accounts'
            , kpi_type= '#'
        )

        
        net_deposits , net_deposits_chng, nd_rep = metric_rep(
            health_all.select(['DUM','pct_change_DUM'])
            , 'Net Deposits'
        )
        
        health_churn = health_churn.with_columns(pl.col(['TPV']).mul(-1) )
        lost_deposits , lost_deposits_chng, ld_rep = metric_rep(
            health_churn.select(['TPV','pct_change_TPV'])
            , 'Lost Deposits'
            , delta_type= 'inverse'
        )
        
        renewed_deposits , renewed_deposits_chng, rd_rep = metric_rep(
            health_renew.select(['ATPV','pct_change_ATPV'])
            , 'Renewed Deposits'
        )
        
        health_cols[1].title("")
        health_cols[1].title("")
        
        health_cols[1].metric( * ac_rep )
        health_cols[1].metric( * cc_rep )
        health_cols[1].metric( * rc_rep )
        
        
        health_cols[2].title("")
        health_cols[2].title("")
        health_cols[2].metric( * aa_rep )
        health_cols[2].metric( * ca_rep )
        health_cols[2].metric( * ra_rep )
        
        health_cols[3].title("")
        health_cols[3].title("")
        health_cols[3].metric( * nd_rep )
        health_cols[3].metric( * ld_rep )
        health_cols[3].metric( * rd_rep )
        
    # =============================================================================
    #   Flow
    # =============================================================================
        

        flow_cols = st.columns([1] * len(services))

        x_map = dict()
        level_0 = channels
        level_mapper = { x: 0 for x in level_0 }
        x_map.update(level_mapper)
        # level_1 = services
        # level_mapper = { x: 1 for x in level_1 }
        # x_map.update(level_mapper)
        level_1 = [ 'open' , 'cross-open'  ]
        level_mapper = { x: 1 for x in level_1 }
        x_map.update(level_mapper)
        
        
        
        level_3 = ['renew' ]
        level_mapper = { x: 2 for x in level_3 }
        x_map.update(level_mapper)
        
        level_5 = ['ongoing' , 'mature', 'break' ]
        level_mapper = { x: 3 for x in level_5 }
        x_map.update(level_mapper)
        
        level_2 = ['re-open']
        level_mapper = { x: 4 for x in level_2 }
        x_map.update(level_mapper)
        
        extra_cols = [ 'service' ]

        for i , x in  enumerate( services ):
            
            service_pdf = filtered_trxns.filter( pl.col('service') == x )
            
            # channel_flow = service_pdf.filter(
            #     pl.col('trxn_type' ).is_in(['open' , 'cross-open'])
            # ).select( 
            #     pl.col(['user_id' , 'service' , 'trxn_datetime'])
            #     , pl.col('channel').alias('trxn_type')
        
            # )
            
            trxn_flow = service_pdf.filter(
                ~ pl.col('trxn_type' ).is_in(['top_up' , 'withdraw'])
            )
            
            flow = pl.concat( 
                [  trxn_flow]
                , how = 'diagonal'
            ).with_columns(
                pl.col('trxn_type').shift(1).over([* extra_cols, 'user_id' ]).alias('source')
                , pl.col( 'trxn_type' ).alias('target')
            ).with_columns(
                pl.when( 
                    pl.col('source').is_null() 
                ).then( pl.col('channel') ).otherwise(pl.col('source')).alias('source')
            ).group_by([* extra_cols, 'source','target']).len().drop_nulls().collect().to_pandas()
            

            unmatched = flow.groupby(
                [* extra_cols,'target']
                ).len.sum().rename_axis(
                    [ *extra_cols,'source']
                ) - flow.groupby(
                        [* extra_cols,'source']
                    ).len.sum()
            unmatched = unmatched[unmatched!= 0].dropna().rename('len').reset_index().rename(columns={'index':'source'})
            unmatched.loc[ unmatched.source.isin(gains), 'target'] = 'ongoing'
            unmatched = unmatched.dropna()
            flow = pd.concat( [flow , unmatched] )

            # unmatched.loc[ unmatched.source.isin(level_2), 'target'] = 'others'
        
            
            flow = flow.sort_values(['source' , 'target'])

            nodes = flow.melt( 
                id_vars = 'service' , value_vars = ['source','target'] , value_name = 'node' 
            ).drop(columns = 'variable').drop_duplicates( ignore_index =True).reset_index() 
            
            flow = flow.reset_index().merge(
                nodes
                , how = 'left'
                , left_on = ['service' , 'source']
                , right_on = ['service' ,'node']
                , suffixes = ('' , '_source')
            )
        
            flow = flow.merge(
                nodes
                , how = 'left'
                , left_on = ['service' , 'target']
                , right_on = ['service' ,'node']
                , suffixes = ('' , '_target')
            )
            
            nodes['color'] = nodes.node.map(color_map)
            flow['color'] = flow.target.map(color_map)

            nodes['x'] = nodes.node.map(x_map)
            nodes['x'] = ( (nodes.x +1) / (nodes.x.max() + 2) ).clip(0.05 , 0.95)
            
            nodes['y'] = nodes.groupby('x').cumcount()
            nodes['y'] = nodes.groupby('x').y.transform( lambda x: ( x + 1 ) / ( x.max() + 2 )  ).fillna(0.5).clip(0.05 , 0.95)
                
            nodes.loc[nodes.node == 're-open' , 'y'] = 0.33
            # nodes.loc[nodes.node == 're-open' , 'x'] = 0.9
            
            fig = go.Figure(
                go.Sankey(
                    arrangement='snap',
                    
                    node = dict(
                        pad = 10,
                        label = nodes.node,
                        color =  nodes.color
                        , x = nodes.x
                        , y = nodes.y
                    ),
                    
                    link = dict(
                        source = flow.index_source,  
                        target = flow.index_target,
                        value = flow.len
                        , color =  'rgba' + flow.color.apply( matplotlib.colors.to_rgb).apply(lambda x: tuple([*x , 0.5])).astype(str)
                        
            
                    )
                )
            )
            
            fig.update_layout(title_text=f'Clients Flow for {x} Service', font_size=10)
            
            
            flow_cols[i].plotly_chart(fig , use_container_width=True)
        
        # ==
            
            
            
        
# =============================================================================
# Expansion Streategy
# =============================================================================
    with tabs["Market Expansion Plan"]:

        # st.title( "Market Expansion Plan" )

        expansion_cols = st.columns([3.5 , 3])

        cac = acqs.filter(
            (pl.col('country') != 'all')
            & (pl.col('channel') == 'all')
            & (pl.col('trxn_type') == 'verify')
            # & (pl.col('date') == pl.col('date').max() )

        ).group_by( pl.col( 'country') ).agg( 
            pl.col( 'cum_U' ).last().alias('client_base')
            , pl.col( 'A_CAC' ).mean().alias('A_CAC') 
        )

        dum = trxns.filter(
            (pl.col('country') != 'all')
            & (pl.col('service') == 'all')
            & (pl.col('trxn_group') == 'all')
            & (pl.col('trxn_type') == 'all')
            # & (pl.col('date') == pl.col('date').max() )

        ).unique( 
            subset= 'country', keep = 'last'
        ).select( 
            pl.col( [ 'country' ,'service' , 'DUM' , 'ACTIVE_C', 'DUMpC'] ) 
        ).rename( { 'ACTIVE_C' : 'active_clients' } )

        pops = st.session_state.lcns_pdf.collect()

        penetration = pops.join( 
            dum , on = 'country' , how = 'left').join(
                cac , on = 'country' , how = 'left')

        penetration = penetration.with_columns(
            ( pl.col('client_base').mul(K_frac) /  pl.col('market_size') ).mul(100).alias('penetration_rate'))

        penetration = penetration.drop_nulls()
        fig = px.scatter(
            
            penetration.to_pandas()
            , x = 'A_CAC'
            , y = 'DUMpC'
            , size = 'market_size'
            , text= 'country'
            , color = 'penetration_rate'
            , title= "Market Opprotunity Plot (Size: Market Size)"
            ,labels=dict(
                
                market_size="Market Size (M#)"
                , A_CAC="Avg. Customer Acq. Cost (€)"
                , peneteration_rate="Peneteration Rate"
                , DUMpC="Depoists per Client (€)"
                , country = 'Country'
            )
            
            , height = 550
            , color_continuous_scale='rdylgn_r'
            
        )
        fig.update_layout( 
            coloraxis_colorbar={"title": 'Penetration Rate' }
        )
        fig.update_traces(textposition='top center')
        # fig.update_traces(textfont_color='black' , textfont_size = 9)
        fig.update_traces(textfont_size = 9)

        expansion_cols[0].plotly_chart(fig , use_container_width=True)

        penetration = penetration.select( 
            pl.col( [ 'country','market_size' ,'client_base' , 'active_clients' 
                    , 'DUM' , 'DUMpC', 'penetration_rate' , 'A_CAC'] ) 
        )
        penetration= penetration.with_columns(
            pl.col('client_base').mul(K_frac).alias('client_base')
            , pl.col('active_clients').mul(K_frac).alias('active_clients')
            , pl.col('DUM').mul(M_frac).alias('DUM')
            , pl.col('DUMpC').mul(K_frac).alias('DUMpC')
        ).rename( { 'country' : 'Country'
                ,'market_size' : 'Market Size'
                ,'client_base' : 'Client Base'
                , 'active_clients' : 'Active Clients'
                , 'DUM'  : 'Deposits under Management'
                , 'DUMpC' : 'Deposit per Client'
                , 'penetration_rate'  : 'Penetration Rate'
                , 'A_CAC' : 'Avg. Customer Acq. Cost'
                }
                    )

        f = {
            'Market Size':'{:.2f} M#'
            , 'Client Base':'{:.3f} K#'
            , 'Active Clients':'{:.3f} K#'
            , 'Deposits under Management':'{:.2f} M€'
            , 'Deposit per Client':'{:.2f} K€'
            , 'Avg. Customer Acq. Cost':'{:.2f} €'
            , 'Penetration Rate':'{:.2f} %'
            }
        expansion_cols[1].title("")
        expansion_cols[1].title("")
        expansion_cols[1].title("")
        penetration = penetration.to_pandas()
        penetration.index += 1
        expansion_cols[1].dataframe(
            penetration.style.format(f).highlight_max( 
                color = 'green', subset = ['Market Size' , 'Client Base' , 'Active Clients'] 
                ).highlight_max( 
                color = 'red' , subset = ['Penetration Rate' , 'Avg. Customer Acq. Cost' ] 
                ).highlight_max( 
                color = 'green' , subset = ['Deposits under Management' , 'Deposit per Client' ] 
                ).highlight_min( 
                color = 'red', subset = ['Market Size' , 'Client Base' , 'Active Clients'] 
                ).highlight_min( 
                color = 'green' , subset = ['Penetration Rate' , 'Avg. Customer Acq. Cost' ] 
                ).highlight_min( 
                color = 'red' , subset = ['Deposits under Management' , 'Deposit per Client' ] 
                )
            )


        # expansion_cols[1].dataframe(
        #     penetration.to_pandas().style.format(f).text_gradient( 
        #         axis = 0 , cmap = 'Blues' , subset = ['market_size' ]  , vmin = -50 , vmax = 100
        #         ).text_gradient( 
        #         axis = 0 , cmap = 'YlGn' , subset = [ 'client_base' , 'active_clients']  , vmin = -10 , vmax = 10
        #         ).text_gradient( 
        #         axis = 0 , cmap = 'RdYlGn_r' , subset = [ 'A_CAC' ]  , vmin = 5 #, vmax = 15
        #         ).text_gradient( 
        #         axis = 0 , cmap = 'RdYlGn_r' , subset = ['penetration_rate' ]  , vmin = 1  , vmax = 10
        #         ).text_gradient( 
        #         axis = 0 , cmap = 'YlGn' , subset = [ 'DUMpC' ] , vmin = -100 , vmax =200
        #         ).text_gradient( 
        #         axis = 0 , cmap = 'Greens' , subset = ['DUM' ] , vmin = -500 , vmax = 500
        #         )
        #     )



    # =============================================================================
    # Expansion Streategy by service type : Active Clients --> active accounts
    # =============================================================================


        expansion_cols = st.columns([3.5 , 3])

        
        dum = trxns.filter(
            (pl.col('country') != 'all')
            & (pl.col('service') != 'all')
            & (pl.col('trxn_group') == 'all')
            & (pl.col('trxn_type') == 'all')
            # & (pl.col('date') == pl.col('date').max() )

        ).unique( 
            subset= ['country' , 'service'], keep = 'last'
        ).select( 
            pl.col( [ 'country' ,'service' , 'DUM' , 'ACTIVE_A', 'DUMpC'] ) 
            ).rename( { 'ACTIVE_A' : 'active_clients' } )


        penetration = pops.join( 
            dum , on = 'country' , how = 'left').join(
                cac , on = 'country' , how = 'left')

        penetration = penetration.with_columns(
            ( pl.col('active_clients').mul(K_frac) /  pl.col('market_size') ).mul(100).alias('penetration_rate'))
        
        penetration = penetration.with_columns(
            ( pl.col('active_clients').mul(K_frac) /  pl.col('client_base') ).mul(100).alias('AC_CB'))
        
        penetration = penetration.drop_nulls()

        fig = px.scatter(
            
            penetration.to_pandas()
            , x = 'A_CAC'
            , y = 'DUMpC'
            , size = 'market_size'
            , text= 'country'
            , color = 'penetration_rate'
            , title= "Market Opprotunity Plot for each Service (Size: Market Size)"
            ,labels=dict(
                
                market_size="Market Size (M#)"
                , A_CAC="Avg. Customer Acq. Cost (€)"
                , peneteration_rate=" Peneteration Rate"
                , DUMpC=" Depoists per Client (€)"
                , country = 'Country'
                , service = 'Service'
            )
            , category_orders= { 'service' : [ 'Flexible9' , 'Locked14' ,'Fixed14']}
            , facet_col= 'service'
            # , facet_col_wrap= 2 
            , facet_col_spacing= 0.05
            # , facet_row_spacing= 0.03
            , height = 550
            # , width = 800
            , color_continuous_scale='rdylgn_r'
            
        )
        fig.update_layout( 
            coloraxis_colorbar={"title": 'Penetration Rate' }
        )
        fig.update_traces(textposition='top center')
        fig.update_traces( textfont_size = 9)
        # fig.update_traces(textfont_color='black' , textfont_size = 9)

        st.plotly_chart(fig , use_container_width=True)



    # Cross-selling
        product_list = penetration.select('service').unique().to_series().sort().to_list()
        combs = [ x for x in itertools.combinations( product_list, 2) ]

        relative_data = []

        for i , x in enumerate(combs):

            a = penetration.filter(
                pl.col('service') == x[0] 
                ).select(
                    pl.all().prefix('S1_') 
                ).with_columns( pl.lit(x[0]).alias('S1') )
            b = penetration.filter(
                pl.col('service') == x[1] 
                ).select( 
                    pl.all().prefix('S2_') 
                ).with_columns( pl.lit(x[1]).alias('S2') )
            c = a.join(b
                , left_on= 'S1_country'
                , right_on= 'S2_country'
            ).with_columns( pl.lit('S1(' + ') vs. S2('.join(x) + ')').alias('product_set') )
            relative_data.append(c)
        
        if len(relative_data ) == 0:
            st.write('Selected Filters Contain Not Enough Product for Cross-selling Analysis')
        else:
            relative_data = pl.concat( relative_data ) 
            dumpc = trxns.filter(
                (pl.col('country') != 'all')
                & (pl.col('service') == 'all')
                & (pl.col('trxn_group') == 'all')
                & (pl.col('trxn_type') == 'all')
                # & (pl.col('date') == pl.col('date').max() )
            ).unique( 
                subset= ['country' ], keep = 'last'
            ).select( 
                pl.col( [ 'country'  , 'DUMpC'] ) 
            )
            relative_data = relative_data.join(
                dumpc 
                , left_on = 'S1_country'
                , right_on = 'country'
            )
            
            
            fig = px.scatter(
                
                relative_data.to_pandas()
                , x = 'S1_AC_CB'
                , y = 'S2_AC_CB'
                , size = 'S1_market_size'
                , text= 'S1_country'
                , color = 'DUMpC' 
                , title= "Cross-sell Opprotunity Plot (Size: Market Size)"
                ,labels=dict(
                    
                    S1_market_size="Market Size (M#)"
                    , S1_A_CAC="Avg. Customer Acq. Cost (€)"
                    , S1_AC_CB="S1 Active Clients / Client Base"
                    , S2_AC_CB="S2 Active Clients / Client Base"
                    , S1_country = "Country"
                    , product_set = "Product Set"
                    , DUMpC = 'Deposit per Client'
                )
                , facet_col= 'product_set'
                # , facet_row= 'S1'
                # , facet_col_wrap= 2 
                , facet_col_spacing= 0.05 
                , facet_row_spacing= 0.03
                , height = 550
                # , width = 800
                , color_continuous_scale= [ 'yellow', 'green']
                
            )   
            # fig.update_layout( 
            #     coloraxis_colorbar={"title": 'Customer Acq. Cost (€)' }
            # )
            fig.update_traces(textposition='top center')
            fig.update_traces(textfont_size = 9)

            st.plotly_chart(fig , use_container_width=True)

            # st.dataframe( trxns.filter(
            #     ( pl.col('country')== 'Montenegro' )
            #     # & ( pl.col('service')== 'Fixed14' )
            #     # & ( pl.col('trxn_group')== 'all' )
            #     # & ( pl.col('trxn_type')== 'all' ) 
            #     )
            # )

# =============================================================================
# =============================================================================
# # 
# =============================================================================
# =============================================================================

  

# ==
    
    
# # =============================================================================
# # =============================================================================
# # #     
# # =============================================================================
# # =============================================================================


    
#     x_map = dict()
#     level_map = { x: 0 for x in channels }
#     x_map.update(level_0)
#     level = { x: 0 for x in services }
#     x_map.update(level_1)
#     level = { x: 1 for x in ['open' , 'cross-open', 're-open'] }
#     x_map.update(level)
#     level_3 = { x: 2 for x in ['renew'] }
#     x_map.update(level)
#     level = { x: 3 for x in ['ongoing' , 'mature', 'break'] }
#     x_map.update(level)
    
    
#     extra_cols = ['service']
    
#     srvice_flow = filtered_trxns.filter(
#         pl.col('trxn_type' ).is_in( ['open' , 'cross-open'])
#     ).select( 
#         pl.col([* extra_cols  ,'user_id' , 'trxn_datetime'])
#         , pl.col('service').alias('trxn_type')

#     )
    
#     trxn_flow = filtered_trxns.filter(
#         ~pl.col('trxn_type' ).is_in( ['open' , 'cross-open'])
#     ).filter(
#         ~ pl.col('trxn_type' ).is_in(['top_up' , 'withdraw'])
#     )
    
#     flow = pl.concat( 
#         [ srvice_flow , trxn_flow]
#         , how = 'diagonal'
#     ).with_columns(
#         pl.col( 'trxn_type' ).alias('source')
#         , pl.col('trxn_type').shift(-1).over(['user_id', 'service']).alias('target')
#     ).with_columns(
#         pl.when( 
#             pl.col('target').is_null() 
#             & ( pl.col('trxn_group') == 'gain' )
#         ).then( pl.lit('ongoing') ).otherwise(pl.col('target')).alias('target')
        
#     ).group_by(['service' , 'source','target']).len().drop_nulls().collect().to_pandas()
    
        
        
#     nodes = flow.melt( 
#         id_vars = 'service' , value_vars = ['source','target'] , value_name = 'node' 
#     ).drop(columns = 'variable').drop_duplicates( ignore_index =True).reset_index() 
    
#     flow = flow.reset_index().merge(
#         nodes
#         , how = 'left'
#         , left_on = ['service' , 'source']
#         , right_on = ['service' ,'node']
#         , suffixes = ('' , '_source')
#     )

#     flow = flow.merge(
#         nodes
#         , how = 'left'
#         , left_on = ['service' , 'target']
#         , right_on = ['service' ,'node']
#         , suffixes = ('' , '_target')
#     )
    
#     nodes['color'] = nodes.node.map(color_map)
    
#     flow['color'] = flow.target.map(color_map)

#     nodes['x'] = nodes.node.map(x_map)
#     nodes['x'] = ( (nodes.x +1) / (nodes.x.max() + 2) ).clip(0.05 , 0.95)
    
#     nodes['y'] = nodes.groupby('x').cumcount()

#     nodes['y'] = nodes.groupby('x').y.transform( lambda x: ( x + 1 ) / ( x.max() + 2 )  ).fillna(0.5).clip(0.05 , 0.95)
            
#     fig = go.Figure(
#         go.Sankey(
#             arrangement='snap',
            
#             node = dict(
#                 pad = 10,
#                 # thickness = 20,
#                 # line = dict(color = "black", width = 0.5),
#                 label = nodes.node,
#                 # color = "blue" ,
#                 # align="right"
#                 color =  nodes.color
#                 # , x = nodes.x
#                 # , y = nodes.y
#             ),
            
#             link = dict(
#                 # arrowlen=15,
#                 source = flow.index_source,  
#                 target = flow.index_target,
#                 value = flow.len
#                 , color =  'rgba' + flow.color.apply( matplotlib.colors.to_rgb).apply(lambda x: tuple([*x , 0.5])).astype(str)
                
      
#             )
#         )
#     )
    
#     fig.update_layout(title_text=f'Clients Flow for {x} Service', font_size=10)
    
    
#     st.plotly_chart(fig , use_container_width=True)

# # # ==
    
    
    
    
    
# =============================================================================
# =============================================================================
# # 
# =============================================================================
# =============================================================================

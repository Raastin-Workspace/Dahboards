# =============================================================================
# libs list
# =============================================================================
# main ones
import polars as pl
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
# others
import itertools
import functools
import datetime as dt


# =============================================================================
# Dashboard Config Setting
# =============================================================================
st.set_page_config(
    page_title= 'GTM Dashboard'
    , page_icon= ':chart_with_upwards_trend:'
    , layout= 'wide'
    # ,initial_sidebar_state="collapsed"
    
)

st.title(':chart_with_upwards_trend: GTM Dashboard')
st.write('The Main Go-to-Market Dashboard for new Deposit Service in Digital Banking')
st.write('Implemented by Polars (instead of Pandas) for Maximum Performance on a Single Cluster.')
st.write('Keywords: Dashboard, GTM Strategy, Python (Polars, Streamlit, Dash, Plotly)')
st.markdown('<style> div.block-container{padding-top:1rem;}</style>', unsafe_allow_html = True)


# =============================================================================
# Sidebar
# =============================================================================
# with st.sidebar.expander("Filters"):
    
#     f1 = st.file_uploader(":file_folder: Upload the file" , type = (['csv' , 'xlsx']))
    
    


# =============================================================================
# Data 
# =============================================================================
# pdf = pl.DataFrame()


# if f1 is None:
#     # show user message
#     pdf = pl.scan_csv('saving.csv')
#     # st.write('Please Upload Your File')
# else:

#     st.write('Your File Has Uploaded Succesfully')

    
#     file_name = f1.name
#     # st.write( ' File is {}'.format(file_name) )
#     pdf = pl.scan_csv( file_name)


pdf = pl.scan_csv('saving.csv')



# =============================================================================
# 
# Data Prepration
# 
# =============================================================================

# 
K_frac = 10**(-3)
M_frac = 10**(-6)
B_frac = 10**(-9)
 
pdf =  pdf.with_columns(pl.col("population") * M_frac)
# 
pdf = pdf.with_columns( pl.col([ 'acc_maturity_date' , 'trxn_datetime' , 'acc_opening_date']).str.to_datetime())

pdf = pdf.with_columns(
    pl.col('trxn_datetime').dt.date().alias('date')
    , pl.col('acc_maturity_date').dt.date().alias('mdate')
).sort( by = [ 'country' ,'service'  , 'date' , 'trxn_type'])



pdf = pdf.with_columns(  
    pl.col('date').dt.strftime("%y-W%W").alias('week')
    , pl.col('date').dt.strftime("%y-%m").alias('month')
    , pl.col('date').dt.quarter().alias('quarter')
    , pl.col('date').dt.strftime("%Y").alias('year')
    
    , pl.col('mdate').dt.strftime("%y-W%W").alias('mweek')
    , pl.col('mdate').dt.strftime("%y-%m").alias('mmonth')
    , pl.col('mdate').dt.quarter().alias('mquarter')
    , pl.col('mdate').dt.strftime("%Y").alias('myear')
)


pdf = pdf.with_columns(
    pl.concat_str(
        [
            pl.col("year")
            , pl.col("quarter")
        ]
        , separator="-Q",
    ).alias('quarter')
    
    , pl.concat_str(
        [
            pl.col("myear")
            , pl.col("mquarter")
        ]
        , separator="-Q",
    ).alias('mquarter')
)


# 

first_day = pdf.select('date').min().collect().to_series().min()
last_day = pdf.select( 'date').max().collect().to_series().max()
duration = (last_day - first_day).days
# 

pdf = pdf.with_columns( 
    pl.when( 
        pl.col('trxn_type') == 'renew'
           
    ).then( 
        pl.col('balance')
    ).otherwise(
        pl.col('tpv')
    ).alias('atpv') # artificial tpv, since renewal's tpv is 0
)
# 

pdf = pdf.collect().lazy()


# =============================================================================
# Dashboard Inputs
# =============================================================================
period_cols= [  'quarter' , 'month' , 'week' , 'date' ]
period_names= [  'Quarter' , 'Month' , 'Week' , 'Day' ]

mperiod_cols= [ 'm'+x  for x in period_cols ]

period_prefixes = [  'Q' , 'M' , 'W', 'D' ]
period_labels = [  'Quarterly' , 'Monthly' , 'Weekly' , 'Daily' ]
preiod_index = { x: i for i ,x in enumerate(period_labels)}



with st.expander("Filters:"):
    
    cols = st.columns(4)
    
    
    # date_container = cols[0].container()
    # all_dates = cols[0].checkbox("Whole Period",value=1 )
    # first_day_filter , last_day_filter = first_day , last_day
    
    # if all_dates:
    #     date_container.date_input(
    #         'Analysis Interval'
    #         , value= (first_day , last_day)
    #         , min_value= first_day
    #         ,  max_value= last_day
    #         , disabled= True)
    #     # first_day_filter , last_day_filter = first_day , last_day
    # else:
    #     dates = date_container.date_input(
    #         'Analysis Interval'
    #         , value= (first_day , last_day)
    #         , min_value= first_day
    #         ,  max_value= last_day
    #     )
    #     if len(dates) == 1:
    #         first_day_filter = dates[0]
    #     else:
    #         first_day_filter , last_day_filter = dates
    
    
    
    freq = cols[-4].selectbox('Frequency', period_labels, index= 2 )
    freq_index = preiod_index[freq]
    
    
    
    # last_day_filter = cols[1].date_input(
    #     'Analysis Closing Date'
    #     , value=  last_day
    #     , min_value= first_day
    #     ,  max_value= last_day
    # )
    
    
    country_list = pdf.select('country').unique().collect().to_pandas()['country'].sort_values().tolist()
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
        
    service_list = pdf.select('service').unique().collect().to_pandas().dropna()['service'].sort_values().tolist()
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
        
    channel_list = pdf.select('channel').unique().collect().to_pandas().dropna()['channel'].sort_values().tolist()
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


# pdf = pdf.filter( pl.col('trxn_datetime') >= first_day_filter )
# pdf = pdf.filter( pl.col('trxn_datetime') <= last_day_filter )

# st.dataframe(pdf.sort('trxn_datetime').collect())

  
    
if len(selected_locations) > 0:
    pdf = pdf.filter( pl.col('country').is_in( selected_locations))

# null is added to keep signups in the data
if len(selected_services) > 0:
    pdf = pdf.filter(
        pl.col('service').is_in( selected_services ) \
            | pl.col('service').is_null()
            )
if len(selected_channels) > 0:
    pdf = pdf.filter( pl.col('channel').is_in( selected_channels))


# st.write(selected_services + [ None ])
# st.dataframe(pdf.sort('trxn_datetime').collect())


# =============================================================================
# Feature Extraction - locations
# =============================================================================
plocations = pdf.select( pl.col(['country', 'population'  ]) ).unique().cache()
plocations = plocations.select( pl.all().sort_by('population' , descending=True) )
plocations = plocations.with_columns( country_category = pl.lit('small'))

plocations = plocations.with_columns(
pl.when( pl.col('population') > 20 )
            .then(pl.lit('XBig'))
        .when( pl.col('population') > 10 )
            .then(pl.lit('Big'))
        .when( pl.col('population') > 5 )
            .then(pl.lit('Med'))
        .otherwise(pl.lit('Small'))
    .alias('country_category')
)

# =============================================================================
# Feature Extraction - acquisition KPIs
# =============================================================================

cols = [   'country'  ,'channel', 'verified']

acq_kpis = {}

for i , period in enumerate( period_cols ):
    
    print(period)
    print ()

    
    combs = itertools.chain.from_iterable(
        itertools.combinations(cols, r) 
        for r in range( len(cols)+1 , -1 , -1)
    )

   
    kpis = [ 
        pdf.filter( pl.col('trxn_type').is_in(['signup' ] )).group_by( [*x ,period  ] ).agg( 
            pl.col('user_id').n_unique().alias( 'U')
            , pl.col('customer_acquisition_cost').mean().alias('A_CAC')
            , (pl.col('acc_opening_date') - pl.col('trxn_datetime')).dt.total_hours().mean().alias('A_TAT')
        ) for x in combs 
    ] 
    kpis = pl.concat(kpis , how= 'diagonal')
    kpis = kpis.with_columns(pl.col(cols).fill_null('all')).sort( by = cols + [ period ]  )

    kpis = kpis.rename({period : 'date'})
    
    kpis = kpis.with_columns(
        pl.col([ 'date', 'country' ,'channel' , 'verified'  ])
        , pl.col("U").cum_sum().over([ 'country' ,'channel' , 'verified']).alias("cum_U")
    )
    

    kpis = kpis.with_columns(
    pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
    .pct_change().over(cols).mul(100).round(1)\
    .name.prefix("pct_change_") )
        
    acq_kpis[ period_labels[i] ] = kpis.collect().lazy()

    
       
    

# =============================================================================
# Feature Extraction - transaction KPIs
# =============================================================================
cols = [   'country' , 'service' ,'trxn_type'   ]

trxn_kpis = {}

for i , period in enumerate( period_cols ):
    print(period)
    print ()
    
    combs = itertools.chain.from_iterable(
        itertools.combinations(cols, r) 
        for r in range( len(cols)+1 , -1 , -1)
    )

    kpis = [ 
        pdf.filter( ~ pl.col('trxn_type').is_in(['signup' ,'reject']) ).group_by(  [*x , period]  ).agg( 
            pl.col('user_id').n_unique().alias( 'U')
            , pl.col('trxn_id').count().alias(  'T')
            , pl.col('tpv').sum().alias(  'TPV')
            , pl.col('atpv').sum().alias(  'ATPV')
        ) for x in combs 
    ] 
    kpis = pl.concat(kpis , how= 'diagonal')
    kpis = kpis.with_columns(pl.col(cols).fill_null('all')).sort( by = cols  + [ period ])
    kpis = kpis.rename({period : 'date'})
    
#     if period != None :
#         kpis = kpis.rename({period : 'date'})
#         kpis = kpis.sort('date')
#     else:
#         kpis = kpis.rename({'literal' : 'date'}).with_columns( pl.col('date').fill_null(0) )
    
    
    kpis = kpis.with_columns(
        pl.col([ 'date', 'service' ,'trxn_type' , 'country'  ])
        , pl.col("TPV").cum_sum().over([ 'service' ,'trxn_type' , 'country']).alias("DUM")
        , pl.col("U").cum_sum().over([ 'service' ,'trxn_type' , 'country']).alias("cum_U")
        , pl.col("T").cum_sum().over([ 'service' ,'trxn_type' , 'country']).alias("cum_T")
    )
    
    kpis = kpis.with_columns(
        ( pl.col(  'T') / pl.col(  'U') ).alias(  'TpU')
        , ( pl.col(  'cum_T') / pl.col(  'cum_U') ).alias(  'cum_TpU')
        , ( pl.col(  'TPV') / pl.col(  'U') ).alias(  'TPVpU')
        , ( pl.col(  'TPV') / pl.col(  'T') ).alias(  'TPVpT')
        , ( pl.col(  'DUM') / pl.col(  'U') ).alias(  'DUMpU')
        , ( pl.col(  'DUM') / pl.col(  'T') ).alias(  'DUMpT')
    )
    
    kpis = kpis.with_columns(
    pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
    .pct_change().over(cols).mul(100).round(1)\
    .name.prefix("pct_change_") )
    
    trxn_kpis[ period_labels[i] ] = kpis

# =============================================================================
# Feature Extraction - Due KPIs
# =============================================================================



cols = [  'service' ,'trxn_type' , 'country'  ]
due_kpis = {}
for i , period in enumerate( period_cols ):
    
    print(period)
    print ()
    mperiod = 'm' + period
    combs = itertools.chain.from_iterable(
        itertools.combinations(cols, r) 
        for r in range( len(cols)+1 , -1 , -1)
    )

    kpis = [ 
        pdf.filter( ~ pl.col('trxn_type').is_in(['signup' ,'reject']) ).group_by(  [ *x , mperiod, period] ).agg( 
            pl.col('user_id').n_unique().alias( 'U')
            , pl.col('trxn_id').n_unique().alias(  'T')
            , pl.col('atpv').sum().alias(  'ATPV')
        ) for x in combs 
    ] 
    kpis = pl.concat(kpis , how= 'diagonal')
    kpis = kpis.with_columns(pl.col(cols).fill_null('all')).sort( by = cols  + [ mperiod , period ])
    kpis = kpis.rename({mperiod : 'mdate' , period : 'date'})
    
#     if period != None :
#         kpis = kpis.rename({period : 'date'})
#         kpis = kpis.sort('date')
#     else:
#         kpis = kpis.rename({'literal' : 'date'}).with_columns( pl.col('date').fill_null(0) )
    
    
    kpis = kpis.with_columns(
        pl.col([ 'mdate', 'date', 'service' ,'trxn_type' , 'country'  ])
        , pl.col("ATPV").cum_sum().over([ 'mdate' , 'service' ,'trxn_type' , 'country']).alias("DD")#.mul(-1)
    )
    
    kpis = kpis.with_columns(
        ( pl.col(  'T') / pl.col(  'U') ).alias(  'TpU')
        , ( pl.col(  'DD') / pl.col(  'U') ).alias(  'DDpU')
        , ( pl.col(  'DD') / pl.col(  'T') ).alias(  'DDpT')
    )

    
    due_kpis[ period_labels[i] ] = kpis


# =============================================================================
# Let's Go
# =============================================================================
trxns = trxn_kpis[freq].sort(by = 'date').collect()
dues = due_kpis[freq].sort(by = ['mdate', 'date']).collect()
acqs = acq_kpis[freq].sort(by = 'date').collect()

if trxns.is_empty() or dues.is_empty():
    st.write('No Transaction Occured During This Interval.')
    st.write('Please Select Another Interval.')
else:


    title_cols = st.columns(2)
    metric_cols = st.columns(4)
    fig_cols = st.columns(2)


# =============================================================================
# North Star Metrics
# =============================================================================


    title_cols[0].title('North Star Metrics')



    data = trxns
    
    
    country = ['all']
    trxn_type = ['all']
    service = ['all']
    
    data = data.filter( 
        (pl.col('country').is_in(country)  )
        & ( pl.col('trxn_type').is_in(trxn_type)  ) 
        & ( pl.col('service').is_in(service))
    ).sort('date')
    
    last = data.select( pl.all().last() )
    DUM ,DUM_chng    = last.select( pl.col(['DUM', 'pct_change_DUM']) ).row(0)
    
    DUM = round(DUM * B_frac, 2) 
    # st.write(DUM_chng)
    # if DUM_chng == None:
    #     st.write('yes')
    # DUM_chng = round(DUM_chng , 1)
    
    
    metric_cols[0].metric(
        "{} Deposits under Management ({}DUM)".format(
            freq, period_prefixes[freq_index])
        , "{} B€".format( DUM )
        , "{} % vs. prev. {}".format( 
            DUM_chng
            , period_names[freq_index] 
            ) if DUM_chng!= None else "No prev. stat"
        , delta_color= "normal" if DUM_chng!= None else "off"
    )
    # 
    data = trxns
    
    country = ['all']
    trxn_type = ['open']
    service = ['all']
    
    data = data.filter( 
        (pl.col('country').is_in(country)  )
        & ( pl.col('trxn_type').is_in(trxn_type)  ) 
        & ( pl.col('service').is_in(service))
    ).sort('date')
    
    last = data.select( pl.all().last() )
    NACC ,NACC_chng    = last.select( pl.col(['T', 'pct_change_T']) ).row(0)
    NACC = round(NACC * K_frac, 2) 
    # NACC_chng = round(NACC_chng , 1)
    
    metric_cols[1].metric(
        "{} Account Activation ({}AA)".format(
            freq, period_prefixes[freq_index])
        , "{} K#".format( NACC )
        , "{} % vs. prev. {}".format( 
            NACC_chng
            , period_names[freq_index]
            )  if NACC_chng != None else "No prev. stat"
        , delta_color= "normal" if DUM_chng!= None else "off"
    )
    
    
    data = trxns
    country = ['all']
    trxn_type = ['all']
    service = data.select('service').unique()#.sort(by ='service')
    data = data.filter( 
        (pl.col('country').is_in(country)  )
        & ( pl.col('trxn_type').is_in(trxn_type)  ) 
        & ( pl.col('service').is_in(service))
    ).sort('date').to_pandas()
    # 
    fig = px.ecdf( data, x= 'date' , y = "TPV" , color = 'service' ,   ecdfnorm=None, markers=True
                  , color_discrete_map= { 'all' : 'green' ,'Fixed14' : 'purple' , 'Flexible9' :'orange' , 'Locked14' : 'blue'}
    
                  , title= freq + ' Deposits under Management')
    fig.for_each_yaxis(lambda y: y.update(title = '€ DUM'))
    
    
    fig_cols[0].plotly_chart(fig,use_container_width=True)
    

# =============================================================================
# Counter Metrics
# =============================================================================

    title_cols[1].title('Counter Metrics')



        
    due_data = dues.filter( pl.col('country') == 'all')\
    .filter( pl.col('trxn_type') == 'all')\
    .filter( pl.col('service') == 'all')\
    .filter ( pl.col('mdate') != pl.col('date') )\
    .unique(subset='mdate', keep="last").sort( by = ['mdate','date'])
    
    cols = [  'service' ,'trxn_type' , 'country'  ]
    
    due_data = due_data.with_columns(
        pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
        .pct_change().over(cols).mul(100).round(1)\
        .name.prefix("pct_change_") 
        ).sort( by = ['mdate','date'])
    
    last = due_data.filter( 
        pl.col('mdate') > pl.col('date').max()
        ).unique(subset='date', keep="first")
    
    DD ,DD_chng    = last.select( pl.col(['DD', 'pct_change_DD']) ).row(0)
    DD = round(DD * M_frac, 1) 
    
    
    # DD_chng = round(DD_chng , 1)
    
    metric_cols[2].metric(
        "Comming {} Due Deposits ({}DD)".format(
            period_names[freq_index] , period_prefixes[freq_index])
        , "{} M€".format( DD )
        , "{} % vs. curr. {}".format( 
            DD_chng
            , period_names[freq_index]) if DD_chng != None else "No prev. stat"
        , delta_color= 'inverse'  if DD_chng!= None else "off"
    
    )
     
    # 
    data = trxns
    
    country = ['all']
    trxn_type = ['break']
    service = ['all']
    
    data = data.filter( 
        (pl.col('country').is_in(country)  )
        & ( pl.col('trxn_type').is_in(trxn_type)  ) 
        & ( pl.col('service').is_in(service))
    ).sort('date')
    
    last = data.select( pl.all().last() )
    DACC ,DACC_chng    = last.select( pl.col(['T', 'pct_change_T']) ).row(0)
    DACC = round(DACC * K_frac, 2) 
    # DACC_chng = round(DACC_chng , 1)
    
    metric_cols[3].metric(
        "{} Account Deactivation ({}AD)".format(
            freq, period_prefixes[freq_index])
        , "{} K#".format( DACC )
        , "{} % vs. prev. {}".format( 
            DACC_chng
            , period_names[freq_index]
            ) if DACC_chng!= None else "No prev. stat"
        , delta_color= 'inverse' if DACC_chng!= None else "off"
    )
    
    
    trxns_data = trxns.filter( 
        pl.col('trxn_type').is_in( [ 'break', 'withdraw' , 'mature', 'renew'] )
        & 
        pl.col('country').is_in( ['all'] )
        & pl.col('service').is_in( ['all'] )
    ).with_columns( (pl.col('ATPV').abs()).alias('pATPV') )
    
    due_data = due_data.to_pandas()
    
    trxns_data = trxns_data.to_pandas()
    
    
    fig = px.bar( 
        trxns_data 
        , x = 'date' 
        , y = "pATPV" 
        , color = 'trxn_type' 
       
        , category_orders = { 
            'trxn_type' : 
            ['mature' ,'break' , 'withdraw', 'renew'] 
        }
       
        , color_discrete_map= { 
            'mature': 'black' 
            , 'break':'red' 
            , 'withdraw' :'orange'
            , 'renew' : 'green'
        }  
        , title = '{} Due Deposits & Realized Transactions'.format(freq)    
                         
    )
    fig.for_each_yaxis(lambda y: y.update(title = '€ DD'))

    
    fig.add_trace( go.Scatter(
            x = due_data.mdate
            , y = due_data.DD
            , name = 'Due Deposits'
            , mode= 'lines+markers'
            ,line=dict(color='grey')
            , stackgroup='one'
            
             
        ) )
    
    fig_cols[1].plotly_chart(fig,use_container_width=True)


# =============================================================================
# Health Metrics
# =============================================================================

    
    st.title( "Health Metrics" )
    
    health_cols = st.columns([4,1,1,1])


    # #####################    

    DUM = trxns
    DUM = DUM.filter( pl.col('date') == pl.col('date').max() )
    DUM = DUM.filter( 
        (pl.col('country') != 'all')
        & (pl.col('trxn_type') == 'all')
        & (pl.col('service') != 'all')
    ) 
    
    DUM = DUM.melt( id_vars= [ 'country' ,'service' ,'trxn_type'] , value_vars= ['DUM'] ,)
    
    
    DD = dues
    DD = DD.filter( 
        (pl.col('country') != 'all')
        & (pl.col('trxn_type') == 'all')
        & (pl.col('service') != 'all')
    )
    DD = DD.filter ( pl.col('mdate') == pl.col('date').max() )
    # DD = DD.filter ( pl.col('date') != pl.col('date').max() )
    DD = DD.unique(subset=[ 'country' ,'service' ,'trxn_type','mdate'], keep="last")

    DD = DD.melt( id_vars= [ 'country' ,'service' ,'trxn_type'] , value_vars= ['DD'] )
    data = pl.concat([DUM , DD])
    
    data = data.filter( pl.col('value') != 0 )
    data = data.with_columns(
        pl.when( pl.col('variable') == 'DUM').then( pl.col('value').log10())\
            .otherwise(pl.col('value').log10().mul(-1) ).alias('signed_value')
            )
    data = data.with_columns( pl.col('value') * M_frac)
    
    fig = px.treemap(
        data.to_pandas(), path= [px.Constant('Europe'),'country'  , 'service' ,'variable']
        , values = "value" 
        , color = 'signed_value'
        , color_continuous_scale='rdylgn'
        , color_continuous_midpoint= 0
        
        , title = 'DUM & DD by Segment(click for zoom in/out)'
        
        
    )
    
    fig.update(layout_coloraxis_showscale=False)
    # fig.data[0].texttemplate = "%{label}<br>%{value:.2f} M€<br>%{percentRoot}"
    
    
    health_cols[0].plotly_chart(fig,use_container_width=True)



    # #####################
    trxns_pivot_clients = trxns.pivot(index = ['country' , 'service' ,'date' ]  , columns = 'trxn_type' , values = 'U').fill_null(0)
    trxns_pivot_clients = trxns_pivot_clients.with_columns(
        pl.sum_horizontal(['mature','break']).alias('churned')
    )
    
    trxns_pivot_clients = trxns_pivot_clients.with_columns(
        ( pl.col('open') - pl.col('churned') ).alias('active')
    )
    
    trxns_pivot_clients = trxns_pivot_clients.with_columns(
            pl.col([  'service'  , 'country' , 'date'  ])
            , pl.selectors.by_dtype(pl.NUMERIC_DTYPES ).cum_sum().over([  'service'  , 'country' ]).name.prefix("cum_")
        )
    
    trxns_pivot_clients = trxns_pivot_clients.with_columns(
        pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
        .pct_change().over(['country' , 'service']).mul(100).round(1)\
        .name.prefix("pct_change_") )
    
    
    # #####################    
    trxns_pivot_accounts = trxns.pivot(index = ['country' , 'service' ,'date' ]  , columns = 'trxn_type' , values = 'T').fill_null(0)
    trxns_pivot_accounts = trxns_pivot_accounts.with_columns(
        pl.sum_horizontal(['mature','break']).alias('churned')
    )
    
    trxns_pivot_accounts = trxns_pivot_accounts.with_columns(
        ( pl.col('open') - pl.col('churned') ).alias('active')
    )
    
    trxns_pivot_accounts = trxns_pivot_accounts.with_columns(
            pl.col([  'service'  , 'country' , 'date'  ])
            , pl.selectors.by_dtype(pl.NUMERIC_DTYPES ).cum_sum().over([  'service'  , 'country' ]).name.prefix("cum_")
        )
    
    trxns_pivot_accounts = trxns_pivot_accounts.with_columns(
        pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
        .pct_change().over(['country' , 'service']).mul(100).round(1)\
        .name.prefix("pct_change_") )
    
    
        
    # #####################    
    trxns_pivot_accounts = trxns.pivot(index = ['country' , 'service' ,'date' ]  , columns = 'trxn_type' , values = 'T').fill_null(0)
    trxns_pivot_accounts = trxns_pivot_accounts.with_columns(
        pl.sum_horizontal(['mature','break']).alias('churned')
    )
    
    trxns_pivot_accounts = trxns_pivot_accounts.with_columns(
        ( pl.col('open') - pl.col('churned') ).alias('active')
    )
    
    trxns_pivot_accounts = trxns_pivot_accounts.with_columns(
            pl.col([  'service'  , 'country' , 'date'  ])
            , pl.selectors.by_dtype(pl.NUMERIC_DTYPES ).cum_sum().over([  'service'  , 'country' ]).name.prefix("cum_")
        )
    
    trxns_pivot_accounts = trxns_pivot_accounts.with_columns(
        pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
        .pct_change().over(['country' , 'service']).mul(100).round(1)\
        .name.prefix("pct_change_") )
    
    # #####################    
        
    trxns_pivot_deposits = trxns.pivot(index = ['country' , 'service' ,'date' ]  , columns = 'trxn_type' , values = 'ATPV')
    trxns_pivot_deposits = trxns_pivot_deposits.with_columns(
        pl.sum_horizontal(['withdraw','mature','break']).abs().alias('lost')
        , pl.sum_horizontal(['open','top_up']).alias('gained')
        
    )
    
    trxns_pivot_deposits = trxns_pivot_deposits.with_columns(
        ( pl.col('gained') - pl.col('lost') ).alias('net')
    )
    
    trxns_pivot_deposits = trxns_pivot_deposits.with_columns(
            pl.col([  'service'  , 'country' , 'date'  ])
            , pl.selectors.by_dtype(pl.NUMERIC_DTYPES ).cum_sum().over([  'service'  , 'country' ]).name.prefix("cum_")
        )
    
    trxns_pivot_deposits = trxns_pivot_deposits.with_columns(
        pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
        .pct_change().over(['country' , 'service']).mul(100).round(1)\
        .name.prefix("pct_change_") )
    
    
    # #####################  
    
    trxns_pivot_clients = trxns_pivot_clients.filter( pl.col('date') == pl.col('date').max() )
    trxns_pivot_clients = trxns_pivot_clients.filter( 
        (pl.col('country') == 'all')
        & (pl.col('service') == 'all')
    ).tail(1)
    
    trxns_pivot_accounts = trxns_pivot_accounts.filter( pl.col('date') == pl.col('date').max() )
    trxns_pivot_accounts = trxns_pivot_accounts.filter( 
        (pl.col('country') == 'all')
        & (pl.col('service') == 'all')
    ).tail(1)
    
    trxns_pivot_deposits = trxns_pivot_deposits.filter( pl.col('date') == pl.col('date').max() )
    trxns_pivot_deposits = trxns_pivot_deposits.filter( 
        (pl.col('country') == 'all')
        & (pl.col('service') == 'all')
    ).tail(1)
    # #####################
    
    active_clients , active_clients_chng = trxns_pivot_clients.select(['cum_active','pct_change_cum_active']).row(-1)    
    churned_clients , churned_clients_chng = trxns_pivot_clients.select(['churned','pct_change_churned']).row(-1)
    renewed_clients , renewed_clients_chng = trxns_pivot_clients.select(['renew','pct_change_renew']).row(-1)    
    
    active_accounts , active_account_chng = trxns_pivot_accounts.select(['cum_active','pct_change_cum_active']).row(-1)
    churned_accounts , churned_account_chng = trxns_pivot_accounts.select(['churned','pct_change_churned']).row(-1)
    renewed_accounts , renewed_account_chng = trxns_pivot_accounts.select(['renew','pct_change_renew']).row(-1)

    
    net_deposits , net_deposits_chng = trxns_pivot_deposits.select(['cum_net','pct_change_cum_net']).row(-1)
    lost_deposits , lost_deposits_chng = trxns_pivot_deposits.select(['lost','pct_change_lost']).row(-1)
    renewed_deposits , renewed_deposits_chng = trxns_pivot_deposits.select(['renew','pct_change_renew']).row(-1)

    # #####################

    active_clients *= K_frac
    churned_clients *= K_frac
    renewed_clients *= K_frac
    active_accounts *= K_frac
    churned_accounts *= K_frac
    renewed_accounts *= K_frac
    
    net_deposits *= M_frac
    lost_deposits *= M_frac
    renewed_deposits *= M_frac
    
    health_cols[1].title("")
    health_cols[1].title("")
    health_cols[2].title("")
    health_cols[2].title("")
    health_cols[3].title("")
    health_cols[3].title("")
    
    health_cols[1].metric(
        'Active Clients', '{:.1f} K#'.format(active_clients) 
        , '{:.1f} % vs. prev. {}'.format( 
            active_clients_chng
            , period_names[freq_index]
            ) if active_clients_chng!= None else "No prev. stat"
    
        , delta_color= "normal" if active_clients_chng!= None else "off"
    )
    
    health_cols[1].metric(
        '{} Churned Clients'.format(period_labels[freq_index]), '{:.1f} K#'.format(churned_clients) 
        , '{:.1f} % vs. prev. {}'.format( 
            churned_clients_chng
            , period_names[freq_index]
            ) if churned_clients_chng!= None else "No prev. stat"
        , delta_color= 'inverse' if churned_clients_chng!= None else "off"
           
    )
    
    health_cols[1].metric(
        '{} Retained Clients'.format(period_labels[freq_index]), '{:.1f} K#'.format(renewed_clients) 
        , '{:.1f} % vs. prev. {}'.format( 
            renewed_clients_chng
            , period_names[freq_index]
            ) if renewed_clients_chng!= None else "No prev. stat"
        
        , delta_color= "normal" if active_clients_chng!= None else "off" 
    )
    
    
    health_cols[2].metric(
        'Active Accounts', '{:.1f} K#'.format(active_accounts) 
        , '{:.1f} % vs. prev. {}'.format( 
            active_account_chng
            , period_names[freq_index]
            ) if active_account_chng!= None else "No prev. stat"
        , delta_color= "normal" if active_account_chng!= None else "off"
    )
    health_cols[2].metric(
        '{} Closed Accounts'.format(period_labels[freq_index]), '{:.1f} K#'.format( churned_accounts) 
        , '{:.1f} % vs. prev. {}'.format( 
            churned_account_chng
            , period_names[freq_index]
            ) if churned_account_chng!= None else "No prev. stat"
        , delta_color= 'inverse' if churned_account_chng!= None else "off"
        
    )
    health_cols[2].metric(
        '{} Renewed Accounts'.format(period_labels[freq_index]), '{:.1f} K#'.format(renewed_accounts) 
        , '{:.1f} % vs. prev. {}'.format( 
            renewed_account_chng
            , period_names[freq_index]
            ) if renewed_account_chng!= None else "No prev. stat"
        , delta_color= "normal" if renewed_account_chng!= None else "off"
        
           
    )
    
    health_cols[3].metric(
        'Net Deposits (DUM)', '{:.1f} M€'.format(net_deposits) 
        , '{:.1f} % vs. prev. {}'.format( 
            net_deposits_chng
            , period_names[freq_index]
            ) if net_deposits_chng!= None else "No prev. stat"
        , delta_color= "normal" if net_deposits_chng!= None else "off"
    )
    health_cols[3].metric(
        '{} Lost Deposits'.format(period_labels[freq_index]), '{:.1f} M€'.format( lost_deposits) 
        , '{:.1f} % vs. prev. {}'.format( 
            lost_deposits_chng
            , period_names[freq_index]
            ) if lost_deposits_chng!= None else "No prev. stat"
        , delta_color= 'inverse' if lost_deposits_chng!= None else "off"
        
    )
    health_cols[3].metric(
        '{} Renewed Deposits'.format(period_labels[freq_index]), '{:.1f} M€'.format(renewed_deposits) 
        , '{:.1f} % vs. prev. {}'.format( 
            renewed_deposits_chng
            , period_names[freq_index]
            ) if renewed_deposits_chng!= None else "No prev. stat"
        , delta_color= "normal" if renewed_deposits_chng!= None else "off"
        
           
    )
    
    
# =============================================================================
# Acquisition
# =============================================================================
    

    st.title( "Acquisition Metrics" )
    
    acq_cols = st.columns([3.5 ,1,1,1])
    
    data = acqs.filter( 
        (pl.col('country') != 'all')
        & (pl.col('verified') != 'all')
        & (pl.col('channel') != 'all')
    ).with_columns( pl.col('verified').replace( { 'Yes' : 'Verified' , 'No' : 'Rejected'}))
    
    data = data.with_columns( pl.col('U') * K_frac)
    
    
    
    fig = px.treemap(
        data.to_pandas(), path= [px.Constant('Europe'),'country' ,'channel' , 'verified']
        , values = "U" 
        , color = 'A_CAC'
        
        , color_continuous_scale='rdylgn_r'

        
    , title= 'Current {} Acquisitions ( & avg. customer acq. cost: A_CAC)'.format(period_names[freq_index])
    )
    
    fig.data[0].texttemplate = "%{label}<br>%{value:.2f} K#<br>%{percentRoot}"
    
    
    # fig.update_traces(root_color="grey")
    fig.update_traces(legendgrouptitle_text="AA" )
    
    acq_cols[0].plotly_chart(fig,use_container_width=True)


# =============================================================================
# 
# =============================================================================
    
    acq_cols[1].title("")
    acq_cols[1].title("")
    acq_cols[2].title("")
    acq_cols[2].title("")
    acq_cols[3].title("")
    acq_cols[3].title("")
    
    acqs_pivot = acqs.pivot(index = ['country' , 'channel' ,'date' ]  , columns = 'verified' , values = 'U').fill_null(0)

    acqs_pivot = acqs_pivot.with_columns(
        ( pl.col('Yes') / pl.col('all') ).alias('CR')
    )
    
    acqs_pivot = acqs_pivot.with_columns(
        pl.selectors.by_dtype(pl.NUMERIC_DTYPES)\
        .pct_change().over(['country' , 'channel']).mul(100).round(1)\
        .name.prefix("pct_change_") )
        
        
    
    acqs_pivot = acqs_pivot.filter( pl.col('date') == pl.col('date').max() )
    acqs_pivot = acqs_pivot.filter( pl.col('country') == 'all' )
    acqs_pivot = acqs_pivot.filter( pl.col('channel') == 'all' )


    data = acqs
    
    data = data.filter( pl.col('date') == pl.col('date').max() )
    data = data.filter( pl.col('country') == 'all' )
    data = data.filter( pl.col('channel') == 'all' )
    
    qualified_leads ,qualified_leads_chng , A_TAT , A_TAT_chng , A_CAC , A_CAC_chng= data.filter(
        pl.col('verified') == 'all' ).select(
            ['U' , 'pct_change_U', 'A_TAT', 'pct_change_A_TAT' , 'A_CAC', 'pct_change_A_CAC']).row(0)
   
    verified_clients ,  verified_clients_chng = data.filter( pl.col('verified') == 'Yes' ).select(['U' , 'pct_change_U' ]).row(0)
    conversion_rate = verified_clients / qualified_leads * 100
    
    conversion_rate_chng ,  = acqs_pivot.select('pct_change_CR').row(0)
    
    new_deposits, new_deposits_chng = trxns_pivot_deposits.select('open' , 'pct_change_open' ).row(0)

    qualified_leads *= K_frac
    verified_clients *= K_frac
    new_deposits *= M_frac
    
    acq_cols[1].metric(
        'New Qualified Leads'
        , '{:.2f} K#'.format(qualified_leads)
        , '{:.1f} % vs. prev. {}'.format( 
            qualified_leads_chng
            , period_names[freq_index]
            ) if qualified_leads_chng!= None else "No prev. stat"
    
        , delta_color= "normal" if qualified_leads_chng!= None else "off"
    )
    
    acq_cols[1].metric(
        'avg. Turn Around Time'
        ,'{:.1f} Hours'.format(A_TAT) 
        , '{:.1f} % vs. prev. {}'.format( 
            A_TAT_chng
            , period_names[freq_index]
            ) if A_TAT_chng!= None else "No prev. stat"
        , delta_color= "inverse" if A_TAT_chng!= None else "off"
    )
    acq_cols[1].metric(
        'avg. Customer Acq. Cost'
        ,'{:.1f} €'.format(A_CAC) 
        , '{:.1f} % vs. prev. {}'.format( 
            A_CAC_chng
            , period_names[freq_index]
            ) if A_CAC_chng!= None else "No prev. stat"
        , delta_color= "inverse" if A_CAC_chng!= None else "off"
    )
    
    acq_cols[2].metric(
        'New Clients'
        ,'{:.2f} K#'.format(verified_clients)
        , '{:.1f} % vs. prev. {}'.format( 
            verified_clients_chng
            , period_names[freq_index]
            ) if verified_clients_chng!= None else "No prev. stat"
    
        , delta_color= "normal" if verified_clients_chng!= None else "off"
    )
    acq_cols[2].metric(
        'Conversion Rate'
        ,'{:.1f} %'.format(conversion_rate)
        , '{:.1f} % vs. prev. {}'.format( 
            conversion_rate_chng
            , period_names[freq_index]
            ) if conversion_rate_chng!= None else "No prev. stat"
    
        , delta_color= "normal" if conversion_rate_chng!= None else "off"
        )
    acq_cols[3].metric(
        'New Deposits'
        ,'{:.1f} M€'.format(new_deposits)
        , '{:.1f} % vs. prev. {}'.format( 
            new_deposits_chng
            , period_names[freq_index]
            ) if new_deposits_chng!= None else "No prev. stat"
    
        , delta_color= "normal" if new_deposits_chng!= None else "off"
        )
    # acq_cols[2].metric('Expected Revenue', 1,1)
    
    # acq_cols[3].metric('New Deposits', 1,1)
# metric_cols[3].metric('New Cross-Sell Accounts', 1,1)



! pip install plotly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import itertools
import functools

import polars as pl
import datetime as dt


st.set_page_config(
    page_title= 'Dashboard'
    , page_icon= ':chart_with_upwards_trend:'
    , layout= 'wide'
    # ,initial_sidebar_state="collapsed"
    
)

st.title(':chart_with_upwards_trend: GTM Dashboard')

st.markdown('<style> div.block-container{padding-top:1rem;}</style>', unsafe_allow_html = True)


with st.sidebar.expander("Filters"):
    
    f1 = st.file_uploader(":file_folder: Upload the file" , type = (['csv' , 'xlsx']))
    
    last_day_filter = st.date_input('last_day' 
        , value= dt.date(2023, 10, 1)
        , min_value= dt.date(2023, 1, 1)
        ,  max_value= dt.date(2024, 1, 1)
    )



pdf = pl.DataFrame()


if f1 is None:
    # show user message
    pdf = pl.read_csv('saving.csv')
    st.write('Please Upload Your File')
else:

    st.write('Your File Has Uploaded Succesfully')

    
    file_name = f1.name
    # st.write( ' File is {}'.format(file_name) )
    pdf = pl.read_csv( file_name)



pdf = pdf.filter( pl.col('trxn_datetime') < str(last_day_filter ))
# =============================================================================
# 
# 
# 
# =============================================================================

period_cols= [ None , 'year' , 'quarter' , 'month' , 'week' , 'date']
period_prefixes = [ '' , 'Y' , 'Q' , 'M' , 'W' , 'D']
period_labels = [ 'Overal' , 'Yearly' , 'Quarterly' , 'Monthly' , 'Weekly' , 'Daily']

population_frac =   1 / 10**6


pdf =  pdf.with_columns(pl.col("population") * population_frac)


plocations = pdf.select( pl.col(['country', 'population'  ]) ).unique()
plocations = plocations.select( pl.all().sort_by('population' , descending=True) )
plocations = plocations.with_columns( country_category = pl.lit('small'))

plocations = plocations.with_columns(
    pl.when( pl.col('population') > 20 )
                .then(4)
            .when( pl.col('population') > 10 )
                .then(3)
            .when( pl.col('population') > 5 )
                .then(2)
            .otherwise(1)
        .alias('country_category')
)


trxns = pdf

trxns = trxns.with_columns( pl.col('trxn_datetime').str.to_datetime())
trxns = trxns.with_columns(  pl.col('trxn_datetime').dt.date().alias('date') )
trxns = trxns.sort( by = 'date')


first_day = trxns.select( pl.col('date').min()).row(0)[0]
last_day = trxns.select( pl.col('date').max()).row(0)[0]
duration = (last_day - first_day).days

time_frame = pl.date_range(
    first_day
    , last_day
    , interval = '1d'
    , eager= True
).alias('date').to_frame()



trxns = trxns.with_columns(  pl.col('date').dt.strftime("%y-W%W").alias('week') )
trxns = trxns.with_columns(  pl.col('date').dt.strftime("%y-%m").alias('month') )
trxns = trxns.with_columns(  pl.col('date').dt.quarter().alias('quarter') )
trxns = trxns.with_columns(  pl.col('date').dt.strftime("%Y").alias('year') )

trxns = trxns.with_columns(
    pl.concat_str(
        [
            pl.col("year")
            , pl.col("quarter")


        ],
        separator="-Q",
    ).alias('quarter')
)

# =============================================================================
# 
# =============================================================================

cols = [  'service' ,'trxn_type' , 'country'  ]
combs = itertools.chain.from_iterable(itertools.combinations(cols, r) for r in range( len(cols)+1 , 0 , -1))

agg_kpis = {}
for i , period in enumerate( period_cols ):
    print(period)
    print ()
    
    combs = itertools.chain.from_iterable(itertools.combinations(cols, r) for r in range( len(cols)+1 , -1 , -1))

    kpis = [ 
        trxns.group_by( [period , *x ] ).agg( 
            pl.col('user_id').n_unique().alias( 'U')
            , pl.col('trxn_id').n_unique().alias(  'T')
            , pl.col('tpv').sum().alias(  'TPV')
            , pl.col('balance').sum().alias(  'B')
        ) for x in combs 
    ] 
    
    kpis = pl.concat(kpis , how= 'diagonal')
    kpis = kpis.with_columns(pl.col(cols).fill_null('all'))

    if period != None :
        kpis = kpis.rename({period : 'date'})
        kpis = kpis.sort('date')
    else:
        kpis = kpis.rename({'literal' : 'date'}).with_columns( pl.col('date').fill_null(0) )
    
    
    kpis = kpis.with_columns(
        pl.col([ 'date', 'service' ,'trxn_type' , 'country'  ]),
        pl.col("TPV").sort_by('date').cum_sum().over([ 'service' ,'trxn_type' , 'country']).alias("DUM")
    )
    
    kpis = kpis.with_columns(
        ( pl.col(  'T') / pl.col(  'U') ).alias(  'TpU')
        , ( pl.col(  'TPV') / pl.col(  'U') ).alias(  'TPVpU')
        , ( pl.col(  'TPV') / pl.col(  'T') ).alias(  'TPVpT')
        , ( pl.col(  'DUM') / pl.col(  'U') ).alias(  'DUMpU')
        , ( pl.col(  'DUM') / pl.col(  'T') ).alias(  'DUMpT')
        , ( pl.col(  'B') / pl.col(  'U') ).alias(  'BpU')
        , ( pl.col(  'B') / pl.col(  'T') ).alias(  'BpT')
    )
    
    agg_kpis[ period_labels[i] ] = kpis
    


# =============================================================================
# 
# =============================================================================

# agg_activity = {}
# for k , v in agg_kpis.items():
#     print(k)
#     agg_activity[ k ] = v.unique().to_pandas()

cols = [  'channel' , 'country'  , 'verified']
combs = itertools.chain.from_iterable(itertools.combinations(cols, r) for r in range( len(cols)+1 , 0 , -1))

acq_kpis = {}
for i , period in enumerate( period_cols ):
    print(period)
    print ()
    
    combs = itertools.chain.from_iterable(itertools.combinations(cols, r) for r in range( len(cols)+1 , -1 , -1))

    kpis = [ 
        trxns.filter( pl.col('trxn_type').is_in(['signup']) ).group_by( [period , *x ] ).agg( 
            pl.col('user_id').n_unique().alias( 'U')
            , pl.col('customer_acquisition_cost').mean().alias(  'CAC')
        ) for x in combs 
    ] 
    kpis = pl.concat(kpis , how= 'diagonal')
    kpis = kpis.with_columns(pl.col(cols).fill_null('all'))

    

    # kpis = kpis.select( pl.all().name.prefix(period_prefixes[i]) )
#     kpis = kpis.with_columns( period = pl.lit(period))
    if period != None :
        kpis = kpis.rename({period : 'date'})
        kpis = kpis.sort('date')
    else:
        kpis = kpis.rename({'literal' : 'date'}).with_columns( pl.col('date').fill_null(0) )

    
    acq_kpis[ period_labels[i] ] = kpis

# =============================================================================
# 
# =============================================================================
period = st.selectbox('Period', period_labels, index= 4 )


data = agg_kpis[period]
country = ['all']
trxn_type = ['all']
service = ['all']
data = data.filter( 
    (pl.col('country').is_in(country)  )
    & ( pl.col('trxn_type').is_in(trxn_type)  ) 
    & ( pl.col('service').is_in(service))
).sort('date')

col1, col2 = st.columns(2,)

col1.metric(
    "North Star: Deposit under Management", "{} B$".format(
        round(
            data.select( pl.col('DUM').last() 
                    ).row(0)[0] * ( 10**-9) , 2 )
        )
    , "{} %".format(
        round(
            ( data.select('DUM').row(-1)[0] -  data.select('DUM').row(-2)[0] ) / ( data.select('DUM').row(-2)[0] ) *100 
            , 2)
        )
    )


# =============================================================================
# 
# =============================================================================

data = agg_kpis[period]
country = ['all']
trxn_type = ['all']
service = data.select('service').unique().sort(by ='service')
data = data.filter( 
    (pl.col('country').is_in(country)  )
    & ( pl.col('trxn_type').is_in(trxn_type)  ) 
    & ( pl.col('service').is_in(service))
).sort('date').to_pandas()

fig = px.ecdf( data, x= 'date' , y = "TPV" ,   ecdfnorm=None, color = 'service' 
              , color_discrete_map= { 'all' : 'green' ,'Fixed14' : 'purple' , 'Flexible9' :'orange' , 'Locked14' : 'blue'}

              , title= period + ' Deposit under Management Growth')
fig.for_each_yaxis(lambda y: y.update(title = 'DUM'))

col1.plotly_chart(fig)

# =============================================================================
# 
# =============================================================================
data = agg_kpis[period]
country = ['all']
trxn_type1 = ['renew']
trxn_type2 = ['withdraw' ,'mature', 'break']
service = ['all']
data = data.filter( 
    (pl.col('country').is_in(country)  )
    # & ( pl.col('trxn_type').is_in(trxn_type)  ) 
    & ( pl.col('service').is_in(service))
).sort('date')


data = pl.concat(
    [
        data.filter( pl.col('trxn_type').is_in(trxn_type1) ).with_columns(pl.col('B').alias('DD'))
        , data.filter( pl.col('trxn_type').is_in(trxn_type2) ).with_columns(pl.col('DUM').alias('DD'))
    ]
).sort(['trxn_type','date']).to_pandas()

col2.metric(
    "Counter: Due Deposit", "{} B$".format(
        round(
            data.groupby('date').DD.sum().iloc[-1] * ( 10**-9) , 2 )
        )
    , "{} %".format(
        round(
            ( data.groupby('date').DD.sum().iloc[-1] -  data.groupby('date').DD.sum().iloc[-2] ) / ( data.groupby('date').DD.sum().iloc[-2] ) *100 
            , 2)
        )
)


fig = px.bar( data, x= 'date' , y = "DD" , color = 'trxn_type' 
             , category_orders= {
                 'trxn_type' : [ 'renew'  ,'mature','withdraw' , 'break']
                 ,'date' : data['date'].drop_duplicates().sort_values()
                 }
             , color_discrete_map= { 'renew' : 'green' ,'break' : 'red' , 'withdraw' :'orange' , 'mature' : 'blue'}
             , title= period + ' Due Deposit Growth'
             )
col2.plotly_chart(fig)
 
# =============================================================================
# 
# =============================================================================

data = agg_kpis[period]
country = data.filter(pl.col('country') != 'all').select('country').unique()
trxn_type = data.filter(pl.col('trxn_type') != 'all').select('trxn_type').unique()
service = data.filter(pl.col('service') != 'all').select('service').unique()


data = data.filter( 
    (pl.col('country').is_in(country)  )
    & ( pl.col('trxn_type').is_in(trxn_type)  ) 
    & ( pl.col('service').is_in(service))
).sort('date').filter(pl.col('date') == pl.last('date')).with_columns(pl.col('DUM').abs()).to_pandas()

fig = px.treemap(
    data, path= ['country' , 'service','trxn_type']
    , values = "DUM" 
    , color = 'trxn_type'
    , color_discrete_map={
        'withdraw':'orange'
        ,'top_up' : 'green'
        , 'mature' : 'blue'
        , 'open' : 'green'
        , 'renew' : 'green'
        , 'break' : 'red'

    }
    , title= 'Deposit under Management by Region in the Last Period'
    # ,textinfo = " DUM "
)
col1.plotly_chart(fig)



# =============================================================================
# 
# =============================================================================




data = acq_kpis[period]
country = ['all']
channel = data.filter(pl.col('channel') != 'all').select('channel').unique()
verified = ['Yes' ,'No']
data = data.filter( 
    (pl.col('country').is_in(country)  )
    & ( pl.col('channel').is_in(channel)  ) 
    & ( pl.col('verified').is_in(verified))
).sort('date').to_pandas()

fig = px.bar( data, x= 'date' , y = "U", color = 'channel' , facet_row= 'verified' , title= period + ' Acquisition by Channel')
fig.for_each_yaxis(lambda y: y.update(title = '' ))

fig.update_annotations( x = 0.8 ,  textangle = 0 )

col2.plotly_chart(fig)








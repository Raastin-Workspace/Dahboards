# =============================================================================
# to do list
# =============================================================================
# adding hash functions for polars data cache /done using random id
# adding regrouping to segmentation part
# adding clustring models to segmentation / first implementation done

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
import numpy as np
import seaborn as sns

# for peak-hours calculation
import datetime as dt
import holidays
from business_duration import businessDuration
import time

# to detoure hashing whole dataframes and files:
import uuid

# clustring
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

# handling universal maps
from collections import defaultdict
# handling colors more elegantly
from functools import reduce
# =============================================================================
# Dashboard Config Setting & Header
# =============================================================================

st.set_page_config(
    page_title= 'Jedlix Dashboard'
    , page_icon= ':chart_with_upwards_trend:'
    , layout= 'wide'
    # ,initial_sidebar_state="collapsed"
)

st.title(':chart_with_upwards_trend: Studying the Charging Behavior')

# # =============================================================================
# # Cached Resources (Shared for all users)
# # =============================================================================

@st.cache_resource
def get_period_truncs():
    return [  '1q' , '1mo' , '1w', '1d' ]
period_truncs = get_period_truncs()

@st.cache_resource
def get_period_labels():
    return [  'Quarterly' , 'Monthly' , 'Weekly' , 'Daily' ]
period_labels = get_period_labels()

@st.cache_resource
def get_period_index():
    return { x: i for i ,x in enumerate(period_labels)}
period_index = get_period_index()

@st.cache_resource
def get_dataset_vars():
    return [ 'ConnectedTime' , 'ChargeTime' , 'TotalEnergy' ,'MaxPower' ]
dataset_vars = get_dataset_vars()

@st.cache_resource
def get_engineered_vars():
    return [ 
        'PeakConnectedTime' 
        , 'AvgChargePower' , 'Utilization' ,'Throughput' , 'PeakhourShare' 
         
          ]
engineered_vars = get_engineered_vars()

@st.cache_resource
def get_transformed_vars():
    return [ 
        'ConnectedTimeLog1p' , 'ChargeTimeLog1p' , 'PeakConnectedTimeLog1p'
         
          ]
transformed_vars = get_transformed_vars()

@st.cache_resource
def get_analysis_vars():
    return [
        'ConnectedTime' , 'ChargeTime' , 'PeakConnectedTime' 
        , 'TotalEnergy' ,'MaxPower' , 'AvgChargePower'
        , 'Utilization' ,'Throughput' , 'PeakhourShare' 
    ]
analysis_vars = get_analysis_vars()



@st.cache_resource
def get_analysis_short_vars():
    return [
        'CntdTm' , 'ChrgTm' , 'PkCnTm' 
        , 'TotEgy' ,'MaxPwr' , 'ACgPwr'
        , 'Utilzn' ,'Thrput' , 'PkhShr'
        , 'CndTLg', 'ChgTLg' , 'PCnTLg' 
    ]
analysis_short_vars = get_analysis_short_vars()

@st.cache_resource
def get_short_vars_map():
    short_vars_map = { x:y for x,y in zip(analysis_vars + transformed_vars , analysis_short_vars ) }
    return defaultdict( lambda x : x[:6] , short_vars_map)
short_vars_map = get_short_vars_map()

@st.cache_resource
def get_result_metric():
    result_metric = {
          'ConnectedTime':'h'
        , 'ChargeTime':'h'
        , 'PeakConnectedTime':'h'
        , 'TotalEnergy':'kWh'
        , 'MaxPower':'kW'
        , 'AvgChargePower':'kW'
        , 'Utilization':'%'
        , 'Throughput':'kW'
        , 'PeakhourShare':'%'

        , 'ConnectedTimeLog1p':'L(1+h)'
        , 'ChargeTimeLog1p':'L(1+h)'
        , 'PeakConnectedTimeLog1p':'L(1+h)'

        # , 'SegmentShare':'%'
        # , 'EnergyShare':'%'
        
    }

    avg_metric = { ('Avg. '+x) : y for x , y in result_metric.items() }
    short_metric = { short_vars_map[x] : y  for x , y in result_metric.items() if x in short_vars_map.keys() }
    result_metric.update(avg_metric)
    result_metric.update(short_metric)
    return defaultdict( lambda : '{:1.f}' , result_metric )

result_metric = get_result_metric()

@st.cache_resource
def get_result_format():
    # result_format = {
    #       'ConnectedTime':'{:.1f} h'
    #     , 'ChargeTime':'{:.1f} h'
    #     , 'PeakConnectedTime':'{:.1f} h'
    #     , 'TotalEnergy':'{:.1f} kWh'
    #     , 'MaxPower':'{:.1f} kW'
    #     , 'AvgChargePower':'{:.1f} kW'
    #     , 'Utilization':'{:.1f} %'
    #     , 'Throughput':'{:.1f} kW'
    #     , 'PeakhourShare':'{:.1f} %'

    #     , 'ConnectedTimeLog1p':'{:.1f} L(h)'
    #     , 'ChargeTimeLog1p':'{:.1f} L(h)'
    #     , 'PeakConnectedTimeLog1p':'{:.1f} L(h)'

    #     # , 'SegmentShare':'{:.1f} %'
    #     # , 'EnergyShare':'{:.1f} %'
        
    # }

    # avg_format = { ('Avg. '+x) : y for x , y in result_format.items() }
    # short_format = { short_vars_map[x] : y  for x , y in result_format.items() if x in short_vars_map.keys() }
    # result_format.update(avg_format)
    # result_format.update(short_format)
    # return defaultdict( lambda : '{:1.f}' , result_format )
    return '{:.1f}'

result_format = get_result_format()

@st.cache_resource
def get_result_colors():

    color_palette = "tab10"
    result_color = { k:v for k , v in zip (analysis_vars , sns.color_palette( color_palette , n_colors=  len (analysis_vars) ).as_hex() ) } 

    avg_color = { ('Avg. '+x) : y for x , y in result_color.items() }
    short_color = { short_vars_map[x] : y  for x , y in result_color.items() if x in short_vars_map.keys() }
    transformed_color = { k:v for k , v in zip (transformed_vars , sns.color_palette(color_palette , n_colors=  len (transformed_vars) ).as_hex() ) } 
    short_transformed_color = { short_vars_map[x] : y  for x , y in transformed_color.items() if x in short_vars_map.keys() }

    result_color.update(avg_color)
    result_color.update(short_color)
    result_color.update(transformed_color)
    result_color.update(short_transformed_color)
    return defaultdict( lambda: 'white' , result_color)

result_colors = get_result_colors()
    
# # =============================================================================
# # Functions & Decorators & Constant Variables
# # =============================================================================
def clear_regrouping():

    if 'nr_of_segments' in st.session_state:
        for i  in range(st.session_state.nr_of_segments):
            if f'segment{i}_grps' in st.session_state:
                st.session_state[f'segment{i}_grps'] = []


def table2html( table , vmin = 0):
    styler = table.style\
    .set_properties(**{'text-align': 'left'})\
    .format(result_format)\
    .bar(color = 'black', vmin = 0,height = 30, align = 'zero' , axis = 0)\
    .set_table_styles(
        [
            {
                'selector': 'th',   'props': [('background-color', 'white') , ('min-width', '75px')]
            }
        ]
    )#.set_sticky().set_sticky(axis="columns")
    styler = reduce(    
        
        lambda a, b: 
            a.background_gradient( 
                cmap = sns.light_palette( result_colors[b] , as_cmap= True )
                , subset = [b]
                , vmin = vmin
            )
            ,  [ styler , * table.columns ] 
        )
    styler = styler.format_index( lambda x : f'{x} ({result_metric[x]})' , axis = 'columns')
    # .set_properties(**{'background-color': 'lightgrey'})\
    return styler
    

# # =============================================================================
# # 
# # Data Prepration & Cached Data (Not Shared)
# # 
# # =============================================================================

# # =============================================================================
# # # Loading The Data
@st.cache_data( max_entries = 1 , ttl = 3600)
def load_data( _input_file , id ):

    return pl.read_csv(_input_file)

# # =============================================================================
# # # Stage 1: Data Cleaning
@st.cache_data(max_entries = 1 , ttl = 3600)
def trxns_stage1( _trxns , id ):

    _trxns = _trxns.with_columns( pl.col(['UTCTransactionStart' , 'UTCTransactionStop'] ).str.to_datetime(format="%d/%m/%Y%H:%M"))
    _trxns = _trxns.with_columns(
        pl.col('UTCTransactionStart').dt.replace_time_zone('UTC').dt.convert_time_zone('CET').alias('CETTransactionStart')
        , pl.col('UTCTransactionStop').dt.replace_time_zone('UTC').dt.convert_time_zone('CET').alias('CETTransactionStop')
    ).sort(pl.col('UTCTransactionStart') )

    # I assumed here that chargin starts when transaction starts and stops right after ChargingTime, just to calculate charging volume over time
    _trxns = _trxns.with_columns(
        ( pl.col('UTCTransactionStart') + pl.duration(seconds= pl.col("ChargeTime") * 3600 ) ).clip(upper_bound= pl.col('UTCTransactionStop')).alias('UTCChargeStop')
    )
    _trxns = _trxns.with_columns(
        pl.col('UTCChargeStop').dt.replace_time_zone('UTC').dt.convert_time_zone('CET').alias('CETChargeStop')
    )
    _trxns = _trxns.with_columns(
        (( pl.col('UTCTransactionStop') - pl.col('UTCTransactionStart') ).dt.total_seconds()/ 3600).alias('ConnectedTimeRC')
        , (( pl.col('UTCChargeStop') - pl.col('UTCTransactionStart') ).dt.total_seconds()/ 3600).alias('ChargeTimeRC')

    )
    return _trxns

# # =============================================================================
# # # Stage2: Feature Engineering
@st.cache_data(max_entries = 1 , ttl = 3600)
def trxns_stage2( _trxns , _public_holidays , id ):

    _trxns= _trxns.with_columns(
        pl.struct( 
            ['CETTransactionStart' , 'CETTransactionStop' ] 
        ).map_elements(
            lambda x : businessDuration(
                startdate= x['CETTransactionStart']
                ,enddate= x['CETTransactionStop']
                ,starttime= dt.time( hour = 7 )
                ,endtime= dt.time( hour = 23 )
                ,holidaylist= _public_holidays
                ,unit='hour'
            )
            , return_dtype = pl.Float64
        ).alias('PeakConnectedTime')
    )
    
    _trxns = _trxns.with_columns(
        ( pl.col('TotalEnergy') / pl.col('ConnectedTimeRC') ).alias('Throughput')
        , ( pl.col('ChargeTimeRC') / pl.col('ConnectedTimeRC') *100 ).alias('Utilization')
        , ( pl.col('PeakConnectedTime') / pl.col('ConnectedTimeRC') *100 ).alias('PeakhourShare')
        , ( pl.col('TotalEnergy') / pl.col('ChargeTimeRC') ).alias('AvgChargePower')
        , ( pl.col('ConnectedTimeRC').log1p() ).alias('ConnectedTimeLog1p')
        , ( pl.col('ChargeTimeRC').log1p() ).alias('ChargeTimeLog1p')
        , ( pl.col('PeakConnectedTime').log1p() ).alias('PeakConnectedTimeLog1p')
    )
    return _trxns

# # =============================================================================
# # # Extracting Granular Changes in the Network
@st.cache_data(max_entries = 1 , ttl = 3600)
def get_granular_changes( _trxns , id ):
    _granular = _trxns.melt(

        id_vars= 'AvgChargePower'
        , value_vars= ['UTCTransactionStart' , 'UTCChargeStop']
        , value_name= 'UTCDatetime'
        , variable_name = 'Point'
    ).rename(  { 'AvgChargePower' : 'AvgChargePowerChange' } )

    #just to avoid relabling plots from 10k kwh to 10M Wh
    _granular = _granular.with_columns( 
        pl.col(['AvgChargePowerChange']) * \
        1000 * \
        pl.col('Point').replace(
            { 
                'UTCTransactionStart' : 1
                , 'UTCChargeStop' : -1
            }
            , return_dtype = pl.Int8
        )
    ).drop('Point' ).sort( pl.col('UTCDatetime') ) 
    return _granular

# # =============================================================================
# # # First Level of Smoothing Granular Changes in the Network
@st.cache_data(max_entries = 1 , ttl = 3600)
def get_smoothed_changes( _granular , id ):
    freq_min = 5
    freq = f'{freq_min}m'
    kwh_corrector = freq_min / 60
    _smoothed = _granular
    _smoothed = _smoothed.with_columns( pl.col('AvgChargePowerChange').cum_sum().alias('AvgChargePower') ) 
    
    
    # _smoothed = _smoothed.with_columns( pl.col('UTCDatetime').dt.truncate(freq).alias('UTCDatetime') )
    _smoothed = _smoothed.sort('UTCDatetime').group_by_dynamic('UTCDatetime' , every= freq , label='left' ).agg(
        pl.col('AvgChargePower').mean().alias('AvgChargePower_Mean')
        , pl.col('AvgChargePower').max().alias('AvgChargePower_Max')
        , pl.col('AvgChargePower').min().alias('AvgChargePower_Min')
    )
    _smoothed = _smoothed.with_columns(
        ( pl.col('AvgChargePower_Mean') * kwh_corrector ).alias('AvgChargeVolume')
    )

    peak_hours_pdf = peakhours(starting_datetime_UTC , end_datetime_UTC)
    _smoothed = _smoothed.join( 
        peak_hours_pdf  , how = 'outer_coalesce' , left_on = 'UTCDatetime' , right_on = 'UTCDatetime' 
    ).sort('UTCDatetime').upsample('UTCDatetime' , every= freq)

    _smoothed = _smoothed.with_columns(
        pl.all().forward_fill()
    )

    _smoothed = _smoothed.with_columns(
        pl.col('Peakhour').fill_null(0)
        , pl.col('UTCDatetime').dt.replace_time_zone('UTC').dt.convert_time_zone('CET').alias('CETDatetime')
    )
    return _smoothed

# # =============================================================================
# # Second Level of Smoothing the Changes in Network
@st.cache_data(max_entries = 4 , ttl = 3600)
def get_coarse_changes( _smoothed , coarse_freq , id ):
    _coarse = _smoothed.with_columns( pl.col('CETDatetime').dt.date().dt.truncate(coarse_freq).alias('Period') )

    _coarse = _coarse.group_by(['Period' , 'Peakhour']).agg(
        pl.col('AvgChargePower_Mean').mean().alias('AvgChargePower_Mean')
        # ,pl.col('AvgChargePower_Mean').count().alias('AvgChargePower_count')
        , pl.col('AvgChargePower_Max').max().alias('AvgChargePower_Max')
        , pl.col('AvgChargePower_Min').min().alias('AvgChargePower_Min')
        , pl.col('AvgChargeVolume').sum().alias('AvgChargeVolume')
    ).sort('Period')

    #it could raise error if sum is exactly zeros ( I add 1 day to the end!), handle?
    def get_peakhours_share(x):
        try:
            return x/x.sum()
        except:
            return [ pd.NA ] * len(x)    
    _coarse = _coarse.with_columns( ( pl.col('AvgChargeVolume') /   pl.col('AvgChargeVolume').sum()).over('Period').alias( 'PeakhoursShare' )  )

    # # just to plotly treated as categorical. why now? speed in previous groupbys!
    _coarse = _coarse.with_columns( pl.col('Peakhour').cast(pl.String) )

    return _coarse

# # =============================================================================
# # Analysis Interval
@st.cache_data(max_entries = 1 , ttl = 3600)
def interval_extraction( _trxns , id ):
    starting_datetime_UTC = _trxns.select( [ 'UTCTransactionStart' , 'UTCTransactionStop' ]).min().min_horizontal().item()
    end_datetime_UTC = _trxns.select( [ 'UTCTransactionStart' , 'UTCTransactionStop' ]).max().max_horizontal().item()

    return starting_datetime_UTC , end_datetime_UTC
# # =============================================================================
# # Public Holidays During Analysis Interval
@st.cache_data(max_entries = 1 , ttl = 3600)
def get_public_holidays( starting_datetime_UTC , end_datetime_UTC ):
    
    public_holidays = holidays.country_holidays(
        country = 'NL'
        , years=range(
            starting_datetime_UTC.year
            , end_datetime_UTC.year + 1
        ) 
        , language= 'en_US'
        , observed= True
    )
    return list(public_holidays)
# # =============================================================================
# # Starting and End Time of Peakhours During Analysis Interval
@st.cache_data(max_entries = 1 , ttl = 3600)
def peakhours( starting_datetime_UTC , end_datetime_UTC ):

    public_holidays = get_public_holidays(  starting_datetime_UTC , end_datetime_UTC  )
    # since it's peak-hours, vs. off-peak-hours, calculation is not affected by "daylightsaving observation"
    peak_hours_start = dt.time( hour = 7 )
    peak_hours_end = dt.time( hour = 23 )
    # daily_peak_hours_duration = 16
    peak_hours_offset = pd.offsets.CustomBusinessHour(
        start = peak_hours_start
        , end = peak_hours_end
        , holidays = public_holidays
    )

    peak_days_offset = pd.offsets.CustomBusinessDay(
        holidays = public_holidays
        # , normalize = True
    )
    normalized_peak_days_offset = pd.offsets.CustomBusinessDay(
        holidays = public_holidays
        , normalize = True
    )

    peak_starting_hours_offset = pd.offsets.CustomBusinessHour(
        start = peak_hours_start
        , end = dt.time( hour = peak_hours_start.hour , minute = 1 )
        , holidays = public_holidays
    )

    peak_starting_hours = pd.Series(
        data = 1
        , index = pd.date_range(
            start= peak_starting_hours_offset.rollforward( starting_datetime_UTC.date() )
            , end = peak_starting_hours_offset.rollforward( end_datetime_UTC.date() )
            , freq= peak_days_offset
        )
    )

    peak_end_hours_offset = pd.offsets.CustomBusinessHour(
        start = peak_hours_end
        , end = dt.time( hour = peak_hours_end.hour , minute = 1 )
        , holidays = public_holidays
    )

    peak_end_hours = pd.Series(
        data = 0
        , index = pd.date_range(
            start= peak_end_hours_offset.rollforward( starting_datetime_UTC.date() )
            , end = peak_end_hours_offset.rollforward( end_datetime_UTC.date() )
            , freq= peak_days_offset
        )
    )
    peak_hours = pd.concat( [ peak_starting_hours , peak_end_hours ] ).rename('Peakhour')
    del peak_end_hours , peak_starting_hours
    # peak_hours.index = peak_hours.index.map( lambda x : x.tz_localize('UTC') )
    peak_hours.index = peak_hours.index.tz_localize('CET').tz_convert('UTC').tz_localize(None)
    peak_hours_pdf = pl.from_pandas(peak_hours.reset_index() ).rename({"index": "UTCDatetime"})

    peak_hours_pdf = peak_hours_pdf.with_columns(pl.col("UTCDatetime").dt.cast_time_unit("us"))
    
    return peak_hours_pdf



# # =============================================================================
# # =============================================================================
# # Visible Page
# # =============================================================================
# # =============================================================================

# # =============================================================================
# # Reciving the Data File from User
# # =============================================================================

with st.expander("File"):

    f1 = st.file_uploader(":file_folder: Upload the file" , type = (['csv' , 'xlsx']))
    
    if 'f_copy' not in st.session_state:
        st.session_state['f_copy'] = None
    
    if 'file_id' not in st.session_state:
        st.session_state['file_id'] = None


    if f1 is None:
        st.write('Please Upload Your File')
        

    else:
        if st.session_state.f_copy is None or st.session_state.f_copy != f1:
            

            st.session_state.f_copy = f1
            st.session_state.file_id = uuid.uuid1() 
            # st.write( st.session_state.file_id)

            st.write('Your File Has Uploaded Succesfully')
            # file_name = f1.name
        
        global data
        data = load_data(f1 , st.session_state.file_id)

        global trxns
        trxns = trxns_stage1( data , st.session_state.file_id )
        
        starting_datetime_UTC , end_datetime_UTC = interval_extraction(trxns , st.session_state.file_id)
        
        public_holidays = get_public_holidays(starting_datetime_UTC , end_datetime_UTC)
        
        trxns = trxns_stage2( trxns , public_holidays  , st.session_state.file_id)

        global granular_change
        granular_change = get_granular_changes(trxns , st.session_state.file_id)
        
        global smoothed_change
        smoothed_change = get_smoothed_changes( granular_change  , st.session_state.file_id)
        
        

# # =============================================================================
# # Dashboard Inputs
# # =============================================================================


    
# # =============================================================================
# # Appling Filters
# # =============================================================================



# # =============================================================================
# # Let's Go EDA
# # =============================================================================
@st.experimental_fragment
def EDA(vars , shorten_columns = False):
    summary_stats = trxns.select(
            vars
        ).describe(
            percentiles= [ 0.05 , 0.5 , 0.95 ]
        ).to_pandas().loc[2:].set_index('statistic').rename(
            columns = lambda x: short_vars_map[x] if shorten_columns else x 
        )

    
    summary_styler = table2html(summary_stats)
    st.write('Summary Statistics')
    st.components.v1.html( summary_styler.to_html() ,scrolling=True , height= 32* (len(summary_stats )+ 2 ))
    
    st.write('Summary Statistics')
    fig = px.histogram(
        trxns.select( vars ).to_pandas().rename(
            columns = lambda x: short_vars_map[x] if shorten_columns else x 
        )
        , facet_col= 'variable'
        # , facet_col_wrap= 3
        # , facet_row_spacing= 0.1
        , histnorm= 'percent'
        , marginal= 'box'
        # , title = 'Variables Distribution'
        , color_discrete_map = result_colors #['blue', 'red', 'green' , 'purple']
    )
    # fig.update_yaxes(matches=None , showticklabels=True , title_text="")
    fig.update_xaxes(matches=None, showticklabels=True , title_text="")
    fig.update_layout(showlegend = False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.for_each_annotation(lambda a: a.update(text= f'{a.text} ({result_metric[a.text]})' ))
    st.plotly_chart( fig , use_container_width= True )

if f1 is not None:
    with st.expander("EDA"):
        st.write('Dataset Variables:')
        EDA(dataset_vars)
        st.write('Engineered Variables:')
        EDA(engineered_vars + transformed_vars , True)
    

# # =============================================================================
# # Charge Power Over Time
# # =============================================================================
@st.experimental_fragment
def charging_power():
    with st.expander("Charge Power Over Time"):
        
        freq_condition = 0
        freq = st.selectbox('Frequency', period_labels[freq_condition:], index= 1  )
        freq_index = period_index[freq]
        
        coarse_freq = period_truncs[freq_index]
        
        coarse_change = get_coarse_changes( smoothed_change , coarse_freq , st.session_state.file_id)
        
        fig = px.bar( 
            coarse_change
            , x = 'Period' 
            , y = 'AvgChargeVolume'
            , color = 'Peakhour'
            , title= 'Avg. Charging Volume Over Time'
            , color_discrete_sequence= [ 'red' , 'green' ]
            , category_orders= { 'Peakhour': [ '1' , '0']}
        )
        # fig.update_xaxes(tickangle=-80)
        fig.update_yaxes(title_text = 'Avg. Charge Volume (Wh)' )
        st.plotly_chart(fig,use_container_width=True)

        fig = px.bar( 
            coarse_change
            , x = 'Period' 
            , y = 'PeakhoursShare'
            , color = 'Peakhour'
            , title= 'Peakhours Share of Avg. Charging Volume Over Time'
            , color_discrete_sequence= [ 'red' , 'green' ]
            , category_orders= { 'Peakhour': [ '1' , '0' ]}
        )
        # fig.update_xaxes(tickangle=-80)
        fig.update_yaxes(title_text = 'PeakhourShare (%)' )
        st.plotly_chart(fig,use_container_width=True)

        fig = px.line(
            coarse_change
            , x = 'Period' 
            , y = ['AvgChargePower_Min', 'AvgChargePower_Mean' , 'AvgChargePower_Max']
            , facet_row = 'Peakhour'
            , title= 'Avg. Charging Power Over Time'
            , color_discrete_sequence= ['green' , 'blue' , 'red' ]
            , category_orders = { 'Peakhour': [  '0' , '1' ] }
            , line_shape='hvh'
            , height= 500
        )
        fig.update_yaxes(matches=None, showticklabels=True , title_text="ChargePower(W)")
        st.plotly_chart(fig,use_container_width=True)
        # fig.update_xaxes(tickangle=-80)

if f1 is not None:
    charging_power()

# # =============================================================================
# # Segmentation
# # =============================================================================
   
@st.experimental_fragment
def regrouping(selected_trxns ,selected_vars ):
    st.write('Re-grouping')
    ns = st.number_input( 'Number of Segments' , step = 1 , min_value = 2 , max_value = 5 , key = 'nr_of_segments' )
    cols = st.columns(ns)
    sub_segments = set( selected_trxns.select('SubSegment').unique().sort('SubSegment').to_series().to_list())

    def all_selected_subs(k):
        l = []
        for i  in range(ns):
            if k != i and f'segment{i}_grps' in st.session_state:
                l.extend(st.session_state[f'segment{i}_grps'])
        return l
    
    def selected_subs(k):
        l = []
        
        if f'segment{i}_grps' in st.session_state:
                l.extend(st.session_state[f'segment{i}_grps'])
        return l
    

    for i  in range(ns):
        name = cols[i].text_input( 
            f'Label' 
            , value= f'Segment{i+1}' 
            , key = f'segment{i}_label' 
        )
        # st.write(i)
        # st.write(all_selected_subs(i))
        # st.write(selected_subs(i))
        selected = cols[i].multiselect(
            f'Sub Segments' 
            , sorted( sub_segments - set(all_selected_subs(i)) )
            , key = f'segment{i}_grps'
            , default = selected_subs(i)
        )

    d = {}
    for i in range(ns):
        for x in selected_subs(i):
            d[x] = st.session_state[f'segment{i}_label']
    
    selected_trxns = selected_trxns.with_columns(
        pl.col('SubSegment').replace(d).alias('Segment')
    )


    avgs_pdf = trxns.group_by(selected_trxns.select('Segment')).agg( pl.col(analysis_vars + transformed_vars).mean() ).to_pandas().set_index('Segment')

    final_table = pd.concat( 
        [
            avgs_pdf.rename(columns =short_vars_map)#.add_prefix('Avg. ')
        ]
        , axis  = 1
    ).sort_index()

    st.components.v1.html( table2html( final_table ).to_html() ,scrolling=True , height= 32* (len(final_table )+ 2 ))
    var_nr = len(selected_vars)

    if var_nr > 2:

        fig = px.scatter_matrix(
            selected_trxns
            ,  dimensions= selected_vars
            , color = 'Segment'
            # , color_discrete_map= color_map
            , title='Distributions'
            , width=800
            , height=800
            , category_orders = { f'Segment': sorted(selected_trxns.select('Segment').unique().to_series().to_list() )  }

        )
        fig.update_traces(
            showupperhalf = False
            , diagonal_visible=False
        )
        st.plotly_chart(fig,use_container_width=True)

    elif var_nr ==2:
        fig = px.scatter(
            selected_trxns
            , x = selected_vars[0]
            , y = selected_vars[1]
            , color = 'Segment'
            # , color_discrete_map= color_map
            , title='Distributions'
            , width=400
            , height=400
            , category_orders = { f'Segment': sorted(selected_trxns.select('Segment').unique().to_series().to_list() )  }

        )
        st.columns(4)[1].plotly_chart(fig,use_container_width=False)

    else:
        fig = px.box(
            selected_trxns
            , x = selected_vars[0]
            , color = 'Segment'
            # , color_discrete_map= color_map
            , title='Distributions'
            , width=800
            , height=800

        )
        st.plotly_chart( fig,use_container_width=True)

if f1 is not None:
    with st.expander("Segmentation"):

        methods = [ 'Direct' , 'KMeans' , 'Hierarchical']
        selected_method = st.radio(
            'Clustring Method' 
            , methods 
            , horizontal= True
            , on_change = clear_regrouping
        )


        selected_vars = st.multiselect(
            'Segmentation Variables'
            , analysis_vars  + transformed_vars
            , max_selections = 4
            , on_change = clear_regrouping
        )
        var_nr = len(selected_vars)
        
        
        if var_nr > 0:
            
            
            selected_trxns = trxns.select( selected_vars )#.sample(2500*var_nr).sort(selected_vars)
            split_list = []
            if selected_method == 'Direct' :

                cols = st.columns(  var_nr )
                
                
                for i , var_name in enumerate(selected_vars):
                    
                    short_name = short_vars_map[selected_vars[i]]
                    splits = cols[i].number_input( var_name , step = 1 , min_value = 1 , max_value = 4 )
                    split_list.append( splits )
                    quantile_rank = cols[i].checkbox(f'{short_name} Rank')

                    bins = [ i/splits for i in range( splits+1 ) ][1:-1]

                    if quantile_rank:
                        selected_trxns = selected_trxns.with_columns(
                            (pl.col(var_name).rank( method = 'random')/ pl.col(var_name).count() ).cut( bins , labels=[ f'{short_name}{i+1}' for i in range(splits) ]).alias(f'{short_name}Grp')
                        )
    
                    else:
                        selected_trxns = selected_trxns.with_columns(
                            ((pl.col(var_name) - pl.col(var_name).min()) / (pl.col(var_name).max() - pl.col(var_name).min()) ).cut( bins , labels=[ f'{short_name}{i+1}' for i in range(splits) ]).alias(f'{short_name}Grp')
                        )

                selected_trxns = selected_trxns.with_columns(
                    pl.concat_str(
                        pl.col(
                            [ f'{short_vars_map[x]}Grp' for  x in selected_vars]
                        )
                        , separator ='_'
                    ).alias(f'SubSegment')
                )

            elif selected_method == 'KMeans' :
                n_clusters = st.number_input( 'Number of Clusters' , step = 1 , min_value = 2 , max_value = 10 )

                split_list = [1] * n_clusters
                model_kmeans =  KMeans( 
                    n_clusters= n_clusters 
                    , n_init= 30
                    , init= "random"
                )
                model_kmeans = model_kmeans.fit(selected_trxns)
                selected_trxns = selected_trxns.with_columns(
                    pl.Series( 
                        name = 'SubSegment'
                        , values=  model_kmeans.labels_.astype('str')
                    )
                )
                selected_trxns = selected_trxns.with_columns(
                     ('Grp' + pl.col('SubSegment') ).alias(f'SubSegment')
                    
                )
            
            elif selected_method == 'Hierarchical' :
                
                n_clusters = st.number_input( 'Number of Clusters' , step = 1 , min_value = 2 , max_value = 10 )
                split_list = [1] * n_clusters

                model_hrch = AgglomerativeClustering( n_clusters= n_clusters )
                model_hrch = model_hrch.fit( selected_trxns )
                selected_trxns = selected_trxns.with_columns(
                    pl.Series( 
                        name = 'SubSegment'
                        , values= model_hrch.labels_.astype('str')
                    )
                )

                selected_trxns = selected_trxns.with_columns(
                    ('Grp' + pl.col('SubSegment') ).alias(f'SubSegment')
                    
                )

            if var_nr > 2:

                fig = px.scatter_matrix(
                    selected_trxns
                    ,  dimensions= selected_vars
                    , color = 'SubSegment'
                    # , color_discrete_map= color_map
                    , title='Distributions'
                    , width=800
                    , height=800
                    , category_orders = { f'SubSegment': sorted(selected_trxns.select('SubSegment').unique().to_series().to_list() )  }

                )
                fig.update_traces(
                    showupperhalf = False
                    , diagonal_visible=False
                )
                st.plotly_chart(fig,use_container_width=True)

            elif var_nr ==2:
                fig = px.scatter(
                    selected_trxns
                    , x = selected_vars[0]
                    , y = selected_vars[1]
                    , color = 'SubSegment'
                    # , color_discrete_map= color_map
                    , title='Distributions'
                    , width=400
                    , height=400
                    , category_orders = { f'SubSegment': sorted(selected_trxns.select('SubSegment').unique().to_series().to_list() )  }
    
                )
                st.columns(4)[1].plotly_chart(fig,use_container_width=False)

            else:
                fig = px.box(
                    selected_trxns
                    , x = selected_vars[0]
                    , color = 'SubSegment'
                    # , color_discrete_map= color_map
                    , title='Distributions'
                    , width=800
                    , height=800

                )
                st.plotly_chart( fig,use_container_width=True)
            avgs_pdf = trxns.group_by(selected_trxns.select('SubSegment')).agg( pl.col(analysis_vars + transformed_vars ).mean() ).to_pandas().set_index('SubSegment')

            final_table = pd.concat( 
                [
                    avgs_pdf.rename(columns =short_vars_map)#.add_prefix('Avg. ')
                ]
                , axis  = 1
            ).sort_index()

            table2html( final_table )

            # cols = st.columns(  var_nr )
            # for i , var_name in enumerate(selected_vars):

            #     short_name = short_vars_map[selected_vars[i]]
            #     cols[i].plotly_chart( 
            #         px.histogram( 
            #             selected_trxns 
            #             , x = var_name
            #             , histnorm='percent'
            #             # , title = f'{var_name} Distribution'
            #             , color = 'SubSegment'
            #         ) 
            #         , use_container_width=True
            #     )

# # # # # # regrouping
            regrouping(selected_trxns ,selected_vars ) 
            


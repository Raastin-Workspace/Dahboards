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

# check if my function works correctly, it does but not as efficient as this lib
from business_duration import businessDuration
import time

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
# # Functions & Decorators & Constant Variables
# # =============================================================================

period_cols= [  'quarter' , 'month' , 'week' , 'date' ]
period_names= [  'Quarter' , 'Month' , 'Week' , 'Day' ]

mperiod_cols= [ 'm'+x  for x in period_cols ]

period_prefixes = [  'Q' , 'M' , 'W', 'D' ]
period_truncs = [  '1q' , '1mo' , '1w', '1d' ]

period_labels = [  'Quarterly' , 'Monthly' , 'Weekly' , 'Daily' ]
period_index = { x: i for i ,x in enumerate(period_labels)}

main_vars = [ 'ConnectedTime' , 'ChargeTime' , 'TotalEnergy' ,'MaxPower' ]
result_format = {
        'SegmentShare':'{:.1f}%'
        , 'EnergyShare':'{:.1f}%'
        , 'ConnectedTime':'{:.2f}h'
        , 'ChargeTime':'{:.2f}h'
        , 'Utilization':'{:.2f}%'
        , 'PeakhourShare':'{:.2f}%'
        , 'TotalEnergy':'{:.2f}kWh'
        , 'Throughput':'{:.2f}kW'
        , 'MaxPower':'{:.2f}kW'
        , 'Avg. ConnectedTime':'{:.2f}h'
        , 'Avg. Utilization':'{:.1f}%'
        , 'Avg. PeakhourShare':'{:.1f}%'
        , 'Avg. TotalEnergy':'{:.2f}kWh'
        , 'Avg. Throughput':'{:.2f}kW'
}
segment_map = { 
        'LowUtlzOffOnPeak' : 'EnergyPool'
        , 'MidUtlzOffOnPeak' : 'EnergyPool'
        , 'HghUtlzOffOnPeak' : 'Busy'
        , 'LowUtlzOnPeak' : 'Queue'
        , 'MidUtlzOnPeak' : 'Queue'
        , 'HghUtlzOnPeak' : 'Busy'
        , 'LowUtlzOffPeak' : 'Frugal'
        , 'MidUtlzOffPeak' : 'Frugal'
        , 'HghUtlzOffPeak' : 'Frugal'
}
color_map = { 
        'EnergyPool' : 'green'
        , 'Frugal' : 'grey'
        , 'Busy' : 'gold'
        , 'Queue' : 'red'
        
}

def color_background(cell):
  for value, color in cell_bg_colors.items():
    if value == cell:
      return "background-color: {}".format(color)
  return ""  # default: do nothing

def get_color_map( xdf ):
    table_color_map = xdf.stack().reset_index()
    table_color_map['SubSegment'] = table_color_map.UtilizationSegment.str.cat(table_color_map.PeakhourShareSegment)
    table_color_map['Color'] = table_color_map.SubSegment.map(segment_map).map(color_map)
    return table_color_map.set_index(0).to_dict()['Color']


# # =============================================================================
# # 
# # Data Prepration
# # 
# # =============================================================================

with st.expander("File"):

    f1 = st.file_uploader(":file_folder: Upload the file" , type = (['csv' , 'xlsx']))

    if f1 is None:
        # show user message
        st.write('Please Upload Your File')
    else:

        st.write('Your File Has Uploaded Succesfully')
        file_name = f1.name
        
    if 'trxns' not in st.session_state and f1 is not None:
    

        st.session_state.trxns = pl.read_csv(f1)
        
        st.session_state.trxns = st.session_state.trxns.with_columns( pl.col(['UTCTransactionStart' , 'UTCTransactionStop'] ).str.to_datetime(format="%d/%m/%Y%H:%M"))
        st.session_state.trxns = st.session_state.trxns.with_columns(
            pl.col('UTCTransactionStart').dt.replace_time_zone('UTC').dt.convert_time_zone('CET').alias('CETTransactionStart')
            , pl.col('UTCTransactionStop').dt.replace_time_zone('UTC').dt.convert_time_zone('CET').alias('CETTransactionStop')
        ).sort(pl.col('UTCTransactionStart') )

        # I assumed here that chargin starts when transaction starts and stops right after ChargingTime, just to calculate charging volume over time
        st.session_state.trxns = st.session_state.trxns.with_columns(
            ( pl.col('UTCTransactionStart') + pl.duration(seconds= pl.col("ChargeTime") * 3600 ) ).clip(upper_bound= pl.col('UTCTransactionStop')).alias('UTCChargeStop')
        )
        st.session_state.trxns = st.session_state.trxns.with_columns(
            pl.col('UTCChargeStop').dt.replace_time_zone('UTC').dt.convert_time_zone('CET').alias('CETChargeStop')
        )
        st.session_state.trxns = st.session_state.trxns.with_columns(
            (( pl.col('UTCTransactionStop') - pl.col('UTCTransactionStart') ).dt.total_seconds()/ 3600).alias('ConnectedTimeRC')
            , (( pl.col('UTCChargeStop') - pl.col('UTCTransactionStart') ).dt.total_seconds()/ 3600).alias('ChargeTimeRC')

        )

        starting_datetime_UTC = st.session_state.trxns.select( [ 'UTCTransactionStart' , 'UTCTransactionStop' ]).min().min_horizontal().item()
        end_datetime_UTC = st.session_state.trxns.select( [ 'UTCTransactionStart' , 'UTCTransactionStop' ]).max().max_horizontal().item()

        public_holidays = holidays.country_holidays(
            country = 'NL'
            , years=range(
                starting_datetime_UTC.year
                , end_datetime_UTC.year + 1
            ) 
            , language= 'en_US'
            , observed= True
        )


        # since it's peak-hours, vs. off-peak-hours, calculation is not affected by "daylightsaving observation"
        peak_hours_start = dt.time( hour = 7 )
        peak_hours_end = dt.time( hour = 23 )
        # daily_peak_hours_duration = 16
        peak_hours_offset = pd.offsets.CustomBusinessHour(
            start = peak_hours_start
            , end = peak_hours_end
            , holidays = public_holidays.keys()
        )

        peak_days_offset = pd.offsets.CustomBusinessDay(
            holidays = public_holidays.keys()
            # , normalize = True
        )
        normalized_peak_days_offset = pd.offsets.CustomBusinessDay(
            holidays = public_holidays.keys()
            , normalize = True
        )


        peak_starting_hours_offset = pd.offsets.CustomBusinessHour(
            start = peak_hours_start
            , end = dt.time( hour = peak_hours_start.hour , minute = 1 )
            , holidays = public_holidays.keys()
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
            , holidays = public_holidays.keys()
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
        
        if 'peak_hours_pdf' not in st.session_state:
            st.session_state.peak_hours_pdf = peak_hours_pdf


        st.session_state.trxns= st.session_state.trxns.with_columns(
            pl.struct( 
                ['CETTransactionStart' , 'CETTransactionStop' ] 
            ).map_elements(
                lambda x : businessDuration(
                    startdate= x['CETTransactionStart']
                    ,enddate= x['CETTransactionStop']
                    ,starttime= peak_hours_start
                    ,endtime= peak_hours_end
                    ,holidaylist=public_holidays.keys()
                    ,unit='hour'
                )
                , return_dtype = pl.Float64
            ).alias('PeakConnectedTime')
        )

        st.session_state.trxns = st.session_state.trxns.with_columns(
            ( pl.col('TotalEnergy') / pl.col('ConnectedTimeRC') ).alias('Throughput')
            , ( pl.col('ChargeTimeRC') / pl.col('ConnectedTimeRC') ).alias('Utilization')
            , ( pl.col('PeakConnectedTime') / pl.col('ConnectedTimeRC') ).alias('PeakhourShare')
            , ( pl.col('TotalEnergy') / pl.col('ChargeTimeRC') ).alias('AvgChargePower')
        )

        st.session_state.trxns = st.session_state.trxns.with_columns(
            pl.col('Throughput').cut( [0.333 , 0.666] , labels=[ 'LowThpt','MidThpt' ,'HghThpt']).alias('ThroughputSegment')
            , pl.col('Utilization').cut( [0.333 , 0.666] , labels=[ 'LowUtlz','MidUtlz' ,'HghUtlz' ]).alias('UtilizationSegment')
            , pl.col('PeakhourShare').cut( [0.333 , 0.666] , labels=[  'OffPeak', 'OffOnPeak', 'OnPeak' ]).alias('PeakhourShareSegment')
            , pl.col('ConnectedTime').qcut( [0.025 ,0.35 , 0.65, 0.975] , labels=[ 'Quck', 'Shrt' ,'Midd' , 'Long', 'Dead' ]).alias('ConnectedTimeSegment')

        )

    
        st.session_state.trxns = st.session_state.trxns.with_columns(
            pl.concat_str( pl.col(['UtilizationSegment' , 'PeakhourShareSegment' ]) ).alias('SubSegment')

        )

        st.session_state.trxns = st.session_state.trxns.with_columns(
            pl.col('SubSegment').replace(segment_map).alias('Segment')
        )


if 'trxns' in st.session_state and 'energy_pdf' not in st.session_state:
    st.session_state.energy_pdf = st.session_state.trxns.melt(

        id_vars= 'AvgChargePower'
        , value_vars= ['UTCTransactionStart' , 'UTCChargeStop']
        , value_name= 'UTCDatetime'
        , variable_name = 'Point'
    ).rename(  { 'AvgChargePower' : 'AvgChargePowerChange' } )

    #just to avoid relabling plots from 10k kwh to 10M Wh
    st.session_state.energy_pdf = st.session_state.energy_pdf.with_columns( 
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

if 'energy_pdf' in st.session_state and 'smoothed_energy_pdf' not in st.session_state:
    freq_min = 5
    freq = f'{freq_min}m'
    kwh_corrector = freq_min / 60
    st.session_state.smoothed_energy_pdf = st.session_state.energy_pdf
    st.session_state.smoothed_energy_pdf = st.session_state.smoothed_energy_pdf.with_columns( pl.col('AvgChargePowerChange').cum_sum().alias('AvgChargePower') ) 
    # st.session_state.smoothed_energy_pdf = st.session_state.smoothed_energy_pdf.with_columns( pl.col('UTCDatetime').dt.truncate(freq).alias('UTCDatetime') )


    st.session_state.smoothed_energy_pdf = st.session_state.smoothed_energy_pdf.group_by_dynamic('UTCDatetime' , every= freq , label='left' ).agg(
        pl.col('AvgChargePower').mean().alias('AvgChargePower_Mean')
        , pl.col('AvgChargePower').max().alias('AvgChargePower_Max')
        , pl.col('AvgChargePower').min().alias('AvgChargePower_Min')
    )
    st.session_state.smoothed_energy_pdf = st.session_state.smoothed_energy_pdf.with_columns(
        ( pl.col('AvgChargePower_Mean') * kwh_corrector ).alias('AvgChargeVolume')
    )
    st.session_state.smoothed_energy_pdf = st.session_state.smoothed_energy_pdf.join( 
        peak_hours_pdf  , how = 'outer_coalesce' , left_on = 'UTCDatetime' , right_on = 'UTCDatetime' 
    ).sort('UTCDatetime').upsample('UTCDatetime' , every= freq)

    st.session_state.smoothed_energy_pdf = st.session_state.smoothed_energy_pdf.with_columns(
        pl.all().forward_fill()
    )

    st.session_state.smoothed_energy_pdf = st.session_state.smoothed_energy_pdf.with_columns(
        pl.col('Peakhour').fill_null(0)
        , pl.col('UTCDatetime').dt.replace_time_zone('UTC').dt.convert_time_zone('CET').alias('CETDatetime')
    )

    

# # =============================================================================
# # Dashboard Inputs
# # =============================================================================


    
# # =============================================================================
# # Appling Filters
# # =============================================================================



# # =============================================================================
# # Let's Go EDA
# # =============================================================================
if 'trxns' in st.session_state:
    with st.expander("EDA"):
        
        summary_statistics = st.session_state.trxns.select(main_vars).describe(percentiles=[0.05,0.5,0.95]).to_pandas()\
        .loc[2:]\
        .style.format(result_format )\
        .set_properties(**{'text-align': 'left'})\
        .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])\
        .bar(color = 'grey', vmin = 0,height = 30, align = 'zero')\
        .background_gradient( subset = ['ConnectedTime'] , cmap = 'Blues' , vmin =0 )\
        .background_gradient( subset = ['ChargeTime'] , cmap = 'Reds' , vmin =0 )\
        .background_gradient( subset = ['TotalEnergy'] , cmap = 'Greens' , vmin =0 )\
        .background_gradient( subset = ['MaxPower'] , cmap = 'Purples' , vmin =0 )\
        .hide(axis="index").set_table_styles(
            [
                {
                    'selector': 'th',   'props': [('background-color', 'white')]
                }
            ]
        )\
        .set_properties(**{'background-color': 'white'}, subset=['statistic'])
        cols = st.columns((2, 5, 1))
        with cols[1]:
            st.components.v1.html(summary_statistics.to_html() ,scrolling=True, height=200)


        fig = px.histogram(
            st.session_state.trxns.select(main_vars).to_pandas()#.select_dtypes('number')\
            .rename( columns = { 'ConnectedTime' : 'ConnectedTime(hours)'} )\
            .rename( columns = { 'ChargeTime' : 'ChargeTime(hours)'} )\
            .rename( columns = { 'TotalEnergy' : 'TotalEnergy(kWh)'} )\
            .rename( columns = { 'MaxPower' : 'MaxPower(kW)'} )
            , facet_col= 'variable'
            # , facet_col_wrap= 2
            # , facet_row_spacing= 0.1
            , histnorm= 'percent'
            , marginal= 'box'
            , title = 'Main Variables Distribution'
            , color_discrete_sequence = ['blue', 'red', 'green' , 'purple']
        )
        # fig.update_yaxes(matches=None , showticklabels=True , title_text="")
        fig.update_xaxes(matches=None, showticklabels=True , title_text="")
        fig.update_layout(showlegend = False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig,use_container_width=True)

# # =============================================================================
# # Charge Power Over Time
# # =============================================================================


if 'smoothed_energy_pdf' in st.session_state:
    with st.expander("Charge Power Over Time"):

        freq_condition = 0
        freq = st.selectbox('Frequency', period_labels[freq_condition:], index= 1  )
        freq_index = period_index[freq]
        coarse_freq = period_truncs[freq_index]

        coarse_energy_pdf = st.session_state.smoothed_energy_pdf
        
        coarse_energy_pdf = coarse_energy_pdf.with_columns( pl.col('CETDatetime').dt.date().dt.truncate(coarse_freq).alias('Period') )

        coarse_energy_pdf = coarse_energy_pdf.group_by(['Period' , 'Peakhour']).agg(
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
        coarse_energy_pdf = coarse_energy_pdf.with_columns( ( pl.col('AvgChargeVolume') /   pl.col('AvgChargeVolume').sum()).over('Period').alias( 'PeakhoursShare' )  )

        # # just to plotly treated as categorical. why now? speed in previous groupbys!
        coarse_energy_pdf = coarse_energy_pdf.with_columns( pl.col('Peakhour').cast(pl.String) )

        fig = px.bar( 
            coarse_energy_pdf
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
            coarse_energy_pdf
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
            coarse_energy_pdf
            , x = 'Period' 
            , y = ['AvgChargePower_Min', 'AvgChargePower_Mean' , 'AvgChargePower_Max']
            , facet_row = 'Peakhour'
            , title= 'Avg. Charging Power Over Time'
            , color_discrete_sequence= ['green' , 'blue' , 'red' ]
            , line_shape='hvh'
            , height= 500
        )
        fig.update_yaxes(matches=None, showticklabels=True , title_text="ChargePower(W)")
        st.plotly_chart(fig,use_container_width=True)
        # fig.update_xaxes(tickangle=-80)

# # =============================================================================
# # Segmentation
# # =============================================================================
 
if 'trxns' in st.session_state:
    with st.expander("Segmentation"):

        fig = px.scatter(
            st.session_state.trxns
            , x = 'Utilization'
            , y = 'PeakhourShare'
            , color = 'Segment'
            , width=600
            , height=500
            , color_discrete_map= color_map

        )
        cols = st.columns((2, 5, 1))
        cols[1].plotly_chart(fig,use_container_width=False)

        sub_segment_session_share_pdf = st.session_state.trxns.group_by( 
            [ 'PeakhourShareSegment' , 'UtilizationSegment' ]
        ).agg( pl.col('TotalEnergy').count()/ 100 )
        sub_segment_session_share_pdf = sub_segment_session_share_pdf.to_pandas().set_index(
            [ 'PeakhourShareSegment' , 'UtilizationSegment' ]
        ).unstack().droplevel(0, axis =1).sort_index(ascending=False).reindex( ['LowUtlz', 'MidUtlz', 'HghUtlz'] , axis=1)


        sub_segment_energy_share_pdf = st.session_state.trxns.group_by( 
            [ 'PeakhourShareSegment' , 'UtilizationSegment' ]
        ).agg( pl.col('TotalEnergy').sum()/ st.session_state.trxns.select('TotalEnergy').sum().item() * 100 )
        sub_segment_energy_share_pdf = sub_segment_energy_share_pdf.to_pandas().set_index(
            [ 'PeakhourShareSegment' , 'UtilizationSegment' ]
        ).unstack().droplevel(0, axis =1).sort_index(ascending=False).reindex( ['LowUtlz', 'MidUtlz', 'HghUtlz'] , axis=1)

        cell_bg_colors = get_color_map( sub_segment_session_share_pdf )

        summary = sub_segment_session_share_pdf.style.format('{:.2f}%')\
        .set_properties(**{'text-align': 'left'})\
        .bar(color = 'black', vmin = 0,height = 30, align = 'zero' , axis = None)\
        .applymap(color_background)


        segment_session_share_pdf = st.session_state.trxns.group_by( [ 'Segment' ]).agg( pl.col('TotalEnergy').count() / 100).rename({'TotalEnergy':'SegmentShare'}).to_pandas().set_index('Segment')

        segment_energy_share_pdf = st.session_state.trxns.group_by( [ 'Segment' ]).agg( pl.col('TotalEnergy').sum() / st.session_state.trxns.select('TotalEnergy').sum().item() * 100).rename({'TotalEnergy':'EnergyShare'}).to_pandas().set_index('Segment')
        
        vars = ['Segment' ,'ConnectedTime', 'Utilization', 'PeakhourShare' , 'TotalEnergy', 'Throughput' ]
        avgs_pdf = st.session_state.trxns.group_by('Segment').agg( pl.col(vars).mean() ).to_pandas().set_index('Segment')
        avgs_pdf[['Utilization' ,'PeakhourShare']] *= 100

        final_table = pd.concat( 
            [
                segment_session_share_pdf
                , segment_energy_share_pdf 
                , avgs_pdf.add_prefix('Avg. ')
            ]
            , axis  = 1
        ).sort_index()
        index_names = final_table.index.to_list()
        final_table = final_table.reset_index(drop=True)
        final_table.index = index_names
        
        summary = final_table.style\
        .set_properties(**{'text-align': 'left'})\
        .format(result_format)\
        .bar(color = 'black', vmin = 0,height = 30, align = 'zero' , axis = 0)\
        .set_properties(**{'background-color': 'gold'}, subset=pd.IndexSlice[ ['Busy'] , :])\
        .set_properties(**{'background-color': 'green'}, subset=pd.IndexSlice[ ['EnergyPool'] , :])\
        .set_properties(**{'background-color': 'grey'}, subset=pd.IndexSlice[ ['Frugal'] , :])\
        .set_properties(**{'background-color': 'red'}, subset=pd.IndexSlice[ ['Queue'] , :])\
        .set_table_styles(
            [
                {
                    'selector': 'th',   'props': [('background-color', 'white')]
                }
            ]
        )

        st.components.v1.html(summary.to_html() ,scrolling=True, height=200)

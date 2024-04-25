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

segmentation_vars = [
    'ConnectedTime' , 'ChargeTime' , 'PeakConnectedTime'
    , 'TotalEnergy' ,'MaxPower' , 'AvgChargePower'
    , 'Utilization' ,'Throughput' , 'PeakhourShare' 
    ]

segmentation_short_vars = [
    'CntdTm' , 'ChrgTm' , 'PkCdTm'
    , 'TotEgy' ,'MaxPwr' , 'ACgPwr'
    , 'Utilzn' ,'Thrput' , 'PkhShr' 
    ]

short_map = { x:y for x,y in zip(segmentation_vars , segmentation_short_vars) }

result_format = {
        'SegmentShare':'{:.1f}%'
        , 'EnergyShare':'{:.1f}%'
        , 'ConnectedTime':'{:.1f}h'
        , 'PeakConnectedTime':'{:.1f}h'
        , 'ChargeTime':'{:.1f}h'
        , 'Utilization':'{:.1f}%'
        , 'PeakhourShare':'{:.1f}%'
        , 'TotalEnergy':'{:.1f}kWh'
        , 'Throughput':'{:.1f}kW'
        , 'MaxPower':'{:.1f}kW'
        , 'AvgChargePower':'{:.1f}kW'
        , 'Avg. ConnectedTime':'{:.1f}h'
        , 'Avg. PeakConnectedTime':'{:.1f}h'
        , 'Avg. ChargeTime':'{:.1f}h'
        , 'Avg. Utilization':'{:.1f}%'
        , 'Avg. PeakhourShare':'{:.1f}%'
        , 'Avg. TotalEnergy':'{:.1f}kWh'
        , 'Avg. Throughput':'{:.1f}kW'
        , 'Avg. MaxPower':'{:.1f}kW'

    , 'CntdTm':'{:.1f}h' 
    , 'ChrgTm' :'{:.1f}h'
    , 'PkCdTm':'{:.1f}h'
    , 'TotEgy' :'{:.1f}kWh'
    ,'MaxPwr' :'{:.1f}kW'
    , 'ACgPwr':'{:.1f}kW'
    , 'Utilzn' :'{:.1f}%'
    ,'Thrput' :'{:.1f}kW'
    , 'PkhShr' :'{:.1f}%'
        
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
            , ( pl.col('ChargeTimeRC') / pl.col('ConnectedTimeRC') *100 ).alias('Utilization')
            , ( pl.col('PeakConnectedTime') / pl.col('ConnectedTimeRC') *100 ).alias('PeakhourShare')
            , ( pl.col('TotalEnergy') / pl.col('ChargeTimeRC') ).alias('AvgChargePower')
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
            , category_orders = { 'Peakhour': [  '0' , '1' ] }
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

        selected_vars = st.multiselect(
            'Segmentation Variables'
            , segmentation_vars  
            , max_selections = 4
        )
        var_nr = len(selected_vars)
        
        
        if var_nr > 0:
            st.write(' Number of Segments:')
            cols = st.columns(  var_nr )


            segmentation_pdf = st.session_state.trxns#.select( selected_vars )#.sample(2500*var_nr).sort(selected_vars)

            split_list = []

            for i , var_name in enumerate(selected_vars):
                # var_name = selected_vars[i]
                short_name = short_map[selected_vars[i]]
                p = cols[i].number_input( var_name , step = 1 , min_value = 1 , max_value = 4 )
                # split_list.append(p)
                quantile_rank = cols[i].checkbox(f'{short_name} Rank')

                bins = [ i/p for i in range(p+1) ][1:-1]

                if quantile_rank:
                    segmentation_pdf = segmentation_pdf.with_columns(
                        (pl.col(var_name).rank( method = 'random')/ pl.col(var_name).count() ).cut( bins , labels=[ f'{short_name}{i+1}' for i in range(p) ]).alias(f'{short_name}Grp')
                    )
 
                else:
                    segmentation_pdf = segmentation_pdf.with_columns(
                        ((pl.col(var_name) - pl.col(var_name).min()) / (pl.col(var_name).max() - pl.col(var_name).min()) ).cut( bins , labels=[ f'{short_name}{i+1}' for i in range(p) ]).alias(f'{short_name}Grp')
                    )


                cols[i].plotly_chart( 
                    px.histogram( 
                        segmentation_pdf 
                        , x = var_name
                        , color = f'{short_name}Grp'
                        ,histnorm='percent'
                        , category_orders = { f'{short_name}Grp': sorted(segmentation_pdf.select(f'{short_name}Grp').unique().to_series().to_list() )  }
                    ) 
                    , use_container_width=True
                )
            


            segmentation_pdf = segmentation_pdf.with_columns(
                pl.concat_str(
                    pl.col(
                        [ f'{short_map[x]}Grp' for  x in selected_vars]
                    )
                    , separator ='_'
                ).alias(f'SubSegment')
            )

            if var_nr > 2:


                # fig = px.imshow(
                #     segmentation_pdf.select( pl.selectors.numeric() ).corr().with_columns( pl.all().round(2) )

                #     , height= 150 * var_nr
                #     , width = 150 * var_nr
                #     , zmin = -1
                #     , text_auto=True
                #     , title = 'Correlation Matrix'
                # )
                # st.plotly_chart(fig,use_container_width=True)
                

                fig = px.scatter_matrix(
                    segmentation_pdf
                    ,  dimensions= selected_vars
                    , color = 'SubSegment'
                    # , color_discrete_map= color_map
                    , title='Distributions'
                    , width=800
                    , height=800
                    , category_orders = { f'SubSegment': sorted(segmentation_pdf.select('SubSegment').unique().to_series().to_list() )  }

                )
                fig.update_traces(
                    showupperhalf = False
                    , diagonal_visible=False
                )
                st.plotly_chart(fig,use_container_width=True)
            elif var_nr ==2:

                # fig = px.imshow(
                #     segmentation_pdf.select( pl.selectors.numeric() ).corr().with_columns( pl.all().round(2) )

                #     , height= 150 * var_nr
                #     , width = 150 * var_nr
                #     , zmin = -1
                #     , text_auto=True
                #     , title = 'Correlation Matrix'
                # )
                # st.plotly_chart(fig,use_container_width=True)
                fig = px.scatter(
                    segmentation_pdf
                    , x = selected_vars[0]
                    , y = selected_vars[1]
                    , color = 'SubSegment'
                    # , color_discrete_map= color_map
                    , title='Distributions'
                    , width=400
                    , height=400
                    , category_orders = { f'SubSegment': sorted(segmentation_pdf.select('SubSegment').unique().to_series().to_list() )  }

                )
                
                st.columns(4)[1].plotly_chart(fig,use_container_width=False)
            else:
                fig = px.box(
                    segmentation_pdf
                    , x = selected_vars[0]
                    , color = 'SubSegment'
                    # , color_discrete_map= color_map
                    , title='Distributions'
                    , width=800
                    , height=800

                )
                
                st.plotly_chart(fig,use_container_width=True)

            avgs_pdf = segmentation_pdf.group_by('SubSegment').agg( pl.col(segmentation_vars).mean() ).to_pandas().set_index('SubSegment')

            final_table = pd.concat( 
                [
                    avgs_pdf.rename(columns =short_map)#.add_prefix('Avg. ')
                ]
                , axis  = 1
            ).sort_index()

            summary = final_table.style\
            .set_properties(**{'text-align': 'left'})\
            .format(result_format)\
            .bar(color = 'black', vmin = 0,height = 30, align = 'zero' , axis = 0)\
            .set_properties(**{'background-color': 'white'})\
            .set_table_styles(
                [
                    {
                        'selector': 'th',   'props': [('background-color', 'white') , ('min-width', '120px')]
                    }
                ]
            )#.set_sticky(axis="columns").set_sticky()  
            st.components.v1.html(summary.to_html() ,scrolling=True, height=200 )
        
        # ns = st.number_input( 'Number of Segments' , step = 1 , min_value = 1 , max_value = 5 )

        # cols = st.columns(ns)
        # sub_segments = set( segmentation_pdf.select('SubSegment').unique().sort('SubSegment').to_series().to_list())
        # st.write(sub_segments)

        # if 'all_selected' not in st.session_state:
            
        #     st.write('Babe')

        # # if 'clicked_clear' not in st.session_state:
        # #     st.session_state.clicked_clear = False

        # def click_button():
        #     # st.session_state.clicked_clear = True
        #     st.session_state.all_selected = []

        # def add_segment():
        #     # st.session_state.clicked_clear = True
        #     st.session_state.all_selected = []

        # st.button('Clear Lists', on_click=click_button)

        # st.write(st.session_state.all_selected)

        # for i  in range(ns):
        #     selected = cols[i].multiselect(f'Segment{i+1}' ,sorted (sub_segments - set(st.session_state.all_selected)) )
        #     for x in selected:
        #         st.session_state.all_selected.append(x)
        
        

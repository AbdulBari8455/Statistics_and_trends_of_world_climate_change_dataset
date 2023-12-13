import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def lineplot(x, y, xlabel, ylabel, title, labels):
    """ Funtion to Create Lineplot. Arguments:
        list of values for xaxis
        list of values for yaxis
        xlabel, ylabel and titel value
        color name
        label value
    """
    plt.style.use('tableau-colorblind10')
    plt.figure(figsize=(7, 5))
    line_styles = itertools.cycle(['-', '--', '-.', ':'])
    markers = itertools.cycle(['o', '^', 's', 'D', 'p'])
    for i, (y_value, line_style, marker) in enumerate(
            zip(y, line_styles, markers)):
        plt.plot(
            x,
            y_value,
            label=labels[i],
            linestyle=line_style,
            marker=marker)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    return


def barplot(dataframe, xlabel, ylabel, title):
    """
    Function to Create Bar Plot

    Parameters
    ----------
    dataframe : Pandas Dataframe
        Using pandas dataframe to plot graph.
    xlabel : string
        x_label value.
    ylabel : string
        ylabel value.
    title : string
        title of Visulization.

    Returns
    -------
    None.

    """
    dataframe.plot(kind='bar', figsize=(10, 6))
    # Set plot labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('barplot.jpg', dpi=500)
    plt.show()
    return


def draw_scatter_plot(
        x,
        y,
        title='Scatter Plot',
        xlabel='X-Axis',
        ylabel='Y-Axis',
        color='blue',
        marker='o',
        label=None):
    """
    Draw a scatter plot using Matplotlib.

    Parameters:
        x (list): X-axis data points.
        y (list): Y-axis data points.
        title (str): Title of the scatter plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        color (str): Color of the markers.
        marker (str): Marker style.
        label (str): Label for the legend.
    """
    plt.scatter(x, y, color=color, marker=marker, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if label is not None:
        plt.legend()
    plt.show()


def make_corr_heat_map(df, title, cmap='viridis'):
    """
    Function to Create corr heatmap

    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe with features to correlate.
    title : String
        Name of the country to Show.
    cmap : Color schme to follow.
        DESCRIPTION. The default is 'viridis'.

    Returns
    -------
    None.

    """
    # Finding Correlation among data features.
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.pcolormesh(
        correlation_matrix,
        cmap=cmap,
        edgecolors='w',
        linewidth=0.5)
    cbar = plt.colorbar(heatmap)
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            ax.text(j + 0.5, i + 0.5, f'{correlation_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', color='white')
    ax.set_xticks(np.arange(len(correlation_matrix.columns)) + 0.5)
    ax.set_yticks(np.arange(len(correlation_matrix.columns)) + 0.5)
    ax.set_xticklabels(correlation_matrix.columns, rotation=90, ha='center')
    ax.set_yticklabels(correlation_matrix.columns, va='center')
    plt.title(title)
    plt.show()


def get_refine_data(df, countries, start_year, up_to_year):
    """
    Function to refine and extract data base on countries,
    from start to end year

    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe to refine and extract Data.
    countries : list of Countries
        Which Countries data needed to be Extracted.
    start_year : string
        year.
    up_to_year : string
        Year.

    Returns
    -------
    selected_data : DataFrame
        Refine DatadFrame containing mentioned countries Data.

    """

    # Taking the Transpose of the DataFrame
    df = df.T
    # Droping the Unnecessary Rows
    df = df.drop(['Country Code', 'Indicator Name', 'Indicator Code'])
    # Giving name(name of countries) to the columns of new dataframe
    df.columns = df.iloc[0]
    # Droping the countries row
    df = df.drop(['Country Name'])
    df = df.reset_index()
    df['Years'] = df['index']
    df = df.drop('index', axis=1)
    # Slicing the Data Based on Start and end year
    df = df[(df['Years'] >= start_year) & (df['Years'] <= up_to_year)]
    # slicing Data frame for only required Countries.
    selected_data = df[countries]
    # Taking mean of Whole column of fill the missing Values
    selected_data = selected_data.fillna(selected_data.iloc[:, :-1].mean())
    return selected_data


def get_data_for_one_country(
        data_frame_list,
        country_name,
        names,
        start_year,
        end_year):
    """
    Function to get data of specific countty.

    Parameters
    ----------
    data_frame_list : DataFrame List
        List of Python DataFrames for each selected indicator.
    country_name : List of required countries.
        Countries.
    names : List of indicators
        names of Selected indicators.
    start_year : String
        Start Year.
    end_year : String
        Ending year.

    Returns
    -------
    country_data : DataFrame
        Getting the Dataframe showing all the indicators for specific country.

    """
    country_data = []
    for i, data in enumerate(data_frame_list):
        # choosing the required countries data
        data = data[country_name]
        data = data.rename(columns={country_name[0]: names[i]})
        country_data.append(data)
    country_data = pd.concat(country_data, axis=1)
    country_data = country_data.T.drop_duplicates().T
    # Dropping the Years columns as its not required.
    country_data = country_data.drop('Years', axis=1)
    return country_data


def get_lists(df, cols):
    """
    Funtion to convert Dataframe columns to lists.
    Parameters
    ----------
    df : pandas dataframe
        Dataframe to convert in to lists.
    cols: List of required countries.
        Countries.
     Returns
     -------
     column_lists : list
         Getting lists of columns data.

    """
    column_lists = [df[col].tolist() for col in cols[:-1]]
    return column_lists


def data_for_bar(df, years):
    """
    Function to get the data for bar plot.

    Parameters
    ----------
    df : Pandas dataframe
        data needed to use for bar.
    years : list of years
        Required data for the specific Years.

    Returns
    -------
    df : Data Frame
        Data Ready to plot bar graph.

    """
    # Looking for the required years.
    df = df[df['Years'].isin(years)]
    # Taking Transpose as we need years as columns.
    df = df.T
    df.columns = df.iloc[-1]
    df = df.drop(['Years'])
    return df


def data_description(dfs, country_name, names, start_year, end_year):
    """
    Funtion for the Description of speciic country.

    Parameters
    ----------
    dfs : Pandas DataFrame
        DataFrames For the indicators.
    Country_name : list.
        Name of the required conutry.
    names : list.
        list of Indicators to rename coloumns.
    start_year : string
        start from year.
    end_year : string
        end on year.
    """

    df = get_data_for_one_country(
        dfs, country_name, names, start_year, end_year)
    df.columns.name = country_name[0]
    df = df.apply(pd.to_numeric, errors='coerce')
    print(df.describe())


def extract_data_for_ind(df, Indicators):
    """
    Funtion to extract Data For the Required years.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame which needs to be sliced.
    Indicators : List of indicators.
        List of required indicators.

    Returns
    -------
    selected_ind_data : Pandas Dataframe
        Refine Dataframe of Required Indicators.

    """
    selected_ind_data = []
    for ind in Indicators:
        # Extracting Data Base on Indicator type.
        data = df[df['Indicator Name'] == ind]
        # Appending Data in the list for each Dataframe.
        selected_ind_data.append(data)
    return selected_ind_data


def get_data_for_specific_countries(ind_lis, country_list, s_year, e_year):
    """
    To Extract Data For a Specific country.
    Parameters
    ----------
    ind_lis : list of Pandas Dataframe
        List of Dataframes for each indicator.
    country_list : countories names
        name of the countries that we required from each indicator.
    s_year : Start Year
        Start Year.
    e_year : End Year
        Start Year.

    Returns
    -------
    data_for_countries : Dataframes List
        clean data frames for required and countries and each indicator.

    """
    data_for_countries = []
    for df in ind_lis:
        data = get_refine_data(df, country_list, s_year, e_year)
        data_for_countries.append(data)
    return data_for_countries


def refine_data_and_make_graphs(
        df,
        Indicators,
        cols,
        years,
        names,
        start_year,
        end_year):
    """
    Function to refine the data and make graphs.

    Parameters
    ----------
    df : Pandas DataFrame.
        Original DataFrame.
    Indicators : list
        List of indicator we want to explore.
    cols : list
        List of Countries we want to explore.
    years : list
        list of years we are intrested about.
    names : list
        names of indicators.
    start_year : string
        start from year.
    end_year : string
        end on year.

    Returns
    -------
    None.

    """
    dataframes = extract_data_for_ind(df, Indicators)
    dataframes = get_data_for_specific_countries(
        dataframes, cols, start_year, end_year)
    lineplot(list(dataframes[1]['Years']),
             get_lists(dataframes[3],
                       cols),
             'Years',
             '(kWh per capita)',
             'Electric power consumption (kWh per capita)',
             cols[:-1])
    lineplot(list(dataframes[3]['Years']),
             get_lists(dataframes[5],
                       cols),
             'Years',
             '(metric tons per capita)',
             'CO2 emissions (metric tons per capita)',
             cols[:-1])
    barplot(
        data_for_bar(
            dataframes[4],
            years),
        'Countries',
        "(kt of CO2 equivalent)",
        "Total greenhouse gas emissions (kt of CO2 equivalent)")
    barplot(
        data_for_bar(
            dataframes[6],
            years),
        'Countries',
        'Population',
        'Urban Population')
    country_name = [cols[-7], 'Years']
    get_data_for_one_country(
        dataframes,
        country_name,
        names,
        start_year,
        end_year)
    make_corr_heat_map(
        get_data_for_one_country(
            dataframes,
            country_name,
            names,
            '1990',
            '2020'),
        country_name[0],
        'cool')
    country_name = [cols[-5], 'Years']
    get_data_for_one_country(
        dataframes,
        country_name,
        names,
        start_year,
        end_year)
    make_corr_heat_map(
        get_data_for_one_country(
            dataframes,
            country_name,
            names,
            '1990',
            '2020'),
        country_name[0],
        'tab20')
    country_name = [cols[-3], 'Years']
    get_data_for_one_country(
        dataframes,
        country_name,
        names,
        start_year,
        end_year)
    make_corr_heat_map(
        get_data_for_one_country(
            dataframes,
            country_name,
            names,
            '1990',
            '2020'),
        country_name[0],
        'turbo')
    country_name = [cols[-7], 'Years']
    data_description(dataframes, country_name, names, start_year, end_year)


world_bank_data = pd.read_csv('World_Climate data.csv', skiprows=4)
names = [
    'Agricultural_land',
    'Electric power consumption',
    'Forest area',
    'CO2_emissions',
    'Total greenhouse gas emissions',
    'GDP',
    'Urban population']
Indicators = [
    "Agricultural land (sq. km)",
    "Electric power consumption (kWh per capita)",
    "Forest area (sq. km)",
    "CO2 emissions (metric tons per capita)",
    "Total greenhouse gas emissions (kt of CO2 equivalent)",
    "Agriculture, forestry, and fishing, value added (% of GDP)",
    "Urban population"]
cols = [
    'United Kingdom',
    'Malaysia',
    'Canada',
    'Australia',
    'Colombia',
    'South Africa',
    'Morocco',
    'Years']
years = ['1995', '2000', '2005', '2010', '2015', '2020']
start_year = '1990'
end_year = '2021'
refine_data_and_make_graphs(
    world_bank_data,
    Indicators,
    cols,
    years,
    names,
    start_year,
    end_year)

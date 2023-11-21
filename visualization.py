import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px



def count_plot(name,df):
    sns.countplot(x=name,data=df)
    plt.show()
    
def plot_hist(name,df):
    df[name].hist()
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.title('Histogram of' + name)
    plt.show()
    
def StormCountByCategory(df):
    grouped = df.groupby('Nature (N/A)')['Serial_Num (N/A)'].unique().to_frame().reset_index()
    grouped['Num'] = grouped['Serial_Num (N/A)'].str.len()
    sns.barplot(x='Nature (N/A)', y='Num', data=grouped)
    #grouped.plot.bar(x='Nature (N/A)', y='Num')
    plt.ylabel('Number of Storms')
    plt.show()
    
def HurricanePointbyYear(hurricane_data):
    count = hurricane_data['ISO_time'].dt.year.value_counts()
    count_df = pd.DataFrame(count)
    count_df = count_df.reset_index()
    count_df = count_df.rename(columns={"ISO_time": "Year", "count": "Count"})
    count_df = count_df.sort_values(by = "Year", ascending = True)
    count_df.plot(x = "Year",y = "Count")
    plt.show()
    
def NumHurricanebyYear(hurricane_data):
    grouped_by_year = hurricane_data.groupby('Season (Year)')['Serial_Num (N/A)'].unique().to_frame().reset_index()
    grouped_by_year['NumStorms'] = grouped_by_year['Serial_Num (N/A)'].str.len()
    grouped_by_year['NumStorms'] = grouped_by_year['NumStorms'].astype('int')
    grouped_by_year.plot(x='Season (Year)', y='NumStorms')
    plt.ylabel('No. of Storms')
    plt.show()
    
def HurricanePointbyMonth(hurricane_data):
    count = hurricane_data['ISO_time'].dt.month.value_counts()
    count_df = pd.DataFrame(count)
    count_df = count_df.reset_index()
    count_df = count_df.rename(columns={"ISO_time": "Month", "count": "Count"})
    for month in range(1,13):
        if month not in count_df['Month'].values:
            count_df = count_df._append({'Month': month, 'Count': 0}, ignore_index=True)
        
    count_df = count_df.sort_values(by = "Month", ascending = True)
    count_df.plot.bar(x = "Month",y = "Count")
    plt.ylabel('No. of Storms Point')
    plt.show()
    

def NumHurricanebyMonth(hurricane_data):
    grouped_by_month = hurricane_data.groupby('month')['Serial_Num (N/A)'].unique().to_frame().reset_index()
    grouped_by_month['NumStorms'] = grouped_by_month['Serial_Num (N/A)'].str.len()
    grouped_by_month['NumStorms'] = grouped_by_month['NumStorms'].astype('int')
    for month in range(1,13):
        if month not in  grouped_by_month['month'].values:
            grouped_by_month =  grouped_by_month._append({'month': month, 'Serial_Num (N/A)': [], 'NumStorms':0}, ignore_index=True)
    grouped_by_month = grouped_by_month.sort_values(by = "month", ascending = True)
    sns.barplot(x='month', y='NumStorms', data=grouped_by_month)
    plt.ylabel('No. of Storms')
    plt.show()
    
def scatter_plot(x,y,x_lab,y_lab,title):
    plt.scatter(x, y)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()
    
def Line_plot(x,y,x_lab,y_lab,title):
    plt.plot(x, y)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()


def plot_(df,xlab,ylab):
    df.plot(x = xlab,y = ylab)
    plt.show()
    
    
#########################VISUALIZATION FOR DASHBOARD ONLY#############################################


def YearTrend(hurricane_data):
    grouped_by_year = hurricane_data.groupby('Season (Year)')['Serial_Num (N/A)'].unique().to_frame().reset_index()
    grouped_by_year['NumStorms'] = grouped_by_year['Serial_Num (N/A)'].str.len()
    grouped_by_year['NumStorms'] = grouped_by_year['NumStorms'].astype('int')
    fig = px.line(grouped_by_year, x='Season (Year)', y='NumStorms')
    
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='No of Storms')
    return fig
    

def HurrSeasonality(hurricane_data):
    grouped_by_month = hurricane_data.groupby('month')['Serial_Num (N/A)'].unique().to_frame().reset_index()
    grouped_by_month['NumStorms'] = grouped_by_month['Serial_Num (N/A)'].str.len()
    grouped_by_month['NumStorms'] = grouped_by_month['NumStorms'].astype('int')
    for month in range(1,13):
        if month not in  grouped_by_month['month'].values:
            grouped_by_month =  grouped_by_month._append({'month': month, 'Serial_Num (N/A)': [], 'NumStorms':0}, ignore_index=True)
    grouped_by_month = grouped_by_month.sort_values(by = "month", ascending = True)
    fig = px.bar(grouped_by_month, x='month', y='NumStorms')
    return fig

def HurrCategory(df):
    grouped = df.groupby('Nature (N/A)')['Serial_Num (N/A)'].unique().to_frame().reset_index()
    grouped['NumStorms'] = grouped['Serial_Num (N/A)'].str.len()
    fig = px.bar(grouped, x='Nature (N/A)', y='NumStorms',color = 'Nature (N/A)')
    return fig


    
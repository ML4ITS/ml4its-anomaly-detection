from pandas import read_csv
from matplotlib import pyplot
import pandas as pd

def plot(dataset):
    # line plots
    # line plot for each variable
    print('Historical Spot price visualization:')
    pyplot.figure(figsize = (25,5))
    pyplot.plot(dataset)
    pyplot.title('Henry Hub Spot Price (Daily frequency)')
    pyplot.xlabel ('Date_time')
    pyplot.ylabel ('Price ($/Mbtu)')
    pyplot.show()

def plot_yearly(dataset, column="price"):
    # yearly line plots
    # plot active power for each year
    years = pd.DatetimeIndex(dataset.index).year.unique().astype(str)
    feature = column
    pyplot.figure(figsize=(30,22))
    for i in range(len(years)):
        # prepare subplot
        ax = pyplot.subplot(len(years), 1, i+1)
        # determine the year to plot
        year = years[i]
        # get all observations for the year
        result = dataset[str(year)]
        # plot the active power for the year
        pyplot.plot(result[feature])
        # add a title to the subplot
        pyplot.title(str(year), y=0, loc='left')
    pyplot.show()

def plot_monthly(dataset, column="price", year="2020"):
    # monthly line plots
    # plot active power for each year
    months = [x for x in range(1, 13)]
    pyplot.figure(figsize=(30,22))
    for i in range(len(months)):
        # prepare subplot
        ax = pyplot.subplot(len(months), 1, i+1)
        # determine the month to plot
        month = str(year) + '-' + str(months[i])
        # get all observations for the month
        result = dataset[month]
        # plot the active power for the month
        pyplot.plot(result[column])
        # add a title to the subplot
        pyplot.title(month, y=0, loc='left')
    pyplot.show()

def plot_daily(dataset, column="price", year="2020", month="1"):
    # daily line plots
    # plot active power for each year
    days = [x for x in range(1, 20)]
    pyplot.figure(figsize=(30,28))
    for i in range(len(days)):
        # prepare subplot
        ax = pyplot.subplot(len(days), 1, i+1)
        # determine the day to plot
        day = str(year) + "-" + str(month) + '-' + str(days[i])
        # get all observations for the day
        result = dataset[day]
        # plot the active power for the day
        pyplot.plot(result[column])
        # add a title to the subplot
        pyplot.title(day, y=0, loc='left')
    pyplot.show()
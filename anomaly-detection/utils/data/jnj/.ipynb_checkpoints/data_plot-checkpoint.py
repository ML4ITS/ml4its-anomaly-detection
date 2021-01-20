from pandas import read_csv
from matplotlib import pyplot
import pandas as pd

def plot(dataset):
    # line plots
    # line plot for each variable
    pyplot.figure(figsize=(30,12))
    for i in range(len(dataset.columns)):
        pyplot.subplot(len(dataset.columns), 1, i+1)
        name = dataset.columns[i]
        pyplot.plot(dataset[name])
        pyplot.title(name, y=0)
    pyplot.show()

def plot_yearly(dataset, column):
    # yearly line plots
    # plot active power for each year
    years = ['2007', '2008', '2009', '2010']
    feature = column
    pyplot.figure(figsize=(30,12))
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

def plot_monthly(dataset, column):
    # monthly line plots
    # plot active power for each year
    months = [x for x in range(1, 13)]
    pyplot.figure(figsize=(30,12))
    for i in range(len(months)):
        # prepare subplot
        ax = pyplot.subplot(len(months), 1, i+1)
        # determine the month to plot
        month = '2007-' + str(months[i])
        # get all observations for the month
        result = dataset[month]
        # plot the active power for the month
        pyplot.plot(result[column])
        # add a title to the subplot
        pyplot.title(month, y=0, loc='left')
    pyplot.show()

def plot_daily(dataset, column):
    # daily line plots
    # plot active power for each year
    days = [x for x in range(1, 20)]
    pyplot.figure(figsize=(30,18))
    for i in range(len(days)):
        # prepare subplot
        ax = pyplot.subplot(len(days), 1, i+1)
        # determine the day to plot
        day = '2007-01-' + str(days[i])
        # get all observations for the day
        result = dataset[day]
        # plot the active power for the day
        pyplot.plot(result[column])
        # add a title to the subplot
        pyplot.title(day, y=0, loc='left')
    pyplot.show()
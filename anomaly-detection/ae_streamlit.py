from utils.data.henryhub_gas_spotprice import data_reader 
from utils.data.henryhub_gas_spotprice import data_plot 
from utils.data import data_util
import pandas as pd
from matplotlib import pyplot
import math
import sklearn
import numpy as np
import streamlit
from tensorflow import keras
from models import keras_lstm_ae
from sklearn import preprocessing 
import altair as alt

import streamlit as st

n_steps = 30

@st.cache()
def train_model(train_pressed, epochs, dataset):
    if True:
        print("DO")
        scaler = preprocessing.RobustScaler(quantile_range=(25, 75))
        train, test = data_util.split(dataset, dataset.columns[0], scaler=scaler, test_size=0.25) #0.05

        # Create training/test data
        col_name = dataset.columns[0]
        X_train, y_train = data_util.create_dataset(train[[col_name]], train[col_name], n_steps)
        X_test, y_test = data_util.create_dataset(test[[col_name]], test[col_name], n_steps)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)
        print("-------------------")

        n_timesteps = X_train.shape[1]
        n_features = X_train.shape[2]
        units = 64
        dropout = 0.2
        loss = 'mae'
        optimizer = 'adam'
        #epochs = 10
        model = keras_lstm_ae.create_lstm_ae(input_shape=(n_timesteps, n_features), dropout=dropout, units=units)
        #model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        model.compile(loss = loss, optimizer = optimizer)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')], shuffle=False) #X_train
        return model, history, train, test, X_train, y_train, X_test, y_test
    
@st.cache()
def load_data(add_selectbox_dataset):
    if(add_selectbox_dataset == 'Gas Spot Price'):
        reader = data_reader.ReadData()
        dataset = reader.load_data()
        dataset = dataset.loc['2000-01-01':,['price']] 
        dataset = dataset.loc[dataset.index<'2020-09-01']
    else: 
        dataset = None
    return dataset

@st.cache()
def calculate_mae_loss(model, X_test):
    # Calculate Threshold
    # MAE on the test data:
    X_test_pred = model.predict(X_test)
    print('Predict shape (X_test_pred):', X_test_pred.shape); print();
    mae = np.mean(np.abs(X_test_pred - X_test), axis=1)
    # reshaping prediction
    pred = X_test_pred.reshape((X_test_pred.shape[0] * X_test_pred.shape[1]), X_test_pred.shape[2])
    print('Prediction:', pred.shape); print();
    print('Test data shape (X_test):', X_test.shape); print();
    # reshaping test data
    X_test_reshape = X_test.reshape((X_test.shape[0] * X_test.shape[1]), X_test.shape[2])
    print('Test data (X_test_reshape):', X_test_reshape.shape); print();
    # error computation
    errors = X_test_reshape - pred
    print('Error:', errors.shape); print();
    # rmse on test data
    RMSE = math.sqrt(sklearn.metrics.mean_squared_error(X_test_reshape, pred))
    #st.write('Test RMSE: %.3f' % RMSE);

    test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)
    return test_mae_loss;
    


def main():
    st.set_page_config(layout="wide")
    
    # Add a selectbox to the sidebar:
    add_selectbox_dataset = st.sidebar.selectbox(
        'Dataset?',
        ('Gas Spot Price', 'House Hold Power', 'JNJ')
    )

    add_selectbox_model = st.sidebar.selectbox(
        'Model',
        ('LSTM-AE', 'CNN-AE')
    )

    st.title(add_selectbox_dataset + " - Dataset")
    dataset = load_data(add_selectbox_dataset)
    #fig, ax = pyplot.subplots(figsize=(20,10))
    #ax.plot(dataset)
    #streamlit.pyplot(fig)
    st.line_chart(dataset)
    
    #fig, ax = data_util.vis_behaviour(dataset, dataset.columns[0])
    #fig.set_size_inches(11,5)
    #streamlit.pyplot(fig)
    
    epochs = st.sidebar.slider(
        'Epochs',
        0, 100, 10,1
    )
    train_pressed = st.sidebar.button("Train")
    print(train_pressed)
    model, history, train, test, X_train, y_train, X_test, y_test = train_model(
        False, 
        epochs, 
        dataset)
    
    
    # Space out the maps so the first one is 2x the size of the other three
    c1, c2 = st.beta_columns((2, 1))
    with c1:
        st.write("Validation and Training Loss")
        st.line_chart(history.history)
    with c2:
        # Calculate Threshold an plot MAE
        test_mae_loss = calculate_mae_loss(model, X_test)
        #fig, ax = pyplot.subplots(figsize=(4,1))
        #ax.hist(test_mae_loss, bins=50)
        #ax.set_xlabel('Test MAE loss')
        #ax.set_ylabel('Number of samples');
        #st.sidebar.pyplot(fig)
        
        st.write("Test MAE Loss")
        import altair as alt
        import pandas as pd
        import numpy as np
        df = pd.DataFrame();
        df['loss'] = test_mae_loss.reshape(len(test_mae_loss))
        base = alt.Chart(df)
        hist = base.mark_bar().encode( 
            x=alt.X('loss:Q', bin=alt.BinParams(maxbins=80), axis=alt.Axis(title='Test MAE Loss'), title='Loss'), y='count()') 
        st.altair_chart(hist,use_container_width=True)


    st.write(np.min(test_mae_loss), np.max(test_mae_loss), np.max(test_mae_loss))
    
    # Choose threhold
    threshold = st.slider(
        'Threshold',
        float(np.min(test_mae_loss)), float(np.max(test_mae_loss)), float(np.mean(test_mae_loss))
    )
    test_score_df = pd.DataFrame(test[n_steps:])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    test_score_df[dataset.columns[0]] = test[n_steps:][dataset.columns[0]]

    st.line_chart(test_score_df[["loss","threshold"]],use_container_width=True)
        
    anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
    a = alt.Chart(test_score_df.reset_index()).mark_line().encode( 
    x="Day", y=dataset.columns[0]) 
    b = alt.Chart(anomalies.reset_index()).mark_point(color="red").encode( 
        x="Day", 
        y=dataset.columns[0],
        color=alt.Color('anomaly:N', scale=alt.Scale(scheme='reds'), legend=alt.Legend(title="Anomaly"))
    ) 
    st.altair_chart((a+b),use_container_width=True)
    #fig,ax = pyplot.subplots(figsize=(35,10))
    #ax.plot(test_score_df.index, (test_score_df[dataset.columns[0]]))
    #ax.scatter(anomalies.index, (anomalies[dataset.columns[0]]), marker='o', color="red")
    #st.write(fig)
        
    
main()
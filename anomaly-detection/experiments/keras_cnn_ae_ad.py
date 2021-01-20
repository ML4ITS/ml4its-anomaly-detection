from utils.data.householdpower import data_reader 
from utils.data.householdpower import data_util 
from tensorflow import keras

def main():
    reader = data_reader.ReadData()
    dataset = reader.load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = data_util.train_val_test_split(dataset)

    input_shape = x_train.shape[1:]

    conv_layers = 3
    dense_layers = 1
    dense_units = 64
    n_kernels = 16
    learning_rate = 1.e-5
    kernel_size = (3, 3)
    max_epochs = 30
    batch_size = 32
    verbose = 1

    model = create_simple_convnet(input_shape, conv_layers, dense_layers, n_kernels, dense_units, kernel_size)

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mae", optimizer=opt)

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                           epochs=max_epochs, batch_size=batch_size,
                           verbose=verbose, callbacks=[es])

    cnn_mae = test_loss(x_test, y_test, model)

    # print(f"FCNN MAE: {fcnn_mae:.4f}")
    print(f"CNN MAE: {cnn_mae:.4f}")
    # print(f"CNN/FCNN: {cnn_mae / fcnn_mae:.4f}")
    return model;
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='householdpower')
    args = parser.parse_args()
    main(args.dataset)
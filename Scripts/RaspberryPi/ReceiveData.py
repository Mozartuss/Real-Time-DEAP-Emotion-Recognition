import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpysocket import NumpySocket
import tflite_runtime.interpreter as tflite

# from tensorflow import lite as tflite

logger = logging.getLogger('PI Emotion Recognition')
logger.setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)

root_path = Path(__file__).parent


# root_path = ROOT_PATH


def bin_power_optimized(x_vector, band, fs):
    c = np.abs(np.fft.fft(x_vector))
    indices = np.floor(band / fs * len(x_vector)).astype(int)
    power = np.zeros(len(band) - 1)
    for freq_index in range(len(band) - 1):
        power[freq_index] = np.sum(c[indices[freq_index]:indices[freq_index + 1]])
    power_ratio = power / np.sum(power)
    return power, power_ratio


def preprocess_data(data, label):
    logger.info(f"preprocessing data")
    window_channel_data = []
    for window in range(0, data.shape[1], 16):
        channel_data = []
        for channel in range(data.shape[0]):
            # Slice raw data over 2 sec, at interval of 0.125 sec
            window_data = data[channel, window: window + 256]
            x = bin_power_optimized(window_data, np.array([4, 8, 12, 16, 25, 45]), 128)
            channel_data.append(x[0])
        channel_data = np.concatenate(channel_data)
        window_channel_data.append(channel_data)
    data = np.array(window_channel_data)
    label_mapping = {"HAHV": [1, 0, 0, 0], "HALV": [0, 1, 0, 0], "LAHV": [0, 0, 1, 0], "LALV": [0, 0, 0, 1]}
    label = label_mapping[label.item(0)]
    label = np.array(label)
    data = data.reshape((data.shape[0], data.shape[1], 1))
    return data, label


def load_model(model_path: str):
    path = Path(root_path, model_path)
    if not path.exists():
        logger.warning(f"model in {path} does not exist")
        return
    else:
        logger.info(f"model in {model_path} exists")
        interpreter = tflite.Interpreter(model_path=str(path))

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        logger.info("\n== Input details ==")
        logger.info("name:", input_details[0]['name'])
        logger.info("shape:", input_details[0]['shape'])
        logger.info("type:", input_details[0]['dtype'])

        logger.info("\n== Output details ==")
        logger.info("name:", output_details[0]['name'])
        logger.info("shape:", output_details[0]['shape'])
        logger.info("type:", output_details[0]['dtype'])

        interpreter.resize_tensor_input(input_details[0]['index'], (32, 35, 1))
        interpreter.resize_tensor_input(output_details[0]['index'], (32, 4))

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        logger.info("\n== Input details ==")
        logger.info("name:", input_details[0]['name'])
        logger.info("shape:", input_details[0]['shape'])
        logger.info("type:", input_details[0]['dtype'])

        logger.info("\n== Output details ==")
        logger.info("name:", output_details[0]['name'])
        logger.info("shape:", output_details[0]['shape'])
        logger.info("type:", output_details[0]['dtype'])

        interpreter.allocate_tensors()

        return interpreter, input_details, output_details


def receive_data_and_predict(interpreter, input_details, output_details):
    with NumpySocket() as s:
        s.bind(('', 12345))
        s.listen()
        conn, addr = s.accept()
        logger.info(f"connected: {addr}")
        try:
            while True:
                frame = conn.recv()
                if frame.size != 0:
                    logger.info("data received")
                    start_time = datetime.now()
                    data = np.array(frame[0])
                    label = np.array(frame[1])
                    data, label = preprocess_data(data, label)
                    data = data.astype(np.float32)
                    interpreter.set_tensor(input_details[0]['index'], data)
                    interpreter.invoke()
                    results = interpreter.get_tensor(output_details[0]['index'])
                    end_time = datetime.now()
                    elapsed = end_time - start_time
                    logger.info(f"Finished prediction in {elapsed.total_seconds()} seconds")
                    yield results, label
                else:
                    break
        finally:
            conn.close()

        logger.info(f"disconnected: {addr}")


if __name__ == '__main__':
    interpreter, input_details, output_details = load_model("Real_Time_Model_CNN_LSTM_BIG.tflite")
    results = receive_data_and_predict(interpreter, input_details, output_details)
    labels = []
    predictions = []
    for result, label in results:
        result = [np.argmax(i) for i in result]
        batch_result = np.argmax(np.bincount(result))
        labels.append(np.argmax(label))
        predictions.append(batch_result)
        logger.info("========================================\n")
    labels = np.array(labels)
    predictions = np.array(predictions)
    accuracy = np.sum(labels == predictions) / len(labels)
    df = pd.DataFrame({"labels": labels, "predictions": predictions})
    df.to_csv("predictions.csv")
    logger.info(f"Validation accuracy: {accuracy}")

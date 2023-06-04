import argparse
import logging
from .model import model
from .utils import utils
from .model.data import DataProcessor
from os import path
import numpy as np

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Wind Turbine Power Prediction')
    subparsers = parser.add_subparsers(dest='command')

    # Sub-parser for the generate_data command
    generate_parser = subparsers.add_parser('generate_data', help='Generate wind turbine data')
    generate_parser.add_argument('--size', type=int, default=1000, help='Size of the data to generate')
    generate_parser.add_argument('--cut_in', type=float, default=3.5, help='Cut-in wind speed')
    generate_parser.add_argument('--rated_speed', type=float, default=15, help='Rated wind speed')
    generate_parser.add_argument('--rated_power', type=float, default=2, help='Rated power output')
    generate_parser.add_argument('--cut_out', type=float, default=25, help='Cut-out wind speed')
    generate_parser.add_argument('--shape', type=float, default=2, help='Shape parameter of the Weibull distribution')
    generate_parser.add_argument('--scale', type=float, default=10, help='Scale parameter of the Weibull distribution')
    generate_parser.add_argument('--random_seed', type=int, default=42, help='Random seed for data generation')
    generate_parser.add_argument('--out_path', type=str, required=True, help='Path to save the generated data')

    # Sub-parser for the train_model command
    train_parser = subparsers.add_parser('train_model', help='Train the prediction model')
    train_parser.add_argument('--data_path', type=str, required=True, help='Path to the training data')
    train_parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')

    # Sub-parser for the predict command
    predict_parser = subparsers.add_parser('predict', help='Predict power output for given wind speed, wind direction, and temperature')
    predict_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    predict_parser.add_argument('wind_speed', type=float, help='Wind speed to predict power output for')
    predict_parser.add_argument('wind_direction', type=float, help='Wind direction to predict power output for')
    predict_parser.add_argument('temperature', type=float, help='Temperature to predict power output for')

    args = parser.parse_args()

    turbine_model = model.TurbineModel()
    data_processor = DataProcessor()

    if args.command == 'generate_data':
        try:
            data_processor.generate_and_save_data(int(args.size), args.cut_in, args.rated_speed, args.rated_power, args.cut_out, args.shape, args.scale, args.out_path, args.random_seed)
            logging.info(f'Successfully generated data and saved to {args.out_path}')
        except Exception as e:
            logging.error(f'Failed to generate data: {e}')

    elif args.command == 'train_model':
        try:
            assert path.exists(args.data_path), "Training data file not found"
            turbine_model.train(args.data_path)
            turbine_model.save_model(args.model_path)
            logging.info(f'Successfully trained model and saved to {args.model_path}')
        except AssertionError as error:
            logging.error(error)
        except Exception as e:
            logging.error(f'Failed to train model: {e}')

    elif args.command == 'predict':
        try:
            assert path.exists(args.model_path), "Model file not found"
            turbine_model = utils.load_model(args.model_path)
            data = np.array([[args.wind_speed, args.wind_direction, args.temperature]])
            prediction = turbine_model.predict(data)
            print(f'Predicted power output for wind speed {args.wind_speed}, wind direction {args.wind_direction}, and temperature {args.temperature}: {prediction[0]}')
        except AssertionError as error:
            logging.error(error)
        except Exception as e:
            logging.error(f'Failed to predict: {e}')

if __name__ == '__main__':
    main()

import argparse
import logging
from .model import model, data
from .utils import utils
from os import path

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Wind Turbine Power Prediction')
    subparsers = parser.add_subparsers(dest='command')

    # Sub-parser for the generate_data command
    generate_parser = subparsers.add_parser('generate_data', help='Generate wind turbine data')
    generate_parser.add_argument('--size', type=int, default=1000, help='Size of the data to generate')
    generate_parser.add_argument('--min_speed', type=float, default=3.5, help='Minimum wind speed')
    generate_parser.add_argument('--max_speed', type=float, default=15, help='Maximum wind speed')
    generate_parser.add_argument('--std_dev', type=float, default=2, help='Standard deviation of the normal distribution')
    generate_parser.add_argument('--power_coeff', type=float, default=25, help='Power coefficient')
    generate_parser.add_argument('--out_path', type=str, required=True, help='Path to save the generated data')

    # Sub-parser for the train_model command
    train_parser = subparsers.add_parser('train_model', help='Train the prediction model')
    train_parser.add_argument('--data_path', type=str, required=True, help='Path to the training data')
    train_parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')

    # Sub-parser for the predict command
    predict_parser = subparsers.add_parser('predict', help='Predict power output for a given wind speed')
    predict_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    predict_parser.add_argument('wind_speed', type=float, help='Wind speed to predict power output for')

    args = parser.parse_args()

    if args.command == 'generate_data':
        try:
            wind_speeds, power_outputs = data.generate_turbine_data(args.size, args.min_speed, args.max_speed, args.std_dev, args.power_coeff)
            utils.save_data((wind_speeds, power_outputs), args.out_path)
            logging.info(f'Successfully generated data and saved to {args.out_path}')
        except Exception as e:
            logging.error(f'Failed to generate data: {e}')

    elif args.command == 'train_model':
        try:
            assert path.exists(args.data_path), "Training data file not found"
            wind_speeds, power_outputs = utils.load_data(args.data_path)
            train_speeds, test_speeds, train_power, test_power = model.split_data(wind_speeds, power_outputs)
            data_processor = model.DataProcessor()
            scaled_train_speeds = data_processor.fit_transform(train_speeds.reshape(-1, 1))
            scaled_test_speeds = data_processor.transform(test_speeds.reshape(-1, 1))
            trained_model = model.train_model(scaled_train_speeds, train_power)
            utils.save_model(trained_model, args.model_path)
            utils.save_model(data_processor, 'data_processor.joblib')
            logging.info(f'Successfully trained model and saved to {args.model_path}')
        except AssertionError as error:
            logging.error(error)
        except Exception as e:
            logging.error(f'Failed to train model: {e}')

    elif args.command == 'predict':
        try:
            assert path.exists(args.model_path), "Model file not found"
            trained_model = utils.load_model(args.model_path)
            data_processor = utils.load_model
            ('data_processor.joblib')
            scaled_speed = data_processor.transform([[args.wind_speed]])
            prediction = trained_model.predict(scaled_speed)
            print(f'Predicted power output for wind speed {args.wind_speed}: {prediction[0]}')
        except AssertionError as error:
            logging.error(error)
        except Exception as e:
            logging.error(f'Failed to predict: {e}')

if __name__ == '__main__':
    main()

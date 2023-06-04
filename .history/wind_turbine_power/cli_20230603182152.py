import argparse
import logging
from .model import model
from .utils import utils
from .model.data import DataProcessor
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
    generate_parser.add_argument('--shape', type=float, default=2, help='Shape parameter of the Weibull distribution')
    generate_parser.add_argument('--scale', type=float, default=2, help='Scale parameter of the Weibull distribution')
    generate_parser.add_argument('--wind_direction_range', type=float, nargs=2, default=[0, 360], help='Range of wind direction')
    generate_parser.add_argument('--temperature_range', type=float, nargs=2, default=[-20, 50], help='Range of temperature')
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

    data_processor = DataProcessor()

    if args.command == 'generate_data':
        try:
            wind_speeds, power_outputs, wind_directions, temperatures = data_processor.generate_turbine_data(args.size, args.min_speed, args.max_speed, rated_power=args.power_coeff, cut_out=args.std_dev, shape=args.shape, scale=args.scale, wind_direction_range=args.wind_direction_range, temperature_range=args.temperature_range)
            utils.save_data((wind_speeds, power_outputs, wind_directions, temperatures), args.out_path)
            logging.info(f'Successfully generated data and saved to {args.out_path}')
        except Exception as e:
            logging.error(f'Failed to generate data: {e}')


    elif args.command == 'train_model':
        try:
            assert path.exists(args.data_path), "Training data file not found"
            data, power_outputs = utils.load_data(args.data_path)
            train_data, test_data, train_power, test_power = model.split_data(data, power_outputs)
            scaled_train_data = data_processor.fit_transform(train_data)
            scaled_test_data = data_processor.transform(test_data)
            trained_model = model.train_model(scaled_train_data, train_power)
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
            data_processor = utils.load_model('data_processor.joblib')
            data = np.array([[args.wind_speed, args.wind_direction, args.temperature]])
            scaled_data = data_processor.transform(data)
            prediction = trained_model.predict(scaled_data)
            print(f'Predicted power output for wind speed {args.wind_speed}, wind direction {args.wind_direction}, and temperature {args.temperature}: {prediction[0]}')
        except AssertionError as error:
            logging.error(error)
        except Exception as e:
            logging.error(f'Failed to predict: {e}')

if __name__ == '__main__':
    main()

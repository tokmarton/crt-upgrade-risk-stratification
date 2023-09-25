from argparse import ArgumentParser
from sys import argv
from training import train_and_validate_internally
from risk_stratification import risk_stratify


def main():
    parser = ArgumentParser()
    action_choices = ['train', 'risk_stratify']
    parser.add_argument('action',
                        help='Train and internally validate a new model or '
                             'risk stratify new patients using a trained model',
                        choices=action_choices)
    parser.add_argument('-d', '--data',
                        help='Path to the CSV file containing the dataset',
                        required=True, type=str)
    parser.add_argument('-t', '--target_folder',
                        help='Folder where the results will be saved',
                        required=False, default=r'.\results', type=str)
    parser.add_argument('-c', '--config_path',
                        help='Path to the YAML file containing the configurations for the training',
                        default=None, required=('train' in argv), type=str)
    parser.add_argument('-m', '--model_path',
                        help='Path to the trained model',
                        required=('risk_stratify' in argv), type=str)
    args = parser.parse_args()

    if args.action == 'train':
        train_and_validate_internally(args.data, args.config_path, args.target_folder)
    elif args.action == 'risk_stratify':
        risk_stratify(args.data, args.model_path, args.target_folder)


if __name__ == '__main__':
    main()

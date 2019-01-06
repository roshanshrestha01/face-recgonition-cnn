import argparse

parser = argparse.ArgumentParser(description='Data sample parameters.')

parser.add_argument('ratio', metavar='N', type=int, nargs='+',
                    help='an integer for data separation')

args = parser.parse_args()

if len(args.ratio) == 2:
    datasets = ['train', 'test']
elif len(args.ratio) == 3:
    datasets = ['train', 'validate', 'test']
else:
    raise ValueError('Ratio should be 2 or 3 values i.e. training and testing or training, validation and testing')

if not sum(args.ratio) == 10:
    raise ValueError('Sum of ratio should be equal 10')



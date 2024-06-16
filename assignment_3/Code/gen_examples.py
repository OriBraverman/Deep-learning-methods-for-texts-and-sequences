"""
This file is used to generate examples for the assignment.

Positive sentence examples (regex format):
-------------------------------------------
[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+

Negative sentence examples (regex format):
-------------------------------------------
[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+

"""
import random
import argparse
import os

# Constants
SEED = 1
POS, NEG = "POS", "NEG"
TYPE_MAP = {POS: ['a', 'b', 'c', 'd'], NEG: ['a', 'c', 'b', 'd']}
MAX_NUM_DIGITS = 15
MAX_NUM_LETTERS = 15
SEP = '\t'


def generate_example(type, max_num_digits=MAX_NUM_DIGITS, max_num_letters=MAX_NUM_LETTERS):
    """
    @brief: Generate a positive or negative example.
    @param type: Type of example to generate (positive or negative).
    @return: Example string.
    """
    example = []
    for i in range(4):
        example += [str(random.randint(1, 9)) for _ in range(random.randint(1, max_num_digits))]
        example += TYPE_MAP[type][i] * random.randint(1, max_num_letters)
    example += [str(random.randint(1, 9)) for _ in range(random.randint(1, max_num_digits))]
    return ''.join(example)


def generate_examples(num_examples, name, output_dir, type):
    """
    @brief: Generate examples of a given type (positive or negative).
    @param num_examples: Number of examples to generate.
    @param name: Name of the output file.
    @param output_dir: Directory to save examples.
    @param type: Type of examples to generate (positive or negative).
    """
    with open(f'{output_dir}/{name}', 'w') as f:
        for i in range(num_examples):
            f.write(f'{generate_example(type)}\n')


def generate_data(num_examples, output_file):
    """
    @brief: Generate the data for the assignment.
    @param num_examples: Number of examples to generate.
    @param output_file: Output file to save examples.
    """
    with open(f'{output_file}', 'w') as f:
        for i in range(num_examples):
            if i % 2 == 0:
                f.write(f'{generate_example(POS)}{SEP}{POS}\n')
            else:
                f.write(f'{generate_example(POS)}{SEP}{NEG}\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate examples for the assignment.')
    parser.add_argument('--num_examples', type=int, default=1000,
                        help='Number of examples to generate (half positive, half negative).')
    parser.add_argument('--pos_output_file', type=str, default='pos_examples',
                        help='Output file for positive examples.')
    parser.add_argument('--neg_output_file', type=str, default='neg_examples',
                        help='Output file for negative examples.')
    parser.add_argument('--output_dir', type=str, default='Data/Pos_Neg_Examples',
                        help='Directory to save examples.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for reproducibility.')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # If there are no output files, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate positive examples
    generate_examples(num_examples=args.num_examples // 2, name=args.pos_output_file,
                      output_dir=args.output_dir, type=POS)

    # Generate negative examples
    generate_examples(num_examples=args.num_examples // 2, name=args.neg_output_file,
                      output_dir=args.output_dir, type=NEG)

    # Generate training data
    generate_data(num_examples=args.num_examples, output_file=f'{args.output_dir}/train')

    # Generate test data
    generate_data(num_examples=args.num_examples, output_file=f'{args.output_dir}/test')


if __name__ == '__main__':
    main()

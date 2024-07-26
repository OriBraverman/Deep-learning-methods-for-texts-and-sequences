"""
This file is used to generate examples for part 2 of the assignment.
The goal is to generate examples that the RNN cannot distinguish.

Sigma_1 = {'a',...,'z'}
Sigma_2 = {'(', ')', '{', '}', '[', ']'}
Max length = 50

L1 = {wtw^R | w in Sigma_1*, |w|<=Max length, t = Sigma_1 or t = epsilon}
L2 = {ww | w in Sigma_1*, |w|<=Max length}
L3 = {w | w in Sigma_2* , |w|<=2*Max length, w is a correct parenthesis string}
"""
import random
import argparse
import os
import string

# Constants
SEED = 1
LANGUAGE = ['L1', 'L2', 'L3']
POS, NEG = "POS", "NEG"
MAX_LENGTH = 50
SEP = '\t'

# Alphabets
Sigma_1 = string.ascii_lowercase
Sigma_2 = "(){}[]"

def generate_example_L1(type, max_length=MAX_LENGTH):
    """
    @brief: Generate a positive or negative example for L1.
    The language L1 is defined as {wtw^R | w in Sigma_1*, |w|<=Max length, t = Sigma_1 or t = epsilon}.
    @param type: Type of example to generate (positive or negative).
    @return: Example string.
    """
    example = []
    w = [random.choice(Sigma_1) for _ in range(random.randint(1, max_length))]
    t = [random.choice([random.choice(Sigma_1), ''])]
    if type == POS:
        example = w + t + w[::-1]
    else:
        example = w + t + w
    return ''.join(example)

def generate_example_L2(type, max_length=MAX_LENGTH):
    """
    @brief: Generate a positive or negative example for L2.
    The language L2 is defined as {ww | w in Sigma_1*, |w|<=Max length}.
    @param type: Type of example to generate (positive or negative).
    @return: Example string.
    """
    example = []
    w = [random.choice(Sigma_1) for _ in range(random.randint(1, max_length))]
    if type == POS:
        example = w + w
    else:
        example = w + [random.choice(Sigma_1) for _ in range(random.randint(1, max_length))]
    return ''.join(example)

def generate_example_L3(type, max_length=MAX_LENGTH):
    """
    @brief: Generate a positive or negative example for L3.
    The language L3 is defined as {w | w in Sigma_2* , |w|<=2*Max length, w is a correct parenthesis string}.
    A correct parenthesis string can be generated recursively.
    The string can be defined recursively for input_t-1, input_t= ()input_t-1, (input_t-1), input_t-1()
    @param type: Type of example to generate (positive or negative).
    @return: Example string.
    """
    def generate_parenthesis_string(max_length):
        if max_length == 0:
            return ''
        choice = random.randint(0, 2)
        left_parenthesis = random.choice('({[')
        right_parenthesis = {'(': ')', '{': '}', '[': ']'}[left_parenthesis]

        if choice == 0:  # ()input
            return left_parenthesis + right_parenthesis + generate_parenthesis_string(max_length - 1)
        elif choice == 1:  # (input)
            return left_parenthesis + generate_parenthesis_string(max_length - 1) + right_parenthesis
        else:  # input()
            return generate_parenthesis_string(max_length - 1) + left_parenthesis + right_parenthesis

    example = generate_parenthesis_string(random.randint(1, max_length))
    if type == NEG:
        # split into two random parts
        split = random.randint(0, len(example))

        # swap the two parts and put open parenthesis in the middle
        example = example[split:] + random.choice('({[') + example[:split]
    return example

def generate_example(language, type, max_length=MAX_LENGTH):
    """
    @brief: Generate a positive or negative example for a given language.
    @param language: Language to generate examples for.
    @param type: Type of example to generate (positive or negative).
    @return: Example string.
    """
    if language == LANGUAGE[0]:
        return generate_example_L1(type, max_length)
    elif language == LANGUAGE[1]:
        return generate_example_L2(type, max_length)
    elif language == LANGUAGE[2]:
        return generate_example_L3(type, max_length)

def generate_examples(num_examples, name, output_dir, language, type):
    """
    @brief: Generate examples of a given type (positive or negative) for a given language.
    @param num_examples: Number of examples to generate.
    @param name: Name of the output file.
    @param output_dir: Directory to save examples.
    @param language: Language to generate examples for.
    @param type: Type of examples to generate (positive or negative).
    """
    with open(f'{output_dir}/{name}', 'w') as f:
        for i in range(num_examples):
            f.write(f'{generate_example(language, type)}\n')

def generate_data(num_examples, output_file):
    """
    @brief: Generate the data for the assignment.
    @param num_examples: Number of examples to generate.
    @param output_file: Output file to save examples.
    """
    with open(f'{output_file}', 'w') as f:
        for i in range(num_examples):
            if i % 2 == 0:
                f.write(f'{generate_example(LANGUAGE[0], POS)}{SEP}{POS}\n')
            else:
                f.write(f'{generate_example(LANGUAGE[0], NEG)}{SEP}{NEG}\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Generate examples for the assignment.')
    parser.add_argument('--num_train', type=int, default=1000,
                        help='Number of training examples to generate.')
    parser.add_argument('--num_test', type=int, default=500,
                        help='Number of test examples to generate.')
    parser.add_argument('--output_dir', type=str, default='Data',
                        help='Directory to save examples.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for random number generator.')
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    # If the output directory does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # If the language directories do not exist, create them
    for language in LANGUAGE:
        if not os.path.exists(f'{args.output_dir}/{language}'):
            os.makedirs(f'{args.output_dir}/{language}')

    # Generate training and test data
    for language in LANGUAGE:
        generate_data(num_examples=args.num_train, output_file=f'{args.output_dir}/{language}/train')
        generate_data(num_examples=args.num_test, output_file=f'{args.output_dir}/{language}/test')


if __name__ == '__main__':
    main()
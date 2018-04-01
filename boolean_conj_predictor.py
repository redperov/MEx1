import numpy as np
import sys


class Literal(object):
    """
    The class represents a literal object.
    """

    def __init__(self, index, is_negation):
        """
        Constructor.
        :param value: literal value
        :param index: literal index
        :param is_negation: is the literal in a negation form
        """

        self._index = index
        self._is_negation = is_negation

    def get_index(self):
        """
        Index getter.
        :return: int
        """
        return self._index

    def get_is_negation(self):
        """
        Is negation getter.
        :return: boolean
        """
        return self._is_negation

    def evaluate(self, value):
        """
        Evaluates the given value on the literal.
        :param value: value
        :return: evaluated value
        """
        # Check if literal is in negation form.
        if self.get_is_negation():

            # Return the opposite of value.
            if value == 1:
                return 0
            else:
                return 1
        else:
            return value

    def __str__(self):
        """
        Converts the literal to string.
        :return: string
        """
        index = self.get_index() + 1

        # Check if literal is in negation form.
        if self.get_is_negation():
            return "not(x{0})".format(index)
        else:
            return "x{0}".format(index)


def create_all_negative_hypothesis(num_of_literals):
    """
    Creates the all negative hypothesis for a given size.
    :param num_of_literals: number of literals
    :return: hypothesis
    """
    hypothesis = []

    for i in range(0, num_of_literals):
        x = Literal(i, False)
        x_not = Literal(i, True)
        hypothesis.append(x)
        hypothesis.append(x_not)

    return hypothesis


def evaluate_hypothesis(hypothesis, training_example):
    """
    Evaluate the hypothesis on a given training example.
    :param hypothesis: hypothesis
    :param training_example: training example
    :return: 0 or 1
    """
    current_index = hypothesis[0].get_index()
    max_length = len(training_example)
    counter = 0

    for literal in hypothesis:

        # Check if need to take the next value from the training example.
        if literal.get_index() != current_index:
            current_index = literal.get_index()
            counter += 1
            if counter >= max_length:
                break

        # Evaluate literal with the training example.
        evaluated_value = literal.evaluate(training_example[counter])

        # If encountered a zero, no need to continue.
        if evaluated_value == 0:
            return 0

    return 1


def remove_literal(hypothesis, index, is_negation):
    """
    Tries to remove a literal from a hypothesis.
    :param hypothesis: hypothesis
    :param index: index
    :param is_negation: is the literal in negation form
    :return: boolean, was the literal found and removed
    """
    for literal in hypothesis:
        if literal.get_index() == index and literal.get_is_negation() == is_negation:
            hypothesis.remove(literal)
            return True

    return False


def consistency_algorithm(X, Y):
    """
    Performs the consistency algorithm.
    :param X: Training examples matrix
    :param Y: labels vector
    :return: conjunction prediction
    """
    # Get number of columns in the examples set.
    try:
        num_of_literals = X.shape[1]
    except IndexError:
        num_of_literals = len(X)

    # Initialize hypothesis.
    h_t = create_all_negative_hypothesis(num_of_literals)

    for t, y_t in zip(X, Y):
        y_t_hat = evaluate_hypothesis(h_t, t)

        if y_t == 1 and y_t_hat == 0:

            for i in range(0, len(t)):
                x_t_i = t[i]

                if x_t_i == 1:
                    # Remove not(x_i) from hypothesis.
                    remove_literal(h_t, i, True)
                if x_t_i == 0:
                    # Remove x_i from hypothesis.
                    remove_literal(h_t, i, False)

    return h_t


def write_answer_to_file(raw_output):
    """
    Writes the answer to a file.
    :param raw_output: raw output
    :return: None
    """
    str_output = []

    # Turn all the items to strings.
    for item in raw_output:
        str_output.append(str(item))

    # Separate the output values with a comma.
    output = ",".join(str_output)

    # Write output to file.
    with open("output.txt", 'w') as output_file:
        output_file.write(output)


if __name__ == "__main__":
    # Load data from input file.
    file_path = sys.argv[1]
    training_examples = np.loadtxt(file_path)

    # Separate the training data to two containers.
    # X - examples
    # Y - classifications
    X = training_examples[:, : -1]
    Y = training_examples[:, -1]

    # Execute the consistency algorithm.
    result = consistency_algorithm(X, Y)

    # Write the answer to a file.
    write_answer_to_file(result)

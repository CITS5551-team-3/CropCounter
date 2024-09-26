from params import Params
import time
import itertools
from count2 import count
import os
def metric(x):
    actual = 208
    return (actual - x)

def evaluate():
    # Create a list of valid parameters
    erosion_iterations = range(6, 7)  # 6 to 8
    dilation_iterations = range(8, 9)  # 8 to 10
    split_scale_factor = [round(i * 0.1, 1) for i in range(14, 15)]  # 1.5 to 1.7
    minimum_width_threshold = range(40, 41, 10)  # 40 to 50, step 10

    # Generate all combinations of parameters
    param_combinations = itertools.product(
        erosion_iterations,
        dilation_iterations,
        split_scale_factor,
        minimum_width_threshold
    )

    best_value = float('-inf')  # Initialize to negative infinity
    best_params = None
    best_count = 0

    # Count combinations using a generator
    num_combinations = sum(1 for _ in itertools.product(
        erosion_iterations,
        dilation_iterations,
        split_scale_factor,
        minimum_width_threshold
    ))

    # Print the number of combinations
    print(f"Number of combinations: {num_combinations}")



    # Loop through all parameter combinations
    for i, combo in enumerate(param_combinations):
        # Create a Params object with the current combination
        temp_params = Params(
            ei=combo[0],
            di=combo[1],
            ssf=combo[2],
            mwt=combo[3]
        )
        
        # Call the count function with current parameters
        count_value = count(params=temp_params, headless=True)
        print(i, count_value)
        # Evaluate the count value
        eval_value = metric(count_value)
        
        # Check if the evaluated value is closest to zero
        if abs(eval_value) < abs(best_value):
            best_value = eval_value
            best_params = temp_params
            best_count = count_value

    return best_count, best_params


if __name__ == "__main__":
    best_count, params = evaluate()
    print(best_count)
    print(params.__dict__)
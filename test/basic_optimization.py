# Tutorial on using simplesvgd


import numpy as np

import simplesvgd

# We define our test function here. The actual scalar value of the function is
# never used in the optimization process, so it isn't necessary to implement.
# Change the smoothing factor to be smaller to create strongly isolated modes.


smoothing = 30


def himmelblau(input_array: np.ndarray) -> np.ndarray:

    # As this is a 2-dimensional function, assert that the passed input_array
    # is correct.
    assert input_array.shape[1] == 2

    # To simplify reading this function, we do this step in between. It is not
    # the most optimal way to program this.
    x = input_array[:, 0, None]
    y = input_array[:, 1, None]

    output_array = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    # As the output should be a scalar function, assert that the
    # output is also length 1 in dim 2 of the array.
    assert output_array.shape == (input_array.shape[0], 1)

    return output_array / smoothing


def himmelblau_grad(input_array: np.ndarray) -> np.ndarray:

    # As this is a 2-dimensional function, assert that the passed input_array
    # is correct.
    assert input_array.shape[1] == 2

    # To simplify reading this function, we do this step in between. It is not
    # the most optimal way to program this.
    x = input_array[:, 0, None]
    y = input_array[:, 1, None]

    # Compute partial derivatives and combine them
    output_array_dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    output_array_dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    output_array = np.hstack((output_array_dx, output_array_dy))

    # Check if the output shape is correct
    assert output_array.shape == input_array.shape

    return output_array / smoothing


# Now we optimize the starting samples using SVGD, and pass the function on
# evaluated on the grid as a nice background for the animation.

rng = np.random.default_rng()
initial_samples = rng.normal(0, 3, [1000, 2])

state = simplesvgd.update(
    initial_samples,
    himmelblau_grad,
    n_iter=130,
    stepsize=1e0,
)
final_samples = state.particles

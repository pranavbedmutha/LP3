import numpy as np
import matplotlib.pyplot as plt

# Function
def function(x):
    return (x + 3) ** 2

# Derivative of the function
def derivative(x):
    return 2 * (x + 3)

# Starting point
x = 2

# Learning rate
learning_rate = 0.1

# Number of iterations
iterations = 20

# Store values for plotting
x_values = []
y_values = []

# Gradient Descent
for i in range(iterations):

    # Store current values
    x_values.append(x)
    y_values.append(function(x))

    # Calculate gradient
    gradient = derivative(x)

    # Update x
    x = x - learning_rate * gradient

    print(
        "Iteration:", i + 1,
        "x =", round(x, 6),
        "y =", round(function(x), 6)
    )

# Final result
print("\nLocal Minimum:")
print("x =", x)
print("y =", function(x))

# Plot the function
x_plot = np.linspace(-6, 4, 100)
y_plot = function(x_plot)

plt.plot(x_plot, y_plot, label="y = (x + 3)^2")

plt.scatter(
    x_values,
    y_values,
    color="red",
    label="Gradient Descent"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Gradient Descent")
plt.legend()
plt.grid()
plt.show()

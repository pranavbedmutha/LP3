#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Fractional Knapsack using Greedy Algorithm
double fractionalKnapsack(vector<int> value,
                          vector<int> weight,
                          int capacity,
                          vector<double>& fractions) {

    int n = value.size();

    // Calculate value-to-weight ratio
    vector<double> ratio(n);

    for (int i = 0; i < n; i++) {
        ratio[i] = (double)value[i] / weight[i];
    }

    // Create list of item indices
    vector<int> index(n);

    for (int i = 0; i < n; i++) {
        index[i] = i;
    }

    // Sort items in decreasing order of ratio
    sort(index.begin(), index.end(),
         [&](int a, int b) {
             return ratio[a] > ratio[b];
         });

    double maxValue = 0;

    // Select items according to highest ratio
    for (int i : index) {

        if (weight[i] <= capacity) {

            // Take complete item
            fractions[i] = 1.0;

            maxValue += value[i];

            capacity -= weight[i];
        }
        else {

            // Take fraction of item
            fractions[i] = (double)capacity / weight[i];

            maxValue += value[i] * fractions[i];

            break;
        }
    }

    return maxValue;
}

int main() {

    int n;

    // Accept number of items
    cout << "Enter number of items: ";
    cin >> n;

    vector<int> value(n);
    vector<int> weight(n);

    // Accept values
    cout << "Enter the values of the " << n << " item(s): ";

    for (int i = 0; i < n; i++) {
        cin >> value[i];
    }

    // Accept weights
    cout << "Enter the positive weights of the " << n << " item(s): ";

    for (int i = 0; i < n; i++) {
        cin >> weight[i];
    }

    // Accept maximum capacity
    int capacity;

    cout << "Enter maximum weight: ";
    cin >> capacity;

    // Store selected fractions
    vector<double> fractions(n, 0.0);

    // Calculate maximum value
    double maxValue = fractionalKnapsack(
        value, weight, capacity, fractions
    );

    // Display results
    cout << "The maximum value of items that can be carried: "
         << maxValue << endl;

    cout << "The fractions in which the items should be taken: ";

    for (int i = 0; i < n; i++) {
        cout << fractions[i] << " ";
    }

    cout << endl;

    return 0;
}

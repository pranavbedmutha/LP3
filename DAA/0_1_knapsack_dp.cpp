#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int knapsack(vector<int> values, vector<int> weights,
             int capacity, vector<int>& selected) {

    int n = values.size();

    vector<vector<int>> dp(n + 1,
                           vector<int>(capacity + 1, 0));

    // Build the DP table
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {

            if (weights[i - 1] <= w) {
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]],
                    dp[i - 1][w]
                );
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }

    // Find selected items
    int w = capacity;

    for (int i = n; i > 0; i--) {
        if (dp[i][w] != dp[i - 1][w]) {
            selected.push_back(i);
            w -= weights[i - 1];
        }
    }

    reverse(selected.begin(), selected.end());

    return dp[n][capacity];
}

int main() {
    int n, capacity;

    cout << "Enter number of items: ";
    cin >> n;

    if (n <= 0) {
        cout << "Invalid number of items." << endl;
        return 0;
    }

    vector<int> values(n), weights(n);

    cout << "Enter item values: ";
    for (int i = 0; i < n; i++) {
        cin >> values[i];
    }

    cout << "Enter item weights: ";
    for (int i = 0; i < n; i++) {
        cin >> weights[i];
    }

    cout << "Enter knapsack capacity: ";
    cin >> capacity;

    for (int i = 0; i < n; i++) {
        if (weights[i] <= 0) {
            cout << "Weights must be positive." << endl;
            return 0;
        }
    }

    if (capacity < 0) {
        cout << "Capacity must be non-negative." << endl;
        return 0;
    }

    vector<int> selected;

    int maximumProfit = knapsack(
        values, weights, capacity, selected
    );

    cout << "Maximum profit: " << maximumProfit << endl;

    cout << "Selected item numbers: [";
    for (int i = 0; i < selected.size(); i++) {
        cout << selected[i];

        if (i < selected.size() - 1)
            cout << ", ";
    }
    cout << "]" << endl;

    cout << "Time complexity: O(n * capacity)" << endl;
    cout << "Space complexity: O(n * capacity)" << endl;

    return 0;
}

#include <iostream>
using namespace std;

int main() {
    int nterms;

    cout << "Enter number of terms: ";
    cin >> nterms;

    // First two terms
    int n1 = 0, n2 = 1;
    int count = 0;

    // Check if the number of terms is valid
    if (nterms <= 0) {
        cout << "Please enter a positive integer";
    }

    // If there is only one term
    else if (nterms == 1) {
        cout << "Fibonacci sequence upto " << nterms << ":" << endl;
        cout << n1;
    }

    // Generate Fibonacci sequence
    else {
        cout << "Fibonacci sequence:" << endl;

        while (count < nterms) {
            cout << n1 << " ";

            int nth = n1 + n2;
            n1 = n2;
            n2 = nth;

            count++;
        }
    }

    return 0;
}

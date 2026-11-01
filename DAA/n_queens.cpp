#include <bits/stdc++.h>
using namespace std;

#define N 4

// Print the chessboard solution
void printSolution(int board[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << board[i][j] << " ";
        }
        cout << endl;
    }
}

// Check whether a queen can be placed safely
bool isSafe(int board[N][N], int row, int col) {
    int i, j;

    // Check the left side of the current row
    for (i = 0; i < col; i++) {
        if (board[row][i])
            return false;
    }

    // Check upper-left diagonal
    for (i = row, j = col;
         i >= 0 && j >= 0; i--, j--) {
        if (board[i][j])
            return false;
    }

    // Check lower-left diagonal
    for (i = row, j = col;
         j >= 0 && i < N; i++, j--) {
        if (board[i][j])
            return false;
    }

    return true;
}

// Recursively place queens column by column
bool solveNQUtil(int board[N][N], int col) {

    // If all queens are placed
    if (col >= N)
        return true;

    // Try placing queen in each row
    for (int i = 0; i < N; i++) {

        if (isSafe(board, i, col)) {

            // Place the queen
            board[i][col] = 1;

            // Recursively place remaining queens
            if (solveNQUtil(board, col + 1))
                return true;

            // Backtracking
            board[i][col] = 0;
        }
    }

    return false;
}

// Solve N-Queens problem
bool solveNQ() {

    // Initialize chessboard with zeros
    int board[N][N] = {0};

    if (!solveNQUtil(board, 0)) {
        cout << "Solution does not exist." << endl;
        return false;
    }

    // Display solution
    printSolution(board);

    return true;
}

// Main function
int main() {
    solveNQ();
    return 0;
}

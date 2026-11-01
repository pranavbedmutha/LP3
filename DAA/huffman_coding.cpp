#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Node class for Huffman Tree
class Node {
public:
    int freq;
    string symbol;
    Node* left;
    Node* right;
    string huff;

    Node(int f, string s, Node* l = nullptr, Node* r = nullptr) {
        freq = f;
        symbol = s;
        left = l;
        right = r;
        huff = "";
    }
};

// Function to print Huffman codes
void printNodes(Node* root, string val = "") {
    string newVal = val + root->huff;

    if (root->left) {
        printNodes(root->left, newVal);
    }

    if (root->right) {
        printNodes(root->right, newVal);
    }

    // If leaf node
    if (!root->left && !root->right) {
        cout << root->symbol << " -> " << newVal << endl;
    }
}

// Function to build Huffman Tree
int main() {

    // Characters
    vector<string> chars = {"a", "b", "c", "d", "e", "f", "g"};

    // Frequencies
    vector<int> freq = {4, 7, 12, 14, 17, 43, 54};

    // List containing unused nodes
    vector<Node*> nodes;

    // Create Huffman Tree nodes
    for (int i = 0; i < chars.size(); i++) {
        nodes.push_back(new Node(freq[i], chars[i]));
    }

    // Build Huffman Tree
    while (nodes.size() > 1) {

        // Sort nodes in ascending order
        sort(nodes.begin(), nodes.end(),
             [](Node* a, Node* b) {
                 return a->freq < b->freq;
             });

        // Pick two smallest nodes
        Node* left = nodes[0];
        Node* right = nodes[1];

        // Assign directional values
        left->huff = "0";
        right->huff = "1";

        // Create new parent node
        Node* newNode = new Node(
            left->freq + right->freq,
            left->symbol + right->symbol,
            left,
            right
        );

        // Remove two nodes
        nodes.erase(nodes.begin(), nodes.begin() + 2);

        // Add parent node
        nodes.push_back(newNode);
    }

    // Print Huffman Codes
    printNodes(nodes[0]);

    return 0;
}

// Carbon Language Project
// Apache License v2.0 with LLVM Exceptions
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>

// Output Validation
// -----------------
// Expected output:
// - "something"
// - Empty line
// - Current line number: [[@LINE+1]]
// - Previous line number: [[@LINE-1]]
// - Special characters: +*[]{}
// - Function: F{{.+}}z

// Function declaration
void F{{.+}}z();

int main() {
    // Output "something"
    std::cout << "something" << std::endl;

    // Output an empty line
    std::cout << std::endl;

    // Output the current line number using line deltas
    std::cout << "[[@LINE+1]]: Line delta" << std::endl;

    // Output the previous line number using negative line deltas
    std::cout << "[[@LINE-1]]: Negative line delta" << std::endl;

    // Output special characters
    std::cout << "+*[]{}" << std::endl;

    // Call function F{{.+}}z
    F{{.+}}z();

    return 0;
}

// Function definition
void F{{.+}}z() {
    // Function implementation goes here
    std::cout << "Function: F{{.+}}z" << std::endl;
}

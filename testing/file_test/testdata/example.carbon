// Carbon Language Project
// Apache License v2.0 with LLVM Exceptions
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <fstream>

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

    // Check for content before the first split file
    std::ifstream file("testing/file_test/file_test_base.cpp");
    std::string line;
    bool found_content_pre_split = false;

    // Iterate over the lines until the first split file is found
    while (std::getline(file, line)) {
        if (!line.empty()) {
            found_content_pre_split = true;
            break;
        }
    }

    // Output the result
    std::cout << "Value of: found_content_pre_split" << std::endl;
    std::cout << "  Actual: " << found_content_pre_split << std::endl;
    std::cout << "Expected: " << (!found_content_pre_split) << std::endl;
    
    return 0;
}

// Function definition
void F{{.+}}z() {
    // Function implementation goes here
    std::cout << "Function: F{{.+}}z" << std::endl;
}

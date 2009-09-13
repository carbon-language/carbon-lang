// Test that we can compile C++ code.
// RUN: llvmc %s -o %t
// RUN: %abs_tmp | grep hello
#include <iostream>

int main() {
    std::cout << "hello" << '\n';
}

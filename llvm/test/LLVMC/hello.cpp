// Test that we can compile C++ code.
// RUN: llvmc2 --linker=c++ %s -o %t
// RUN: ./%t | grep hello
#include <iostream>

int main() {
    std::cout << "hello" << '\n';
}

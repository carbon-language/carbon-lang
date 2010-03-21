// Test that we can compile C++ code.
// RUN: llvmc %s -o %t
// RUN: %abs_tmp | grep hello
// XFAIL: vg
#include <iostream>

int main() {
    std::cout << "hello" << '\n';
}

// Test that the -c flag works.
// RUN: llvmc -c %s -o %t.o
// RUN: llvmc --linker=c++ %t.o -o %t
// RUN: %abs_tmp | grep hello
// XFAIL: vg
#include <iostream>

int main() {
    std::cout << "hello" << '\n';
}

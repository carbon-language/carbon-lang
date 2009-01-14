// RUN: xcc -ccc-cxx %s -o %t &&
// RUN: %t | grep "Hello, World"
// XFAIL

#include <iostream>

int main() {
  std::cout << "Hello, World!\n";
  return 0;
}

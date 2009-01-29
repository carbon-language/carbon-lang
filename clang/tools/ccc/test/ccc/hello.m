// RUN: xcc -ccc-no-clang %s -o %t &&
// RUN: %t | grep "Hello, World" &&
// RUN: xcc %s -o %t &&
// RUN: %t | grep "Hello, World"

int main() {
  printf("Hello, World!\n");
  return 0;
}

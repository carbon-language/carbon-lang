// RUN: xcc %s -o %t &&
// RUN: %t | grep "Hello, World" &&
// RUN: xcc %s -o %t -pipe &&
// RUN: %t | grep "Hello, World" &&
// RUN: xcc -ccc-clang %s -o %t &&
// RUN: %t | grep "Hello, World"

int main() {
  printf("Hello, World!\n");
  return 0;
}

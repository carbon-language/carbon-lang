// RUN: clang -ccc-echo -o %t %s 2> %t.log &&

// Make sure we used clang.
// RUN: grep 'clang-cc" .*hello.c' %t.log &&

// RUN: %t > %t.out &&
// RUN: grep "I'm a little driver, short and stout." %t.out

#include <stdio.h>

int main() {
  printf("I'm a little driver, short and stout.");
  return 0;
}

// RUN: clang %s -emit-llvm -o %t -fblocks -f__block
#include <stdio.h>

void test1() {
  __block int a;
  int b=2;
  a=1;
  printf("a is %d, b is %d\n", a, b);
  ^{ a = 10; printf("a is %d, b is %d\n", a, b); }();
  printf("a is %d, b is %d\n", a, b);
  a = 1;
  printf("a is %d, b is %d\n", a, b);
}


void test2() {
  __block int a;
  a=1;
  printf("a is %d\n", a);
  ^{
    ^{
      a = 10;
    }();
  }();
  printf("a is %d\n", a);
  a = 1;
  printf("a is %d\n", a);
}

int main() {
  test1();
  test2();
  return 0;
}

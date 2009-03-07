// RUN: clang %s -emit-llvm -o %t -fblocks -f__block &&
// RUN: grep "_Block_object_dispose" %t | count 10 &&
// RUN: grep "__copy_helper_block_" %t | count 6 &&
// RUN: grep "__destroy_helper_block_" %t | count 6 &&
// RUN: grep "__Block_byref_id_object_copy_" %t | count 2 &&
// RUN: grep "__Block_byref_id_object_dispose_" %t | count 2 &&
// RUN: grep "i32 135)" %t | count 2 &&
// RUN: grep "_Block_object_assign" %t | count 6

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

void test3() {
  __block int k;
  __block int (^j)(int);
  ^{j=0; k=0;}();
}

int main() {
  test1();
  test2();
  test3();
  return 0;
}

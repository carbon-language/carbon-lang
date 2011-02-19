// RUN: %clang_cc1 -fexceptions -emit-llvm-only -verify %s -Wno-unreachable-code

int val = 42;
int& test1() {
  return throw val, val;
}

int test2() {
  return val ? throw val : val;
}

// rdar://problem/8608801
void test3() {
  throw false;
}

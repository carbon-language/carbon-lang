// RUN: clang-cc -emit-llvm-only -verify %s

int val = 42;
int& test1() {
  return throw val, val;
}

int test2() {
  return val ? throw val : val;
}

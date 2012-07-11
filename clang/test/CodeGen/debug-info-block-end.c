// RUN: %clang_cc1 -g -S -emit-llvm %s -o - | FileCheck %s

int bar();

int foo(int i) {
  int j = 0;
  if (i) {
    j = bar();
  }
  else {
    j = bar() + 2;
  }
  return j;
}

// Make sure we don't have a line table entry for a block with no cleanups.
// CHECK-NOT: i32 9, i32 3, metadata

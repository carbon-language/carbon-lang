// REQUIRES: ppc64-registered-target
// RUN: %clang_cc1 %s -triple=powerpc64-unknown-linux-gnu -S -o - | FileCheck %s

typedef char v8qi  __attribute__((vector_size (8)));

extern v8qi x, y;

v8qi foo (void) {
  return x + y;
}

// CHECK-NOT: lvx

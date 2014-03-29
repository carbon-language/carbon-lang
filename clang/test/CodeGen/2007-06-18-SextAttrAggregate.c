// RUN: %clang_cc1 %s -o - -emit-llvm | FileCheck %s
// XFAIL: aarch64, arm64

// PR1513

// AArch64 ABI actually requires the reverse of what this is testing: the callee
// does any extensions and remaining bits are unspecified.

// Technically this test wasn't written to test that feature, but it's a
// valuable check nevertheless.

struct s{
long a;
long b;
};

void f(struct s a, char *b, signed char C) {
  // CHECK: i8 signext

}

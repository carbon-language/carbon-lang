// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

struct A;
typedef int A::*param_t;
struct {
  const char *name;
  param_t par;
} *ptr;
void test_ptr() { (void) ptr; } // forced use

// CHECK: type { i8*, {{i..}} }

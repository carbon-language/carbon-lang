// RUN: clang-cc -emit-llvm %s -o - | FileCheck %s

struct A;
typedef int A::*param_t;
struct {
  const char *name;
  param_t par;
} *ptr;

// CHECK: type { i8*, {{i..}} }

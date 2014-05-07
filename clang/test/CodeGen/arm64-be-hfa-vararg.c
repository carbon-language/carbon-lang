// RUN:  %clang_cc1 -triple arm64_be-linux-gnu -ffreestanding -emit-llvm -O0 -o - %s | FileCheck %s

#include <stdarg.h>

// A single member HFA must be aligned just like a non-HFA register argument.
double callee(int a, ...) {
// CHECK: = add i64 %{{.*}}, 8
  va_list vl;
  va_start(vl, a);
  double result = va_arg(vl, struct { double a; }).a;
  va_end(vl);
  return result;
}

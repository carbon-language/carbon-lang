// RUN: %clang -target xcore -O1 -o - -emit-llvm -S %s | FileCheck %s

// CHECK: target datalayout = "e-p:32:32:32-a0:0:32-n32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f16:16:32-f32:32:32-f64:32:32"
// CHECK: target triple = "xcore"

#include <stdarg.h>
struct x { int a; };
void testva (int n, ...) {
  va_list ap;
  // CHECK: %ap = alloca i8*, align 4

  char* v1 = va_arg (ap, char*);
  // CHECK: %0 = va_arg i8** %ap, i8*

  int v2 = va_arg (ap, int);
  // CHECK: %1 = va_arg i8** %ap, i32

  long long int v3 = va_arg (ap, long long int);
  // CHECK: %2 = va_arg i8** %ap, i64

  //struct x t = va_arg (ap, struct x);
  //cannot compile aggregate va_arg expressions yet
}

void testbuiltin (void) {
// CHECK: %0 = tail call i32 @llvm.xcore.getid()
// CHECK: %1 = tail call i32 @llvm.xcore.getps(i32 %0)
// CHECK: %2 = tail call i32 @llvm.xcore.bitrev(i32 %1)
// CHECK: tail call void @llvm.xcore.setps(i32 %0, i32 %2)
  int i = __builtin_getid();
  unsigned int ui = __builtin_getps(i);
  ui = __builtin_bitrev(ui);
  __builtin_setps(i,ui);

}

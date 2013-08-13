// RUN: %clang -target xcore -O1 -o - -emit-llvm -S %s | FileCheck %s

// CHECK: target datalayout = "e-p:32:32:32-a0:0:32-n32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f16:16:32-f32:32:32-f64:32:32"
// CHECK: target triple = "xcore"

#include <stdarg.h>
struct x { int a; };
void testva (int n, ...) {
  va_list ap;
  // CHECK: [[AP:%[a-z0-9]+]] = alloca i8*, align 4

  char* v1 = va_arg (ap, char*);
  // CHECK: va_arg i8** [[AP]], i8*

  int v2 = va_arg (ap, int);
  // CHECK: va_arg i8** [[AP]], i32

  long long int v3 = va_arg (ap, long long int);
  // CHECK: va_arg i8** [[AP]], i64

  //struct x t = va_arg (ap, struct x);
  //cannot compile aggregate va_arg expressions yet
}

void testbuiltin (void) {
// CHECK: [[I:%[a-z0-9]+]] = tail call i32 @llvm.xcore.getid()
// CHECK: [[UI:%[a-z0-9]+]] = tail call i32 @llvm.xcore.getps(i32 [[I]])
// CHECK: [[UI2:%[a-z0-9]+]] = tail call i32 @llvm.xcore.bitrev(i32 [[UI]])
// CHECK: tail call void @llvm.xcore.setps(i32 [[I]], i32 [[UI2]])
  int i = __builtin_getid();
  unsigned int ui = __builtin_getps(i);
  ui = __builtin_bitrev(ui);
  __builtin_setps(i,ui);

}

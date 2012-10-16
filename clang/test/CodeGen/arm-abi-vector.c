// RUN: %clang_cc1 -triple armv7-apple-darwin -target-abi aapcs -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7-apple-darwin -target-abi apcs-gnu -emit-llvm -o - %s | FileCheck -check-prefix=APCS-GNU %s

#include <stdarg.h>

typedef __attribute__(( ext_vector_type(2) ))  int __int2;

// Passing legal vector types as varargs.
double varargs_vec_2i(int fixed, ...) {
// CHECK: varargs_vec_2i
// CHECK: %c3 = alloca <2 x i32>, align 8
// CHECK: %3 = and i32 %2, -8
// CHECK: %ap.align = inttoptr i32 %3 to i8*
// CHECK: %ap.next = getelementptr i8* %ap.align, i32 8
// CHECK: bitcast i8* %ap.align to <2 x i32>*
// APCS-GNU: varargs_vec_2i
// APCS-GNU: %c3 = alloca <2 x i32>, align 8
// APCS-GNU: %var.align = alloca <2 x i32>
// APCS-GNU: %ap.next = getelementptr i8* %ap.cur, i32 8
// APCS-GNU: %1 = bitcast <2 x i32>* %var.align to i8*
// APCS-GNU: call void @llvm.memcpy
// APCS-GNU: %2 = load <2 x i32>* %var.align
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __int2 c3 = va_arg(ap, __int2);
  sum = sum + c3.x + c3.y;
  va_end(ap);
  return sum;
}

double test_2i(__int2 *in) {
// CHECK: test_2i
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_2i(i32 3, <2 x i32> %1)
// APCS-GNU: test_2i
// APCS-GNU: call double (i32, ...)* @varargs_vec_2i(i32 3, <2 x i32> %1)
  return varargs_vec_2i(3, *in);
}

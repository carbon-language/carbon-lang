// RUN: %clang_cc1 -triple armv7-apple-darwin -target-abi aapcs -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7-apple-darwin -target-abi apcs-gnu -emit-llvm -o - %s | FileCheck -check-prefix=APCS-GNU %s

#include <stdarg.h>

typedef __attribute__(( ext_vector_type(2) ))  int __int2;
typedef __attribute__(( ext_vector_type(3) ))  char __char3;
typedef __attribute__(( ext_vector_type(5) ))  char __char5;
typedef __attribute__(( ext_vector_type(9) ))  char __char9;
typedef __attribute__(( ext_vector_type(19) )) char __char19;
typedef __attribute__(( ext_vector_type(3) ))  short __short3;
typedef __attribute__(( ext_vector_type(5) ))  short __short5;

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

double varargs_vec_3c(int fixed, ...) {
// CHECK: varargs_vec_3c
// CHECK: %c3 = alloca <3 x i8>, align 4
// CHECK: %ap.next = getelementptr i8* %ap.cur, i32 4
// CHECK: %1 = bitcast i8* %ap.cur to <3 x i8>*
// APCS-GNU: varargs_vec_3c
// APCS-GNU: %c3 = alloca <3 x i8>, align 4
// APCS-GNU: %ap.next = getelementptr i8* %ap.cur, i32 4
// APCS-GNU: bitcast i8* %ap.cur to <3 x i8>*
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char3 c3 = va_arg(ap, __char3);
  sum = sum + c3.x + c3.y;
  va_end(ap);
  return sum;
}

double test_3c(__char3 *in) {
// CHECK: test_3c
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_3c(i32 3, i32 %2)
// APCS-GNU: test_3c
// APCS-GNU: call double (i32, ...)* @varargs_vec_3c(i32 3, i32 %2)
  return varargs_vec_3c(3, *in);
}

double varargs_vec_5c(int fixed, ...) {
// CHECK: varargs_vec_5c
// CHECK: %c5 = alloca <5 x i8>, align 8
// CHECK: %3 = and i32 %2, -8
// CHECK: %ap.align = inttoptr i32 %3 to i8*
// CHECK: %ap.next = getelementptr i8* %ap.align, i32 8
// CHECK: bitcast i8* %ap.align to <5 x i8>*
// APCS-GNU: varargs_vec_5c
// APCS-GNU: %c5 = alloca <5 x i8>, align 8
// APCS-GNU: %var.align = alloca <5 x i8>
// APCS-GNU: %ap.next = getelementptr i8* %ap.cur, i32 8
// APCS-GNU: %1 = bitcast <5 x i8>* %var.align to i8*
// APCS-GNU: call void @llvm.memcpy
// APCS-GNU: %2 = load <5 x i8>* %var.align
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char5 c5 = va_arg(ap, __char5);
  sum = sum + c5.x + c5.y;
  va_end(ap);
  return sum;
}

double test_5c(__char5 *in) {
// CHECK: test_5c
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_5c(i32 5, <2 x i32> %3)
// APCS-GNU: test_5c
// APCS-GNU: call double (i32, ...)* @varargs_vec_5c(i32 5, <2 x i32> %3)
  return varargs_vec_5c(5, *in);
}

double varargs_vec_9c(int fixed, ...) {
// CHECK: varargs_vec_9c
// CHECK: %c9 = alloca <9 x i8>, align 16
// CHECK: %var.align = alloca <9 x i8>
// CHECK: %3 = and i32 %2, -8
// CHECK: %ap.align = inttoptr i32 %3 to i8*
// CHECK: %ap.next = getelementptr i8* %ap.align, i32 16
// CHECK: %4 = bitcast <9 x i8>* %var.align to i8*
// CHECK: call void @llvm.memcpy
// CHECK: %5 = load <9 x i8>* %var.align
// APCS-GNU: varargs_vec_9c
// APCS-GNU: %c9 = alloca <9 x i8>, align 16
// APCS-GNU: %var.align = alloca <9 x i8>
// APCS-GNU: %ap.next = getelementptr i8* %ap.cur, i32 16
// APCS-GNU: %1 = bitcast <9 x i8>* %var.align to i8*
// APCS-GNU: call void @llvm.memcpy
// APCS-GNU: %2 = load <9 x i8>* %var.align
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char9 c9 = va_arg(ap, __char9);
  sum = sum + c9.x + c9.y;
  va_end(ap);
  return sum;
}

double test_9c(__char9 *in) {
// CHECK: test_9c
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_9c(i32 9, <4 x i32> %3)
// APCS-GNU: test_9c
// APCS-GNU: call double (i32, ...)* @varargs_vec_9c(i32 9, <4 x i32> %3)
  return varargs_vec_9c(9, *in);
}

double varargs_vec_19c(int fixed, ...) {
// CHECK: varargs_vec_19c
// CHECK: %ap.next = getelementptr i8* %ap.cur, i32 4
// CHECK: %1 = bitcast i8* %ap.cur to i8**
// CHECK: %2 = load i8** %1
// CHECK: bitcast i8* %2 to <19 x i8>*
// APCS-GNU: varargs_vec_19c
// APCS-GNU: %ap.next = getelementptr i8* %ap.cur, i32 4
// APCS-GNU: %1 = bitcast i8* %ap.cur to i8**
// APCS-GNU: %2 = load i8** %1
// APCS-GNU: bitcast i8* %2 to <19 x i8>*
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __char19 c19 = va_arg(ap, __char19);
  sum = sum + c19.x + c19.y;
  va_end(ap);
  return sum;
}

double test_19c(__char19 *in) {
// CHECK: test_19c
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_19c(i32 19, <19 x i8>* %tmp)
// APCS-GNU: test_19c
// APCS-GNU: call double (i32, ...)* @varargs_vec_19c(i32 19, <19 x i8>* %tmp)
  return varargs_vec_19c(19, *in);
}

double varargs_vec_3s(int fixed, ...) {
// CHECK: varargs_vec_3s
// CHECK: %c3 = alloca <3 x i16>, align 8
// CHECK: %3 = and i32 %2, -8
// CHECK: %ap.align = inttoptr i32 %3 to i8*
// CHECK: %ap.next = getelementptr i8* %ap.align, i32 8
// CHECK: bitcast i8* %ap.align to <3 x i16>*
// APCS-GNU: varargs_vec_3s
// APCS-GNU: %c3 = alloca <3 x i16>, align 8
// APCS-GNU: %var.align = alloca <3 x i16>
// APCS-GNU: %ap.next = getelementptr i8* %ap.cur, i32 8
// APCS-GNU: %1 = bitcast <3 x i16>* %var.align to i8*
// APCS-GNU: call void @llvm.memcpy
// APCS-GNU: %2 = load <3 x i16>* %var.align
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __short3 c3 = va_arg(ap, __short3);
  sum = sum + c3.x + c3.y;
  va_end(ap);
  return sum;
}

double test_3s(__short3 *in) {
// CHECK: test_3s
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_3s(i32 3, <2 x i32> %2)
// APCS-GNU: test_3s
// APCS-GNU: call double (i32, ...)* @varargs_vec_3s(i32 3, <2 x i32> %2)
  return varargs_vec_3s(3, *in);
}

double varargs_vec_5s(int fixed, ...) {
// CHECK: varargs_vec_5s
// CHECK: %c5 = alloca <5 x i16>, align 16
// CHECK: %var.align = alloca <5 x i16>
// CHECK: %3 = and i32 %2, -8
// CHECK: %ap.align = inttoptr i32 %3 to i8*
// CHECK: %ap.next = getelementptr i8* %ap.align, i32 16
// CHECK: %4 = bitcast <5 x i16>* %var.align to i8*
// CHECK: call void @llvm.memcpy
// CHECK: %5 = load <5 x i16>* %var.align
// APCS-GNU: varargs_vec_5s
// APCS-GNU: %c5 = alloca <5 x i16>, align 16
// APCS-GNU: %var.align = alloca <5 x i16>
// APCS-GNU: %ap.next = getelementptr i8* %ap.cur, i32 16
// APCS-GNU: %1 = bitcast <5 x i16>* %var.align to i8*
// APCS-GNU: call void @llvm.memcpy
// APCS-GNU: %2 = load <5 x i16>* %var.align
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  __short5 c5 = va_arg(ap, __short5);
  sum = sum + c5.x + c5.y;
  va_end(ap);
  return sum;
}

double test_5s(__short5 *in) {
// CHECK: test_5s
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_vec_5s(i32 5, <4 x i32> %3)
// APCS-GNU: test_5s
// APCS-GNU: call double (i32, ...)* @varargs_vec_5s(i32 5, <4 x i32> %3)
  return varargs_vec_5s(5, *in);
}

// Pass struct as varargs.
typedef struct
{
  __int2 i2;
  float f;
} StructWithVec;

double varargs_struct(int fixed, ...) {
// CHECK: varargs_struct
// CHECK: %3 = and i32 %2, -8
// CHECK: %ap.align = inttoptr i32 %3 to i8*
// CHECK: %ap.next = getelementptr i8* %ap.align, i32 16
// CHECK: bitcast i8* %ap.align to %struct.StructWithVec*
// APCS-GNU: varargs_struct
// APCS-GNU: %var.align = alloca %struct.StructWithVec
// APCS-GNU: %ap.next = getelementptr i8* %ap.cur, i32 16
// APCS-GNU: %1 = bitcast %struct.StructWithVec* %var.align to i8*
// APCS-GNU: call void @llvm.memcpy
  va_list ap;
  double sum = fixed;
  va_start(ap, fixed);
  StructWithVec c3 = va_arg(ap, StructWithVec);
  sum = sum + c3.i2.x + c3.i2.y + c3.f;
  va_end(ap);
  return sum;
}

double test_struct(StructWithVec* d) {
// CHECK: test_struct
// CHECK: call arm_aapcscc double (i32, ...)* @varargs_struct(i32 3, [2 x i64] %3)
// APCS-GNU: test_struct
// APCS-GNU: call double (i32, ...)* @varargs_struct(i32 3, [2 x i64] %3)
  return varargs_struct(3, *d);
}

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

typedef double * __attribute__((align_value(64))) aligned_double;

void foo(aligned_double x, double * y __attribute__((align_value(32))),
         double & z __attribute__((align_value(128)))) { };
// CHECK: define void @_Z3fooPdS_Rd(double* align 64 %x, double* align 32 %y, double* dereferenceable(8) align 128 %z)


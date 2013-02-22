// RUN: %clang_cc1 -triple i386-unknown-unknown -Werror -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Werror -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7-unknown-unknown -Werror -emit-llvm -o - %s | FileCheck %s

void __attribute__((coldcc)) f1(void);

void f2(void) {
  f1();
// CHECK: call coldcc void @f1()
}

// CHECK: declare coldcc void @f1()

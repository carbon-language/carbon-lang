// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-pth -o %t1 %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-pth -o - %s > %t2
// RUN: cmp %t1 %t2
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-pth -o - %s | \
// RUN: FileCheck %s

// CHECK: cfe-pth

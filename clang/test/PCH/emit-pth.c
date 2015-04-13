// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-pth -o %t1 %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-pth -o - %s > %t2
// RUN: cmp %t1 %t2
// RUN: not %clang_cc1 -triple i386-unknown-unknown -emit-pth -o - %s 2>&1 | \
// RUN: FileCheck %s

// CHECK: PTH requires a seekable file for output!

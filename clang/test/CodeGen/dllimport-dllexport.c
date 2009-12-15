// RUN: %clang_cc1 -emit-llvm < %s -o %t
// RUN: grep 'dllexport' %t | count 1
// RUN: not grep 'dllimport' %t

void __attribute__((dllimport)) foo1();
void __attribute__((dllexport)) foo1(){}
void __attribute__((dllexport)) foo2();

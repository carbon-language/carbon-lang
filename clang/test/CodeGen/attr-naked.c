// RUN: %clang_cc1 -g -emit-llvm -o %t %s
// RUN: grep 'naked' %t

void t1() __attribute__((naked));

void t1()
{
}


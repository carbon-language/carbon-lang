// RUN: clang -emit-llvm -S -o %t %s &&
// RUN: grep '@x = common global' %t &&
// RUN: clang -fno-common -emit-llvm -S -o %t %s &&
// RUN: grep '@x = global' %t

int x;

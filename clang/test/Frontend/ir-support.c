// Test that we can consume LLVM IR/bitcode in the frontend and produce
// identical output to a standard compilation.

// Reference output:
// RUN: %clang_cc1 -S -o %t.s %s

// LLVM bitcode:
// RUN: %clang_cc1 -emit-llvm-bc -o %t.bc %s
// RUN: %clang_cc1 -S -o - %t.bc > %t.bc.s
// RUN: diff %t.s %t.bc.s

// LLVM IR source code:
// RUN: %clang_cc1 -emit-llvm-bc -o %t.ll %s
// RUN: %clang_cc1 -S -o - %t.ll > %t.ll.s
// RUN: diff %t.s %t.ll.s

int f() { return 0; }

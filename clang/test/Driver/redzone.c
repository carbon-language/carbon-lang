// RUN: clang -mno-red-zone %s -S -emit-llvm -o %t.log &&
// RUN: grep 'noredzone' %t.log 
// RUN: clang -mred-zone %s -S -emit-llvm -o %t.log &&
// RUN: grep -v 'noredzone' %t.log 

int foo() { return 42; }

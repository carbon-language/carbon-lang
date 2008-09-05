// RUN: clang -emit-llvm -o %t.clang.ll %s &&
// RUN: %llvmgcc -c --emit-llvm -o - %s | llvm-dis -f -o %t.gcc.ll &&
// RUN: grep "define" %t.clang.ll | sort > %t.clang.defs &&
// RUN: grep "define" %t.gcc.ll | sort > %t.gcc.defs &&
// RUN: diff %t.clang.defs %t.gcc.defs

signed char f0(int x) { return x; }

unsigned char f1(int x) { return x; }

void f2(signed char x) { }

void f3(unsigned char x) { }

signed short f4(int x) { return x; }

unsigned short f5(int x) { return x; }

void f6(signed short x) { }

void f7(unsigned short x) { }


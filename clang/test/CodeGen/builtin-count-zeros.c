// RUN: clang-cc -emit-llvm %s -o - | grep 'cttz' | count 2 &&
// RUN: clang-cc -emit-llvm %s -o - | grep 'ctlz' | count 2

int a(int a) {return __builtin_ctz(a) + __builtin_clz(a);}

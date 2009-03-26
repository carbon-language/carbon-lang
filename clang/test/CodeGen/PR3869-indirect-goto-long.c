// RUN: clang-cc -emit-llvm-bc -o - %s
// PR3869
int a(long long b) { goto *b; }


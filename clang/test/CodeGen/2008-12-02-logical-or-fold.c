// RUN: clang-cc -emit-llvm -o - %s | grep "store i32 1"
// PR3150

int a() {return 1||1;}

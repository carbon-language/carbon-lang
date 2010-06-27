// RUN: %clang_cc1 -emit-llvm -o - %s | grep "ret i32 1"
// PR3150

int a() {return 1||1;}

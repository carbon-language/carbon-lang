// RUN: %clang_cc1 %s -emit-llvm -g -o /dev/null
// XTARGET: darwin,linux
// XFAIL: *
void foo() {}


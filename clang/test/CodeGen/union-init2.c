// RUN: %clang_cc1 -emit-llvm %s -o - -triple i686-pc-linux-gnu | grep "bitcast (%0\* @r to %union.x\*), \[4 x i8\] undef"

// Make sure we generate something sane instead of a ptrtoint
union x {long long b;union x* a;} r = {.a = &r};

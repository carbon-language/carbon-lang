// RUN: clang-cc -emit-llvm %s -o - | not grep ptrtoint

// Make sure we generate something sane instead of a ptrtoint
union x {long long b;union x* a;} r = {.a = &r};

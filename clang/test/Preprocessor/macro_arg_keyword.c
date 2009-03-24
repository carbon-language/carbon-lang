// RUN: clang-cc -E %s | grep xxx-xxx

#define foo(return) return-return

foo(xxx)


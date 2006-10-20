// RUN: clang -E %s | grep xxx-xxx

#define foo(return) return-return

foo(xxx)


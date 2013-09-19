// RUN: %clang -target i386-pc-win32 -fms-extensions -fsyntax-only %s

#include <Intrin.h>

template <typename T>
void foo(T V) {}

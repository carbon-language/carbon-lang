// RUN: %clang -target i386-pc-win32 -fms-extensions -ffreestanding -fsyntax-only %s

// Intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;

#include <Intrin.h>

// Use some C++ to make sure we closed the extern "C" brackets.
template <typename T>
void foo(T V) {}

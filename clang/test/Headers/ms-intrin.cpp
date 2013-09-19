// RUN: %clang -target i386-pc-win32 -fms-extensions -fsyntax-only %s

// Get size_t, but avoid including mm_malloc.h which includes stdlib.h which may
// not exist.
#include <stdint.h>
#undef __STDC_HOSTED__

#include <Intrin.h>

template <typename T>
void foo(T V) {}

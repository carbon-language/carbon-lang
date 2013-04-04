// RUN: %clang -fsyntax-only -std=c++11 %s
// RUN: %clang -fsyntax-only -std=c++11 -fmodules %s

#include <stdalign.h>

#if defined alignas
#error alignas should not be defined in C++
#endif

#if defined alignof
#error alignof should not be defined in C++
#endif

static_assert(__alignas_is_defined, "");
static_assert(__alignof_is_defined, "");

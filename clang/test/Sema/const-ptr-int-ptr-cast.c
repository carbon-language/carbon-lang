// RUN: clang-cc -fsyntax-only -verify %s

#include <stdint.h>

char *a = (void*)(uintptr_t)(void*)&a;

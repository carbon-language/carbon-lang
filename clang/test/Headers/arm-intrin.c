// RUN: %clang -target armv7-windows -I %S/Inputs/include -Xclang -verify -E %s
// expected-no-diagnostics

typedef __SIZE_TYPE__ size_t;
#include <Intrin.h>


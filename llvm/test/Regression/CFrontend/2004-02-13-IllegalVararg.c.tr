// RUN: %llvmgcc -xc %s -c -o - | llc

#include <stdarg.h>

float test(int X, ...) {
  va_list ap;
  float F;
  va_start(ap, X);
  F = va_arg(ap, float);
  return F;
}

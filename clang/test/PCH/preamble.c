// Check that using the preamble option actually skips the preamble.

// RUN: %clang_cc1 -emit-pch -o %t %S/Inputs/preamble.h
// RUN: %clang_cc1 -include-pch %t -preamble-bytes=278,1 -DFOO=f -verify %s

float f(int); // Not an error, because we skip this via the preamble!












int g(int x) {
  return FOO(x);
}

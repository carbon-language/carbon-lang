// REQUIRES: powerpc-registered-target
// XFAIL: powerpc

// RUN: %clang -faltivec -target powerpc64le-unknown-unknown  -mcpu=power8 \
// RUN: -Wall -Wextra -c %s
// RUN: %clang -faltivec -target powerpc64-unknown-unknown  -mcpu=power8 \
// RUN: -Wall -Wextra -c %s

// Expect the compile to fail with "cannot compile this builtin function yet"
extern vector signed int vsi;
extern vector unsigned char vuc;

vector unsigned long long testExtractWord(void) {
  return  __builtin_vsx_extractuword(vuc, 12);
}

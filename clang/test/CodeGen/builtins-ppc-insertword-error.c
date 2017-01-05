// REQUIRES: powerpc-registered-target
// XFAIL: powerpc

// RUN: %clang -faltivec -target powerpc64le-unknown-unknown -mcpu=power8 \
// RUN: -Wall -Werror -c %s

// RUN: %clang -faltivec -target powerpc64-unknown-unknown -mcpu=power8 \
// RUN: -Wall -Werror -c %s

// expect to fail  with diagnostic: "cannot compile this builtin function yet"
extern vector signed int vsi;
extern vector unsigned char vuc;

vector  unsigned char testInsertWord(void) {
  return __builtin_vsx_insertword(vsi, vuc, 0);
}

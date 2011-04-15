// RUN: %llvmgcc -S %s -o - | FileCheck %s
extern int printf(const char *, ...);
static void bad(unsigned int v1, unsigned int v2) {
  printf("%u\n", 1631381461u * (((v2 - 1273463329u <= v1 - 1273463329u) ? v2 : v1) - 1273463329u) + 121322179u);
}
// Radar 8198362
// GCC FE wants to convert the above to
//   1631381461u * MIN(v2 - 1273463329u, v1 - 1273463329u)
// and then to
//   MIN(1631381461u * v2 - 4047041419, 1631381461u * v1 - 4047041419)
//
// 1631381461u * 1273463329u = 2077504466193943669, but 32-bit overflow clips
// this to 4047041419. This breaks the comparison implicit in the MIN().
// Two multiply operations suggests the bad optimization is happening;
// one multiplication, after the MIN(), is correct.
// CHECK: mul
// CHECK-NOT: mul
// CHECK: ret

#include <stdio.h>
#include <math.h>

int target_isinf(double x) {
  union {
    double d;
    struct {
      unsigned mantissa2;
      unsigned mantissa1 : 20;
      unsigned exponent  : 11;
      unsigned sign      :  1;
    } big_endian;
  } u;

  u.d = x;
  return (u.big_endian.exponent == 2047 && u.big_endian.mantissa1 == 0 && u.big_endian.mantissa2 == 0);
}

int main() {
  printf("%d %d\n", target_isinf(1234.42), target_isinf(1.0/1.0e-1000));
  return 0;
}

// RUN: clang-cc -emit-llvm -o - %s > %t
// RUN: ! grep "__builtin" %t

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void test(long double a, int b) {
  printf("%Lf**%d: %08x %08x %016Lx\n", 
         a, b,
         __builtin_powi(a, b),
         __builtin_powif(a, b),
         __builtin_powil(a, b)
         );
}

int main() {
  int i;

  test(-1,-1LL);
  test(0,0);
  test(1,1);

  for (i=0; i<3; i++) {
    test(random(), i);
  }

  return 0;
}

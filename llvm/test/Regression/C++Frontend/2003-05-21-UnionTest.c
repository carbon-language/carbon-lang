#include <stdio.h>

int __signbit (double __x) {
  union { double __d; int __i[3]; } __u = { __d: __x };
  return __u.__i[1] < 0;
}

int main() {
  printf("%d %d\n", __signbit(-1), __signbit(2.0));
  return 0;
}

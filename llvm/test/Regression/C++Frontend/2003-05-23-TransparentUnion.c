#include <stdio.h>

typedef union {
  float *__fptr;
  int *__iptr;
} UNION __attribute__ ((__transparent_union__));

int try(UNION U) {
  return 1;
}
int test() {
  int I;
  float F;
  return try(&I) | try(&F);
}

int main() {
  if (test()) printf("ok");
  return 0;
}

// RUN: %libomptarget-compile-generic -fopenmp-version=51
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic


#include <stdio.h>

int main() {
  int i;

  // CHECK: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  fprintf(stderr, "addr=%p, size=%ld\n", &i, sizeof i);

  // CHECK-NOT: Libomptarget
#pragma omp target enter data map(alloc: i)
#pragma omp target enter data map(present, alloc: i)
#pragma omp target exit data map(delete: i)

  // CHECK: i is present
  fprintf(stderr, "i is present\n");

  // CHECK: Libomptarget message: device mapping required by 'present' map type modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK: Libomptarget error: Call to getOrAllocTgtPtr returned null pointer ('present' map type modifier).
  // CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory
#pragma omp target enter data map(present, alloc: i)

  // CHECK-NOT: i is present
  fprintf(stderr, "i is present\n");

  return 0;
}

// Check profile with a single new call and set of loads and stores. Ensures
// we get the same profile regardless of whether the memory is deallocated
// before exit.

// RUN: %clangxx_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_memprof -DFREE -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr %run %t 2>&1 | FileCheck %s

// Try again with callbacks instead of inline sequences
// RUN: %clangxx_memprof -mllvm -memprof-use-callbacks -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr %run %t 2>&1 | FileCheck %s

// This is actually:
//  Memory allocation stack id = STACKID
//   alloc_count 1, size (ave/min/max) 40.00 / 40 / 40
// but we need to look for them in the same CHECK to get the correct STACKID.
// CHECK:  Memory allocation stack id = [[STACKID:[0-9]+]]{{[[:space:]].*}}alloc_count 1, size (ave/min/max) 40.00 / 40 / 40
// CHECK-NEXT:  access_count (ave/min/max): 20.00 / 20 / 20
// CHECK-NEXT:  lifetime (ave/min/max): [[AVELIFETIME:[0-9]+]].00 / [[AVELIFETIME]] / [[AVELIFETIME]]
// CHECK-NEXT:  num migrated: {{[0-1]}}, num lifetime overlaps: 0, num same alloc cpu: 0, num same dealloc_cpu: 0
// CHECK: Stack for id [[STACKID]]:
// CHECK-NEXT: #0 {{.*}} in operator new
// CHECK-NEXT: #1 {{.*}} in main {{.*}}:[[@LINE+6]]

#include <stdio.h>
#include <stdlib.h>

int main() {
  int *p = new int[10];
  for (int i = 0; i < 10; i++)
    p[i] = i;
  int j = 0;
  for (int i = 0; i < 10; i++)
    j += p[i];
#ifdef FREE
  delete[] p;
#endif

  return 0;
}

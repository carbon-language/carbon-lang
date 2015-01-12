// RUN: %clang_cl_asan -O0 %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdio.h>

char bigchunk[1 << 30];

int main() {
  printf("Hello, world!\n");
  scanf("%s", bigchunk);
// CHECK-NOT: Hello, world!
// CHECK: Shadow memory range interleaves with an existing memory mapping.
// CHECK: Dumping process modules:
// CHECK-DAG: 0x{{[0-9a-f]*}}-0x{{[0-9a-f]*}} {{.*}}shadow_mapping_failure
// CHECK-DAG: 0x{{[0-9a-f]*}}-0x{{[0-9a-f]*}} {{.*}}kernel32.dll
// CHECK-DAG: 0x{{[0-9a-f]*}}-0x{{[0-9a-f]*}} {{.*}}ntdll.dll
}

// RUN: %clangxx_asan -O0 %p/dll_host.cc -Fe%t
// RUN: %clangxx_asan -LD -O0 %s -Fe%t.dll
// FIXME: 'cat' is needed due to PR19744.
// RUN: not %run %t %t.dll 2>&1 | cat | FileCheck %s

#include <malloc.h>

extern "C" __declspec(dllexport)
int test_function() {
  char *buffer = (char*)malloc(42);
  free(buffer);
  buffer[0] = 42;
// CHECK: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
// CHECK:       test_function {{.*}}dll_malloc_uaf.cc:[[@LINE-3]]
// CHECK-NEXT:  main {{.*}}dll_host
// CHECK: [[ADDR]] is located 0 bytes inside of 42-byte region
// CHECK-LABEL: freed by thread T0 here:
// CHECK:       free
// CHECK:       test_function {{.*}}dll_malloc_uaf.cc:[[@LINE-9]]
// CHECK-NEXT:  main {{.*}}dll_host
// CHECK-LABEL: previously allocated by thread T0 here:
// CHECK:       malloc
// CHECK:       test_function {{.*}}dll_malloc_uaf.cc:[[@LINE-14]]
// CHECK-NEXT:  main {{.*}}dll_host
  return 0;
}

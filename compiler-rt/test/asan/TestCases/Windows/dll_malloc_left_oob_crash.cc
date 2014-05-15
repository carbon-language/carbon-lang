// RUN: %clangxx_asan -O0 %p/dll_host.cc -Fe%t
// RUN: %clangxx_asan -LD -O0 %s -Fe%t.dll
// FIXME: 'cat' is needed due to PR19744.
// RUN: not %run %t %t.dll 2>&1 | cat | FileCheck %s

#include <malloc.h>
extern "C" __declspec(dllexport)
int test_function() {
  char *buffer = (char*)malloc(42);
  buffer[-1] = 42;
// CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
// CHECK:   test_function {{.*}}dll_malloc_left_oob_crash.cc:[[@LINE-3]]
// CHECK:   main {{.*}}dll_host.cc
// CHECK: [[ADDR]] is located 1 bytes to the left of 42-byte region
// CHECK-LABEL: allocated by thread T0 here:
// CHECK:   malloc
// CHECK:   test_function {{.*}}dll_malloc_left_oob_crash.cc:[[@LINE-9]]
// CHECK:   main {{.*}}dll_host.cc
// CHECK-LABEL: SUMMARY
  free(buffer);
  return 0;
}

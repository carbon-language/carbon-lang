// RUN: %clang_cl_asan -O0 %p/dll_host.cc -Fe%t
// RUN: %clang_cl_asan -LD -O0 %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

#include <string.h>

extern "C" __declspec(dllexport)
int test_function() {
  char buff[6] = "Hello";

  memchr(buff, 'z', 7);
// CHECK: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: READ of size 7 at [[ADDR]] thread T0
// CHECK-NEXT:  __asan_wrap_memchr
// CHECK-NEXT:  memchr
// CHECK-NEXT:  test_function {{.*}}dll_intercept_memchr.cc:[[@LINE-5]]
// CHECK: Address [[ADDR]] is located in stack of thread T0 at offset {{.*}} in frame
// CHECK-NEXT:  test_function {{.*}}dll_intercept_memchr.cc
// CHECK: 'buff' <== Memory access at offset {{.*}} overflows this variable
  return 0;
}

// Check that memset() call from a shared library gets intercepted.

// RUN: %clangxx_asan -O0 %s -DSHARED_LIB \
// RUN:     -shared -o %T/libinterception-in-shared-lib-test.so \
// RUN:     -fPIC
// TODO(glider): figure out how to set rpath in a more portable way and unite
// this test with ../Linux/interception-in-shared-lib-test.cc.
// RUN: %clangxx_asan -O0 %s -o %t -Wl,-rpath,@executable-path -L%T -linterception-in-shared-lib-test && \
// RUN:     not %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <string.h>

#if defined(SHARED_LIB)
extern "C"
void my_memset(void *p, size_t sz) {
  memset(p, 0, sz);
}
#else
extern "C" void my_memset(void *p, size_t sz);

int main(int argc, char *argv[]) {
  char buf[10];
  my_memset(buf, 11);
  // CHECK: {{.*ERROR: AddressSanitizer: stack-buffer-overflow}}
  // CHECK: {{WRITE of size 11 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in my_memset .*interception-in-shared-lib-test.cc:17}}
  return 0;
}
#endif

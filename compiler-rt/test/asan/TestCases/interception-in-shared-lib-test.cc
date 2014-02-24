// Check that memset() call from a shared library gets intercepted.

// RUN: %clangxx_asan -O0 %p/SharedLibs/shared-lib-test-so.cc \
// RUN:     -shared -o %T/libinterception-in-shared-lib-test.so \
// RUN:     -fPIC -Wl,--soname,libinterception-in-shared-lib-test.so
// RUN: %clangxx_asan -O0 %s -o %t -Wl,-R,\$ORIGIN -L%T -linterception-in-shared-lib-test && \
// RUN:     not %t 2>&1 | FileCheck %s

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include <string>

using std::string;

typedef void (fun_t)(void *p, size_t sz);

int main(int argc, char *argv[]) {
  string path = string(argv[0]) + "-so.so";
  printf("opening %s ... \n", path.c_str());
  void *lib = dlopen(path.c_str(), RTLD_NOW);
  if (!lib) {
    printf("error in dlopen(): %s\n", dlerror());
    return 1;
  }
  fun_t *my_memset = (fun_t*)dlsym(lib, "my_memset");
  if (!my_memset) return 1;
  printf("ok\n");
  char buf[10];
  my_memset(buf, 11);
  // CHECK: {{.*ERROR: AddressSanitizer: stack-buffer-overflow}}
  // CHECK: {{WRITE of size 11 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in my_memset .*shared-lib-test-so.cc:31}}
  return 0;
}

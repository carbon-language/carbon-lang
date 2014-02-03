// Check that --gc-sections does not throw away (or localize) parts of sanitizer
// interface.
// RUN: %clang_asan -m64 %s -Wl,--gc-sections -o %t
// RUN: %clang_asan -m64 %s -DBUILD_SO -fPIC -o %t-so.so -shared
// RUN: %t 2>&1

#ifndef BUILD_SO
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  char path[4096];
  snprintf(path, sizeof(path), "%s-so.so", argv[0]);

  void *handle = dlopen(path, RTLD_LAZY);
  if (!handle) fprintf(stderr, "%s\n", dlerror());
  assert(handle != 0);

  typedef void (*F)();
  F f = (F)dlsym(handle, "call_rtl_from_dso");
  printf("%s\n", dlerror());
  assert(dlerror() == 0);
  f();
  
  dlclose(handle);
  return 0;
}

#else // BUILD_SO

#include <sanitizer/msan_interface.h>
extern "C" void call_rtl_from_dso() {
  volatile int32_t x;
  volatile int32_t y = __sanitizer_unaligned_load32((void *)&x);
}

#endif // BUILD_SO

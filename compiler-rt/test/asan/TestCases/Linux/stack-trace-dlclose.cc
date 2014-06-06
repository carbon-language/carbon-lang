// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
//
// RUN: %clangxx_asan -DSHARED %s -shared -o %T/stack_trace_dlclose.so -fPIC
// RUN: %clangxx_asan -DSO_DIR=\"%T\" %s -o %t
// RUN: ASAN_OPTIONS=exitcode=0 %run %t 2>&1 | FileCheck %s
// XFAIL: arm-linux-gnueabi

#include <assert.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <sanitizer/common_interface_defs.h>

#ifdef SHARED
extern "C" {
void *foo() {
  return malloc(1);
}
}
#else
void *handle;

int main(int argc, char **argv) {
  void *handle = dlopen(SO_DIR "/stack_trace_dlclose.so", RTLD_LAZY);
  assert(handle);
  void *(*foo)() = (void *(*)())dlsym(handle, "foo");
  assert(foo);
  void *p = foo();
  assert(p);
  dlclose(handle);

  free(p);
  free(p);  // double-free

  return 0;
}
#endif

// CHECK: {{    #0 0x.* in malloc}}
// CHECK: {{    #1 0x.* \(<unknown module>\)}}
// CHECK: {{    #2 0x.* in main}}

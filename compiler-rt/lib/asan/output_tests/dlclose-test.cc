//===----------- dlclose-test.cc --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Regression test for
// http://code.google.com/p/address-sanitizer/issues/detail?id=19
// Bug description:
// 1. application dlopens foo.so
// 2. asan registers all globals from foo.so
// 3. application dlcloses foo.so
// 4. application mmaps some memory to the location where foo.so was before
// 5. application starts using this mmaped memory, but asan still thinks there
// are globals.
// 6. BOOM
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#include <string>

using std::string;

static const int kPageSize = 4096;

typedef int *(fun_t)();

int main(int argc, char *argv[]) {
  string path = string(argv[0]) + "-so.so";
  printf("opening %s ... \n", path.c_str());
  void *lib = dlopen(path.c_str(), RTLD_NOW);
  if (!lib) {
    printf("error in dlopen(): %s\n", dlerror());
    return 1;
  }
  fun_t *get = (fun_t*)dlsym(lib, "get_address_of_static_var");
  if (!get) {
    printf("failed dlsym\n");
    return 1;
  }
  int *addr = get();
  assert(((size_t)addr % 32) == 0);  // should be 32-byte aligned.
  printf("addr: %p\n", addr);
  addr[0] = 1;  // make sure we can write there.

  // Now dlclose the shared library.
  printf("attempting to dlclose\n");
  if (dlclose(lib)) {
    printf("failed to dlclose\n");
    return 1;
  }
  // Now, the page where 'addr' is unmapped. Map it.
  size_t page_beg = ((size_t)addr) & ~(kPageSize - 1);
  void *res = mmap((void*)(page_beg), kPageSize,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE, 0, 0);
  if (res == (char*)-1L) {
    printf("failed to mmap\n");
    return 1;
  }
  addr[1] = 2;  // BOOM (if the bug is not fixed).
  printf("PASS\n");
  // Check-Common: PASS
  return 0;
}

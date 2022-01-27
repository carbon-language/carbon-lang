// When we link a binary without the -debug flag, ASan should print out VAs
// instead of RVAs. The frames for main and do_uaf should be above 0x400000,
// which is the default image base of an executable.

// RUN: rm -f %t.pdb
// RUN: %clangxx_asan -c -O2 %s -o %t.obj
// RUN: link /nologo /OUT:%t.exe %t.obj %asan_lib %asan_cxx_lib
// RUN: not %run %t.exe 2>&1 | FileCheck %s

#include <stdlib.h>
#include <stdio.h>
int __attribute__((noinline)) do_uaf(void);
int main() {
  int r = do_uaf();
  printf("r: %d\n", r);
  return r;
}
int do_uaf(void) {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: AddressSanitizer: heap-use-after-free
  // CHECK: #0 {{0x[a-f0-9]+ \(.*[\\/]unsymbolized.cpp.*.exe\+(0x40|0x14000)[a-f0-9]{4}\)}}
  // CHECK: #1 {{0x[a-f0-9]+ \(.*[\\/]unsymbolized.cpp.*.exe\+(0x40|0x14000)[a-f0-9]{4}\)}}
}

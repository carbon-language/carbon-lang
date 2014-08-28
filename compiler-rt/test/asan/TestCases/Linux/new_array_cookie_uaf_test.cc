// REQUIRES: asan-64-bits
// RUN: %clangxx_asan -O3 %s -o %t
// DISABLED: ASAN_OPTIONS=poison_array_cookie=1 not %run %t 2>&1  | FileCheck %s --check-prefix=COOKIE
// RUN: ASAN_OPTIONS=poison_array_cookie=0 not %run %t 2>&1  | FileCheck %s --check-prefix=NO_COOKIE
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
int dtor_counter;
struct C {
  int x;
  ~C() {
    fprintf(stderr, "DTOR\n");
    dtor_counter++;
    if (dtor_counter >= 100) {
      fprintf(stderr, "Called DTOR too many times\n");
// NO_COOKIE: Called DTOR too many times
      exit(1);
    }
  }
};

int main(int argc, char **argv) {
  C *buffer = new C[argc];
  delete [] buffer;
  delete [] buffer;
// COOKIE: AddressSanitizer: loaded array cookie from free-d memory
// COOKIE: AddressSanitizer: attempting double-free
}

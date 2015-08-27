// Try a strlen suppression, but force the input file to be DOS format (CRLF).
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: python -c 'import sys; sys.stdout.write("interceptor_name:strlen\r\n")' > %t.supp
// RUN: %env_asan_opts=suppressions='"%t.supp"' %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  char *a = (char *)malloc(6);
  free(a);
  size_t len = strlen(a); // BOOM
  fprintf(stderr, "strlen ignored, len = %zu\n", len);
}

// CHECK-NOT: AddressSanitizer: heap-use-after-free
// CHECK: strlen ignored

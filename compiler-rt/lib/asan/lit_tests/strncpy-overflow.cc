// RUN: %clangxx_asan -m64 -O2 %s -o %t
// RUN: %t 2>&1 | %symbolizer | c++filt > %t.output
// RUN: FileCheck %s < %t.output
// RUN: FileCheck %s --check-prefix=CHECK-%os < %t.output

#include <string.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  char *hello = (char*)malloc(6);
  strcpy(hello, "hello");
  char *short_buffer = (char*)malloc(9);
  strncpy(short_buffer, hello, 10);  // BOOM
  // CHECK: {{WRITE of size 1 at 0x.* thread T0}}
  // CHECK-Linux: {{    #0 0x.* in .*strncpy}}
  // CHECK-Darwin: {{    #0 0x.* in wrap_strncpy}}
  // CHECK: {{    #1 0x.* in main .*strncpy-overflow.cc:12}}
  // CHECK: {{0x.* is located 0 bytes to the right of 9-byte region}}
  // CHECK: {{allocated by thread T0 here:}}

  // CHECK-Linux: {{    #0 0x.* in .*malloc}}
  // CHECK-Linux: {{    #1 0x.* in main .*strncpy-overflow.cc:11}}

  // CHECK-Darwin: {{    #0 0x.* in .*mz_malloc.*}}
  // CHECK-Darwin: {{    #1 0x.* in malloc_zone_malloc.*}}
  // CHECK-Darwin: {{    #2 0x.* in malloc.*}}
  // CHECK-Darwin: {{    #3 0x.* in main .*strncpy-overflow.cc:11}}
  return short_buffer[8];
}

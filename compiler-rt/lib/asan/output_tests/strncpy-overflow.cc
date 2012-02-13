#include <string.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  char *hello = (char*)malloc(6);
  strcpy(hello, "hello");
  char *short_buffer = (char*)malloc(9);
  strncpy(short_buffer, hello, 10);  // BOOM
  return short_buffer[8];
}

// CHECK: {{WRITE of size 1 at 0x.* thread T0}}
// CHECK: {{    #0 0x.* in strncpy}}
// CHECK: {{    #1 0x.* in main .*strncpy-overflow.cc:[78]}}
// CHECK: {{0x.* is located 0 bytes to the right of 9-byte region}}
// CHECK: {{allocated by thread T0 here:}}
// CHECK: {{    #0 0x.* in malloc}}
// CHECK: {{    #1 0x.* in main .*strncpy-overflow.cc:6}}

// Darwin: {{WRITE of size 1 at 0x.* thread T0}}
// Darwin: {{    #0 0x.* in wrap_strncpy}}
// Darwin: {{    #1 0x.* in main .*strncpy-overflow.cc:[78]}}
// Darwin: {{0x.* is located 0 bytes to the right of 9-byte region}}
// Darwin: {{allocated by thread T0 here:}}
// Darwin: {{    #0 0x.* in .*mz_malloc.*}}
// Darwin: {{    #1 0x.* in malloc_zone_malloc.*}}
// Darwin: {{    #2 0x.* in malloc.*}}
// Darwin: {{    #3 0x.* in main .*strncpy-overflow.cc:6}}

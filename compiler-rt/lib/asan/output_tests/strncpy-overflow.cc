#include <string.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  char *hello = (char*)malloc(6);
  strcpy(hello, "hello");
  char *short_buffer = (char*)malloc(9);
  strncpy(short_buffer, hello, 10);  // BOOM
  return short_buffer[8];
}

// Check-Common: {{WRITE of size 1 at 0x.* thread T0}}
// Check-Linux: {{    #0 0x.* in .*strncpy}}
// Check-Darwin: {{    #0 0x.* in wrap_strncpy}}
// Check-Common: {{    #1 0x.* in main .*strncpy-overflow.cc:7}}
// Check-Common: {{0x.* is located 0 bytes to the right of 9-byte region}}
// Check-Common: {{allocated by thread T0 here:}}

// Check-Linux: {{    #0 0x.* in .*malloc}}
// Check-Linux: {{    #1 0x.* in main .*strncpy-overflow.cc:6}}

// Check-Darwin: {{    #0 0x.* in .*mz_malloc.*}}
// Check-Darwin: {{    #1 0x.* in malloc_zone_malloc.*}}
// Check-Darwin: {{    #2 0x.* in malloc.*}}
// Check-Darwin: {{    #3 0x.* in main .*strncpy-overflow.cc:6}}

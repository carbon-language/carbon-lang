#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char*)malloc(10 * sizeof(char));
  memset(x, 0, 10);
  int res = x[argc * 10];  // BOOOM
  free(x);
  return res;
}

// Check-Common: {{READ of size 1 at 0x.* thread T0}}
// Check-Common: {{    #0 0x.* in main .*heap-overflow.cc:6}}
// Check-Common: {{0x.* is located 0 bytes to the right of 10-byte region}}
// Check-Common: {{allocated by thread T0 here:}}

// Check-Linux: {{    #0 0x.* in __xsan_malloc}}
// Check-Linux: {{    #1 0x.* in main .*heap-overflow.cc:[45]}}

// Check-Darwin: {{    #0 0x.* in .*mz_malloc.*}}
// Check-Darwin: {{    #1 0x.* in malloc_zone_malloc.*}}
// Check-Darwin: {{    #2 0x.* in malloc.*}}
// Check-Darwin: {{    #3 0x.* in main heap-overflow.cc:[45]}}

#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
}

// Check-Common: {{.*ERROR: AddressSanitizer heap-use-after-free on address}}
// Check-Common:   {{0x.* at pc 0x.* bp 0x.* sp 0x.*}}
// Check-Common: {{READ of size 1 at 0x.* thread T0}}
// Check-Common: {{    #0 0x.* in main .*use-after-free.cc:5}}
// Check-Common: {{0x.* is located 5 bytes inside of 10-byte region .0x.*,0x.*}}
// Check-Common: {{freed by thread T0 here:}}

// Check-Linux: {{    #0 0x.* in .*free}}
// Check-Linux: {{    #1 0x.* in main .*use-after-free.cc:[45]}}

// Check-Darwin: {{    #0 0x.* in .*mz_free.*}}
// We override free() on Darwin, thus no malloc_zone_free
// Check-Darwin: {{    #1 0x.* in free}}
// Check-Darwin: {{    #2 0x.* in main .*use-after-free.cc:[45]}}

// Check-Common: {{previously allocated by thread T0 here:}}

// Check-Linux: {{    #0 0x.* in .*malloc}}
// Check-Linux: {{    #1 0x.* in main .*use-after-free.cc:3}}

// Check-Darwin: {{    #0 0x.* in .*mz_malloc.*}}
// Check-Darwin: {{    #1 0x.* in malloc_zone_malloc.*}}
// Check-Darwin: {{    #2 0x.* in malloc.*}}
// Check-Darwin: {{    #3 0x.* in main .*use-after-free.cc:3}}

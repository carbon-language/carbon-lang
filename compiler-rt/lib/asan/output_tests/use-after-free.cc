#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
}

// CHECK: {{.*ERROR: AddressSanitizer heap-use-after-free on address 0x.* at pc 0x.* bp 0x.* sp 0x.*}}
// CHECK: {{READ of size 1 at 0x.* thread T0}}
// CHECK: {{    #0 0x.* in main .*use-after-free.cc:5}}
// CHECK: {{0x.* is located 5 bytes inside of 10-byte region .0x.*,0x.*}}
// CHECK: {{freed by thread T0 here:}}
// CHECK: {{    #0 0x.* in free}}
// CHECK: {{    #1 0x.* in main .*use-after-free.cc:[45]}}
// CHECK: {{previously allocated by thread T0 here:}}
// CHECK: {{    #0 0x.* in malloc}}
// CHECK: {{    #1 0x.* in main .*use-after-free.cc:3}}

// Darwin: {{.*ERROR: AddressSanitizer heap-use-after-free on address 0x.* at pc 0x.* bp 0x.* sp 0x.*}}
// Darwin: {{READ of size 1 at 0x.* thread T0}}
// Darwin: {{    #0 0x.* in main .*use-after-free.cc:5}}
// Darwin: {{0x.* is located 5 bytes inside of 10-byte region .0x.*,0x.*}}
// Darwin: {{freed by thread T0 here:}}
// Darwin: {{    #0 0x.* in .*mz_free.*}}
// We override free() on Darwin, thus no malloc_zone_free
// Darwin: {{    #1 0x.* in free}}
// Darwin: {{    #2 0x.* in main .*use-after-free.cc:[45]}}
// Darwin: {{previously allocated by thread T0 here:}}
// Darwin: {{    #0 0x.* in .*mz_malloc.*}}
// Darwin: {{    #1 0x.* in malloc_zone_malloc.*}}
// Darwin: {{    #2 0x.* in malloc.*}}
// Darwin: {{    #3 0x.* in main .*use-after-free.cc:3}}

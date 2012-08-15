#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
}

// CHECK: heap-use-after-free
// CHECK: free
// CHECK: main{{.*}}use-after-free.c:4
// CHECK: malloc
// CHECK: main{{.*}}use-after-free.c:3
// CHECKSLEEP: Sleeping for 1 second
// CHECKSTRIP-NOT: #0 0x{{.*}} ({{[/].*}})

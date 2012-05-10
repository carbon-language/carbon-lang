#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

void *Thread1(void *x) {
  int *p = (int*)x;
  p[0] = 1;
  return NULL;
}

void *Thread2(void *x) {
  usleep(500*1000);
  char *p = (char*)x;
  p[2] = 1;
  return NULL;
}

int main() {
  int data = 42;
  fprintf(stderr, "ptr1=%p\n", &data);
  fprintf(stderr, "ptr2=%p\n", (char*)&data + 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, &data);
  pthread_create(&t[1], NULL, Thread2, &data);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
}

// CHECK: ptr1=[[PTR1:0x[0-9,a-f]+]]
// CHECK: ptr2=[[PTR2:0x[0-9,a-f]+]]
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 1 at [[PTR2]] by thread 2:
// CHECK:   Previous write of size 4 at [[PTR1]] by thread 1:


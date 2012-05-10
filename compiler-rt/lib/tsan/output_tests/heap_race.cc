#include <pthread.h>
#include <stdio.h>
#include <stddef.h>

void *Thread(void *a) {
  ((int*)a)[0]++;
  return NULL;
}

int main() {
  int *p = new int(42);
  pthread_t t;
  pthread_create(&t, NULL, Thread, p);
  p[0]++;
  pthread_join(t, NULL);
  delete p;
}

// CHECK: WARNING: ThreadSanitizer: data race

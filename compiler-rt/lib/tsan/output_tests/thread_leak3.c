#include <pthread.h>

void *Thread(void *x) {
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: thread leak


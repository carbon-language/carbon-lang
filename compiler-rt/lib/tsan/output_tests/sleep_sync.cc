#include <pthread.h>
#include <unistd.h>

int X = 0;

void *Thread(void *p) {
  X = 42;
  return 0;
}

void MySleep() {
  usleep(100*1000);
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  MySleep();  // Assume the thread has done the write.
  X = 43;
  pthread_join(t, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// ...
// CHECK:   As if synchronized via sleep:
// CHECK-NEXT:     #0 usleep
// CHECK-NEXT:     #1 MySleep
// CHECK-NEXT:     #2 main


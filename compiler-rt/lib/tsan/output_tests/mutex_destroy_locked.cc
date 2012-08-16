#include <pthread.h>
#include <unistd.h>

void *Thread(void *p) {
  pthread_mutex_lock((pthread_mutex_t*)p);
  return 0;
}

int main() {
  pthread_mutex_t m;
  pthread_mutex_init(&m, 0);
  pthread_t t;
  pthread_create(&t, 0, Thread, &m);
  usleep(1000*1000);
  pthread_mutex_destroy(&m);
  pthread_join(t, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: destroy of a locked mutex
// CHECK:     #0 pthread_mutex_destroy
// CHECK:     #1 main
// CHECK:   and:
// CHECK:     #0 pthread_mutex_lock
// CHECK:     #1 Thread
// CHECK:   Mutex {{.*}} created at:
// CHECK:     #0 pthread_mutex_init
// CHECK:     #1 main


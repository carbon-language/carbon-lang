// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <unistd.h>

pthread_rwlock_t rwlock;
int GLOB;

void *Thread1(void *p) {
  (void)p;
  pthread_rwlock_rdlock(&rwlock);
  // Write under reader lock.
  sleep(1);
  GLOB++;
  pthread_rwlock_unlock(&rwlock);
  return 0;
}

int main(int argc, char *argv[]) {
  pthread_rwlock_init(&rwlock, NULL);
  pthread_rwlock_rdlock(&rwlock);
  pthread_t t;
  pthread_create(&t, 0, Thread1, 0);
  volatile int x = GLOB;
  (void)x;
  pthread_rwlock_unlock(&rwlock);
  pthread_join(t, 0);
  pthread_rwlock_destroy(&rwlock);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 4 at {{.*}} by thread T1{{.*}}:
// CHECK:     #0 Thread1(void*) {{.*}}write_in_reader_lock.cc:13
// CHECK:   Previous read of size 4 at {{.*}} by main thread{{.*}}:
// CHECK:     #0 main {{.*}}write_in_reader_lock.cc:23

// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <errno.h>
#include <sys/mman.h>

void *SubWorker(void *arg) {
  (void)arg;
  const int kMmapSize =  65536;
  for (int i = 0; i < 500; i++) {
    int *ptr = (int*)mmap(0, kMmapSize, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    *ptr = 42;
    munmap(ptr, kMmapSize);
  }
  return 0;
}

void *Worker1(void *arg) {
  (void)arg;
  pthread_t th[4];
  for (int i = 0; i < 4; i++)
    pthread_create(&th[i], 0, SubWorker, 0);
  for (int i = 0; i < 4; i++)
    pthread_join(th[i], 0);
  return 0;
}

void *Worker(void *arg) {
  (void)arg;
  pthread_t th[4];
  for (int i = 0; i < 4; i++)
    pthread_create(&th[i], 0, Worker1, 0);
  for (int i = 0; i < 4; i++)
    pthread_join(th[i], 0);
  return 0;
}

int main() {
  pthread_t th[4];
  for (int i = 0; i < 4; i++)
    pthread_create(&th[i], 0, Worker, 0);
  for (int i = 0; i < 4; i++)
    pthread_join(th[i], 0);
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE

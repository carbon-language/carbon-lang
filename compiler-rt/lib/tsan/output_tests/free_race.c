#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>

int *mem;
pthread_mutex_t mtx;

void *Thread1(void *x) {
  pthread_mutex_lock(&mtx);
  free(mem);
  pthread_mutex_unlock(&mtx);
  return NULL;
}

void *Thread2(void *x) {
  usleep(1000000);
  pthread_mutex_lock(&mtx);
  mem[0] = 42;
  pthread_mutex_unlock(&mtx);
  return NULL;
}

int main() {
  mem = (int*)malloc(100);
  pthread_mutex_init(&mtx, 0);
  pthread_t t;
  pthread_create(&t, NULL, Thread1, NULL);
  Thread2(0);
  pthread_join(t, NULL);
  pthread_mutex_destroy(&mtx);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race

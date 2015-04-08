// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <semaphore.h>

// Test that signals can be delivered to blocked pthread_cond_wait.
// https://code.google.com/p/thread-sanitizer/issues/detail?id=91

int g_thread_run = 1;
pthread_mutex_t mutex;
pthread_cond_t cond;
sem_t sem;

void sig_handler(int sig) {
  (void)sig;
  write(1, "SIGNAL\n", sizeof("SIGNAL\n") - 1);
  sem_post(&sem);
}

void* my_thread(void* arg) {
  pthread_mutex_lock(&mutex);
  while (g_thread_run)
    pthread_cond_wait(&cond, &mutex);
  pthread_mutex_unlock(&mutex);
  return 0;
}

int main() {
  sem_init(&sem, 0, 0);
  signal(SIGUSR1, &sig_handler);
  pthread_t thr;
  pthread_create(&thr, 0, &my_thread, 0);
  // wait for thread to get inside pthread_cond_wait
  // (can't use barrier_wait for that)
  sleep(1);
  pthread_kill(thr, SIGUSR1);
  while (sem_wait(&sem) == -1 && errno == EINTR) {
  }
  pthread_mutex_lock(&mutex);
  g_thread_run = 0;
  pthread_cond_signal(&cond);
  pthread_mutex_unlock(&mutex);
  pthread_join(thr, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: SIGNAL
// CHECK: DONE

#include <pthread.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>

int bench_nthread;
int bench_niter;
int grow_clock_var;
pthread_barrier_t glow_clock_barrier;

void bench();  // defined by user
void start_thread_group(int nth, void(*f)(int tid));
void grow_clock_worker(int tid);

int main(int argc, char **argv) {
  bench_nthread = 2;
  if (argc > 1)
    bench_nthread = atoi(argv[1]);
  bench_niter = 100;
  if (argc > 2)
    bench_niter = atoi(argv[2]);

  // Grow thread's clock.
  int clock_size = 10;
  if (argc > 1)
    clock_size = 1000;
  pthread_barrier_init(&glow_clock_barrier, 0, clock_size);
  start_thread_group(clock_size, grow_clock_worker);
  pthread_barrier_destroy(&glow_clock_barrier);
  __atomic_load_n(&grow_clock_var, __ATOMIC_ACQUIRE);

  timespec tp0;
  clock_gettime(CLOCK_MONOTONIC, &tp0);
  bench();
  timespec tp1;
  clock_gettime(CLOCK_MONOTONIC, &tp1);
  unsigned long long t =
      (tp1.tv_sec * 1000000000ULL + tp1.tv_nsec) -
      (tp0.tv_sec * 1000000000ULL + tp0.tv_nsec);
  fprintf(stderr, "%llu ns/iter\n", t / bench_niter);
  fprintf(stderr, "DONE\n");
}

void start_thread_group(int nth, void(*f)(int tid)) {
  pthread_t *th = (pthread_t*)malloc(nth * sizeof(pthread_t));
  for (int i = 0; i < nth; i++)
    pthread_create(&th[i], 0, (void*(*)(void*))f, (void*)(long)i);
  for (int i = 0; i < nth; i++)
    pthread_join(th[i], 0);
}

void grow_clock_worker(int tid) {
  int res = pthread_barrier_wait(&glow_clock_barrier);
  if (res == PTHREAD_BARRIER_SERIAL_THREAD)
    __atomic_store_n(&grow_clock_var, 0, __ATOMIC_RELEASE);
}


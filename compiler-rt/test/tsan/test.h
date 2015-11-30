#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <stddef.h>
#include <sched.h>
#include <stdarg.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

// TSan-invisible barrier.
// Tests use it to establish necessary execution order in a way that does not
// interfere with tsan (does not establish synchronization between threads).
// 8 lsb is thread count, the remaining are count of entered threads.
typedef unsigned long long invisible_barrier_t;

void barrier_init(invisible_barrier_t *barrier, unsigned count) {
  if (count >= (1 << 8))
      exit(fprintf(stderr, "barrier_init: count is too large (%d)\n", count));
  *barrier = count;
}

void barrier_wait(invisible_barrier_t *barrier) {
  unsigned old = __atomic_fetch_add(barrier, 1 << 8, __ATOMIC_RELAXED);
  unsigned old_epoch = (old >> 8) / (old & 0xff);
  for (;;) {
    unsigned cur = __atomic_load_n(barrier, __ATOMIC_RELAXED);
    unsigned cur_epoch = (cur >> 8) / (cur & 0xff);
    if (cur_epoch != old_epoch)
      return;
    // Can't use usleep, because it leads to spurious "As if synchronized via
    // sleep" messages which fail some output tests.
    sched_yield();
  }
}

// Default instance of the barrier, but a test can declare more manually.
invisible_barrier_t barrier;

void print_address(const char *str, int n, ...) {
  fprintf(stderr, "%s", str);
  va_list ap;
  va_start(ap, n);
  while (n--) {
    void *p = va_arg(ap, void *);
#if defined(__x86_64__) || defined(__aarch64__)
    // On FreeBSD, the %p conversion specifier works as 0x%x and thus does not
    // match to the format used in the diagnotic message.
    fprintf(stderr, "0x%012lx ", (unsigned long) p);
#elif defined(__mips64)
    fprintf(stderr, "0x%010lx ", (unsigned long) p);
#endif
  }
  fprintf(stderr, "\n");
}

#ifdef __APPLE__
unsigned long long monotonic_clock_ns() {
  static mach_timebase_info_data_t timebase_info;
  if (timebase_info.denom == 0) mach_timebase_info(&timebase_info);
  return (mach_absolute_time() * timebase_info.numer) / timebase_info.denom;
}
#else
unsigned long long monotonic_clock_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (unsigned long long)t.tv_sec * 1000000000ull + t.tv_nsec;
}
#endif

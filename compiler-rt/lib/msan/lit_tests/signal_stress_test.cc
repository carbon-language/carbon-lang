// RUN: %clangxx_msan -O0 %s -o %t && %t

// Test that va_arg shadow from a signal handler does not leak outside.

#include <signal.h>
#include <stdarg.h>
#include <sanitizer/msan_interface.h>
#include <assert.h>
#include <sys/time.h>
#include <stdio.h>

const int kArgCnt = 20;
const int kSigCnt = 100;

volatile int z;

void f(bool poisoned, ...) {
  va_list vl;
  va_start(vl, poisoned);
  for (int i = 0; i < kArgCnt; ++i) {
    void *p = va_arg(vl, void *);
    if (poisoned)
      assert(__msan_test_shadow(&p, sizeof(p)) == 0);
    else 
      assert(__msan_test_shadow(&p, sizeof(p)) == -1);
  }
  va_end(vl);
}

int sigcnt;

void SignalHandler(int signo) {
  assert(signo == SIGPROF);
  void *p;
  void ** volatile q = &p;
  f(true,
      *q, *q, *q, *q, *q,
      *q, *q, *q, *q, *q,
      *q, *q, *q, *q, *q,
      *q, *q, *q, *q, *q,
      *q, *q, *q, *q, *q);
  ++sigcnt;
}

int main() {
  signal(SIGPROF, SignalHandler);

  itimerval itv;
  itv.it_interval.tv_sec = 0;
  itv.it_interval.tv_usec = 100;
  itv.it_value.tv_sec = 0;
  itv.it_value.tv_usec = 100;
  setitimer(ITIMER_PROF, &itv, NULL);

  do {
    f(false,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0);
  } while (sigcnt < kSigCnt);

  itv.it_interval.tv_sec = 0;
  itv.it_interval.tv_usec = 0;
  itv.it_value.tv_sec = 0;
  itv.it_value.tv_usec = 0;
  setitimer(ITIMER_PROF, &itv, NULL);

  signal(SIGPROF, SIG_DFL);
  return 0;
}

// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Test case for longjumping out of signal handler:
// https://code.google.com/p/thread-sanitizer/issues/detail?id=75

// Longjmp assembly has not been implemented for mips64 or aarch64 yet
// XFAIL: mips64
// XFAIL: aarch64

#include <setjmp.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>

sigjmp_buf fault_jmp;
volatile int fault_expected;

void sigfault_handler(int sig) {
  if (!fault_expected)
    abort();

  /* just return from sighandler to proper place */
  fault_expected = 0;
  siglongjmp(fault_jmp, 1);
}

#define MUST_FAULT(code) do { \
  fault_expected = 1; \
  if (!sigsetjmp(fault_jmp, 1)) { \
    code; /* should pagefault -> sihandler does longjmp */ \
    fprintf(stderr, "%s not faulted\n", #code); \
    abort(); \
  } else { \
    fprintf(stderr, "%s faulted ok\n", #code); \
  } \
} while (0)

int main() {
  struct sigaction act;
  act.sa_handler  = sigfault_handler;
  act.sa_flags    = 0;
  if (sigemptyset(&act.sa_mask)) {
    perror("sigemptyset");
    exit(1);
  }

  if (sigaction(SIGSEGV, &act, NULL)) {
    perror("sigaction");
    exit(1);
  }

  void *mem = mmap(0, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANON,
      -1, 0);

  MUST_FAULT(((volatile int *volatile)mem)[0] = 0);
  MUST_FAULT(((volatile int *volatile)mem)[1] = 1);
  MUST_FAULT(((volatile int *volatile)mem)[3] = 1);

  // Ensure that tsan does not think that we are
  // in a signal handler.
  void *volatile p = malloc(10);
  ((volatile int*)p)[1] = 1;
  free((void*)p);

  munmap(p, 4096);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: DONE

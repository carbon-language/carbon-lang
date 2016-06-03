// RUN: %clang_esan_wset -O0 %s -o %t 2>&1
// RUN: %env_esan_opts=verbosity=1 %run %t 2>&1 | FileCheck %s

#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

const int size = 0x1 << 25; // 523288 cache lines

int main(int argc, char **argv) {
  char *buf = (char *)mmap(0, size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  // Try to increase the probability that the sideline thread is
  // scheduled.  Unfortunately we can't do proper synchronization
  // without some form of annotation or something.
  sched_yield();
  // Do enough work to get at least 2 samples.
  for (int i = 0; i < size; ++i)
    buf[i] = i;
  munmap(buf, size);
  // CHECK:      {{.*}}EfficiencySanitizer: snapshot {{.*}}
  // CHECK-NEXT: {{.*}}EfficiencySanitizer: snapshot {{.*}}
  // CHECK: {{.*}} EfficiencySanitizer: the total working set size: 32 MB (5242{{[0-9][0-9]}} cache lines)
  return 0;
}

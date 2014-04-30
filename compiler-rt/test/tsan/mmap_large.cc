// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>

int main() {
  const size_t kLog2Size = 40;
  const uintptr_t kLocation = 0x40ULL << kLog2Size;
  void *p = mmap(
      reinterpret_cast<void*>(kLocation),
      1ULL << kLog2Size,
      PROT_READ|PROT_WRITE,
      MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,
      -1, 0);
  fprintf(stderr, "DONE %p\n", p);
  return p == MAP_FAILED;
}

// CHECK: DONE

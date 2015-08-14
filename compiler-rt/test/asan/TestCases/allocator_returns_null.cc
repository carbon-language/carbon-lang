// Test the behavior of malloc/calloc/realloc when the allocation size is huge.
// By default (allocator_may_return_null=0) the process should crash.
// With allocator_may_return_null=1 the allocator should return 0.
//
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: %env_asan_opts=allocator_may_return_null=1     %run %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mNULL
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t calloc 2>&1 | FileCheck %s --check-prefix=CHECK-cCRASH
// RUN: %env_asan_opts=allocator_may_return_null=1     %run %t calloc 2>&1 | FileCheck %s --check-prefix=CHECK-cNULL
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t calloc-overflow 2>&1 | FileCheck %s --check-prefix=CHECK-coCRASH
// RUN: %env_asan_opts=allocator_may_return_null=1     %run %t calloc-overflow 2>&1 | FileCheck %s --check-prefix=CHECK-coNULL
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t realloc 2>&1 | FileCheck %s --check-prefix=CHECK-rCRASH
// RUN: %env_asan_opts=allocator_may_return_null=1     %run %t realloc 2>&1 | FileCheck %s --check-prefix=CHECK-rNULL
// RUN: %env_asan_opts=allocator_may_return_null=0 not %run %t realloc-after-malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mrCRASH
// RUN: %env_asan_opts=allocator_may_return_null=1     %run %t realloc-after-malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mrNULL

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <limits>
int main(int argc, char **argv) {
  // Disable stderr buffering. Needed on Windows.
  setvbuf(stderr, NULL, _IONBF, 0);

  volatile size_t size = std::numeric_limits<size_t>::max() - 10000;
  assert(argc == 2);
  void *x = 0;
  if (!strcmp(argv[1], "malloc")) {
    fprintf(stderr, "malloc:\n");
    x = malloc(size);
  }
  if (!strcmp(argv[1], "calloc")) {
    fprintf(stderr, "calloc:\n");
    x = calloc(size / 4, 4);
  }

  if (!strcmp(argv[1], "calloc-overflow")) {
    fprintf(stderr, "calloc-overflow:\n");
    volatile size_t kMaxSizeT = std::numeric_limits<size_t>::max();
    size_t kArraySize = 4096;
    volatile size_t kArraySize2 = kMaxSizeT / kArraySize + 10;
    x = calloc(kArraySize, kArraySize2);
  }

  if (!strcmp(argv[1], "realloc")) {
    fprintf(stderr, "realloc:\n");
    x = realloc(0, size);
  }
  if (!strcmp(argv[1], "realloc-after-malloc")) {
    fprintf(stderr, "realloc-after-malloc:\n");
    char *t = (char*)malloc(100);
    *t = 42;
    x = realloc(t, size);
    assert(*t == 42);
    free(t);
  }
  // The NULL pointer is printed differently on different systems, while (long)0
  // is always the same.
  fprintf(stderr, "x: %lx\n", (long)x);
  free(x);
  return x != 0;
}
// CHECK-mCRASH: malloc:
// CHECK-mCRASH: AddressSanitizer's allocator is terminating the process
// CHECK-cCRASH: calloc:
// CHECK-cCRASH: AddressSanitizer's allocator is terminating the process
// CHECK-coCRASH: calloc-overflow:
// CHECK-coCRASH: AddressSanitizer's allocator is terminating the process
// CHECK-rCRASH: realloc:
// CHECK-rCRASH: AddressSanitizer's allocator is terminating the process
// CHECK-mrCRASH: realloc-after-malloc:
// CHECK-mrCRASH: AddressSanitizer's allocator is terminating the process

// CHECK-mNULL: malloc:
// CHECK-mNULL: x: 0
// CHECK-cNULL: calloc:
// CHECK-cNULL: x: 0
// CHECK-coNULL: calloc-overflow:
// CHECK-coNULL: x: 0
// CHECK-rNULL: realloc:
// CHECK-rNULL: x: 0
// CHECK-mrNULL: realloc-after-malloc:
// CHECK-mrNULL: x: 0

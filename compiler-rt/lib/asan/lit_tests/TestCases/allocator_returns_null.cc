// Test the behavior of malloc/calloc/realloc when the allocation size is huge.
// By default (allocator_may_return_null=0) the process shoudl crash.
// With allocator_may_return_null=1 the allocator should return 0.
//
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: ASAN_OPTIONS=allocator_may_return_null=0 not %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mCRASH
// RUN: ASAN_OPTIONS=allocator_may_return_null=1     %t malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mNULL
// RUN: ASAN_OPTIONS=allocator_may_return_null=0 not %t calloc 2>&1 | FileCheck %s --check-prefix=CHECK-cCRASH
// RUN: ASAN_OPTIONS=allocator_may_return_null=1     %t calloc 2>&1 | FileCheck %s --check-prefix=CHECK-cNULL
// RUN: ASAN_OPTIONS=allocator_may_return_null=0 not %t realloc 2>&1 | FileCheck %s --check-prefix=CHECK-rCRASH
// RUN: ASAN_OPTIONS=allocator_may_return_null=1     %t realloc 2>&1 | FileCheck %s --check-prefix=CHECK-rNULL
// RUN: ASAN_OPTIONS=allocator_may_return_null=0 not %t realloc-after-malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mrCRASH
// RUN: ASAN_OPTIONS=allocator_may_return_null=1     %t realloc-after-malloc 2>&1 | FileCheck %s --check-prefix=CHECK-mrNULL

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <limits>
int main(int argc, char **argv) {
  volatile size_t size = std::numeric_limits<size_t>::max() - 10000;
  assert(argc == 2);
  char *x = 0;
  if (!strcmp(argv[1], "malloc")) {
    fprintf(stderr, "malloc:\n");
    x = (char*)malloc(size);
  }
  if (!strcmp(argv[1], "calloc")) {
    fprintf(stderr, "calloc:\n");
    x = (char*)calloc(size / 4, 4);
  }

  if (!strcmp(argv[1], "realloc")) {
    fprintf(stderr, "realloc:\n");
    x = (char*)realloc(0, size);
  }
  if (!strcmp(argv[1], "realloc-after-malloc")) {
    fprintf(stderr, "realloc-after-malloc:\n");
    char *t = (char*)malloc(100);
    *t = 42;
    x = (char*)realloc(t, size);
    assert(*t == 42);
  }
  fprintf(stderr, "x: %p\n", x);
  return x != 0;
}
// CHECK-mCRASH: malloc:
// CHECK-mCRASH: AddressSanitizer's allocator is terminating the process
// CHECK-cCRASH: calloc:
// CHECK-cCRASH: AddressSanitizer's allocator is terminating the process
// CHECK-rCRASH: realloc:
// CHECK-rCRASH: AddressSanitizer's allocator is terminating the process
// CHECK-mrCRASH: realloc-after-malloc:
// CHECK-mrCRASH: AddressSanitizer's allocator is terminating the process

// CHECK-mNULL: malloc:
// CHECK-mNULL: x: (nil)
// CHECK-cNULL: calloc:
// CHECK-cNULL: x: (nil)
// CHECK-rNULL: realloc:
// CHECK-rNULL: x: (nil)
// CHECK-mrNULL: realloc-after-malloc:
// CHECK-mrNULL: x: (nil)

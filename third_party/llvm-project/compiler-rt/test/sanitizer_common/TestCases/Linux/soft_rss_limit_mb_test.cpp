// Check soft_rss_limit_mb. Not all sanitizers implement it yet.
// RUN: %clangxx -O2 %s -o %t
//
// Run with limit should fail:
// RUN: %env_tool_opts=soft_rss_limit_mb=220:quarantine_size=1:allocator_may_return_null=1     %run %t 2>&1 | FileCheck %s -check-prefix=CHECK_MAY_RETURN_1
// RUN: %env_tool_opts=soft_rss_limit_mb=220:quarantine_size=1:allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s -check-prefix=CHECK_MAY_RETURN_0

// This run uses getrusage. We can only test getrusage when allocator_may_return_null=0
// because getrusage gives us max-rss, not current-rss.
// RUN: %env_tool_opts=soft_rss_limit_mb=220:quarantine_size=1:allocator_may_return_null=0:can_use_proc_maps_statm=0 not %run %t 2>&1 | FileCheck %s -check-prefix=CHECK_MAY_RETURN_0
// REQUIRES: stable-runtime

// FIXME: make it work for other sanitizers.
// XFAIL: lsan
// XFAIL: tsan
// XFAIL: msan
// XFAIL: ubsan

// https://github.com/google/sanitizers/issues/981
// UNSUPPORTED: android-26

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const int kMaxNumAllocs = 1 << 9;
static const int kAllocSize = 1 << 20;  // Large enough to go via mmap.

static char *allocs[kMaxNumAllocs];

int main() {
  int num_allocs = kMaxNumAllocs / 4;
  for (int i = 0; i < 3; i++, num_allocs *= 2) {
    fprintf(stderr, "[%d] allocating %d times\n", i, num_allocs);
    int zero_results = 0;
    for (int j = 0; j < num_allocs; j++) {
      if ((j % (num_allocs / 8)) == 0) {
        usleep(100000);
        fprintf(stderr, "  [%d]\n", j);
      }
      allocs[j] = (char*)malloc(kAllocSize);
      if (allocs[j])
        memset(allocs[j], -1, kAllocSize);
      else
        zero_results++;
    }
    if (zero_results)
      fprintf(stderr, "Some of the malloc calls returned null: %d\n",
              zero_results);
    if (zero_results != num_allocs)
      fprintf(stderr, "Some of the malloc calls returned non-null: %d\n",
              num_allocs - zero_results);
    for (int j = 0; j < num_allocs; j++) {
      free(allocs[j]);
    }
  }
}

// CHECK_MAY_RETURN_1: allocating 128 times
// CHECK_MAY_RETURN_1: Some of the malloc calls returned non-null: 128
// CHECK_MAY_RETURN_1: allocating 256 times
// CHECK_MAY_RETURN_1: Some of the malloc calls returned null:
// CHECK_MAY_RETURN_1: Some of the malloc calls returned non-null:
// CHECK_MAY_RETURN_1: allocating 512 times
// CHECK_MAY_RETURN_1: Some of the malloc calls returned null:
// CHECK_MAY_RETURN_1: Some of the malloc calls returned non-null:

// CHECK_MAY_RETURN_0: allocating 128 times
// CHECK_MAY_RETURN_0: Some of the malloc calls returned non-null: 128
// CHECK_MAY_RETURN_0: allocating 256 times
// CHECK_MAY_RETURN_0: {{SUMMARY: .*Sanitizer: rss-limit-exceeded}}

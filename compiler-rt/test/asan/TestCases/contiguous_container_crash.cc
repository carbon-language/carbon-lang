// RUN: %clangxx_asan -O %s -o %t
// RUN: not %run %t crash 2>&1 | FileCheck --check-prefix=CHECK-CRASH %s
// RUN: not %run %t bad-bounds 2>&1 | FileCheck --check-prefix=CHECK-BAD %s
// RUN: ASAN_OPTIONS=detect_container_overflow=0 %run %t crash
//
// Test crash due to __sanitizer_annotate_contiguous_container.

#include <assert.h>
#include <string.h>

extern "C" {
void __sanitizer_annotate_contiguous_container(const void *beg, const void *end,
                                               const void *old_mid,
                                               const void *new_mid);
}  // extern "C"

static volatile int one = 1;

int TestCrash() {
  long t[100];
  t[60] = 0;
  __sanitizer_annotate_contiguous_container(&t[0], &t[0] + 100, &t[0] + 100,
                                            &t[0] + 50);
  return (int)t[60 * one];  // Touches the poisoned memory.
}

void BadBounds() {
  long t[100];
  __sanitizer_annotate_contiguous_container(&t[0], &t[0] + 100, &t[0] + 101,
                                            &t[0] + 50);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1], "crash"))
    return TestCrash();
  else if (!strcmp(argv[1], "bad-bounds"))
    BadBounds();
}
// CHECK-CRASH: AddressSanitizer: container-overflow
// CHECK-BAD: ERROR: AddressSanitizer: bad parameters to __sanitizer_annotate_contiguous_container

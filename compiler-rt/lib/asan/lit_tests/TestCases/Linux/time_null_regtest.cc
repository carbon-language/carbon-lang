// RUN: %clangxx_asan -O0 %s -fsanitize-address-zero-base-shadow -pie -o %t && %t 2>&1 | FileCheck %s

// Zero-base shadow only works on x86_64 and i386.
// REQUIRES: x86_64-supported-target

// A regression test for time(NULL), which caused ASan to crash in the
// zero-based shadow mode on Linux.
// FIXME: this test does not work on Darwin, because the code pages of the
// executable interleave with the zero-based shadow.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  time_t t = time(NULL);
  fprintf(stderr, "Time: %s\n", ctime(&t));  // NOLINT
  // CHECK: {{Time: .* .* .*}}
  return 0;
}

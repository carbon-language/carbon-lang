// RUN: %clangxx_asan -mllvm -asan-coverage=1 %s -o %t
// RUN: rm -rf %T/coverage-maybe-open-file
// RUN: mkdir -p %T/coverage-maybe-open-file && cd %T/coverage-maybe-open-file
// RUN: ASAN_OPTIONS=coverage=1 %run %t | FileCheck %s --check-prefix=CHECK-success
// RUN: ASAN_OPTIONS=coverage=0 %run %t | FileCheck %s --check-prefix=CHECK-fail
// RUN: [ "$(cat test.sancov.packed)" == "test" ]
// RUN: cd .. && rm -rf %T/coverage-maybe-open-file

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <sanitizer/common_interface_defs.h>

int main(int argc, char **argv) {
  int fd = __sanitizer_maybe_open_cov_file("test");
  if (fd > 0) {
    printf("SUCCESS\n");
    const char s[] = "test\n";
    write(fd, s, strlen(s));
    close(fd);
  } else {
    printf("FAIL\n");
  }
}

// CHECK-success: SUCCESS
// CHECK-fail: FAIL

// Make sure ASan doesn't hang in an exec loop if DYLD_INSERT_LIBRARIES is set.
// This is a regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=159

// RUN: %clangxx_asan %s -o %t
// RUN: %clangxx %p/../SharedLibs/darwin-dummy-shared-lib-so.cc \
// RUN:     -dynamiclib -o darwin-dummy-shared-lib-so.dylib

// FIXME: the following command line may hang in the case of a regression.
// RUN: DYLD_INSERT_LIBRARIES=darwin-dummy-shared-lib-so.dylib \
// RUN:     %run %t 2>&1 | FileCheck %s || exit 1
#include <stdio.h>
#include <stdlib.h>

int main() {
  const char kEnvName[] = "DYLD_INSERT_LIBRARIES";
  printf("%s=%s\n", kEnvName, getenv(kEnvName));
  // CHECK: {{DYLD_INSERT_LIBRARIES=.*darwin-dummy-shared-lib-so.dylib.*}}
  return 0;
}

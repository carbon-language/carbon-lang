// Make sure ASan removes the runtime library from DYLD_INSERT_LIBRARIES before
// executing other programs.

// RUN: %clangxx_asan -m64 %s -o %t
// RUN: %clangxx -m64 %p/../SharedLibs/darwin-dummy-shared-lib-so.cc \
// RUN:     -dynamiclib -o darwin-dummy-shared-lib-so.dylib

// Make sure DYLD_INSERT_LIBRARIES doesn't contain the runtime library before
// execl().

// RUN: %t >/dev/null 2>&1
// RUN: DYLD_INSERT_LIBRARIES=darwin-dummy-shared-lib-so.dylib \
// RUN:     %t 2>&1 | FileCheck %s || exit 1
#include <unistd.h>
int main() {
  execl("/bin/bash", "/bin/bash", "-c",
        "echo DYLD_INSERT_LIBRARIES=$DYLD_INSERT_LIBRARIES", NULL);
  // CHECK:  {{DYLD_INSERT_LIBRARIES=.*darwin-dummy-shared-lib-so.dylib.*}}
  return 0;
}

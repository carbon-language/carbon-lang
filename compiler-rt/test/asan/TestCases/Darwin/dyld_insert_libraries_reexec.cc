// When DYLD-inserting the ASan dylib from a different location than the
// original, make sure we don't try to reexec.

// RUN: mkdir -p %T/dyld_insert_libraries_reexec
// RUN: cp `%clang_asan %s -fsanitize=address -### 2>&1 \
// RUN:   | grep "libclang_rt.asan_osx_dynamic.dylib" \
// RUN:   | sed -e 's/.*"\(.*libclang_rt.asan_osx_dynamic.dylib\)".*/\1/'` \
// RUN:   %T/dyld_insert_libraries_reexec/libclang_rt.asan_osx_dynamic.dylib
// RUN: %clangxx_asan %s -o %T/dyld_insert_libraries_reexec/a.out
// RUN: DYLD_INSERT_LIBRARIES=@executable_path/libclang_rt.asan_osx_dynamic.dylib \
// RUN:   ASAN_OPTIONS=verbosity=1 %run %T/dyld_insert_libraries_reexec/a.out 2>&1 \
// RUN:   | FileCheck %s
// RUN: ASAN_OPTIONS=verbosity=1 %run %T/dyld_insert_libraries_reexec/a.out 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOINSERT %s

#include <stdio.h>

int main() {
  printf("Passed\n");
  return 0;
}

// CHECK-NOINSERT: exec()-ing the program with
// CHECK-NOINSERT: DYLD_INSERT_LIBRARIES
// CHECK-NOINSERT: to enable ASan wrappers.
// CHECK-NOINSERT: Passed

// CHECK-NOT: exec()-ing the program with
// CHECK-NOT: DYLD_INSERT_LIBRARIES
// CHECK-NOT: to enable ASan wrappers.
// CHECK: Passed

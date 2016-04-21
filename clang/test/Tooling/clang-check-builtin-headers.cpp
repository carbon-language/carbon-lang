// RUN: rm -rf %t
// RUN: mkdir %t
// Add a path that doesn't exist as argv[0] for the compile command line:
// RUN: echo '[{"directory":".","command":"/random/tool -c %t/test.cpp","file":"%t/test.cpp"}]' | sed -e 's/\\/\//g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: not clang-check -p "%t" "%t/test.cpp" 2>&1|FileCheck %s
// RUN: not clang-check -p "%t" "%t/test.cpp" -extra-arg=-resource-dir=foo 2>&1|FileCheck %s -check-prefix=CHECK-NOHDR
// FIXME: Make the above easier.

#include <stddef.h>

// CHECK: C++ requires
// CHECK-NOHDR: fatal error: 'stddef.h' file not
invalid;

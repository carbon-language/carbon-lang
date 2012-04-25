// RUN: rm -rf %t
// RUN: mkdir %t
// Add a path that doesn't exist as argv[0] for the compile command line:
// RUN: echo '[{"directory":".","command":"/random/tool -c %t/test.cpp","file":"%t/test.cpp"}]' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-check "%t" "%t/test.cpp" 2>&1|FileCheck %s
// FIXME: Make the above easier.

#include <stddef.h>

// CHECK: C++ requires
invalid;

// FIXME: JSON doesn't like path separator '\', on Win32 hosts.
// FIXME: clang-check doesn't like gcc driver on cygming.
// XFAIL: cygwin,mingw32,win32

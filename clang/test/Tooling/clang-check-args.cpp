// RUN: clang-check . "%s" -- -c 2>&1 | FileCheck %s

// CHECK: C++ requires
invalid;

// FIXME: JSON doesn't like path separator '\', on Win32 hosts.
// FIXME: clang-check doesn't like gcc driver on cygming.
// XFAIL: cygwin,mingw32,win32

// RUN: clang-check . "%s" -- -no-integrated-as -c 2>&1 | FileCheck %s

// CHECK: C++ requires
invalid;

// FIXME: clang-check doesn't like gcc driver on cygming.
// XFAIL: cygwin,mingw32,win32


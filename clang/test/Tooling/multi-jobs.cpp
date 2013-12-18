// RUN: not clang-check "%s" -- -no-integrated-as -c 2>&1 | FileCheck %s
// RUN: not clang-check "%s" -- -target x86_64-win32 -no-integrated-as -c 2>&1 | FileCheck %s

// CHECK: C++ requires
invalid;

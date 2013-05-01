// RUN: clang-check "%s" -- -no-integrated-as -c 2>&1 | FileCheck %s

// CHECK: C++ requires
invalid;

// RUN: clang-check . "%s" -- -c 2>&1 | FileCheck %s

// CHECK: C++ requires
invalid;

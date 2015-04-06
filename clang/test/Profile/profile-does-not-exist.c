// RUN: not %clang_cc1 -emit-llvm %s -fprofile-instr-use=%t.nonexistent.profdata 2>&1 | FileCheck %s

// CHECK: error: Could not read profile:
// CHECK-NOT: Assertion failed

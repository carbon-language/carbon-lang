// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --share-cpp-ast %s
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/share_ast.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/share_ast.carbon

// --- a.h
#ifndef A_H
#define A_H
inline int GetA() { return 42; }
#endif

// --- b.h
#ifndef B_H
#define B_H
#include "a.h"
inline int GetB() { return 100; }
#endif

// --- a.carbon
library "[[@TEST_NAME]]";
import Cpp library "a.h";

fn CallA() -> i32 {
  return Cpp.GetA();
}

// --- fail_a_does_not_leak.carbon
library "[[@TEST_NAME]]";
import Cpp;

fn CallLeakedA() -> i32 {
  // This should not be found as we have not imported "a.h".
  // CHECK:STDERR: fail_a_does_not_leak.carbon:[[@LINE+4]]:10: error: member name `GetA` not found in `Cpp`
  // CHECK:STDERR:   return Cpp.GetA();
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR:
  return Cpp.GetA();
}

// --- a_again.carbon
library "[[@TEST_NAME]]";
import Cpp library "a.h";

fn CallAAgain() -> i32 {
  // This should be accepted as we have imported "a.h", even though we will just reuse the version from `a.carbon`.
  return Cpp.GetA();
}

// --- b.carbon
library "[[@TEST_NAME]]";
import Cpp library "b.h";

fn CallAIndirect() -> i32 {
  return Cpp.GetA();
}

fn CallB() -> i32 {
  return Cpp.GetB();
}

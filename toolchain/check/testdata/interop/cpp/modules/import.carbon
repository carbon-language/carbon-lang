// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --clang-arg=-fmodules --clang-arg=-fmodules-cache-path=%t.cache
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/modules/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/modules/import.carbon

// --- module.modulemap

module A {
  header "a.h"
  export *
}

module B {
  header "b.h"
  export *
}

// --- a.h

struct A {};

A *_Nonnull makeA();

// --- b.h

#include "a.h"

struct B {};

B *_Nonnull makeB(A *_Nonnull);

// --- fail_todo_use_a_b.carbon

library "[[@TEST_NAME]]";

import Cpp library "a.h";
import Cpp library "b.h";

// CHECK:STDERR: fail_todo_use_a_b.carbon:[[@LINE+4]]:12: error: member name `B` not found in `Cpp` [MemberNameNotFoundInInstScope]
// CHECK:STDERR: fn Go() -> Cpp.B* {
// CHECK:STDERR:            ^~~~~
// CHECK:STDERR:
fn Go() -> Cpp.B* {
  // CHECK:STDERR: fail_todo_use_a_b.carbon:[[@LINE+8]]:10: error: member name `makeB` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return Cpp.makeB(Cpp.makeA());
  // CHECK:STDERR:          ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use_a_b.carbon:[[@LINE+4]]:20: error: member name `makeA` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return Cpp.makeB(Cpp.makeA());
  // CHECK:STDERR:                    ^~~~~~~~~
  // CHECK:STDERR:
  return Cpp.makeB(Cpp.makeA());
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/import.carbon

// --- header.h

auto foo_short(short x) -> void;
auto foo_int(int x) -> void;

// --- api.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

alias FooShort = Cpp.foo_short;
alias FooInt = Cpp.foo_int;

// --- import_api.carbon

library "[[@TEST_NAME]]";

import library "api";

// TODO: Fix this test as a follow-up of https://github.com/carbon-language/carbon-lang/pull/5891.
fn F() {
  //@dump-sem-ir-begin
  // FooShort(8 as i16);
  // FooInt(9);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

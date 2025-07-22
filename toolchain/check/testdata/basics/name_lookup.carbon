// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/name_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/name_lookup.carbon

// --- fail_not_found.carbon

library "[[@TEST_NAME]]";

fn Main() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_not_found.carbon:[[@LINE+4]]:3: error: name `x` not found [NameNotFound]
  // CHECK:STDERR:   x;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  x;
  //@dump-sem-ir-end
}

// --- fail_qualififier_unsupported.carbon

library "[[@TEST_NAME]]";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_qualififier_unsupported.carbon:[[@LINE+4]]:3: error: type `bool` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   true.b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  true.b;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_not_found.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_qualififier_unsupported.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

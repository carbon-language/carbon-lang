// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/package_expr/not_found.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/package_expr/not_found.carbon

// --- fail_direct.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_direct.carbon:[[@LINE+4]]:23: error: member name `x` not found in `Main` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   var unused y: i32 = package.x;
  // CHECK:STDERR:                       ^~~~~~~~~
  // CHECK:STDERR:
  var unused y: i32 = package.x;
}

// --- fail_nested.carbon
library "[[@TEST_NAME]]";

private interface Private {}

fn F() {
  // CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:3: error: member name `DoesNotExist` not found in `Private` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   package.Private.DoesNotExist;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  package.Private.DoesNotExist;
}

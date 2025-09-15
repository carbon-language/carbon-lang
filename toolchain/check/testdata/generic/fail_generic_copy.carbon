// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/fail_generic_copy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/fail_generic_copy.carbon

// --- fail_return.carbon

library "[[@TEST_NAME]]";

fn F[T:! type](x: T) -> T {
  // CHECK:STDERR: fail_return.carbon:[[@LINE+7]]:10: error: cannot copy value of type `T` [CopyOfUncopyableType]
  // CHECK:STDERR:   return x;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_return.carbon:[[@LINE+4]]:10: note: type `T` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return x;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return x;
}

// --- fail_var.carbon

library "[[@TEST_NAME]]";

fn F[T:! type](x: T) {
  // CHECK:STDERR: fail_var.carbon:[[@LINE+7]]:14: error: cannot copy value of type `T` [CopyOfUncopyableType]
  // CHECK:STDERR:   var y: T = x;
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_var.carbon:[[@LINE+4]]:14: note: type `T` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var y: T = x;
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  var y: T = x;
}

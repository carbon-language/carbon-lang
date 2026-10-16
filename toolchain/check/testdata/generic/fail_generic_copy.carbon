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

fn F[T: type](x: T) -> T {
  // CHECK:STDERR: fail_return.carbon:[[@LINE+4]]:10: error: cannot copy value of type `T` [CopyOfUncopyableType]
  // CHECK:STDERR: fail_return.carbon:[[@LINE+3]]:10: note: type `T` does not implement interface `Core.Copy` [MissingImplForType]
  // CHECK:STDERR: fail_return.carbon:[[@LINE-3]]:24: note: return type of function is `T` [ReturnTypeHereNote]
  // CHECK:STDERR:
  return x;
}

// --- fail_var.carbon

library "[[@TEST_NAME]]";

fn F[T: Core.Destroy](x: T) {
  // CHECK:STDERR: fail_var.carbon:[[@LINE+4]]:21: error: cannot copy value of type `T` [CopyOfUncopyableType]
  // CHECK:STDERR: fail_var.carbon:[[@LINE+3]]:21: note: type `T` does not implement interface `Core.Copy` [MissingImplForType]
  // CHECK:STDERR: fail_var.carbon:[[@LINE+2]]:3-17: note: initializing this binding [InBindingInit]
  // CHECK:STDERR:
  var unused y: T = x;
}

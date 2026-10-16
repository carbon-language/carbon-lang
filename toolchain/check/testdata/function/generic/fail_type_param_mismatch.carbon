// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/fail_type_param_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/fail_type_param_mismatch.carbon

fn F(generic T: type, generic U: type) {
  var p: T*;
  // CHECK:STDERR: fail_type_param_mismatch.carbon:[[@LINE+4]]:21-22: error: cannot implicitly convert expression of type `T` to `U` [ConversionFailure]
  // CHECK:STDERR: fail_type_param_mismatch.carbon:[[@LINE+3]]:21-22: note: type `T` does not implement interface `Core.ImplicitAs(U)` [MissingImplForType]
  // CHECK:STDERR: fail_type_param_mismatch.carbon:[[@LINE+2]]:14-17: note: initializing this binding [InBindingInit]
  // CHECK:STDERR:
  let unused n: U = *p;
}

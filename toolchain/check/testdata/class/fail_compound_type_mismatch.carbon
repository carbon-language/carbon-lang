// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_compound_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_compound_type_mismatch.carbon

class A {
  var a: i32;
}

class B {
  var b: i32;
}

fn AccessBInA(a: A) -> i32 {
  // CHECK:STDERR: fail_compound_type_mismatch.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `A` to `B` [ConversionFailure]
  // CHECK:STDERR:   return a.(B.b);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_compound_type_mismatch.carbon:[[@LINE+4]]:10: note: type `A` does not implement interface `Core.ImplicitAs(B)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a.(B.b);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  return a.(B.b);
}

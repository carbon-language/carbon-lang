// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_convert_to_invalid.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_convert_to_invalid.carbon

class C {
  // CHECK:STDERR: fail_convert_to_invalid.carbon:[[@LINE+4]]:10: error: name `NoSuchType` not found [NameNotFound]
  // CHECK:STDERR:   var a: NoSuchType;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  var a: NoSuchType;
}

fn Make() -> C {
  return {.a = ()};
}

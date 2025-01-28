// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_template.carbon

interface I {
  // CHECK:STDERR: fail_assoc_const_template.carbon:[[@LINE+4]]:7: error: semantics TODO: `HandleTemplate` [SemanticsTodo]
  // CHECK:STDERR:   let template T:! type;
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let template T:! type;
}

// CHECK:STDOUT: --- fail_assoc_const_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = <unexpected>.inst15
// CHECK:STDOUT:   witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: assoc_const @T T:! type;
// CHECK:STDOUT:

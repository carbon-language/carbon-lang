// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_assoc_const_template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_assoc_const_template.carbon

interface I {
  // CHECK:STDERR: fail_assoc_const_template.carbon:[[@LINE+8]]:7: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:   let template T:! type;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_assoc_const_template.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   let template T:! type;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let template T:! type;
}

// CHECK:STDOUT: --- fail_assoc_const_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = <unexpected>.inst17
// CHECK:STDOUT:   witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:

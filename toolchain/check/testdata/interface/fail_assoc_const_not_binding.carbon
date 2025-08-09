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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_assoc_const_not_binding.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_assoc_const_not_binding.carbon

// --- fail_tuple_pattern.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_tuple_pattern.carbon:[[@LINE+8]]:7: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:   let (T:! type, U:! type);
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_tuple_pattern.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   let (T:! type, U:! type);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let (T:! type, U:! type);
}

// --- fail_tuple_pattern_with_default.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_tuple_pattern_with_default.carbon:[[@LINE+8]]:15: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:   default let (T:! type, U:! type) = ({}, {});
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_tuple_pattern_with_default.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   default let (T:! type, U:! type) = ({}, {});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  default let (T:! type, U:! type) = ({}, {});
}

// --- fail_var_pattern.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_var_pattern.carbon:[[@LINE+8]]:7: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:   let var T:! type;
  // CHECK:STDERR:       ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_var_pattern.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   let var T:! type;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let var T:! type;
}

// --- fail_var_pattern_with_default.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_var_pattern_with_default.carbon:[[@LINE+8]]:15: error: expected identifier in associated constant declaration [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:   default let var T:! type = {};
  // CHECK:STDERR:               ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_var_pattern_with_default.carbon:[[@LINE+4]]:3: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   default let var T:! type = {};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  default let var T:! type = {};
}

// CHECK:STDOUT: --- fail_tuple_pattern.carbon
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
// CHECK:STDOUT: --- fail_tuple_pattern_with_default.carbon
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
// CHECK:STDOUT: --- fail_var_pattern.carbon
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
// CHECK:STDOUT: --- fail_var_pattern_with_default.carbon
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

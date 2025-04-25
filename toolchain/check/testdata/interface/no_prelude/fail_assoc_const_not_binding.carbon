// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_not_binding.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_not_binding.carbon

// --- fail_tuple_pattern.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_tuple_pattern.carbon:[[@LINE+4]]:7: error: pattern in associated constant declaration must be a single `:!` binding [ExpectedSymbolicBindingInAssociatedConstant]
  // CHECK:STDERR:   let (T:! type, U:! type);
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let (T:! type, U:! type);
}

// --- fail_tuple_pattern_with_default.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_tuple_pattern_with_default.carbon:[[@LINE+4]]:15: error: pattern in associated constant declaration must be a single `:!` binding [ExpectedSymbolicBindingInAssociatedConstant]
  // CHECK:STDERR:   default let (T:! type, U:! type) = ({}, {});
  // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  default let (T:! type, U:! type) = ({}, {});
}

// --- fail_var_pattern.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_var_pattern.carbon:[[@LINE+8]]:11: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   let var T:! type;
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_var_pattern.carbon:[[@LINE+4]]:19: error: `var` pattern cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
  // CHECK:STDERR:   let var T:! type;
  // CHECK:STDERR:                   ^
  // CHECK:STDERR:
  let var T:! type;
}

// --- fail_var_pattern_with_default.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_var_pattern_with_default.carbon:[[@LINE+8]]:19: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   default let var T:! type = {};
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_var_pattern_with_default.carbon:[[@LINE+4]]:28: error: `var` pattern cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
  // CHECK:STDERR:   default let var T:! type = {};
  // CHECK:STDERR:                            ^
  // CHECK:STDERR:
  default let var T:! type = {};
}

// CHECK:STDOUT: --- fail_tuple_pattern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: assoc_const @T T:! type;
// CHECK:STDOUT:
// CHECK:STDOUT: assoc_const @U U:! type;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_tuple_pattern_with_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: assoc_const @T T:! type;
// CHECK:STDOUT:
// CHECK:STDOUT: assoc_const @U U:! type;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_var_pattern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
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
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = <unexpected>.inst17
// CHECK:STDOUT:   witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:

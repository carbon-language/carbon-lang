// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_todo_no_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_todo_no_params.carbon

// --- fail_no_body.carbon

library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_no_body.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn A;
// CHECK:STDERR: ^~~~~
// CHECK:STDERR:
fn A;

// --- fail_todo_brace_body.carbon

library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_brace_body.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn A {
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
fn A {
}

// --- fail_todo_return_type.carbon

library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_return_type.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn A -> ();
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
fn A -> ();

// --- fail_todo_arrow_body.carbon

// TODO: We don't have parsing support for this yet.
library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_arrow_body.carbon:[[@LINE+8]]:6: error: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR: fn A => 0;
// CHECK:STDERR:      ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_arrow_body.carbon:[[@LINE+4]]:1: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: fn A => 0;
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR:
fn A => 0;

// --- fail_invalid_file_generic_regression_test.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_invalid_file_generic_regression_test.carbon:[[@LINE+8]]:5: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: var x:! () = ();
// CHECK:STDERR:     ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_invalid_file_generic_regression_test.carbon:[[@LINE+4]]:12: error: `var` pattern cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
// CHECK:STDERR: var x:! () = ();
// CHECK:STDERR:            ^
// CHECK:STDERR:
var x:! () = ();

fn A {
  A();
}

// CHECK:STDOUT: --- fail_no_body.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_brace_body.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_10.2: type = converted %.loc7_10.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_arrow_body.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_file_generic_regression_test.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:

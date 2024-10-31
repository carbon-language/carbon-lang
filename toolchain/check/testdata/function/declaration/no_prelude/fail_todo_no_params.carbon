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

// --- fail_todo_arrow_body.carbon

// TODO: We don't have parsing support for this yet.
library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_arrow_body.carbon:[[@LINE+8]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn A => 0;
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_arrow_body.carbon:[[@LINE+4]]:6: error: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR: fn A => 0;
// CHECK:STDERR:      ^~
// CHECK:STDERR:
fn A => 0;

// --- fail_invalid_file_generic_regression_test.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_invalid_file_generic_regression_test.carbon:[[@LINE+4]]:9: error: `var` declaration cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
// CHECK:STDERR: var x:! () = ();
// CHECK:STDERR:         ^~
// CHECK:STDERR:
var x:! () = ();

// CHECK:STDERR: fail_invalid_file_generic_regression_test.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn A {
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
fn A {
  // CHECK:STDERR: fail_invalid_file_generic_regression_test.carbon:[[@LINE+6]]:3: error: cannot deduce value for generic parameter `x` [DeductionIncomplete]
  // CHECK:STDERR:   A();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_invalid_file_generic_regression_test.carbon:[[@LINE-4]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A {
  // CHECK:STDERR: ^~~~~~
  A();
}

// CHECK:STDOUT: --- fail_no_body.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_brace_body.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_arrow_body.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_file_generic_regression_test.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x, 0 [symbolic]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc8_10.2: type = converted %.loc8_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref %.1 = var x
// CHECK:STDOUT:   %x: %.1 = bind_symbolic_name x, 0, %x.var [symbolic = constants.%x]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(file.%x: %.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_15.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc8_15.2: init %.1 = tuple_init () to file.%x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_16: init %.1 = converted %.loc8_15.1, %.loc8_15.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc8_16
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%x) {}
// CHECK:STDOUT:

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
// CHECK:STDERR: fail_no_body.carbon:[[@LINE+4]]:1: ERROR: Semantics TODO: `function with positional parameters`.
// CHECK:STDERR: fn A;
// CHECK:STDERR: ^~~~~
// CHECK:STDERR:
fn A;

// --- fail_todo_brace_body.carbon

library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_brace_body.carbon:[[@LINE+4]]:1: ERROR: Semantics TODO: `function with positional parameters`.
// CHECK:STDERR: fn A {
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
fn A {
}

// --- fail_todo_arrow_body.carbon

// TODO: We don't have parsing support for this yet.
library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_arrow_body.carbon:[[@LINE+7]]:1: ERROR: Semantics TODO: `function with positional parameters`.
// CHECK:STDERR: fn A => 0;
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_arrow_body.carbon:[[@LINE+3]]:6: ERROR: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition.
// CHECK:STDERR: fn A => 0;
// CHECK:STDERR:      ^~
fn A => 0;

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
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {}
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
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {}
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
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:

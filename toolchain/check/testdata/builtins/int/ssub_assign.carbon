// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/ssub_assign.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/ssub_assign.carbon

// --- call.carbon

library "[[@TEST_NAME]]";

fn Builtin(a: i32*, b: i32) = "int.ssub_assign";

fn Call(a: i32*, b: i32) {
  //@dump-sem-ir-begin
  Builtin(a, b);
  //@dump-sem-ir-end
}

// --- fail_unsized.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_unsized.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.ssub_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Builtin(a: Core.IntLiteral()*, b: Core.IntLiteral()) = "int.ssub_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Builtin(a: Core.IntLiteral()*, b: Core.IntLiteral()) = "int.ssub_assign";

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.ssub_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NotPtr(a: i32, b: i32) = "int.ssub_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotPtr(a: i32, b: i32) = "int.ssub_assign";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.ssub_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedTypes(a: i32*, b: i64) = "int.ssub_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedTypes(a: i32*, b: i64) = "int.ssub_assign";

// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Builtin.type: type = fn_type @Builtin [concrete]
// CHECK:STDOUT:   %Builtin: %Builtin.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param: %ptr, %b.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Builtin.ref: %Builtin.type = name_ref Builtin, file.%Builtin.decl [concrete = constants.%Builtin]
// CHECK:STDOUT:   %a.ref: %ptr = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.ssub_assign: init %empty_tuple.type = call %Builtin.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

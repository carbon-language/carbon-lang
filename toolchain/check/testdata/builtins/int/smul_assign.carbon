// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/smul_assign.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/smul_assign.carbon

// --- call.carbon

library "[[@TEST_NAME]]";

fn Builtin(ref a: i32, b: i32) = "int.smul_assign";

fn Call(ref a: i32, b: i32) {
  //@dump-sem-ir-begin
  Builtin(ref a, b);
  //@dump-sem-ir-end
}

// --- fail_unsized.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_unsized.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.smul_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Builtin(ref a: Core.IntLiteral(), b: Core.IntLiteral()) = "int.smul_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Builtin(ref a: Core.IntLiteral(), b: Core.IntLiteral()) = "int.smul_assign";

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.smul_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NotRef(a: i32, b: i32) = "int.smul_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotRef(a: i32, b: i32) = "int.smul_assign";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.smul_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedTypes(ref a: i32, b: i64) = "int.smul_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedTypes(ref a: i32, b: i64) = "int.smul_assign";

// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Builtin.type: type = fn_type @Builtin [concrete]
// CHECK:STDOUT:   %Builtin: %Builtin.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param: %i32, %b.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Builtin.ref: %Builtin.type = name_ref Builtin, file.%Builtin.decl [concrete = constants.%Builtin]
// CHECK:STDOUT:   %a.ref: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %Builtin.call: init %empty_tuple.type = call %Builtin.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

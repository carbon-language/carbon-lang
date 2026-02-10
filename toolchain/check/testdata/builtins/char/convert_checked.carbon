// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char/convert_checked.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char/convert_checked.carbon

// --- builtin.carbon

library "[[@TEST_NAME]]";

fn CharLiteral() -> type = "char_literal.make_type";
fn IntLiteral() -> type = "int_literal.make_type";
fn UInt(n: IntLiteral()) -> type = "int.make_type_unsigned";

fn ToChar(c: CharLiteral()) -> UInt(8) = "char.convert_checked";

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";
import library "builtin";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "char.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn ToCharBarResultU16(c: CharLiteral()) -> UInt(16) = "char.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn ToCharBarResultU16(c: CharLiteral()) -> UInt(16) = "char.convert_checked";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "char.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn ToCharBadResultU9(c: CharLiteral()) -> UInt(9) = "char.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn ToCharBadResultU9(c: CharLiteral()) -> UInt(9) = "char.convert_checked";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "char.convert_checked" [InvalidBuiltinSignature]
// CHECK:STDERR: fn ToCharNoParam() -> UInt(8) = "char.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn ToCharNoParam() -> UInt(8) = "char.convert_checked";

// --- fail_runtime_call.carbon

library "[[@TEST_NAME]]";
import library "builtin";

let c: CharLiteral() = 'a';
// CHECK:STDERR: fail_runtime_call.carbon:[[@LINE+8]]:18: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let d: UInt(8) = ToChar(c);
// CHECK:STDERR:                  ^~~~~~~~~
// CHECK:STDERR: fail_runtime_call.carbon:[[@LINE-6]]:1: in import [InImport]
// CHECK:STDERR: builtin.carbon:8:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn ToChar(c: CharLiteral()) -> UInt(8) = "char.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let d: UInt(8) = ToChar(c);

// --- narrow.carbon

library "[[@TEST_NAME]]";
import library "builtin";

//@dump-sem-ir-begin
let a: UInt(8) = ToChar('\0');
let b: UInt(8) = ToChar('b');
let c: UInt(8) = ToChar('\u{7F}');
//@dump-sem-ir-end

// --- fail_size_small.carbon

library "[[@TEST_NAME]]";
import library "builtin";

// CHECK:STDERR: fail_size_small.carbon:[[@LINE+4]]:18: error: character value U+0080 too large for type `<builtin u8>` [CharTooLargeForType]
// CHECK:STDERR: let c: UInt(8) = ToChar('\u{80}');
// CHECK:STDERR:                  ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: UInt(8) = ToChar('\u{80}');

// --- fail_size_multi_byte.carbon

library "[[@TEST_NAME]]";
import library "builtin";

// CHECK:STDERR: fail_size_multi_byte.carbon:[[@LINE+4]]:18: error: character value U+1E15 too large for type `<builtin u8>` [CharTooLargeForType]
// CHECK:STDERR: let c: UInt(8) = ToChar('\u{1E15}');
// CHECK:STDERR:                  ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: UInt(8) = ToChar('\u{1E15}');

// CHECK:STDOUT: --- narrow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [concrete]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8.builtin: type = int_type unsigned, %int_8 [concrete]
// CHECK:STDOUT:   %pattern_type.456: type = pattern_type %u8.builtin [concrete]
// CHECK:STDOUT:   %ToChar.type: type = fn_type @ToChar [concrete]
// CHECK:STDOUT:   %ToChar: %ToChar.type = struct_value () [concrete]
// CHECK:STDOUT:   %.dc9: Core.CharLiteral = char_value U+0000 [concrete]
// CHECK:STDOUT:   %int_0: %u8.builtin = int_value 0 [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT:   %int_98: %u8.builtin = int_value 98 [concrete]
// CHECK:STDOUT:   %.e28: Core.CharLiteral = char_value U+007F [concrete]
// CHECK:STDOUT:   %int_127: %u8.builtin = int_value 127 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.UInt: %UInt.type = import_ref Main//builtin, UInt, loaded [concrete = constants.%UInt]
// CHECK:STDOUT:   %Main.ToChar: %ToChar.type = import_ref Main//builtin, ToChar, loaded [concrete = constants.%ToChar]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.456 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_14.1: type = splice_block %.loc6_14.3 [concrete = constants.%u8.builtin] {
// CHECK:STDOUT:     %UInt.ref.loc6: %UInt.type = name_ref UInt, imports.%Main.UInt [concrete = constants.%UInt]
// CHECK:STDOUT:     %int_8.loc6: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %UInt.call.loc6: init type = call %UInt.ref.loc6(%int_8.loc6) [concrete = constants.%u8.builtin]
// CHECK:STDOUT:     %.loc6_14.2: type = value_of_initializer %UInt.call.loc6 [concrete = constants.%u8.builtin]
// CHECK:STDOUT:     %.loc6_14.3: type = converted %UInt.call.loc6, %.loc6_14.2 [concrete = constants.%u8.builtin]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_29.1: %u8.builtin = value_of_initializer @__global_init.%ToChar.call.loc6 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc6_29.2: %u8.builtin = converted @__global_init.%ToChar.call.loc6, %.loc6_29.1 [concrete = constants.%int_0]
// CHECK:STDOUT:   %a: %u8.builtin = value_binding a, %.loc6_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.456 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_14.1: type = splice_block %.loc7_14.3 [concrete = constants.%u8.builtin] {
// CHECK:STDOUT:     %UInt.ref.loc7: %UInt.type = name_ref UInt, imports.%Main.UInt [concrete = constants.%UInt]
// CHECK:STDOUT:     %int_8.loc7: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %UInt.call.loc7: init type = call %UInt.ref.loc7(%int_8.loc7) [concrete = constants.%u8.builtin]
// CHECK:STDOUT:     %.loc7_14.2: type = value_of_initializer %UInt.call.loc7 [concrete = constants.%u8.builtin]
// CHECK:STDOUT:     %.loc7_14.3: type = converted %UInt.call.loc7, %.loc7_14.2 [concrete = constants.%u8.builtin]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_28.1: %u8.builtin = value_of_initializer @__global_init.%ToChar.call.loc7 [concrete = constants.%int_98]
// CHECK:STDOUT:   %.loc7_28.2: %u8.builtin = converted @__global_init.%ToChar.call.loc7, %.loc7_28.1 [concrete = constants.%int_98]
// CHECK:STDOUT:   %b: %u8.builtin = value_binding b, %.loc7_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.456 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_14.1: type = splice_block %.loc8_14.3 [concrete = constants.%u8.builtin] {
// CHECK:STDOUT:     %UInt.ref.loc8: %UInt.type = name_ref UInt, imports.%Main.UInt [concrete = constants.%UInt]
// CHECK:STDOUT:     %int_8.loc8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %UInt.call.loc8: init type = call %UInt.ref.loc8(%int_8.loc8) [concrete = constants.%u8.builtin]
// CHECK:STDOUT:     %.loc8_14.2: type = value_of_initializer %UInt.call.loc8 [concrete = constants.%u8.builtin]
// CHECK:STDOUT:     %.loc8_14.3: type = converted %UInt.call.loc8, %.loc8_14.2 [concrete = constants.%u8.builtin]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_33.1: %u8.builtin = value_of_initializer @__global_init.%ToChar.call.loc8 [concrete = constants.%int_127]
// CHECK:STDOUT:   %.loc8_33.2: %u8.builtin = converted @__global_init.%ToChar.call.loc8, %.loc8_33.1 [concrete = constants.%int_127]
// CHECK:STDOUT:   %c: %u8.builtin = value_binding c, %.loc8_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ToChar.ref.loc6: %ToChar.type = name_ref ToChar, imports.%Main.ToChar [concrete = constants.%ToChar]
// CHECK:STDOUT:   %.loc6: Core.CharLiteral = char_value U+0000 [concrete = constants.%.dc9]
// CHECK:STDOUT:   %ToChar.call.loc6: init %u8.builtin = call %ToChar.ref.loc6(%.loc6) [concrete = constants.%int_0]
// CHECK:STDOUT:   %ToChar.ref.loc7: %ToChar.type = name_ref ToChar, imports.%Main.ToChar [concrete = constants.%ToChar]
// CHECK:STDOUT:   %.loc7: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %ToChar.call.loc7: init %u8.builtin = call %ToChar.ref.loc7(%.loc7) [concrete = constants.%int_98]
// CHECK:STDOUT:   %ToChar.ref.loc8: %ToChar.type = name_ref ToChar, imports.%Main.ToChar [concrete = constants.%ToChar]
// CHECK:STDOUT:   %.loc8: Core.CharLiteral = char_value U+007F [concrete = constants.%.e28]
// CHECK:STDOUT:   %ToChar.call.loc8: init %u8.builtin = call %ToChar.ref.loc8(%.loc8) [concrete = constants.%int_127]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

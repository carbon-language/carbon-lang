// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/add_char_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/add_char_literal.carbon

// --- success.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn AddIntChar(n: IntLiteral, c: CharLiteral) -> CharLiteral = "int.add_char_literal";

//@dump-sem-ir-begin
let a: CharLiteral = AddIntChar(1, 'a');
//@dump-sem-ir-end

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn AddIntChar(n: IntLiteral, c: CharLiteral) -> CharLiteral = "int.add_char_literal";

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:22: error: character arithmetic result 4503599627370593 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let a: CharLiteral = AddIntChar(0x10000000000000, 'a');
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: CharLiteral = AddIntChar(0x10000000000000, 'a');

// --- fail_invalid_result.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn AddIntChar(n: IntLiteral, c: CharLiteral) -> CharLiteral = "int.add_char_literal";
fn Negate(n: IntLiteral) -> IntLiteral = "int.snegate";

// CHECK:STDERR: fail_invalid_result.carbon:[[@LINE+4]]:22: error: character arithmetic result 1114112 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let a: CharLiteral = AddIntChar(1, '\u{10FFFF}');
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: CharLiteral = AddIntChar(1, '\u{10FFFF}');

// CHECK:STDERR: fail_invalid_result.carbon:[[@LINE+4]]:22: error: character arithmetic result 55296 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let b: CharLiteral = AddIntChar(1, '\u{D7FF}');
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: CharLiteral = AddIntChar(1, '\u{D7FF}');

// CHECK:STDERR: fail_invalid_result.carbon:[[@LINE+4]]:22: error: character arithmetic result -1 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let c: CharLiteral = AddIntChar(Negate(98), 'a');
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: CharLiteral = AddIntChar(Negate(98), 'a');

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.8c6: type = pattern_type Core.CharLiteral [concrete]
// CHECK:STDOUT:   %AddIntChar.type: type = fn_type @AddIntChar [concrete]
// CHECK:STDOUT:   %AddIntChar: %AddIntChar.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.8c6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.8c6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_39.1: Core.CharLiteral = value_of_initializer @__global_init.%AddIntChar.call [concrete = constants.%.711]
// CHECK:STDOUT:   %.loc12_39.2: Core.CharLiteral = converted @__global_init.%AddIntChar.call, %.loc12_39.1 [concrete = constants.%.711]
// CHECK:STDOUT:   %CharLiteral.ref: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %a: Core.CharLiteral = wrapper_binding a, %.loc12_39.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AddIntChar.ref: %AddIntChar.type = name_ref AddIntChar, file.%AddIntChar.decl [concrete = constants.%AddIntChar]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc12: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %AddIntChar.call: init Core.CharLiteral = call %AddIntChar.ref(%int_1, %.loc12) [concrete = constants.%.711]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

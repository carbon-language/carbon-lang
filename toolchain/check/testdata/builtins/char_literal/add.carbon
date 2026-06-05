// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char_literal/add.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char_literal/add.carbon

// --- success.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn AddCharInt(c: CharLiteral, n: IntLiteral) -> CharLiteral = "char_literal.add";

//@dump-sem-ir-begin
let a: CharLiteral = AddCharInt('a', 1);
//@dump-sem-ir-end

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn AddCharInt(c: CharLiteral, n: IntLiteral) -> CharLiteral = "char_literal.add";

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:22: error: character arithmetic result 4503599627370593 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let a: CharLiteral = AddCharInt('a', 0x10000000000000);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: CharLiteral = AddCharInt('a', 0x10000000000000);

// --- fail_invalid_result.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn AddCharInt(c: CharLiteral, n: IntLiteral) -> CharLiteral = "char_literal.add";
fn Negate(n: IntLiteral) -> IntLiteral = "int.snegate";

// CHECK:STDERR: fail_invalid_result.carbon:[[@LINE+4]]:22: error: character arithmetic result 1114112 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let a: CharLiteral = AddCharInt('\u{10FFFF}', 1);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: CharLiteral = AddCharInt('\u{10FFFF}', 1);

// CHECK:STDERR: fail_invalid_result.carbon:[[@LINE+4]]:22: error: character arithmetic result 55296 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let b: CharLiteral = AddCharInt('\u{D7FF}', 1);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: CharLiteral = AddCharInt('\u{D7FF}', 1);

// CHECK:STDERR: fail_invalid_result.carbon:[[@LINE+4]]:22: error: character arithmetic result -1 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let c: CharLiteral = AddCharInt('a', Negate(98));
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: CharLiteral = AddCharInt('a', Negate(98));

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.8c6: type = pattern_type Core.CharLiteral [concrete]
// CHECK:STDOUT:   %AddCharInt.type: type = fn_type @AddCharInt [concrete]
// CHECK:STDOUT:   %AddCharInt: %AddCharInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.8c6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CharLiteral.ref: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %.loc12_39.1: Core.CharLiteral = value_of_initializer @__global_init.%AddCharInt.call [concrete = constants.%.711]
// CHECK:STDOUT:   %.loc12_39.2: Core.CharLiteral = converted @__global_init.%AddCharInt.call, %.loc12_39.1 [concrete = constants.%.711]
// CHECK:STDOUT:   %a: Core.CharLiteral = value_binding a, %.loc12_39.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AddCharInt.ref: %AddCharInt.type = name_ref AddCharInt, file.%AddCharInt.decl [concrete = constants.%AddCharInt]
// CHECK:STDOUT:   %.loc12: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %AddCharInt.call: init Core.CharLiteral = call %AddCharInt.ref(%.loc12, %int_1) [concrete = constants.%.711]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// It's fine to perform an addition that is in-bounds for `CharLiteral` but out
// of the range of a `char` or a byte.
let b: CharLiteral = AddCharInt('\u{7F}', 1);
let c: CharLiteral = AddCharInt('\u{FF}', 1);
//@dump-sem-ir-end

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn AddCharInt(c: CharLiteral, n: IntLiteral) -> CharLiteral = "char_literal.add";
fn Negate(n: IntLiteral) -> IntLiteral = "int.snegate";

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:22: error: character arithmetic result 4503599627370593 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let a: CharLiteral = AddCharInt('a', 0x10000000000000);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: CharLiteral = AddCharInt('a', 0x10000000000000);

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+11]]:5: error: duplicate name `a` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: let a: CharLiteral = AddCharInt('\u{10FFFF}', 1);
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_overflow.carbon:[[@LINE-5]]:5: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: let a: CharLiteral = AddCharInt('a', 0x10000000000000);
// CHECK:STDERR:     ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:22: error: character arithmetic result 1114112 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let a: CharLiteral = AddCharInt('\u{10FFFF}', 1);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: CharLiteral = AddCharInt('\u{10FFFF}', 1);

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:22: error: character arithmetic result 55296 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let b: CharLiteral = AddCharInt('\u{D7FF}', 1);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: CharLiteral = AddCharInt('\u{D7FF}', 1);

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:22: error: character arithmetic result -1 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
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
// CHECK:STDOUT:   %.e28: Core.CharLiteral = char_value U+007F [concrete]
// CHECK:STDOUT:   %.ae7: Core.CharLiteral = char_value U+0080 [concrete]
// CHECK:STDOUT:   %.ad8: Core.CharLiteral = char_value U+00FF [concrete]
// CHECK:STDOUT:   %.a8b: Core.CharLiteral = char_value U+0100 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.8c6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CharLiteral.ref.loc12: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %.loc12_39.1: Core.CharLiteral = value_of_initializer @__global_init.%AddCharInt.call.loc12 [concrete = constants.%.711]
// CHECK:STDOUT:   %.loc12_39.2: Core.CharLiteral = converted @__global_init.%AddCharInt.call.loc12, %.loc12_39.1 [concrete = constants.%.711]
// CHECK:STDOUT:   %a: Core.CharLiteral = value_binding a, %.loc12_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.8c6 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CharLiteral.ref.loc15: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %.loc15_44.1: Core.CharLiteral = value_of_initializer @__global_init.%AddCharInt.call.loc15 [concrete = constants.%.ae7]
// CHECK:STDOUT:   %.loc15_44.2: Core.CharLiteral = converted @__global_init.%AddCharInt.call.loc15, %.loc15_44.1 [concrete = constants.%.ae7]
// CHECK:STDOUT:   %b: Core.CharLiteral = value_binding b, %.loc15_44.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.8c6 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CharLiteral.ref.loc16: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %.loc16_44.1: Core.CharLiteral = value_of_initializer @__global_init.%AddCharInt.call.loc16 [concrete = constants.%.a8b]
// CHECK:STDOUT:   %.loc16_44.2: Core.CharLiteral = converted @__global_init.%AddCharInt.call.loc16, %.loc16_44.1 [concrete = constants.%.a8b]
// CHECK:STDOUT:   %c: Core.CharLiteral = value_binding c, %.loc16_44.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AddCharInt.ref.loc12: %AddCharInt.type = name_ref AddCharInt, file.%AddCharInt.decl [concrete = constants.%AddCharInt]
// CHECK:STDOUT:   %.loc12: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %AddCharInt.call.loc12: init Core.CharLiteral = call %AddCharInt.ref.loc12(%.loc12, %int_1.loc12) [concrete = constants.%.711]
// CHECK:STDOUT:   %AddCharInt.ref.loc15: %AddCharInt.type = name_ref AddCharInt, file.%AddCharInt.decl [concrete = constants.%AddCharInt]
// CHECK:STDOUT:   %.loc15: Core.CharLiteral = char_value U+007F [concrete = constants.%.e28]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %AddCharInt.call.loc15: init Core.CharLiteral = call %AddCharInt.ref.loc15(%.loc15, %int_1.loc15) [concrete = constants.%.ae7]
// CHECK:STDOUT:   %AddCharInt.ref.loc16: %AddCharInt.type = name_ref AddCharInt, file.%AddCharInt.decl [concrete = constants.%AddCharInt]
// CHECK:STDOUT:   %.loc16: Core.CharLiteral = char_value U+00FF [concrete = constants.%.ad8]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %AddCharInt.call.loc16: init Core.CharLiteral = call %AddCharInt.ref.loc16(%.loc16, %int_1.loc16) [concrete = constants.%.a8b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char_literal/sub_int.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char_literal/sub_int.carbon

// --- success.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn SubCharInt(c: CharLiteral, n: IntLiteral) -> CharLiteral = "char_literal.sub_int";
fn Negate(n: IntLiteral) -> IntLiteral = "int.snegate";

//@dump-sem-ir-begin
let a: CharLiteral = SubCharInt('b', 1);
// It's fine to perform a subtraction that is in-bounds for `CharLiteral` but out
// of the range of a `char` or a byte.
let b: CharLiteral = SubCharInt('\u{7F}', Negate(1));
let c: CharLiteral = SubCharInt('\u{FF}', Negate(1));
//@dump-sem-ir-end

// --- fail_invalid_result.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn SubCharInt(c: CharLiteral, n: IntLiteral) -> CharLiteral = "char_literal.sub_int";
fn Negate(n: IntLiteral) -> IntLiteral = "int.snegate";

// CHECK:STDERR: fail_invalid_result.carbon:[[@LINE+4]]:22: error: character arithmetic result -1 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let a: CharLiteral = SubCharInt('\u{0000}', 1);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: CharLiteral = SubCharInt('\u{0000}', 1);

// CHECK:STDERR: fail_invalid_result.carbon:[[@LINE+4]]:22: error: character arithmetic result 1114112 is not a valid Unicode code point [InvalidCharLiteralArithmeticResult]
// CHECK:STDERR: let b: CharLiteral = SubCharInt('\u{10FFFF}', Negate(1));
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: CharLiteral = SubCharInt('\u{10FFFF}', Negate(1));

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.8c6: type = pattern_type Core.CharLiteral [concrete]
// CHECK:STDOUT:   %SubCharInt.type: type = fn_type @SubCharInt [concrete]
// CHECK:STDOUT:   %SubCharInt: %SubCharInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.8c6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.8c6 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.e28: Core.CharLiteral = char_value U+007F [concrete]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %.ae7: Core.CharLiteral = char_value U+0080 [concrete]
// CHECK:STDOUT:   %c.patt.c30: %pattern_type.8c6 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %.ad8: Core.CharLiteral = char_value U+00FF [concrete]
// CHECK:STDOUT:   %.a8b: Core.CharLiteral = char_value U+0100 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.8c6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_39.1: Core.CharLiteral = value_of_initializer @__global_init.%SubCharInt.call.loc13 [concrete = constants.%.54f]
// CHECK:STDOUT:   %.loc13_39.2: Core.CharLiteral = converted @__global_init.%SubCharInt.call.loc13, %.loc13_39.1 [concrete = constants.%.54f]
// CHECK:STDOUT:   %CharLiteral.ref.loc13: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %a: Core.CharLiteral = wrapper_binding a, %.loc13_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.8c6 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_52.1: Core.CharLiteral = value_of_initializer @__global_init.%SubCharInt.call.loc16 [concrete = constants.%.ae7]
// CHECK:STDOUT:   %.loc16_52.2: Core.CharLiteral = converted @__global_init.%SubCharInt.call.loc16, %.loc16_52.1 [concrete = constants.%.ae7]
// CHECK:STDOUT:   %CharLiteral.ref.loc16: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %b: Core.CharLiteral = wrapper_binding b, %.loc16_52.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.8c6 = value_binding_pattern c [concrete = constants.%c.patt.c30]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_52.1: Core.CharLiteral = value_of_initializer @__global_init.%SubCharInt.call.loc17 [concrete = constants.%.a8b]
// CHECK:STDOUT:   %.loc17_52.2: Core.CharLiteral = converted @__global_init.%SubCharInt.call.loc17, %.loc17_52.1 [concrete = constants.%.a8b]
// CHECK:STDOUT:   %CharLiteral.ref.loc17: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %c: Core.CharLiteral = wrapper_binding c, %.loc17_52.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %SubCharInt.ref.loc13: %SubCharInt.type = name_ref SubCharInt, file.%SubCharInt.decl [concrete = constants.%SubCharInt]
// CHECK:STDOUT:   %.loc13: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %SubCharInt.call.loc13: init Core.CharLiteral = call %SubCharInt.ref.loc13(%.loc13, %int_1.loc13) [concrete = constants.%.54f]
// CHECK:STDOUT:   %SubCharInt.ref.loc16: %SubCharInt.type = name_ref SubCharInt, file.%SubCharInt.decl [concrete = constants.%SubCharInt]
// CHECK:STDOUT:   %.loc16_33: Core.CharLiteral = char_value U+007F [concrete = constants.%.e28]
// CHECK:STDOUT:   %Negate.ref.loc16: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Negate.call.loc16: init Core.IntLiteral = call %Negate.ref.loc16(%int_1.loc16) [concrete = constants.%int_-1]
// CHECK:STDOUT:   %.loc16_51.1: Core.IntLiteral = value_of_initializer %Negate.call.loc16 [concrete = constants.%int_-1]
// CHECK:STDOUT:   %.loc16_51.2: Core.IntLiteral = converted %Negate.call.loc16, %.loc16_51.1 [concrete = constants.%int_-1]
// CHECK:STDOUT:   %SubCharInt.call.loc16: init Core.CharLiteral = call %SubCharInt.ref.loc16(%.loc16_33, %.loc16_51.2) [concrete = constants.%.ae7]
// CHECK:STDOUT:   %SubCharInt.ref.loc17: %SubCharInt.type = name_ref SubCharInt, file.%SubCharInt.decl [concrete = constants.%SubCharInt]
// CHECK:STDOUT:   %.loc17_33: Core.CharLiteral = char_value U+00FF [concrete = constants.%.ad8]
// CHECK:STDOUT:   %Negate.ref.loc17: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Negate.call.loc17: init Core.IntLiteral = call %Negate.ref.loc17(%int_1.loc17) [concrete = constants.%int_-1]
// CHECK:STDOUT:   %.loc17_51.1: Core.IntLiteral = value_of_initializer %Negate.call.loc17 [concrete = constants.%int_-1]
// CHECK:STDOUT:   %.loc17_51.2: Core.IntLiteral = converted %Negate.call.loc17, %.loc17_51.1 [concrete = constants.%int_-1]
// CHECK:STDOUT:   %SubCharInt.call.loc17: init Core.CharLiteral = call %SubCharInt.ref.loc17(%.loc17_33, %.loc17_51.2) [concrete = constants.%.a8b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

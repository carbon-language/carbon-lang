// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char_literal/sub_char.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char_literal/sub_char.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn SubCharCharToIntLiteral(a: CharLiteral, b: CharLiteral) -> IntLiteral = "char_literal.sub_char";
fn SubCharCharToI32(a: CharLiteral, b: CharLiteral) -> i32 = "char_literal.sub_char";

//@dump-sem-ir-begin
let a: IntLiteral = SubCharCharToIntLiteral('b', 'a');
let b: i32 = SubCharCharToI32('b', 'a');
let c: IntLiteral = SubCharCharToIntLiteral('a', 'b');
let d: i32 = SubCharCharToI32('a', 'b');
let max: i32 = SubCharCharToI32('\u{10FFFF}', '\u{0}');
let min: i32 = SubCharCharToI32('\u{0}', '\u{10FFFF}');
//@dump-sem-ir-end

// CHECK:STDOUT: --- sub_char.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %SubCharCharToIntLiteral.type: type = fn_type @SubCharCharToIntLiteral [concrete]
// CHECK:STDOUT:   %SubCharCharToIntLiteral: %SubCharCharToIntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %SubCharCharToI32.type: type = fn_type @SubCharCharToI32 [concrete]
// CHECK:STDOUT:   %SubCharCharToI32: %SubCharCharToI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt.b0e: %pattern_type.dc0 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %b.patt.3f7: %pattern_type.6b6 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.dc0 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %int_-1.638: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.6b6 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %int_-1.4ef: %i32 = int_value -1 [concrete]
// CHECK:STDOUT:   %max.patt: %pattern_type.6b6 = value_binding_pattern max [concrete]
// CHECK:STDOUT:   %.914: Core.CharLiteral = char_value U+10FFFF [concrete]
// CHECK:STDOUT:   %.dc9: Core.CharLiteral = char_value U+0000 [concrete]
// CHECK:STDOUT:   %int_1114111: %i32 = int_value 1114111 [concrete]
// CHECK:STDOUT:   %min.patt: %pattern_type.6b6 = value_binding_pattern min [concrete]
// CHECK:STDOUT:   %int_-1114111: %i32 = int_value -1114111 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.dc0 = value_binding_pattern a [concrete = constants.%a.patt.b0e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24_53.1: Core.IntLiteral = value_of_initializer @__global_init.%SubCharCharToIntLiteral.call.loc24 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc24_53.2: Core.IntLiteral = converted @__global_init.%SubCharCharToIntLiteral.call.loc24, %.loc24_53.1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %IntLiteral.ref.loc24: type = name_ref IntLiteral, %IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %a: Core.IntLiteral = value_binding a, %.loc24_53.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = value_binding_pattern b [concrete = constants.%b.patt.3f7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25_39.1: %i32 = value_of_initializer @__global_init.%SubCharCharToI32.call.loc25 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc25_39.2: %i32 = converted @__global_init.%SubCharCharToI32.call.loc25, %.loc25_39.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %i32.loc25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = value_binding b, %.loc25_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.dc0 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26_53.1: Core.IntLiteral = value_of_initializer @__global_init.%SubCharCharToIntLiteral.call.loc26 [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %.loc26_53.2: Core.IntLiteral = converted @__global_init.%SubCharCharToIntLiteral.call.loc26, %.loc26_53.1 [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %IntLiteral.ref.loc26: type = name_ref IntLiteral, %IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %c: Core.IntLiteral = value_binding c, %.loc26_53.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.6b6 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_39.1: %i32 = value_of_initializer @__global_init.%SubCharCharToI32.call.loc27 [concrete = constants.%int_-1.4ef]
// CHECK:STDOUT:   %.loc27_39.2: %i32 = converted @__global_init.%SubCharCharToI32.call.loc27, %.loc27_39.1 [concrete = constants.%int_-1.4ef]
// CHECK:STDOUT:   %i32.loc27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %d: %i32 = value_binding d, %.loc27_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %max.patt: %pattern_type.6b6 = value_binding_pattern max [concrete = constants.%max.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc28_54.1: %i32 = value_of_initializer @__global_init.%SubCharCharToI32.call.loc28 [concrete = constants.%int_1114111]
// CHECK:STDOUT:   %.loc28_54.2: %i32 = converted @__global_init.%SubCharCharToI32.call.loc28, %.loc28_54.1 [concrete = constants.%int_1114111]
// CHECK:STDOUT:   %i32.loc28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %max: %i32 = value_binding max, %.loc28_54.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %min.patt: %pattern_type.6b6 = value_binding_pattern min [concrete = constants.%min.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc29_54.1: %i32 = value_of_initializer @__global_init.%SubCharCharToI32.call.loc29 [concrete = constants.%int_-1114111]
// CHECK:STDOUT:   %.loc29_54.2: %i32 = converted @__global_init.%SubCharCharToI32.call.loc29, %.loc29_54.1 [concrete = constants.%int_-1114111]
// CHECK:STDOUT:   %i32.loc29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %min: %i32 = value_binding min, %.loc29_54.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %SubCharCharToIntLiteral.ref.loc24: %SubCharCharToIntLiteral.type = name_ref SubCharCharToIntLiteral, file.%SubCharCharToIntLiteral.decl [concrete = constants.%SubCharCharToIntLiteral]
// CHECK:STDOUT:   %.loc24_45: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %.loc24_50: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %SubCharCharToIntLiteral.call.loc24: init Core.IntLiteral = call %SubCharCharToIntLiteral.ref.loc24(%.loc24_45, %.loc24_50) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %SubCharCharToI32.ref.loc25: %SubCharCharToI32.type = name_ref SubCharCharToI32, file.%SubCharCharToI32.decl [concrete = constants.%SubCharCharToI32]
// CHECK:STDOUT:   %.loc25_31: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %.loc25_36: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %SubCharCharToI32.call.loc25: init %i32 = call %SubCharCharToI32.ref.loc25(%.loc25_31, %.loc25_36) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %SubCharCharToIntLiteral.ref.loc26: %SubCharCharToIntLiteral.type = name_ref SubCharCharToIntLiteral, file.%SubCharCharToIntLiteral.decl [concrete = constants.%SubCharCharToIntLiteral]
// CHECK:STDOUT:   %.loc26_45: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %.loc26_50: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %SubCharCharToIntLiteral.call.loc26: init Core.IntLiteral = call %SubCharCharToIntLiteral.ref.loc26(%.loc26_45, %.loc26_50) [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %SubCharCharToI32.ref.loc27: %SubCharCharToI32.type = name_ref SubCharCharToI32, file.%SubCharCharToI32.decl [concrete = constants.%SubCharCharToI32]
// CHECK:STDOUT:   %.loc27_31: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %.loc27_36: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %SubCharCharToI32.call.loc27: init %i32 = call %SubCharCharToI32.ref.loc27(%.loc27_31, %.loc27_36) [concrete = constants.%int_-1.4ef]
// CHECK:STDOUT:   %SubCharCharToI32.ref.loc28: %SubCharCharToI32.type = name_ref SubCharCharToI32, file.%SubCharCharToI32.decl [concrete = constants.%SubCharCharToI32]
// CHECK:STDOUT:   %.loc28_33: Core.CharLiteral = char_value U+10FFFF [concrete = constants.%.914]
// CHECK:STDOUT:   %.loc28_47: Core.CharLiteral = char_value U+0000 [concrete = constants.%.dc9]
// CHECK:STDOUT:   %SubCharCharToI32.call.loc28: init %i32 = call %SubCharCharToI32.ref.loc28(%.loc28_33, %.loc28_47) [concrete = constants.%int_1114111]
// CHECK:STDOUT:   %SubCharCharToI32.ref.loc29: %SubCharCharToI32.type = name_ref SubCharCharToI32, file.%SubCharCharToI32.decl [concrete = constants.%SubCharCharToI32]
// CHECK:STDOUT:   %.loc29_33: Core.CharLiteral = char_value U+0000 [concrete = constants.%.dc9]
// CHECK:STDOUT:   %.loc29_42: Core.CharLiteral = char_value U+10FFFF [concrete = constants.%.914]
// CHECK:STDOUT:   %SubCharCharToI32.call.loc29: init %i32 = call %SubCharCharToI32.ref.loc29(%.loc29_33, %.loc29_42) [concrete = constants.%int_-1114111]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

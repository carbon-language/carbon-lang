// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/char_literal/convert_int.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/char_literal/convert_int.carbon

// --- success.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();

fn ConvertToI32(c: CharLiteral) -> i32 = "char_literal.convert_int";
fn ConvertToU32(c: CharLiteral) -> u32 = "char_literal.convert_int";
fn ConvertToI64(c: CharLiteral) -> i64 = "char_literal.convert_int";
fn ConvertToU64(c: CharLiteral) -> u64 = "char_literal.convert_int";
fn ConvertToU8(c: CharLiteral) -> u8 = "char_literal.convert_int";

//@dump-sem-ir-begin
let a: i32 = ConvertToI32('a');
let b: u32 = ConvertToU32('b');
let c: i64 = ConvertToI64('\u{10FFFF}');
let d: u64 = ConvertToU64('\0');
// Code points in [U+0080, U+00FF] fit in u8 but not in char.
let e: u8 = ConvertToU8('\u{80}');
let f: u8 = ConvertToU8('\u{FF}');
//@dump-sem-ir-end

// --- fail_overflow.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();
fn UInt(n: IntLiteral) -> type = "int.make_type_unsigned";

fn ConvertToU8(c: CharLiteral) -> UInt(8) = "char_literal.convert_int";

// CHECK:STDERR: fail_overflow.carbon:[[@LINE+11]]:13: error: character value U+0100 too large for type `<builtin u8>` [CharTooLargeForType]
// CHECK:STDERR: let g: u8 = ConvertToU8('\u{100}');
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+7]]:13: error: cannot implicitly convert expression of type `<builtin u8>` to `u8` [ConversionFailure]
// CHECK:STDERR: let g: u8 = ConvertToU8('\u{100}');
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:13: note: type `<builtin u8>` does not implement interface `Core.ImplicitAs(u8)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: let g: u8 = ConvertToU8('\u{100}');
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let g: u8 = ConvertToU8('\u{100}');

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ConvertToI32.type: type = fn_type @ConvertToI32 [concrete]
// CHECK:STDOUT:   %ConvertToI32: %ConvertToI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.668: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %ConvertToU32.type: type = fn_type @ConvertToU32 [concrete]
// CHECK:STDOUT:   %ConvertToU32: %ConvertToU32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.f57: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %ConvertToI64.type: type = fn_type @ConvertToI64 [concrete]
// CHECK:STDOUT:   %ConvertToI64: %ConvertToI64.type = struct_value () [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.b35: type = pattern_type %u64 [concrete]
// CHECK:STDOUT:   %ConvertToU64.type: type = fn_type @ConvertToU64 [concrete]
// CHECK:STDOUT:   %ConvertToU64: %ConvertToU64.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.186: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %ConvertToU8.type: type = fn_type @ConvertToU8 [concrete]
// CHECK:STDOUT:   %ConvertToU8: %ConvertToU8.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %int_97: %i32 = int_value 97 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.668 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT:   %int_98: %u32 = int_value 98 [concrete]
// CHECK:STDOUT:   %c.patt.a8f: %pattern_type.f57 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %.914: Core.CharLiteral = char_value U+10FFFF [concrete]
// CHECK:STDOUT:   %int_1114111: %i64 = int_value 1114111 [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.b35 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %.dc9: Core.CharLiteral = char_value U+0000 [concrete]
// CHECK:STDOUT:   %int_0: %u64 = int_value 0 [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.186 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   %.ae7: Core.CharLiteral = char_value U+0080 [concrete]
// CHECK:STDOUT:   %int_128: %u8 = int_value 128 [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type.186 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   %.ad8: Core.CharLiteral = char_value U+00FF [concrete]
// CHECK:STDOUT:   %int_255: %u8 = int_value 255 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_30.1: %i32 = value_of_initializer @__global_init.%ConvertToI32.call [concrete = constants.%int_97]
// CHECK:STDOUT:   %.loc14_30.2: %i32 = converted @__global_init.%ConvertToI32.call, %.loc14_30.1 [concrete = constants.%int_97]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc14_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.668 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_30.1: %u32 = value_of_initializer @__global_init.%ConvertToU32.call [concrete = constants.%int_98]
// CHECK:STDOUT:   %.loc15_30.2: %u32 = converted @__global_init.%ConvertToU32.call, %.loc15_30.1 [concrete = constants.%int_98]
// CHECK:STDOUT:   %u32: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %b: %u32 = wrapper_binding b, %.loc15_30.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.f57 = value_binding_pattern c [concrete = constants.%c.patt.a8f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_39.1: %i64 = value_of_initializer @__global_init.%ConvertToI64.call [concrete = constants.%int_1114111]
// CHECK:STDOUT:   %.loc16_39.2: %i64 = converted @__global_init.%ConvertToI64.call, %.loc16_39.1 [concrete = constants.%int_1114111]
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %c: %i64 = wrapper_binding c, %.loc16_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.b35 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17_31.1: %u64 = value_of_initializer @__global_init.%ConvertToU64.call [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc17_31.2: %u64 = converted @__global_init.%ConvertToU64.call, %.loc17_31.1 [concrete = constants.%int_0]
// CHECK:STDOUT:   %u64: type = type_literal constants.%u64 [concrete = constants.%u64]
// CHECK:STDOUT:   %d: %u64 = wrapper_binding d, %.loc17_31.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.186 = value_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_33.1: %u8 = value_of_initializer @__global_init.%ConvertToU8.call.loc19 [concrete = constants.%int_128]
// CHECK:STDOUT:   %.loc19_33.2: %u8 = converted @__global_init.%ConvertToU8.call.loc19, %.loc19_33.1 [concrete = constants.%int_128]
// CHECK:STDOUT:   %u8.loc19: type = type_literal constants.%u8 [concrete = constants.%u8]
// CHECK:STDOUT:   %e: %u8 = wrapper_binding e, %.loc19_33.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.186 = value_binding_pattern f [concrete = constants.%f.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_33.1: %u8 = value_of_initializer @__global_init.%ConvertToU8.call.loc20 [concrete = constants.%int_255]
// CHECK:STDOUT:   %.loc20_33.2: %u8 = converted @__global_init.%ConvertToU8.call.loc20, %.loc20_33.1 [concrete = constants.%int_255]
// CHECK:STDOUT:   %u8.loc20: type = type_literal constants.%u8 [concrete = constants.%u8]
// CHECK:STDOUT:   %f: %u8 = wrapper_binding f, %.loc20_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ConvertToI32.ref: %ConvertToI32.type = name_ref ConvertToI32, file.%ConvertToI32.decl [concrete = constants.%ConvertToI32]
// CHECK:STDOUT:   %.loc14: Core.CharLiteral = char_value U+0061 [concrete = constants.%.54f]
// CHECK:STDOUT:   %ConvertToI32.call: init %i32 = call %ConvertToI32.ref(%.loc14) [concrete = constants.%int_97]
// CHECK:STDOUT:   %ConvertToU32.ref: %ConvertToU32.type = name_ref ConvertToU32, file.%ConvertToU32.decl [concrete = constants.%ConvertToU32]
// CHECK:STDOUT:   %.loc15: Core.CharLiteral = char_value U+0062 [concrete = constants.%.711]
// CHECK:STDOUT:   %ConvertToU32.call: init %u32 = call %ConvertToU32.ref(%.loc15) [concrete = constants.%int_98]
// CHECK:STDOUT:   %ConvertToI64.ref: %ConvertToI64.type = name_ref ConvertToI64, file.%ConvertToI64.decl [concrete = constants.%ConvertToI64]
// CHECK:STDOUT:   %.loc16: Core.CharLiteral = char_value U+10FFFF [concrete = constants.%.914]
// CHECK:STDOUT:   %ConvertToI64.call: init %i64 = call %ConvertToI64.ref(%.loc16) [concrete = constants.%int_1114111]
// CHECK:STDOUT:   %ConvertToU64.ref: %ConvertToU64.type = name_ref ConvertToU64, file.%ConvertToU64.decl [concrete = constants.%ConvertToU64]
// CHECK:STDOUT:   %.loc17: Core.CharLiteral = char_value U+0000 [concrete = constants.%.dc9]
// CHECK:STDOUT:   %ConvertToU64.call: init %u64 = call %ConvertToU64.ref(%.loc17) [concrete = constants.%int_0]
// CHECK:STDOUT:   %ConvertToU8.ref.loc19: %ConvertToU8.type = name_ref ConvertToU8, file.%ConvertToU8.decl [concrete = constants.%ConvertToU8]
// CHECK:STDOUT:   %.loc19: Core.CharLiteral = char_value U+0080 [concrete = constants.%.ae7]
// CHECK:STDOUT:   %ConvertToU8.call.loc19: init %u8 = call %ConvertToU8.ref.loc19(%.loc19) [concrete = constants.%int_128]
// CHECK:STDOUT:   %ConvertToU8.ref.loc20: %ConvertToU8.type = name_ref ConvertToU8, file.%ConvertToU8.decl [concrete = constants.%ConvertToU8]
// CHECK:STDOUT:   %.loc20: Core.CharLiteral = char_value U+00FF [concrete = constants.%.ad8]
// CHECK:STDOUT:   %ConvertToU8.call.loc20: init %u8 = call %ConvertToU8.ref.loc20(%.loc20) [concrete = constants.%int_255]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/convert_char_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/convert_char_literal.carbon

// --- success.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn ConvertFromIntLiteral(n: IntLiteral) -> CharLiteral = "int.convert_char_literal";
fn ConvertFromI32(n: i32) -> CharLiteral = "int.convert_char_literal";
fn ConvertFromU32(n: u32) -> CharLiteral = "int.convert_char_literal";

//@dump-sem-ir-begin
let a: CharLiteral = ConvertFromIntLiteral(97);
let b: CharLiteral = ConvertFromI32(98);
let c: CharLiteral = ConvertFromU32(99);
//@dump-sem-ir-end

// --- fail_invalid.carbon

library "[[@TEST_NAME]]";

fn MakeCharLiteral() -> type = "char_literal.make_type";
alias CharLiteral = MakeCharLiteral();
fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

fn ConvertFromIntLiteral(n: IntLiteral) -> CharLiteral = "int.convert_char_literal";

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:22: error: integer value -1 is not a valid Unicode code point [InvalidCharLiteralValue]
// CHECK:STDERR: let a: CharLiteral = ConvertFromIntLiteral(-1);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: CharLiteral = ConvertFromIntLiteral(-1);

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:22: error: integer value 55296 is not a valid Unicode code point [InvalidCharLiteralValue]
// CHECK:STDERR: let b: CharLiteral = ConvertFromIntLiteral(0xD800);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: CharLiteral = ConvertFromIntLiteral(0xD800);

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:22: error: integer value 1114112 is not a valid Unicode code point [InvalidCharLiteralValue]
// CHECK:STDERR: let c: CharLiteral = ConvertFromIntLiteral(0x110000);
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: CharLiteral = ConvertFromIntLiteral(0x110000);

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.8c6: type = pattern_type Core.CharLiteral [concrete]
// CHECK:STDOUT:   %ConvertFromIntLiteral.type: type = fn_type @ConvertFromIntLiteral [concrete]
// CHECK:STDOUT:   %ConvertFromIntLiteral: %ConvertFromIntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ConvertFromI32.type: type = fn_type @ConvertFromI32 [concrete]
// CHECK:STDOUT:   %ConvertFromI32: %ConvertFromI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %ConvertFromU32.type: type = fn_type @ConvertFromU32 [concrete]
// CHECK:STDOUT:   %ConvertFromU32: %ConvertFromU32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_97: Core.IntLiteral = int_value 97 [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %int_98.fb6: Core.IntLiteral = int_value 98 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9f1: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.9f1) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet.9f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dc1: <bound method> = bound_method %int_98.fb6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.571: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6e9: <bound method> = bound_method %int_98.fb6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.571 [concrete]
// CHECK:STDOUT:   %int_98.1e6: %i32 = int_value 98 [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT:   %int_99.6d1: Core.IntLiteral = int_value 99 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.efe: type = facet_type <@ImplicitAs, @ImplicitAs(%u32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b68(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.170: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0a1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.42f, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.957: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.b68(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.9db: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.957 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.134: %ImplicitAs.type.efe = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.0a1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b71: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u32, %ImplicitAs.facet.134) [concrete]
// CHECK:STDOUT:   %.22b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b71, %ImplicitAs.facet.134 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.62f: <bound method> = bound_method %int_99.6d1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.9db [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.4ac: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.9db, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.bfd: <bound method> = bound_method %int_99.6d1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.4ac [concrete]
// CHECK:STDOUT:   %int_99.d4e: %u32 = int_value 99 [concrete]
// CHECK:STDOUT:   %.b9b: Core.CharLiteral = char_value U+0063 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.aa5: @Core.IntLiteral.as.ImplicitAs.impl.b68.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.b68.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.170)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.42f = impl_witness_table (%Core.import_ref.aa5), @Core.IntLiteral.as.ImplicitAs.impl.b68 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.8c6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CharLiteral.ref.loc14: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %.loc14_46.1: Core.CharLiteral = value_of_initializer @__global_init.%ConvertFromIntLiteral.call [concrete = constants.%.54f]
// CHECK:STDOUT:   %.loc14_46.2: Core.CharLiteral = converted @__global_init.%ConvertFromIntLiteral.call, %.loc14_46.1 [concrete = constants.%.54f]
// CHECK:STDOUT:   %a: Core.CharLiteral = value_binding a, %.loc14_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.8c6 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CharLiteral.ref.loc15: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %.loc15_39.1: Core.CharLiteral = value_of_initializer @__global_init.%ConvertFromI32.call [concrete = constants.%.711]
// CHECK:STDOUT:   %.loc15_39.2: Core.CharLiteral = converted @__global_init.%ConvertFromI32.call, %.loc15_39.1 [concrete = constants.%.711]
// CHECK:STDOUT:   %b: Core.CharLiteral = value_binding b, %.loc15_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.8c6 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CharLiteral.ref.loc16: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %.loc16_39.1: Core.CharLiteral = value_of_initializer @__global_init.%ConvertFromU32.call [concrete = constants.%.b9b]
// CHECK:STDOUT:   %.loc16_39.2: Core.CharLiteral = converted @__global_init.%ConvertFromU32.call, %.loc16_39.1 [concrete = constants.%.b9b]
// CHECK:STDOUT:   %c: Core.CharLiteral = value_binding c, %.loc16_39.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ConvertFromIntLiteral.ref: %ConvertFromIntLiteral.type = name_ref ConvertFromIntLiteral, file.%ConvertFromIntLiteral.decl [concrete = constants.%ConvertFromIntLiteral]
// CHECK:STDOUT:   %int_97: Core.IntLiteral = int_value 97 [concrete = constants.%int_97]
// CHECK:STDOUT:   %ConvertFromIntLiteral.call: init Core.CharLiteral = call %ConvertFromIntLiteral.ref(%int_97) [concrete = constants.%.54f]
// CHECK:STDOUT:   %ConvertFromI32.ref: %ConvertFromI32.type = name_ref ConvertFromI32, file.%ConvertFromI32.decl [concrete = constants.%ConvertFromI32]
// CHECK:STDOUT:   %int_98: Core.IntLiteral = int_value 98 [concrete = constants.%int_98.fb6]
// CHECK:STDOUT:   %impl.elem0.loc15: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc15_37.1: <bound method> = bound_method %int_98, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dc1]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.571]
// CHECK:STDOUT:   %bound_method.loc15_37.2: <bound method> = bound_method %int_98, %specific_fn.loc15 [concrete = constants.%bound_method.6e9]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i32 = call %bound_method.loc15_37.2(%int_98) [concrete = constants.%int_98.1e6]
// CHECK:STDOUT:   %.loc15_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_98.1e6]
// CHECK:STDOUT:   %.loc15_37.2: %i32 = converted %int_98, %.loc15_37.1 [concrete = constants.%int_98.1e6]
// CHECK:STDOUT:   %ConvertFromI32.call: init Core.CharLiteral = call %ConvertFromI32.ref(%.loc15_37.2) [concrete = constants.%.711]
// CHECK:STDOUT:   %ConvertFromU32.ref: %ConvertFromU32.type = name_ref ConvertFromU32, file.%ConvertFromU32.decl [concrete = constants.%ConvertFromU32]
// CHECK:STDOUT:   %int_99: Core.IntLiteral = int_value 99 [concrete = constants.%int_99.6d1]
// CHECK:STDOUT:   %impl.elem0.loc16: %.22b = impl_witness_access constants.%ImplicitAs.impl_witness.0a1, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.9db]
// CHECK:STDOUT:   %bound_method.loc16_37.1: <bound method> = bound_method %int_99, %impl.elem0.loc16 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.62f]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.4ac]
// CHECK:STDOUT:   %bound_method.loc16_37.2: <bound method> = bound_method %int_99, %specific_fn.loc16 [concrete = constants.%bound_method.bfd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %u32 = call %bound_method.loc16_37.2(%int_99) [concrete = constants.%int_99.d4e]
// CHECK:STDOUT:   %.loc16_37.1: %u32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_99.d4e]
// CHECK:STDOUT:   %.loc16_37.2: %u32 = converted %int_99, %.loc16_37.1 [concrete = constants.%int_99.d4e]
// CHECK:STDOUT:   %ConvertFromU32.call: init Core.CharLiteral = call %ConvertFromU32.ref(%.loc16_37.2) [concrete = constants.%.b9b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

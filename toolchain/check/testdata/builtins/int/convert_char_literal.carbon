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

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+2]]:22-46: error: integer value -1 is not a valid Unicode code point [InvalidCharLiteralValue]
// CHECK:STDERR:
let a: CharLiteral = ConvertFromIntLiteral(-1);

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+2]]:22-50: error: integer value 55296 is not a valid Unicode code point [InvalidCharLiteralValue]
// CHECK:STDERR:
let b: CharLiteral = ConvertFromIntLiteral(0xD800);

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+2]]:22-52: error: integer value 1114112 is not a valid Unicode code point [InvalidCharLiteralValue]
// CHECK:STDERR:
let c: CharLiteral = ConvertFromIntLiteral(0x110000);

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %a.patt: %pattern_type.8c6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_97: Core.IntLiteral = int_value 97 [concrete]
// CHECK:STDOUT:   %.54f: Core.CharLiteral = char_value U+0061 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.8c6 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %int_98.fb6: Core.IntLiteral = int_value 98 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.1, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.91e: <bound method> = bound_method %int_98.fb6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.384: <bound method> = bound_method %int_98.fb6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8 [concrete]
// CHECK:STDOUT:   %int_98.f19: %i32 = int_value 98 [concrete]
// CHECK:STDOUT:   %.711: Core.CharLiteral = char_value U+0062 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.8c6 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %int_99.6d1: Core.IntLiteral = int_value 99 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.f26: type = facet_type <@ImplicitAs, @ImplicitAs(%u32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.be5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.bd8(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.60e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.be5 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.88f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.79a, @Core.IntLiteral.as.ImplicitAs.impl.bd8(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.56f: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert.2, @Core.IntLiteral.as.ImplicitAs.impl.bd8(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.14a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.56f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.1ad: %ImplicitAs.type.f26 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.88f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3d4: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u32, %ImplicitAs.facet.1ad) [concrete]
// CHECK:STDOUT:   %.8ad: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3d4, %ImplicitAs.facet.1ad [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ae1: <bound method> = bound_method %int_99.6d1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.14a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.2f3: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.14a, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1b9: <bound method> = bound_method %int_99.6d1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.2f3 [concrete]
// CHECK:STDOUT:   %int_99.7cf: %u32 = int_value 99 [concrete]
// CHECK:STDOUT:   %.b9b: Core.CharLiteral = char_value U+0063 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl.6b6 [concrete]
// CHECK:STDOUT:   %Core.import_ref.685: @Core.IntLiteral.as.ImplicitAs.impl.bd8.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.be5) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.bd8.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.60e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.79a = impl_witness_table (%Core.import_ref.685), @Core.IntLiteral.as.ImplicitAs.impl.bd8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc14_46.1: Core.CharLiteral = value_of_initializer @__global_init.%ConvertFromIntLiteral.call [concrete = constants.%.54f]
// CHECK:STDOUT:   %.loc14_46.2: Core.CharLiteral = converted @__global_init.%ConvertFromIntLiteral.call, %.loc14_46.1 [concrete = constants.%.54f]
// CHECK:STDOUT:   %CharLiteral.ref.loc14: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %a: Core.CharLiteral = wrapper_binding a, %.loc14_46.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.8c6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_39.1: Core.CharLiteral = value_of_initializer @__global_init.%ConvertFromI32.call [concrete = constants.%.711]
// CHECK:STDOUT:   %.loc15_39.2: Core.CharLiteral = converted @__global_init.%ConvertFromI32.call, %.loc15_39.1 [concrete = constants.%.711]
// CHECK:STDOUT:   %CharLiteral.ref.loc15: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %b: Core.CharLiteral = wrapper_binding b, %.loc15_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.8c6 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_39.1: Core.CharLiteral = value_of_initializer @__global_init.%ConvertFromU32.call [concrete = constants.%.b9b]
// CHECK:STDOUT:   %.loc16_39.2: Core.CharLiteral = converted @__global_init.%ConvertFromU32.call, %.loc16_39.1 [concrete = constants.%.b9b]
// CHECK:STDOUT:   %CharLiteral.ref.loc16: type = name_ref CharLiteral, %CharLiteral [concrete = Core.CharLiteral]
// CHECK:STDOUT:   %c: Core.CharLiteral = wrapper_binding c, %.loc16_39.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.8c6 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ConvertFromIntLiteral.ref: %ConvertFromIntLiteral.type = name_ref ConvertFromIntLiteral, file.%ConvertFromIntLiteral.decl [concrete = constants.%ConvertFromIntLiteral]
// CHECK:STDOUT:   %int_97: Core.IntLiteral = int_value 97 [concrete = constants.%int_97]
// CHECK:STDOUT:   %ConvertFromIntLiteral.call: init Core.CharLiteral = call %ConvertFromIntLiteral.ref(%int_97) [concrete = constants.%.54f]
// CHECK:STDOUT:   %ConvertFromI32.ref: %ConvertFromI32.type = name_ref ConvertFromI32, file.%ConvertFromI32.decl [concrete = constants.%ConvertFromI32]
// CHECK:STDOUT:   %int_98: Core.IntLiteral = int_value 98 [concrete = constants.%int_98.fb6]
// CHECK:STDOUT:   %impl.elem0.loc15: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc15_37.1: <bound method> = bound_method %int_98, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.91e]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert.1(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.5d8]
// CHECK:STDOUT:   %bound_method.loc15_37.2: <bound method> = bound_method %int_98, %specific_fn.loc15 [concrete = constants.%bound_method.384]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i32 = call %bound_method.loc15_37.2(%int_98) [concrete = constants.%int_98.f19]
// CHECK:STDOUT:   %.loc15_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_98.f19]
// CHECK:STDOUT:   %.loc15_37.2: %i32 = converted %int_98, %.loc15_37.1 [concrete = constants.%int_98.f19]
// CHECK:STDOUT:   %ConvertFromI32.call: init Core.CharLiteral = call %ConvertFromI32.ref(%.loc15_37.2) [concrete = constants.%.711]
// CHECK:STDOUT:   %ConvertFromU32.ref: %ConvertFromU32.type = name_ref ConvertFromU32, file.%ConvertFromU32.decl [concrete = constants.%ConvertFromU32]
// CHECK:STDOUT:   %int_99: Core.IntLiteral = int_value 99 [concrete = constants.%int_99.6d1]
// CHECK:STDOUT:   %impl.elem0.loc16: %.8ad = impl_witness_access constants.%ImplicitAs.impl_witness.88f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.14a]
// CHECK:STDOUT:   %bound_method.loc16_37.1: <bound method> = bound_method %int_99, %impl.elem0.loc16 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ae1]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Core.IntLiteral.as.ImplicitAs.impl.Convert.2(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.2f3]
// CHECK:STDOUT:   %bound_method.loc16_37.2: <bound method> = bound_method %int_99, %specific_fn.loc16 [concrete = constants.%bound_method.1b9]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %u32 = call %bound_method.loc16_37.2(%int_99) [concrete = constants.%int_99.7cf]
// CHECK:STDOUT:   %.loc16_37.1: %u32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_99.7cf]
// CHECK:STDOUT:   %.loc16_37.2: %u32 = converted %int_99, %.loc16_37.1 [concrete = constants.%int_99.7cf]
// CHECK:STDOUT:   %ConvertFromU32.call: init Core.CharLiteral = call %ConvertFromU32.ref(%.loc16_37.2) [concrete = constants.%.b9b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

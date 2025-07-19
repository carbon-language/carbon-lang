// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/method.carbon

interface I {
  fn F[self: Self](other: Self) -> Self;
}

impl i32 as I {
  fn F[self: i32](other: i32) -> i32 = "int.sadd";
}

var arr: array(i32, (1 as i32).(I.F)(2));

// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type.b70 [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.loc16 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type.066: type = fn_type @F.loc20 [concrete]
// CHECK:STDOUT:   %F.9ec: %F.type.066 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %i32, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.1, @As(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.3, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.708: type = fn_type @Convert.4, @ImplicitAs.impl.971(%From) [symbolic]
// CHECK:STDOUT:   %Convert.c68: %Convert.type.708 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.5, @As.impl.686(%To) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.6b4: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @As.impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @As.impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%As.impl_witness.6b4) [concrete]
// CHECK:STDOUT:   %.982: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.c1b: <bound method> = bound_method %int_1.5b8, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.aad: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.082: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn.aad [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.5c7: type = fn_type_with_self_type %F.type.cf0, %I.facet [concrete]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %int_1.5d2, %F.9ec [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.2, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.3, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b6f: <specific function> = specific_function %Convert.956, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn.b6f [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a11: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1d9, @ImplicitAs.impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.4, @ImplicitAs.impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f49: %ImplicitAs.type.2fd = facet_value %i32, (%ImplicitAs.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.f49 [concrete]
// CHECK:STDOUT:   %Convert.bound.2d6: <bound method> = bound_method %int_3.822, %Convert.960 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.8a8: <specific function> = specific_function %Convert.960, @Convert.4(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c6f: <bound method> = bound_method %int_3.822, %Convert.specific_fn.8a8 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.5d8: type = pattern_type %array_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.85c: @ImplicitAs.impl.971.%Convert.type (%Convert.type.708) = import_ref Core//prelude/parts/int, loc20_44, loaded [symbolic = @ImplicitAs.impl.971.%Convert (constants.%Convert.c68)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1d9 = impl_witness_table (%Core.import_ref.85c), @ImplicitAs.impl.971 [concrete]
// CHECK:STDOUT:   %Core.import_ref.78a: @As.impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude/parts/int, loc25_39, loaded [symbolic = @As.impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @As.impl.686 [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @I.impl [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@I.impl.%F.decl), @I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %pattern_type.5d8 = binding_pattern arr [concrete]
// CHECK:STDOUT:     %arr.var_patt: %pattern_type.5d8 = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var %arr.var_patt [concrete]
// CHECK:STDOUT:   %.loc23_40: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc23_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_32.loc23_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %impl.elem0.loc23_24: %.982 = impl_witness_access constants.%As.impl_witness.6b4, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:     %bound_method.loc23_24.1: <bound method> = bound_method %int_1, %impl.elem0.loc23_24 [concrete = constants.%Convert.bound.c1b]
// CHECK:STDOUT:     %specific_fn.loc23_24: <specific function> = specific_function %impl.elem0.loc23_24, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn.aad]
// CHECK:STDOUT:     %bound_method.loc23_24.2: <bound method> = bound_method %int_1, %specific_fn.loc23_24 [concrete = constants.%bound_method.082]
// CHECK:STDOUT:     %int.convert_checked.loc23_24: init %i32 = call %bound_method.loc23_24.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc23_24.1: %i32 = value_of_initializer %int.convert_checked.loc23_24 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc23_24.2: %i32 = converted %int_1, %.loc23_24.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.%assoc0 [concrete = constants.%assoc0.82e]
// CHECK:STDOUT:     %impl.elem0.loc23_31: %.5c7 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%F.9ec]
// CHECK:STDOUT:     %bound_method.loc23_31: <bound method> = bound_method %.loc23_24.2, %impl.elem0.loc23_31 [concrete = constants.%F.bound]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc23_38: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc23_38.1: <bound method> = bound_method %int_2, %impl.elem0.loc23_38 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn.loc23_38: <specific function> = specific_function %impl.elem0.loc23_38, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:     %bound_method.loc23_38.2: <bound method> = bound_method %int_2, %specific_fn.loc23_38 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %int.convert_checked.loc23_38: init %i32 = call %bound_method.loc23_38.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc23_38.1: %i32 = value_of_initializer %int.convert_checked.loc23_38 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc23_38.2: %i32 = converted %int_2, %.loc23_38.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %int.sadd: init %i32 = call %bound_method.loc23_31(%.loc23_24.2, %.loc23_38.2) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %impl.elem0.loc23_39: %.0ea = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:     %bound_method.loc23_39.1: <bound method> = bound_method %int.sadd, %impl.elem0.loc23_39 [concrete = constants.%Convert.bound.2d6]
// CHECK:STDOUT:     %specific_fn.loc23_39: <specific function> = specific_function %impl.elem0.loc23_39, @Convert.4(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:     %bound_method.loc23_39.2: <bound method> = bound_method %int.sadd, %specific_fn.loc23_39 [concrete = constants.%bound_method.c6f]
// CHECK:STDOUT:     %.loc23_39.1: %i32 = value_of_initializer %int.sadd [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc23_39.2: %i32 = converted %int.sadd, %.loc23_39.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %int.convert_checked.loc23_39: init Core.IntLiteral = call %bound_method.loc23_39.2(%.loc23_39.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc23_39.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc23_39 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc23_39.4: Core.IntLiteral = converted %int.sadd, %.loc23_39.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %.loc23_39.4, %i32.loc23_16 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var [concrete = %arr.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.loc16 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %self.patt: @F.loc16.%pattern_type (%pattern_type.6de) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.loc16.%pattern_type (%pattern_type.6de) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: @F.loc16.%pattern_type (%pattern_type.6de) = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: @F.loc16.%pattern_type (%pattern_type.6de) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc16.%pattern_type (%pattern_type.6de) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc16.%pattern_type (%pattern_type.6de) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_36: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:     %Self.as_type.loc16_36: type = facet_access_type %Self.ref.loc16_36 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     %.loc16_36: type = converted %Self.ref.loc16_36, %Self.as_type.loc16_36 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     %self.param: @F.loc16.%Self.as_type.loc16_14.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %.loc16_14.2 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref.loc16_14: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc16_14.2: type = facet_access_type %Self.ref.loc16_14 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc16_14.2: type = converted %Self.ref.loc16_14, %Self.as_type.loc16_14.2 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.loc16.%Self.as_type.loc16_14.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @F.loc16.%Self.as_type.loc16_14.1 (%Self.as_type.b70) = value_param call_param1
// CHECK:STDOUT:     %.loc16_27.1: type = splice_block %.loc16_27.2 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref.loc16_27: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc16_27: type = facet_access_type %Self.ref.loc16_27 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc16_27.2: type = converted %Self.ref.loc16_27, %Self.as_type.loc16_27 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @F.loc16.%Self.as_type.loc16_14.1 (%Self.as_type.b70) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @F.loc16.%Self.as_type.loc16_14.1 (%Self.as_type.b70) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.loc16.%Self.as_type.loc16_14.1 (%Self.as_type.b70) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.82e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @I.impl: %i32 as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.066 = fn_decl @F.loc20 [concrete = constants.%F.9ec] {
// CHECK:STDOUT:     %self.patt: %pattern_type.7ce = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7ce = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.7ce = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.7ce = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc20_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20_34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc20_14: type = splice_block %i32.loc20_14 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc20_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc20_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %i32 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc20_26: type = splice_block %i32.loc20_26 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc20_26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc20_26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %i32 = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc16(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc16_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_14.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.loc16.%Self.as_type.loc16_14.1 (%Self.as_type.b70), %other.param: @F.loc16.%Self.as_type.loc16_14.1 (%Self.as_type.b70)) -> @F.loc16.%Self.as_type.loc16_14.1 (%Self.as_type.b70);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc20(%self.param: %i32, %other.param: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc16(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc16_14.1 => constants.%Self.as_type.b70
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc16(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc16_14.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:

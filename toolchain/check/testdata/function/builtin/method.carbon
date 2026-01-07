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
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.d31: type = symbolic_binding_type Self, 0, %Self.ab9 [symbolic]
// CHECK:STDOUT:   %pattern_type.fa0: type = pattern_type %Self.binding.as_type.d31 [symbolic]
// CHECK:STDOUT:   %.1fc: form = init_form %Self.binding.as_type.d31, call_param2 [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.524: %I.assoc_type = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @i32.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.88a: form = init_form %i32, call_param2 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type: type = fn_type @i32.as.I.impl.F [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F: %i32.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %i32, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.99b: type = fn_type @As.Convert, @As(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8ec: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.29b: %Core.IntLiteral.as.As.impl.Convert.type.8ec = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %.97a: type = fn_type_with_self_type %As.Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.29b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.29b, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.290: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.acc: type = fn_type_with_self_type %I.F.type, %I.facet [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.bound: <bound method> = bound_method %int_1.5d2, %i32.as.I.impl.F [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2ed: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.d29: %Int.as.ImplicitAs.impl.Convert.type.2ed = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %.863: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.640: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ea2, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.240: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.dd4: %Int.as.ImplicitAs.impl.Convert.type.240 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.290: %ImplicitAs.type.139 = facet_value %i32, (%ImplicitAs.impl_witness.640) [concrete]
// CHECK:STDOUT:   %.71e: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.290 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.822, %Int.as.ImplicitAs.impl.Convert.dd4 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.dd4, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.17a: <bound method> = bound_method %int_3.822, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.0bc: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2ed) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.d29)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ea2 = impl_witness_table (%Core.import_ref.0bc), @Int.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   impl_decl @i32.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %pattern_type.5d8 = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:     %arr.var_patt: %pattern_type.5d8 = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var %arr.var_patt [concrete]
// CHECK:STDOUT:   %.loc23_40: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc23_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_32.loc23_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %impl.elem0.loc23_24: %.97a = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:     %bound_method.loc23_24.1: <bound method> = bound_method %int_1, %impl.elem0.loc23_24 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc23_24: <specific function> = specific_function %impl.elem0.loc23_24, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc23_24.2: <bound method> = bound_method %int_1, %specific_fn.loc23_24 [concrete = constants.%bound_method.290]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc23_24.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc23_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc23_24.2: %i32 = converted %int_1, %.loc23_24.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.%assoc0 [concrete = constants.%assoc0.524]
// CHECK:STDOUT:     %impl.elem0.loc23_31: %.acc = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%i32.as.I.impl.F]
// CHECK:STDOUT:     %bound_method.loc23_31: <bound method> = bound_method %.loc23_24.2, %impl.elem0.loc23_31 [concrete = constants.%i32.as.I.impl.F.bound]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc23_38: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc23_38.1: <bound method> = bound_method %int_2, %impl.elem0.loc23_38 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc23_38: <specific function> = specific_function %impl.elem0.loc23_38, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc23_38.2: <bound method> = bound_method %int_2, %specific_fn.loc23_38 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc23_38.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc23_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc23_38.2: %i32 = converted %int_2, %.loc23_38.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %i32.as.I.impl.F.call: init %i32 = call %bound_method.loc23_31(%.loc23_24.2, %.loc23_38.2) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %impl.elem0.loc23_39: %.71e = impl_witness_access constants.%ImplicitAs.impl_witness.640, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.dd4]
// CHECK:STDOUT:     %bound_method.loc23_39.1: <bound method> = bound_method %i32.as.I.impl.F.call, %impl.elem0.loc23_39 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc23_39: <specific function> = specific_function %impl.elem0.loc23_39, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc23_39.2: <bound method> = bound_method %i32.as.I.impl.F.call, %specific_fn.loc23_39 [concrete = constants.%bound_method.17a]
// CHECK:STDOUT:     %.loc23_39.1: %i32 = value_of_initializer %i32.as.I.impl.F.call [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc23_39.2: %i32 = converted %i32.as.I.impl.F.call, %.loc23_39.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc23_39.2(%.loc23_39.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc23_39.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc23_39.4: Core.IntLiteral = converted %i32.as.I.impl.F.call, %.loc23_39.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %.loc23_39.4, %i32.loc23_16 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %array_type = ref_binding arr, %arr.var [concrete = %arr.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %self.patt: @I.F.%pattern_type (%pattern_type.fa0) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @I.F.%pattern_type (%pattern_type.fa0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: @I.F.%pattern_type (%pattern_type.fa0) = value_binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: @I.F.%pattern_type (%pattern_type.fa0) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @I.F.%pattern_type (%pattern_type.fa0) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @I.F.%pattern_type (%pattern_type.fa0) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_36: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:     %Self.as_type.loc16_36: type = facet_access_type %Self.ref.loc16_36 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:     %.loc16_36.2: type = converted %Self.ref.loc16_36, %Self.as_type.loc16_36 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:     %.loc16_36.3: form = init_form %.loc16_36.2, call_param2 [symbolic = %.loc16_36.1 (constants.%.1fc)]
// CHECK:STDOUT:     %self.param: @I.F.%Self.binding.as_type (%Self.binding.as_type.d31) = value_param call_param0
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %.loc16_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)] {
// CHECK:STDOUT:       %Self.ref.loc16_14: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:       %Self.as_type.loc16_14: type = facet_access_type %Self.ref.loc16_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:       %.loc16_14.2: type = converted %Self.ref.loc16_14, %Self.as_type.loc16_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.F.%Self.binding.as_type (%Self.binding.as_type.d31) = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: @I.F.%Self.binding.as_type (%Self.binding.as_type.d31) = value_param call_param1
// CHECK:STDOUT:     %.loc16_27.1: type = splice_block %.loc16_27.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)] {
// CHECK:STDOUT:       %Self.ref.loc16_27: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:       %Self.as_type.loc16_27: type = facet_access_type %Self.ref.loc16_27 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:       %.loc16_27.2: type = converted %Self.ref.loc16_27, %Self.as_type.loc16_27 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @I.F.%Self.binding.as_type (%Self.binding.as_type.d31) = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref @I.F.%Self.binding.as_type (%Self.binding.as_type.d31) = out_param call_param2
// CHECK:STDOUT:     %return: ref @I.F.%Self.binding.as_type (%Self.binding.as_type.d31) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.F.decl [concrete = constants.%assoc0.524]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%I.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.I.impl: %i32 as %I.ref {
// CHECK:STDOUT:   %i32.as.I.impl.F.decl: %i32.as.I.impl.F.type = fn_decl @i32.as.I.impl.F [concrete = constants.%i32.as.I.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.7ce = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7ce = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.7ce = value_binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.7ce = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc20_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20_34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc20_34: form = init_form %i32.loc20_34, call_param2 [concrete = constants.%.88a]
// CHECK:STDOUT:     %self.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc20_14: type = splice_block %i32.loc20_14 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc20_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc20_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %i32 = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc20_26: type = splice_block %i32.loc20_26 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc20_26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc20_26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %i32 = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%i32.as.I.impl.F.decl), @i32.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %i32.as.I.impl.F.decl
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d31)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.fa0)]
// CHECK:STDOUT:   %.loc16_36.1: form = init_form %Self.binding.as_type, call_param2 [symbolic = %.loc16_36.1 (constants.%.1fc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.F.%Self.binding.as_type (%Self.binding.as_type.d31), %other.param: @I.F.%Self.binding.as_type (%Self.binding.as_type.d31)) -> @I.F.%Self.binding.as_type (%Self.binding.as_type.d31);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F(%self.param: %i32, %other.param: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.ab9) {
// CHECK:STDOUT:   %Self => constants.%Self.ab9
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.d31
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fa0
// CHECK:STDOUT:   %.loc16_36.1 => constants.%.1fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:   %.loc16_36.1 => constants.%.88a
// CHECK:STDOUT: }
// CHECK:STDOUT:

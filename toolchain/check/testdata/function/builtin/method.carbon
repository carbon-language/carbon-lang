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
// CHECK:STDOUT:   %Self.as_type.a26: type = facet_access_type %Self.ab9 [symbolic]
// CHECK:STDOUT:   %pattern_type.965: type = pattern_type %Self.as_type.a26 [symbolic]
// CHECK:STDOUT:   %.eb0: Core.Form = init_form %Self.as_type.a26 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.08c: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.ab9) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.705: %I.WithSelf.F.type.08c = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.18e: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @i32.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type: type = fn_type @i32.as.I.impl.F [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F: %i32.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %i32, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.f37: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.b05: %I.WithSelf.F.type.f37 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.38c: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8f0: %Core.IntLiteral.as.As.impl.Convert.type.38c = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e0a: <witness> = impl_witness imports.%As.impl_witness_table.2cc, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e40: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.376: %Core.IntLiteral.as.As.impl.Convert.type.e40 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.e0a) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.920: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.ef8: type = fn_type_with_self_type %As.WithSelf.Convert.type.920, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.376 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.376, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.93d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.131: type = fn_type_with_self_type %I.WithSelf.F.type.f37, %I.facet [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.bound: <bound method> = bound_method %int_1.5d2, %i32.as.I.impl.F [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.7d6: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.81b: %Int.as.ImplicitAs.impl.Convert.type.7d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ac5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.614: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ac5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.614) [concrete]
// CHECK:STDOUT:   %.9db: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b91, %ImplicitAs.facet.614 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.419: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.176: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.577, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.c8f: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.741: %Int.as.ImplicitAs.impl.Convert.type.c8f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e7a: %ImplicitAs.type.139 = facet_value %i32, (%ImplicitAs.impl_witness.176) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.00f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.e7a) [concrete]
// CHECK:STDOUT:   %.42f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.00f, %ImplicitAs.facet.e7a [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.822, %Int.as.ImplicitAs.impl.Convert.741 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.741, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.a56: <bound method> = bound_method %int_3.822, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.5d8: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.462: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.b80: %T.as.DefaultOrUnformed.impl.Op.type.462 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.93f: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.725, @T.as.DefaultOrUnformed.impl(%array_type) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.e33: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%array_type) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.0d4: %T.as.DefaultOrUnformed.impl.Op.type.e33 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %array_type, (%DefaultOrUnformed.impl_witness.93f) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.0d4, @T.as.DefaultOrUnformed.impl.Op(%array_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.d74: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.38c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.8f0)]
// CHECK:STDOUT:   %As.impl_witness_table.2cc = impl_witness_table (%Core.import_ref.d74), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e0f: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.7d6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.81b)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.577 = impl_witness_table (%Core.import_ref.e0f), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.dba: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.462) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.b80)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.725 = impl_witness_table (%Core.import_ref.dba), @T.as.DefaultOrUnformed.impl [concrete]
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
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %pattern_type.5d8 = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:     %arr.var_patt: %pattern_type.5d8 = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var %arr.var_patt [concrete]
// CHECK:STDOUT:   %.loc23_40: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc23_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %i32.loc23_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %impl.elem0.loc23_24: %.ef8 = impl_witness_access constants.%As.impl_witness.e0a, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.376]
// CHECK:STDOUT:     %bound_method.loc23_24.1: <bound method> = bound_method %int_1, %impl.elem0.loc23_24 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc23_24: <specific function> = specific_function %impl.elem0.loc23_24, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc23_24.2: <bound method> = bound_method %int_1, %specific_fn.loc23_24 [concrete = constants.%bound_method.93d]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc23_24.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc23_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc23_24.2: %i32 = converted %int_1, %.loc23_24.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0.18e]
// CHECK:STDOUT:     %impl.elem0.loc23_31: %.131 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%i32.as.I.impl.F]
// CHECK:STDOUT:     %bound_method.loc23_31: <bound method> = bound_method %.loc23_24.2, %impl.elem0.loc23_31 [concrete = constants.%i32.as.I.impl.F.bound]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc23_38: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:     %bound_method.loc23_38.1: <bound method> = bound_method %int_2, %impl.elem0.loc23_38 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc23_38: <specific function> = specific_function %impl.elem0.loc23_38, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc23_38.2: <bound method> = bound_method %int_2, %specific_fn.loc23_38 [concrete = constants.%bound_method.419]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc23_38.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc23_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc23_38.2: %i32 = converted %int_2, %.loc23_38.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %i32.as.I.impl.F.call: init %i32 = call %bound_method.loc23_31(%.loc23_24.2, %.loc23_38.2) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %impl.elem0.loc23_39: %.42f = impl_witness_access constants.%ImplicitAs.impl_witness.176, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.741]
// CHECK:STDOUT:     %bound_method.loc23_39.1: <bound method> = bound_method %i32.as.I.impl.F.call, %impl.elem0.loc23_39 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc23_39: <specific function> = specific_function %impl.elem0.loc23_39, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc23_39.2: <bound method> = bound_method %i32.as.I.impl.F.call, %specific_fn.loc23_39 [concrete = constants.%bound_method.a56]
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
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.08c) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.705)] {
// CHECK:STDOUT:     %self.param_patt: @I.WithSelf.F.%pattern_type (%pattern_type.965) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @I.WithSelf.F.%pattern_type (%pattern_type.965) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %other.param_patt: @I.WithSelf.F.%pattern_type (%pattern_type.965) = value_param_pattern [concrete]
// CHECK:STDOUT:     %other.patt: @I.WithSelf.F.%pattern_type (%pattern_type.965) = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @I.WithSelf.F.%pattern_type (%pattern_type.965) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @I.WithSelf.F.%pattern_type (%pattern_type.965) = return_slot_pattern %return.param_patt, %.loc16_36.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_36: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:     %Self.as_type.loc16_36: type = facet_access_type %Self.ref.loc16_36 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.a26)]
// CHECK:STDOUT:     %.loc16_36.2: type = converted %Self.ref.loc16_36, %Self.as_type.loc16_36 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.a26)]
// CHECK:STDOUT:     %.loc16_36.3: Core.Form = init_form %.loc16_36.2 [symbolic = %.loc16_36.1 (constants.%.eb0)]
// CHECK:STDOUT:     %self.param: @I.WithSelf.F.%Self.as_type.loc16_14.1 (%Self.as_type.a26) = value_param call_param0
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %.loc16_14.2 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.a26)] {
// CHECK:STDOUT:       %Self.ref.loc16_14: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:       %Self.as_type.loc16_14.2: type = facet_access_type %Self.ref.loc16_14 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.a26)]
// CHECK:STDOUT:       %.loc16_14.2: type = converted %Self.ref.loc16_14, %Self.as_type.loc16_14.2 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.a26)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.F.%Self.as_type.loc16_14.1 (%Self.as_type.a26) = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: @I.WithSelf.F.%Self.as_type.loc16_14.1 (%Self.as_type.a26) = value_param call_param1
// CHECK:STDOUT:     %.loc16_27.1: type = splice_block %.loc16_27.2 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.a26)] {
// CHECK:STDOUT:       %Self.ref.loc16_27: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:       %Self.as_type.loc16_27: type = facet_access_type %Self.ref.loc16_27 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.a26)]
// CHECK:STDOUT:       %.loc16_27.2: type = converted %Self.ref.loc16_27, %Self.as_type.loc16_27 [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.a26)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @I.WithSelf.F.%Self.as_type.loc16_14.1 (%Self.as_type.a26) = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref @I.WithSelf.F.%Self.as_type.loc16_14.1 (%Self.as_type.a26) = out_param call_param2
// CHECK:STDOUT:     %return: ref @I.WithSelf.F.%Self.as_type.loc16_14.1 (%Self.as_type.a26) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0.18e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.as.I.impl: %i32 as %I.ref {
// CHECK:STDOUT:   %i32.as.I.impl.F.decl: %i32.as.I.impl.F.type = fn_decl @i32.as.I.impl.F [concrete = constants.%i32.as.I.impl.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7ce = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.7ce = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc20_34 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc20_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc20: Core.Form = init_form %i32.loc20_34 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %self.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc20_14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %self: %i32 = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc20_26: type = type_literal constants.%i32 [concrete = constants.%i32]
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
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.ab9)]
// CHECK:STDOUT:   %Self.as_type.loc16_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_14.1 (constants.%Self.as_type.a26)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_14.1 [symbolic = %pattern_type (constants.%pattern_type.965)]
// CHECK:STDOUT:   %.loc16_36.1: Core.Form = init_form %Self.as_type.loc16_14.1 [symbolic = %.loc16_36.1 (constants.%.eb0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.F.%Self.as_type.loc16_14.1 (%Self.as_type.a26), %other.param: @I.WithSelf.F.%Self.as_type.loc16_14.1 (%Self.as_type.a26)) -> out %return.param: @I.WithSelf.F.%Self.as_type.loc16_14.1 (%Self.as_type.a26);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F(%self.param: %i32, %other.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%array_type, (constants.%DefaultOrUnformed.impl_witness.93f) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc23_41.1: %DefaultOrUnformed.type = converted constants.%array_type, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc23_41.1 [concrete = constants.%array_type]
// CHECK:STDOUT:   %.loc23_41.2: type = converted %.loc23_41.1, %as_type [concrete = constants.%array_type]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.0d4, @T.as.DefaultOrUnformed.impl.Op(constants.%array_type) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %.loc23_1: ref %array_type = splice_block file.%arr.var [concrete = file.%arr.var] {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %array_type to %.loc23_1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%arr.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.ab9
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.08c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.705
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.ab9) {
// CHECK:STDOUT:   %Self => constants.%Self.ab9
// CHECK:STDOUT:   %Self.as_type.loc16_14.1 => constants.%Self.as_type.a26
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.965
// CHECK:STDOUT:   %.loc16_36.1 => constants.%.eb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.f37
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.b05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc16_14.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:   %.loc16_36.1 => constants.%.ff5
// CHECK:STDOUT: }
// CHECK:STDOUT:

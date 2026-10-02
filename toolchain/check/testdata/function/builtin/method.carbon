// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/method.carbon

interface I {
  fn F(self, other: Self) -> Self;
}

impl i32 as I {
  fn F(self: i32, other: i32) -> i32 = "int.sadd";
}

var arr: array(i32, (1 as i32).(I.F)(2));


// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c84: type = facet_access_type %Self.37e [symbolic]
// CHECK:STDOUT:   %pattern_type.c60: type = pattern_type %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %self.param_patt.a33: %pattern_type.c60 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.85e: %pattern_type.c60 = wrapper_binding_pattern self, %self.param_patt.a33 [symbolic]
// CHECK:STDOUT:   %other.param_patt.03c: %pattern_type.c60 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %other.patt.1b9: %pattern_type.c60 = wrapper_binding_pattern other, %other.param_patt.03c [symbolic]
// CHECK:STDOUT:   %.8fc: Core.Form = init_form %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %return.param_patt.434: %pattern_type.c60 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.44a: %pattern_type.c60 = return_slot_pattern %return.param_patt.434, %Self.as_type.c84 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.de2: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.20b: %I.WithSelf.F.type.de2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.e24: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.1f3: <witness> = impl_self_witness %i32, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @i32.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %self.param_patt.3f2: %pattern_type.9d9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c63: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.3f2 [concrete]
// CHECK:STDOUT:   %other.param_patt.379: %pattern_type.9d9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %other.patt.b26: %pattern_type.9d9 = wrapper_binding_pattern other, %other.param_patt.379 [concrete]
// CHECK:STDOUT:   %.55c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.13f: %pattern_type.9d9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a79: %pattern_type.9d9 = return_slot_pattern %return.param_patt.13f, %i32 [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.type: type = fn_type @i32.as.I.impl.F [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F: %i32.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %i32, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.94e: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ec8: %I.WithSelf.F.type.94e = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.d76: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0e2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.203: %Core.IntLiteral.as.As.impl.Convert.type.0e2 = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %As.impl_witness.769: <witness> = impl_witness imports.%As.impl_witness_table.dda, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.594: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6bb: %Core.IntLiteral.as.As.impl.Convert.type.594 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d76 = facet_value Core.IntLiteral, (%As.impl_witness.769) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.7d0: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.f25: type = fn_type_with_self_type %As.WithSelf.Convert.type.7d0, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.6bb [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.6bb, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.2c6: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.8e8: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.d5a: type = fn_type_with_self_type %I.WithSelf.F.type.94e, %I.facet [concrete]
// CHECK:STDOUT:   %i32.as.I.impl.F.bound: <bound method> = bound_method %int_1.8e8, %i32.as.I.impl.F [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.500: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bfc: %Int.as.ImplicitAs.impl.Convert.type.500 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eeb: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.eeb) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet.eeb [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3e0: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.cb7: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.af5: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.803: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6ce, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.de1: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.76f: %Int.as.ImplicitAs.impl.Convert.type.de1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eb0: %ImplicitAs.type.5be = facet_value %i32, (%ImplicitAs.impl_witness.803) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.797: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.eb0) [concrete]
// CHECK:STDOUT:   %.917: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.797, %ImplicitAs.facet.eb0 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.af5, %Int.as.ImplicitAs.impl.Convert.76f [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.76f, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.343: <bound method> = bound_method %int_3.af5, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Array.3c4: type = class_type @Array, @Array(%i32, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %pattern_type.ce9: type = pattern_type %Array.3c4 [concrete]
// CHECK:STDOUT:   %arr.patt: %pattern_type.ce9 = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:   %arr.var_patt: %pattern_type.ce9 = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.5f6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.5c0: %T.as.DefaultOrUnformed.impl.Op.type.5f6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.fc4: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.856, @T.as.DefaultOrUnformed.impl(%Array.3c4) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.9d5: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Array.3c4) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.51b: %T.as.DefaultOrUnformed.impl.Op.type.9d5 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Array.3c4, (%DefaultOrUnformed.impl_witness.fc4) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.51b, @T.as.DefaultOrUnformed.impl.Op(%Array.3c4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.501: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0e2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.203)]
// CHECK:STDOUT:   %As.impl_witness_table.dda = impl_witness_table (%Core.import_ref.501), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5b7: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.500) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.bfc)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ce = impl_witness_table (%Core.import_ref.5b7), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.002: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.5f6) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.5c0)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.856 = impl_witness_table (%Core.import_ref.002), @T.as.DefaultOrUnformed.impl [concrete]
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
// CHECK:STDOUT:   %.loc19: <witness> = impl_self_witness @i32.as.I.impl.%i32, @I [concrete = constants.%.1f3]
// CHECK:STDOUT:   %arr.var: ref %Array.3c4 = var_storage %arr.var_patt [concrete]
// CHECK:STDOUT:   %.loc23_40: type = splice_block %Array [concrete = constants.%Array.3c4] {
// CHECK:STDOUT:     %i32.loc23_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %i32.loc23_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %impl.elem0.loc23_24: %.f25 = impl_witness_access constants.%As.impl_witness.769, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.6bb]
// CHECK:STDOUT:     %bound_method.loc23_24.1: <bound method> = bound_method %int_1, %impl.elem0.loc23_24 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc23_24: <specific function> = specific_function %impl.elem0.loc23_24, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc23_24.2: <bound method> = bound_method %int_1, %specific_fn.loc23_24 [concrete = constants.%bound_method.2c6]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc23_24.2(%int_1) [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:     %.loc23_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:     %.loc23_24.2: %i32 = converted %int_1, %.loc23_24.1 [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:     %I.ref: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0.e24]
// CHECK:STDOUT:     %impl.elem0.loc23_31: %.d5a = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%i32.as.I.impl.F]
// CHECK:STDOUT:     %bound_method.loc23_31: <bound method> = bound_method %.loc23_24.2, %impl.elem0.loc23_31 [concrete = constants.%i32.as.I.impl.F.bound]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc23_38: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc23_38.1: <bound method> = bound_method %int_2, %impl.elem0.loc23_38 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc23_38: <specific function> = specific_function %impl.elem0.loc23_38, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc23_38.2: <bound method> = bound_method %int_2, %specific_fn.loc23_38 [concrete = constants.%bound_method.3e0]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc23_38.2(%int_2) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %.loc23_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %.loc23_38.2: %i32 = converted %int_2, %.loc23_38.1 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %i32.as.I.impl.F.call: init %i32 = call %bound_method.loc23_31(%.loc23_24.2, %.loc23_38.2) [concrete = constants.%int_3.af5]
// CHECK:STDOUT:     %impl.elem0.loc23_39: %.917 = impl_witness_access constants.%ImplicitAs.impl_witness.803, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.76f]
// CHECK:STDOUT:     %bound_method.loc23_39.1: <bound method> = bound_method %i32.as.I.impl.F.call, %impl.elem0.loc23_39 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc23_39: <specific function> = specific_function %impl.elem0.loc23_39, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc23_39.2: <bound method> = bound_method %i32.as.I.impl.F.call, %specific_fn.loc23_39 [concrete = constants.%bound_method.343]
// CHECK:STDOUT:     %.loc23_39.1: %i32 = value_of_initializer %i32.as.I.impl.F.call [concrete = constants.%int_3.af5]
// CHECK:STDOUT:     %.loc23_39.2: %i32 = converted %i32.as.I.impl.F.call, %.loc23_39.1 [concrete = constants.%int_3.af5]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc23_39.2(%.loc23_39.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc23_39.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc23_39.4: Core.IntLiteral = converted %i32.as.I.impl.F.call, %.loc23_39.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_3.1ba) [concrete = constants.%Array.3c4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %Array.3c4 = wrapper_binding arr, %arr.var [concrete = %arr.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %pattern_type.ce9 = ref_binding_pattern arr [concrete = constants.%arr.patt]
// CHECK:STDOUT:     %arr.var_patt: %pattern_type.ce9 = var_pattern %arr.patt [concrete = constants.%arr.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.de2) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.20b)] {
// CHECK:STDOUT:     %self.param_patt.loc16_8.1: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %self.param_patt.loc16_8.2 (constants.%self.param_patt.a33)]
// CHECK:STDOUT:     %self.patt.loc16_8.1: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = wrapper_binding_pattern self, %self.param_patt.loc16_8.1 [symbolic = %self.patt.loc16_8.2 (constants.%self.patt.85e)]
// CHECK:STDOUT:     %other.param_patt.loc16_19.1: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %other.param_patt.loc16_19.2 (constants.%other.param_patt.03c)]
// CHECK:STDOUT:     %other.patt.loc16_19.1: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = wrapper_binding_pattern other, %other.param_patt.loc16_19.1 [symbolic = %other.patt.loc16_19.2 (constants.%other.patt.1b9)]
// CHECK:STDOUT:     %return.param_patt.loc16_30.1: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = out_param_pattern [symbolic = %return.param_patt.loc16_30.2 (constants.%return.param_patt.434)]
// CHECK:STDOUT:     %return.patt.loc16_27.1: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = return_slot_pattern %return.param_patt.loc16_30.1, %.loc16_30.2 [symbolic = %return.patt.loc16_27.2 (constants.%return.patt.44a)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_30: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:     %Self.as_type.loc16_30: type = facet_access_type %Self.ref.loc16_30 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:     %.loc16_30.2: type = converted %Self.ref.loc16_30, %Self.as_type.loc16_30 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:     %.loc16_30.3: Core.Form = init_form %.loc16_30.2 [symbolic = %.loc16_30.1 (constants.%.8fc)]
// CHECK:STDOUT:     %self.param: @I.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.c84) = value_param call_param0
// CHECK:STDOUT:     %.loc16_8.1: type = splice_block %.loc16_8.2 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.c84)] {
// CHECK:STDOUT:       %Self.ref.loc16_8: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:       %Self.as_type.loc16_8.2: type = facet_access_type %Self.ref.loc16_8 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:       %.loc16_8.2: type = converted %Self.ref.loc16_8, %Self.as_type.loc16_8.2 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @I.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.c84) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %other.param: @I.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.c84) = value_param call_param1
// CHECK:STDOUT:     %.loc16_21.1: type = splice_block %.loc16_21.2 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.c84)] {
// CHECK:STDOUT:       %Self.ref.loc16_21: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:       %Self.as_type.loc16_21: type = facet_access_type %Self.ref.loc16_21 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:       %.loc16_21.2: type = converted %Self.ref.loc16_21, %Self.as_type.loc16_21 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @I.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.c84) = wrapper_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref @I.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.c84) = out_param call_param2
// CHECK:STDOUT:     %return: ref @I.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.c84) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0.e24]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = value_param_pattern [concrete = constants.%self.param_patt.3f2]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c63]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.9d9 = value_param_pattern [concrete = constants.%other.param_patt.379]
// CHECK:STDOUT:     %other.patt: %pattern_type.9d9 = wrapper_binding_pattern other, %other.param_patt [concrete = constants.%other.patt.b26]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32.loc20_34 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc20_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc20: Core.Form = init_form %i32.loc20_34 [concrete = constants.%.55c]
// CHECK:STDOUT:     %self.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc20_14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %self: %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %other.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc20_26: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %other: %i32 = wrapper_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%i32.as.I.impl.F.decl), @i32.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %i32.as.I.impl.F.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.37e)]
// CHECK:STDOUT:   %Self.as_type.loc16_8.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type.c84)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_8.1 [symbolic = %pattern_type (constants.%pattern_type.c60)]
// CHECK:STDOUT:   %self.param_patt.loc16_8.2: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %self.param_patt.loc16_8.2 (constants.%self.param_patt.a33)]
// CHECK:STDOUT:   %self.patt.loc16_8.2: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = wrapper_binding_pattern self, %self.param_patt.loc16_8.2 [symbolic = %self.patt.loc16_8.2 (constants.%self.patt.85e)]
// CHECK:STDOUT:   %other.param_patt.loc16_19.2: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = value_param_pattern [symbolic = %other.param_patt.loc16_19.2 (constants.%other.param_patt.03c)]
// CHECK:STDOUT:   %other.patt.loc16_19.2: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = wrapper_binding_pattern other, %other.param_patt.loc16_19.2 [symbolic = %other.patt.loc16_19.2 (constants.%other.patt.1b9)]
// CHECK:STDOUT:   %.loc16_30.1: Core.Form = init_form %Self.as_type.loc16_8.1 [symbolic = %.loc16_30.1 (constants.%.8fc)]
// CHECK:STDOUT:   %return.param_patt.loc16_30.2: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = out_param_pattern [symbolic = %return.param_patt.loc16_30.2 (constants.%return.param_patt.434)]
// CHECK:STDOUT:   %return.patt.loc16_27.2: @I.WithSelf.F.%pattern_type (%pattern_type.c60) = return_slot_pattern %return.param_patt.loc16_30.2, %Self.as_type.loc16_8.1 [symbolic = %return.patt.loc16_27.2 (constants.%return.patt.44a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.c84), %other.param: @I.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.c84)) -> out %return.param: @I.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type.c84);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.as.I.impl.F(%self.param: %i32, %other.param: %i32) -> out %return.param: %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Array.3c4, (constants.%DefaultOrUnformed.impl_witness.fc4) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc23_41.1: %DefaultOrUnformed.type = converted constants.%Array.3c4, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc23_41.1 [concrete = constants.%Array.3c4]
// CHECK:STDOUT:   %.loc23_41.2: type = converted %.loc23_41.1, %as_type [concrete = constants.%Array.3c4]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.51b, @T.as.DefaultOrUnformed.impl.Op(constants.%Array.3c4) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn]
// CHECK:STDOUT:   %.loc23_1: ref %Array.3c4 = splice_block file.%arr.var [concrete = file.%arr.var] {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %Array.3c4 to %.loc23_1 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign file.%arr.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.de2
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.20b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.37e) {
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %Self.as_type.loc16_8.1 => constants.%Self.as_type.c84
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c60
// CHECK:STDOUT:   %self.param_patt.loc16_8.2 => constants.%self.param_patt.a33
// CHECK:STDOUT:   %self.patt.loc16_8.2 => constants.%self.patt.85e
// CHECK:STDOUT:   %other.param_patt.loc16_19.2 => constants.%other.param_patt.03c
// CHECK:STDOUT:   %other.patt.loc16_19.2 => constants.%other.patt.1b9
// CHECK:STDOUT:   %.loc16_30.1 => constants.%.8fc
// CHECK:STDOUT:   %return.param_patt.loc16_30.2 => constants.%return.param_patt.434
// CHECK:STDOUT:   %return.patt.loc16_27.2 => constants.%return.patt.44a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.94e
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ec8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc16_8.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9d9
// CHECK:STDOUT:   %self.param_patt.loc16_8.2 => constants.%self.param_patt.3f2
// CHECK:STDOUT:   %self.patt.loc16_8.2 => constants.%self.patt.c63
// CHECK:STDOUT:   %other.param_patt.loc16_19.2 => constants.%other.param_patt.379
// CHECK:STDOUT:   %other.patt.loc16_19.2 => constants.%other.patt.b26
// CHECK:STDOUT:   %.loc16_30.1 => constants.%.55c
// CHECK:STDOUT:   %return.param_patt.loc16_30.2 => constants.%return.param_patt.13f
// CHECK:STDOUT:   %return.patt.loc16_27.2 => constants.%return.patt.a79
// CHECK:STDOUT: }
// CHECK:STDOUT:

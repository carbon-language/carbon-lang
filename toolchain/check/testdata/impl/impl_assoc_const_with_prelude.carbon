// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/impl_assoc_const_with_prelude.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/impl_assoc_const_with_prelude.carbon

// --- same_non_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! {.a: bool, .b: (i32, i32)};
}
impl () as I where .X = {.a = true, .b = (1, 2)} and .X = {.a = not false, .b = (3 - 2, 4 / 2)} {}

// --- fail_two_different_non_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! {.a: bool, .b: (i32, i32)};
}
// CHECK:STDERR: fail_two_different_non_type.carbon:[[@LINE+4]]:12: error: associated constant X given two different values `{.a = true, .b = (1, 2)}` and `{.a = false, .b = (3, 4)}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl () as I where .X = {.a = true, .b = (1, 2)} and .X = {.a = false, .b = (3, 4)} {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I where .X = {.a = true, .b = (1, 2)} and .X = {.a = false, .b = (3, 4)} {}

// CHECK:STDOUT: --- same_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.fe2: type = struct_type {.a: bool, .b: %tuple.type.d07} [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.e7f: %I.assoc_type = assoc_entity element0, @I.%X [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, (%I.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.a.b.fe2 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.aa4: type = struct_type {.a: bool, .b: %tuple.type.f94} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Div.type: type = facet_type <@Div> [concrete]
// CHECK:STDOUT:   %Sub.type: type = facet_type <@Sub> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.fe2 = struct_value (%true, %tuple) [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Op.type.111: type = fn_type @Op.23 [concrete]
// CHECK:STDOUT:   %Sub.impl_witness.37d: <witness> = impl_witness imports.%Sub.impl_witness_table.d7b [concrete]
// CHECK:STDOUT:   %Sub.facet: %Sub.type = facet_value Core.IntLiteral, (%Sub.impl_witness.37d) [concrete]
// CHECK:STDOUT:   %.07b: type = fn_type_with_self_type %Op.type.111, %Sub.facet [concrete]
// CHECK:STDOUT:   %Op.type.2b9: type = fn_type @Op.24 [concrete]
// CHECK:STDOUT:   %Op.667: %Op.type.2b9 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound.645: <bound method> = bound_method %int_3, %Op.667 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Op.type.784: type = fn_type @Op.25 [concrete]
// CHECK:STDOUT:   %Div.impl_witness.ec4: <witness> = impl_witness imports.%Div.impl_witness_table.381 [concrete]
// CHECK:STDOUT:   %Div.facet: %Div.type = facet_value Core.IntLiteral, (%Div.impl_witness.ec4) [concrete]
// CHECK:STDOUT:   %.d13: type = fn_type_with_self_type %Op.type.784, %Div.facet [concrete]
// CHECK:STDOUT:   %Op.type.c38: type = fn_type @Op.26 [concrete]
// CHECK:STDOUT:   %Op.553: %Op.type.c38 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound.2e7: <bound method> = bound_method %int_4, %Op.553 [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %struct> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Sub = %Core.Sub
// CHECK:STDOUT:     .Div = %Core.Div
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Sub: type = import_ref Core//prelude/operators/arithmetic, Sub, loaded [concrete = constants.%Sub.type]
// CHECK:STDOUT:   %Core.import_ref.6c4: %Op.type.2b9 = import_ref Core//prelude/operators/arithmetic, loc98_42, loaded [concrete = constants.%Op.667]
// CHECK:STDOUT:   %Sub.impl_witness_table.d7b = impl_witness_table (%Core.import_ref.6c4), @impl.bf8 [concrete]
// CHECK:STDOUT:   %Core.Div: type = import_ref Core//prelude/operators/arithmetic, Div, loaded [concrete = constants.%Div.type]
// CHECK:STDOUT:   %Core.import_ref.dfd: %Op.type.c38 = import_ref Core//prelude/operators/arithmetic, loc82_42, loaded [concrete = constants.%Op.553]
// CHECK:STDOUT:   %Div.impl_witness_table.381 = impl_witness_table (%Core.import_ref.dfd), @impl.f0e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.9be [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc6_20: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref.loc6_20: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.e7f]
// CHECK:STDOUT:     %.Self.as_type.loc6_20: type = facet_access_type %.Self.ref.loc6_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc6_20: type = converted %.Self.ref.loc6_20, %.Self.as_type.loc6_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0.loc6_20: %struct_type.a.b.fe2 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2.loc6_46: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc6_47.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2.loc6_46)
// CHECK:STDOUT:     %.loc6_48.1: %struct_type.a.b.aa4 = struct_literal (%true, %.loc6_47.1)
// CHECK:STDOUT:     %impl.elem0.loc6_47.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc6_47.1: <bound method> = bound_method %int_1, %impl.elem0.loc6_47.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:     %specific_fn.loc6_47.1: <specific function> = specific_function %impl.elem0.loc6_47.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_47.2: <bound method> = bound_method %int_1, %specific_fn.loc6_47.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:     %int.convert_checked.loc6_47.1: init %i32 = call %bound_method.loc6_47.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc6_47.2: %i32 = value_of_initializer %int.convert_checked.loc6_47.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc6_47.3: %i32 = converted %int_1, %.loc6_47.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc6_47.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc6_47.3: <bound method> = bound_method %int_2.loc6_46, %impl.elem0.loc6_47.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn.loc6_47.2: <specific function> = specific_function %impl.elem0.loc6_47.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_47.4: <bound method> = bound_method %int_2.loc6_46, %specific_fn.loc6_47.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %int.convert_checked.loc6_47.2: init %i32 = call %bound_method.loc6_47.4(%int_2.loc6_46) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc6_47.4: %i32 = value_of_initializer %int.convert_checked.loc6_47.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc6_47.5: %i32 = converted %int_2.loc6_46, %.loc6_47.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %tuple.loc6_47: %tuple.type.d07 = tuple_value (%.loc6_47.3, %.loc6_47.5) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc6_48.2: %tuple.type.d07 = converted %.loc6_47.1, %tuple.loc6_47 [concrete = constants.%tuple]
// CHECK:STDOUT:     %struct.loc6_48: %struct_type.a.b.fe2 = struct_value (%true, %.loc6_48.2) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc6_48.3: %struct_type.a.b.fe2 = converted %.loc6_48.1, %struct.loc6_48 [concrete = constants.%struct]
// CHECK:STDOUT:     %.Self.ref.loc6_54: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref.loc6_54: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.e7f]
// CHECK:STDOUT:     %.Self.as_type.loc6_54: type = facet_access_type %.Self.ref.loc6_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc6_54: type = converted %.Self.ref.loc6_54, %.Self.as_type.loc6_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0.loc6_54: %struct_type.a.b.fe2 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %.loc6_65: bool = not %false [concrete = constants.%true]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %int_2.loc6_86: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc6_84: %.07b = impl_witness_access constants.%Sub.impl_witness.37d, element0 [concrete = constants.%Op.667]
// CHECK:STDOUT:     %bound_method.loc6_84: <bound method> = bound_method %int_3, %impl.elem0.loc6_84 [concrete = constants.%Op.bound.645]
// CHECK:STDOUT:     %int.ssub: init Core.IntLiteral = call %bound_method.loc6_84(%int_3, %int_2.loc6_86) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %int_2.loc6_93: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc6_91: %.d13 = impl_witness_access constants.%Div.impl_witness.ec4, element0 [concrete = constants.%Op.553]
// CHECK:STDOUT:     %bound_method.loc6_91: <bound method> = bound_method %int_4, %impl.elem0.loc6_91 [concrete = constants.%Op.bound.2e7]
// CHECK:STDOUT:     %int.sdiv: init Core.IntLiteral = call %bound_method.loc6_91(%int_4, %int_2.loc6_93) [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc6_94.1: %tuple.type.f94 = tuple_literal (%int.ssub, %int.sdiv)
// CHECK:STDOUT:     %.loc6_95.1: %struct_type.a.b.aa4 = struct_literal (%.loc6_65, %.loc6_94.1)
// CHECK:STDOUT:     %impl.elem0.loc6_94.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc6_94.1: <bound method> = bound_method %int.ssub, %impl.elem0.loc6_94.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:     %specific_fn.loc6_94.1: <specific function> = specific_function %impl.elem0.loc6_94.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_94.2: <bound method> = bound_method %int.ssub, %specific_fn.loc6_94.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:     %.loc6_84.1: Core.IntLiteral = value_of_initializer %int.ssub [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc6_84.2: Core.IntLiteral = converted %int.ssub, %.loc6_84.1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int.convert_checked.loc6_94.1: init %i32 = call %bound_method.loc6_94.2(%.loc6_84.2) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc6_94.2: %i32 = value_of_initializer %int.convert_checked.loc6_94.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc6_94.3: %i32 = converted %int.ssub, %.loc6_94.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc6_94.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc6_94.3: <bound method> = bound_method %int.sdiv, %impl.elem0.loc6_94.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn.loc6_94.2: <specific function> = specific_function %impl.elem0.loc6_94.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_94.4: <bound method> = bound_method %int.sdiv, %specific_fn.loc6_94.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %.loc6_91.1: Core.IntLiteral = value_of_initializer %int.sdiv [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc6_91.2: Core.IntLiteral = converted %int.sdiv, %.loc6_91.1 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %int.convert_checked.loc6_94.2: init %i32 = call %bound_method.loc6_94.4(%.loc6_91.2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc6_94.4: %i32 = value_of_initializer %int.convert_checked.loc6_94.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc6_94.5: %i32 = converted %int.sdiv, %.loc6_94.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %tuple.loc6_94: %tuple.type.d07 = tuple_value (%.loc6_94.3, %.loc6_94.5) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc6_95.2: %tuple.type.d07 = converted %.loc6_94.1, %tuple.loc6_94 [concrete = constants.%tuple]
// CHECK:STDOUT:     %struct.loc6_95: %struct_type.a.b.fe2 = struct_value (%.loc6_65, %.loc6_95.2) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc6_95.3: %struct_type.a.b.fe2 = converted %.loc6_95.1, %struct.loc6_95 [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc6_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6_20, %.loc6_48.3
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6_54, %.loc6_95.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @impl.9be [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %struct_type.a.b.fe2 = impl_witness_assoc_constant constants.%struct [concrete = constants.%struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %X: %struct_type.a.b.fe2 = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%X [concrete = constants.%assoc0.e7f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (%X)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const X:! %struct_type.a.b.fe2;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9be: %.loc6_7.2 as %.loc6_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.fe2: type = struct_type {.a: bool, .b: %tuple.type.d07} [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.e7f: %I.assoc_type = assoc_entity element0, @I.%X [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, (%I.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.a.b.fe2 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.aa4: type = struct_type {.a: bool, .b: %tuple.type.f94} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %struct.682: %struct_type.a.b.fe2 = struct_value (%true, %tuple.21c) [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.ffd: %tuple.type.d07 = tuple_value (%int_3.822, %int_4.940) [concrete]
// CHECK:STDOUT:   %struct.68c: %struct_type.a.b.fe2 = struct_value (%false, %tuple.ffd) [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %struct.682 and %impl.elem0 = %struct.68c> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.dd8 [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc10_20: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref.loc10_20: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.e7f]
// CHECK:STDOUT:     %.Self.as_type.loc10_20: type = facet_access_type %.Self.ref.loc10_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_20: type = converted %.Self.ref.loc10_20, %.Self.as_type.loc10_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0.loc10_20: %struct_type.a.b.fe2 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc10_47.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:     %.loc10_48.1: %struct_type.a.b.aa4 = struct_literal (%true, %.loc10_47.1)
// CHECK:STDOUT:     %impl.elem0.loc10_47.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc10_47.1: <bound method> = bound_method %int_1, %impl.elem0.loc10_47.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:     %specific_fn.loc10_47.1: <specific function> = specific_function %impl.elem0.loc10_47.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_47.2: <bound method> = bound_method %int_1, %specific_fn.loc10_47.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:     %int.convert_checked.loc10_47.1: init %i32 = call %bound_method.loc10_47.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc10_47.2: %i32 = value_of_initializer %int.convert_checked.loc10_47.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc10_47.3: %i32 = converted %int_1, %.loc10_47.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc10_47.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc10_47.3: <bound method> = bound_method %int_2, %impl.elem0.loc10_47.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn.loc10_47.2: <specific function> = specific_function %impl.elem0.loc10_47.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_47.4: <bound method> = bound_method %int_2, %specific_fn.loc10_47.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %int.convert_checked.loc10_47.2: init %i32 = call %bound_method.loc10_47.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc10_47.4: %i32 = value_of_initializer %int.convert_checked.loc10_47.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc10_47.5: %i32 = converted %int_2, %.loc10_47.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %tuple.loc10_47: %tuple.type.d07 = tuple_value (%.loc10_47.3, %.loc10_47.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %.loc10_48.2: %tuple.type.d07 = converted %.loc10_47.1, %tuple.loc10_47 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %struct.loc10_48: %struct_type.a.b.fe2 = struct_value (%true, %.loc10_48.2) [concrete = constants.%struct.682]
// CHECK:STDOUT:     %.loc10_48.3: %struct_type.a.b.fe2 = converted %.loc10_48.1, %struct.loc10_48 [concrete = constants.%struct.682]
// CHECK:STDOUT:     %.Self.ref.loc10_54: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref.loc10_54: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.e7f]
// CHECK:STDOUT:     %.Self.as_type.loc10_54: type = facet_access_type %.Self.ref.loc10_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_54: type = converted %.Self.ref.loc10_54, %.Self.as_type.loc10_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0.loc10_54: %struct_type.a.b.fe2 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %.loc10_82.1: %tuple.type.f94 = tuple_literal (%int_3, %int_4)
// CHECK:STDOUT:     %.loc10_83.1: %struct_type.a.b.aa4 = struct_literal (%false, %.loc10_82.1)
// CHECK:STDOUT:     %impl.elem0.loc10_82.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc10_82.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_82.1 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:     %specific_fn.loc10_82.1: <specific function> = specific_function %impl.elem0.loc10_82.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_82.2: <bound method> = bound_method %int_3, %specific_fn.loc10_82.1 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:     %int.convert_checked.loc10_82.1: init %i32 = call %bound_method.loc10_82.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc10_82.2: %i32 = value_of_initializer %int.convert_checked.loc10_82.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc10_82.3: %i32 = converted %int_3, %.loc10_82.2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %impl.elem0.loc10_82.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc10_82.3: <bound method> = bound_method %int_4, %impl.elem0.loc10_82.2 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:     %specific_fn.loc10_82.2: <specific function> = specific_function %impl.elem0.loc10_82.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_82.4: <bound method> = bound_method %int_4, %specific_fn.loc10_82.2 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:     %int.convert_checked.loc10_82.2: init %i32 = call %bound_method.loc10_82.4(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc10_82.4: %i32 = value_of_initializer %int.convert_checked.loc10_82.2 [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc10_82.5: %i32 = converted %int_4, %.loc10_82.4 [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %tuple.loc10_82: %tuple.type.d07 = tuple_value (%.loc10_82.3, %.loc10_82.5) [concrete = constants.%tuple.ffd]
// CHECK:STDOUT:     %.loc10_83.2: %tuple.type.d07 = converted %.loc10_82.1, %tuple.loc10_82 [concrete = constants.%tuple.ffd]
// CHECK:STDOUT:     %struct.loc10_83: %struct_type.a.b.fe2 = struct_value (%false, %.loc10_83.2) [concrete = constants.%struct.68c]
// CHECK:STDOUT:     %.loc10_83.3: %struct_type.a.b.fe2 = converted %.loc10_83.1, %struct.loc10_83 [concrete = constants.%struct.68c]
// CHECK:STDOUT:     %.loc10_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_20, %.loc10_48.3
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_54, %.loc10_83.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @impl.dd8 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %struct_type.a.b.fe2 = impl_witness_assoc_constant constants.%struct.682 [concrete = constants.%struct.682]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %X: %struct_type.a.b.fe2 = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%X [concrete = constants.%assoc0.e7f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (%X)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const X:! %struct_type.a.b.fe2;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.dd8: %.loc10_7.2 as %.loc10_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%I.facet) {}
// CHECK:STDOUT:

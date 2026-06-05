// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
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
// CHECK:STDERR: fail_two_different_non_type.carbon:[[@LINE+4]]:12: error: associated constant `.(I.X)` given two different values `{.a = true, .b = (1, 2)}` and `{.a = false, .b = (3, 4)}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl () as I where .X = {.a = true, .b = (1, 2)} and .X = {.a = false, .b = (3, 4)} {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I where .X = {.a = true, .b = (1, 2)} and .X = {.a = false, .b = (3, 4)} {}

// CHECK:STDOUT: --- same_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.e55: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.486: type = struct_type {.a: bool, .b: %tuple.type.e55} [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.6fc: %I.assoc_type = assoc_entity element0, @I.WithSelf.%X [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.9e9: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.9e9 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.9e9, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.049: %struct_type.a.b.486 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.aa4: type = struct_type {.a: bool, .b: %tuple.type.f94} [concrete]
// CHECK:STDOUT:   %struct.e57: %struct_type.a.b.aa4 = struct_value (%true, %tuple.ad8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.102: %tuple.type.e55 = tuple_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %struct.247: %struct_type.a.b.486 = struct_value (%true, %tuple.102) [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %SubWith.type.f8d: type = generic_interface_type @SubWith [concrete]
// CHECK:STDOUT:   %SubWith.generic: %SubWith.type.f8d = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.type.4c2: type = facet_type <@SubWith, @SubWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness: <witness> = impl_witness imports.%SubWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %SubWith.facet: %SubWith.type.4c2 = facet_value Core.IntLiteral, (%SubWith.impl_witness) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.663: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(Core.IntLiteral, %SubWith.facet) [concrete]
// CHECK:STDOUT:   %.ec4: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.663, %SubWith.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.SubWith.impl.Op.type: type = fn_type @Core.IntLiteral.as.SubWith.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.SubWith.impl.Op: %Core.IntLiteral.as.SubWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.SubWith.impl.Op.bound: <bound method> = bound_method %int_3, %Core.IntLiteral.as.SubWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %DivWith.type.8b7: type = generic_interface_type @DivWith [concrete]
// CHECK:STDOUT:   %DivWith.generic: %DivWith.type.8b7 = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.type.e2a: type = facet_type <@DivWith, @DivWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness: <witness> = impl_witness imports.%DivWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %DivWith.facet: %DivWith.type.e2a = facet_value Core.IntLiteral, (%DivWith.impl_witness) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.9c3: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(Core.IntLiteral, %DivWith.facet) [concrete]
// CHECK:STDOUT:   %.5ce: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.9c3, %DivWith.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.DivWith.impl.Op.type: type = fn_type @Core.IntLiteral.as.DivWith.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.DivWith.impl.Op: %Core.IntLiteral.as.DivWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.DivWith.impl.Op.bound: <bound method> = bound_method %int_4, %Core.IntLiteral.as.DivWith.impl.Op [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.049 = %struct.247> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_tuple.type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_tuple.type, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .SubWith = %Core.SubWith
// CHECK:STDOUT:     .DivWith = %Core.DivWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.SubWith: %SubWith.type.f8d = import_ref Core//prelude/operators/arithmetic, SubWith, loaded [concrete = constants.%SubWith.generic]
// CHECK:STDOUT:   %Core.import_ref.abd97d.1 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9c3475.1: %Core.IntLiteral.as.SubWith.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.SubWith.impl.Op]
// CHECK:STDOUT:   %SubWith.impl_witness_table = impl_witness_table (%Core.import_ref.abd97d.1, %Core.import_ref.9c3475.1), @Core.IntLiteral.as.SubWith.impl [concrete]
// CHECK:STDOUT:   %Core.DivWith: %DivWith.type.8b7 = import_ref Core//prelude/operators/arithmetic, DivWith, loaded [concrete = constants.%DivWith.generic]
// CHECK:STDOUT:   %Core.import_ref.abd97d.2 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9c3475.2: %Core.IntLiteral.as.DivWith.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.DivWith.impl.Op]
// CHECK:STDOUT:   %DivWith.impl_witness_table = impl_witness_table (%Core.import_ref.abd97d.2, %Core.import_ref.9c3475.2), @Core.IntLiteral.as.DivWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.9e9]
// CHECK:STDOUT:     %.Self.ref.loc6_20: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.9e9]
// CHECK:STDOUT:     %.Self.as_type.loc6_20: type = facet_access_type %.Self.ref.loc6_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc6_20: type = converted %.Self.ref.loc6_20, %.Self.as_type.loc6_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %X.ref.loc6_20: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.6fc]
// CHECK:STDOUT:     %impl.elem0.loc6_20: %struct_type.a.b.486 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.049]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2.loc6_46: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc6_47.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2.loc6_46) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:     %.loc6_48.1: %struct_type.a.b.aa4 = struct_literal (%true, %.loc6_47.1) [concrete = constants.%struct.e57]
// CHECK:STDOUT:     %impl.elem0.loc6_47.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc6_47.1: <bound method> = bound_method %int_1, %impl.elem0.loc6_47.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:     %specific_fn.loc6_47.1: <specific function> = specific_function %impl.elem0.loc6_47.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_47.2: <bound method> = bound_method %int_1, %specific_fn.loc6_47.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_47.1: init %i32 = call %bound_method.loc6_47.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %.loc6_47.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_47.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %.loc6_47.3: %i32 = converted %int_1, %.loc6_47.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %impl.elem0.loc6_47.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc6_47.3: <bound method> = bound_method %int_2.loc6_46, %impl.elem0.loc6_47.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:     %specific_fn.loc6_47.2: <specific function> = specific_function %impl.elem0.loc6_47.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_47.4: <bound method> = bound_method %int_2.loc6_46, %specific_fn.loc6_47.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_47.2: init %i32 = call %bound_method.loc6_47.4(%int_2.loc6_46) [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %.loc6_47.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_47.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %.loc6_47.5: %i32 = converted %int_2.loc6_46, %.loc6_47.4 [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %tuple.loc6_47: %tuple.type.e55 = tuple_value (%.loc6_47.3, %.loc6_47.5) [concrete = constants.%tuple.102]
// CHECK:STDOUT:     %.loc6_48.2: %tuple.type.e55 = converted %.loc6_47.1, %tuple.loc6_47 [concrete = constants.%tuple.102]
// CHECK:STDOUT:     %struct.loc6_48: %struct_type.a.b.486 = struct_value (%true, %.loc6_48.2) [concrete = constants.%struct.247]
// CHECK:STDOUT:     %.loc6_48.3: %struct_type.a.b.486 = converted %.loc6_48.1, %struct.loc6_48 [concrete = constants.%struct.247]
// CHECK:STDOUT:     %.Self.ref.loc6_54: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.9e9]
// CHECK:STDOUT:     %.Self.as_type.loc6_54: type = facet_access_type %.Self.ref.loc6_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc6_54: type = converted %.Self.ref.loc6_54, %.Self.as_type.loc6_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %X.ref.loc6_54: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.6fc]
// CHECK:STDOUT:     %impl.elem0.loc6_54: %struct_type.a.b.486 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.049]
// CHECK:STDOUT:     %impl.elem0.subst: %struct_type.a.b.486 = impl_witness_access_substituted %impl.elem0.loc6_54, %.loc6_48.3 [concrete = constants.%struct.247]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %.loc6_65: bool = not %false [concrete = constants.%true]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %int_2.loc6_86: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem1.loc6_84: %.ec4 = impl_witness_access constants.%SubWith.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.SubWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc6_84: <bound method> = bound_method %int_3, %impl.elem1.loc6_84 [concrete = constants.%Core.IntLiteral.as.SubWith.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.SubWith.impl.Op.call: init Core.IntLiteral = call %bound_method.loc6_84(%int_3, %int_2.loc6_86) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %int_2.loc6_93: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem1.loc6_91: %.5ce = impl_witness_access constants.%DivWith.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.DivWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc6_91: <bound method> = bound_method %int_4, %impl.elem1.loc6_91 [concrete = constants.%Core.IntLiteral.as.DivWith.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.DivWith.impl.Op.call: init Core.IntLiteral = call %bound_method.loc6_91(%int_4, %int_2.loc6_93) [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc6_94.1: %tuple.type.f94 = tuple_literal (%Core.IntLiteral.as.SubWith.impl.Op.call, %Core.IntLiteral.as.DivWith.impl.Op.call) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:     %.loc6_95.1: %struct_type.a.b.aa4 = struct_literal (%.loc6_65, %.loc6_94.1) [concrete = constants.%struct.e57]
// CHECK:STDOUT:     %impl.elem0.loc6_94.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc6_94.1: <bound method> = bound_method %Core.IntLiteral.as.SubWith.impl.Op.call, %impl.elem0.loc6_94.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:     %specific_fn.loc6_94.1: <specific function> = specific_function %impl.elem0.loc6_94.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_94.2: <bound method> = bound_method %Core.IntLiteral.as.SubWith.impl.Op.call, %specific_fn.loc6_94.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:     %.loc6_84.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.SubWith.impl.Op.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc6_84.2: Core.IntLiteral = converted %Core.IntLiteral.as.SubWith.impl.Op.call, %.loc6_84.1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_94.1: init %i32 = call %bound_method.loc6_94.2(%.loc6_84.2) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %.loc6_94.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_94.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %.loc6_94.3: %i32 = converted %Core.IntLiteral.as.SubWith.impl.Op.call, %.loc6_94.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %impl.elem0.loc6_94.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc6_94.3: <bound method> = bound_method %Core.IntLiteral.as.DivWith.impl.Op.call, %impl.elem0.loc6_94.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:     %specific_fn.loc6_94.2: <specific function> = specific_function %impl.elem0.loc6_94.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_94.4: <bound method> = bound_method %Core.IntLiteral.as.DivWith.impl.Op.call, %specific_fn.loc6_94.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:     %.loc6_91.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.DivWith.impl.Op.call [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc6_91.2: Core.IntLiteral = converted %Core.IntLiteral.as.DivWith.impl.Op.call, %.loc6_91.1 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_94.2: init %i32 = call %bound_method.loc6_94.4(%.loc6_91.2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %.loc6_94.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_94.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %.loc6_94.5: %i32 = converted %Core.IntLiteral.as.DivWith.impl.Op.call, %.loc6_94.4 [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %tuple.loc6_94: %tuple.type.e55 = tuple_value (%.loc6_94.3, %.loc6_94.5) [concrete = constants.%tuple.102]
// CHECK:STDOUT:     %.loc6_95.2: %tuple.type.e55 = converted %.loc6_94.1, %tuple.loc6_94 [concrete = constants.%tuple.102]
// CHECK:STDOUT:     %struct.loc6_95: %struct_type.a.b.486 = struct_value (%.loc6_65, %.loc6_95.2) [concrete = constants.%struct.247]
// CHECK:STDOUT:     %.loc6_95.3: %struct_type.a.b.486 = converted %.loc6_95.1, %struct.loc6_95 [concrete = constants.%struct.247]
// CHECK:STDOUT:     %.loc6_14: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type %I.ref
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6_20, %.loc6_48.3
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.subst, %.loc6_95.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %X: %struct_type.a.b.486 = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%X [concrete = constants.%assoc0.6fc]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc6_7.2 as %.loc6_14 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @empty_tuple.type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %struct_type.a.b.486 = impl_witness_assoc_constant constants.%struct.247 [concrete = constants.%struct.247]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %.loc6_14
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.9e9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_non_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.e55: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.486: type = struct_type {.a: bool, .b: %tuple.type.e55} [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.6fc: %I.assoc_type = assoc_entity element0, @I.WithSelf.%X [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.049: %struct_type.a.b.486 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.aa4: type = struct_type {.a: bool, .b: %tuple.type.f94} [concrete]
// CHECK:STDOUT:   %struct.e57: %struct_type.a.b.aa4 = struct_value (%true, %tuple.ad8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.102: %tuple.type.e55 = tuple_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %struct.247: %struct_type.a.b.486 = struct_value (%true, %tuple.102) [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.302: %tuple.type.f94 = tuple_value (%int_3.1ba, %int_4.0c1) [concrete]
// CHECK:STDOUT:   %struct.1ec: %struct_type.a.b.aa4 = struct_value (%false, %tuple.302) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aad: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.b7a: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.4eb: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.e16: %tuple.type.e55 = tuple_value (%int_3.410, %int_4.4eb) [concrete]
// CHECK:STDOUT:   %struct.866: %struct_type.a.b.486 = struct_value (%false, %tuple.e16) [concrete]
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
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc10_20: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc10_20: type = facet_access_type %.Self.ref.loc10_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_20: type = converted %.Self.ref.loc10_20, %.Self.as_type.loc10_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %X.ref.loc10_20: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.6fc]
// CHECK:STDOUT:     %impl.elem0.loc10_20: %struct_type.a.b.486 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.049]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc10_47.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:     %.loc10_48.1: %struct_type.a.b.aa4 = struct_literal (%true, %.loc10_47.1) [concrete = constants.%struct.e57]
// CHECK:STDOUT:     %impl.elem0.loc10_47.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc10_47.1: <bound method> = bound_method %int_1, %impl.elem0.loc10_47.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:     %specific_fn.loc10_47.1: <specific function> = specific_function %impl.elem0.loc10_47.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_47.2: <bound method> = bound_method %int_1, %specific_fn.loc10_47.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_47.1: init %i32 = call %bound_method.loc10_47.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %.loc10_47.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_47.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %.loc10_47.3: %i32 = converted %int_1, %.loc10_47.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %impl.elem0.loc10_47.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc10_47.3: <bound method> = bound_method %int_2, %impl.elem0.loc10_47.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:     %specific_fn.loc10_47.2: <specific function> = specific_function %impl.elem0.loc10_47.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_47.4: <bound method> = bound_method %int_2, %specific_fn.loc10_47.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_47.2: init %i32 = call %bound_method.loc10_47.4(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %.loc10_47.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_47.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %.loc10_47.5: %i32 = converted %int_2, %.loc10_47.4 [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %tuple.loc10_47: %tuple.type.e55 = tuple_value (%.loc10_47.3, %.loc10_47.5) [concrete = constants.%tuple.102]
// CHECK:STDOUT:     %.loc10_48.2: %tuple.type.e55 = converted %.loc10_47.1, %tuple.loc10_47 [concrete = constants.%tuple.102]
// CHECK:STDOUT:     %struct.loc10_48: %struct_type.a.b.486 = struct_value (%true, %.loc10_48.2) [concrete = constants.%struct.247]
// CHECK:STDOUT:     %.loc10_48.3: %struct_type.a.b.486 = converted %.loc10_48.1, %struct.loc10_48 [concrete = constants.%struct.247]
// CHECK:STDOUT:     %.Self.ref.loc10_54: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc10_54: type = facet_access_type %.Self.ref.loc10_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_54: type = converted %.Self.ref.loc10_54, %.Self.as_type.loc10_54 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %X.ref.loc10_54: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.6fc]
// CHECK:STDOUT:     %impl.elem0.loc10_54: %struct_type.a.b.486 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.049]
// CHECK:STDOUT:     %impl.elem0.subst: %struct_type.a.b.486 = impl_witness_access_substituted %impl.elem0.loc10_54, %.loc10_48.3 [concrete = constants.%struct.247]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %.loc10_82.1: %tuple.type.f94 = tuple_literal (%int_3, %int_4) [concrete = constants.%tuple.302]
// CHECK:STDOUT:     %.loc10_83.1: %struct_type.a.b.aa4 = struct_literal (%false, %.loc10_82.1) [concrete = constants.%struct.1ec]
// CHECK:STDOUT:     %impl.elem0.loc10_82.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc10_82.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_82.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:     %specific_fn.loc10_82.1: <specific function> = specific_function %impl.elem0.loc10_82.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_82.2: <bound method> = bound_method %int_3, %specific_fn.loc10_82.1 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_82.1: init %i32 = call %bound_method.loc10_82.2(%int_3) [concrete = constants.%int_3.410]
// CHECK:STDOUT:     %.loc10_82.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_82.1 [concrete = constants.%int_3.410]
// CHECK:STDOUT:     %.loc10_82.3: %i32 = converted %int_3, %.loc10_82.2 [concrete = constants.%int_3.410]
// CHECK:STDOUT:     %impl.elem0.loc10_82.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc10_82.3: <bound method> = bound_method %int_4, %impl.elem0.loc10_82.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aad]
// CHECK:STDOUT:     %specific_fn.loc10_82.2: <specific function> = specific_function %impl.elem0.loc10_82.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_82.4: <bound method> = bound_method %int_4, %specific_fn.loc10_82.2 [concrete = constants.%bound_method.b7a]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_82.2: init %i32 = call %bound_method.loc10_82.4(%int_4) [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:     %.loc10_82.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_82.2 [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:     %.loc10_82.5: %i32 = converted %int_4, %.loc10_82.4 [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:     %tuple.loc10_82: %tuple.type.e55 = tuple_value (%.loc10_82.3, %.loc10_82.5) [concrete = constants.%tuple.e16]
// CHECK:STDOUT:     %.loc10_83.2: %tuple.type.e55 = converted %.loc10_82.1, %tuple.loc10_82 [concrete = constants.%tuple.e16]
// CHECK:STDOUT:     %struct.loc10_83: %struct_type.a.b.486 = struct_value (%false, %.loc10_83.2) [concrete = constants.%struct.866]
// CHECK:STDOUT:     %.loc10_83.3: %struct_type.a.b.486 = converted %.loc10_83.1, %struct.loc10_83 [concrete = constants.%struct.866]
// CHECK:STDOUT:     %.loc10_14: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       requirement_base_facet_type %I.ref
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_20, %.loc10_48.3
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.subst, %.loc10_83.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %X: %struct_type.a.b.486 = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%X [concrete = constants.%assoc0.6fc]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.<error>.impl: %.loc10_7.2 as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.fe2: type = struct_type {.a: bool, .b: %tuple.type.d07} [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.be0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%X [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.1dc: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.1dc [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.1dc, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.358da: %struct_type.a.b.fe2 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.aa4: type = struct_type {.a: bool, .b: %tuple.type.f94} [concrete]
// CHECK:STDOUT:   %struct.e57: %struct_type.a.b.aa4 = struct_value (%true, %tuple.ad8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %struct.682: %struct_type.a.b.fe2 = struct_value (%true, %tuple.21c) [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %SubWith.type.378: type = generic_interface_type @SubWith [concrete]
// CHECK:STDOUT:   %SubWith.generic: %SubWith.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.type.a89: type = facet_type <@SubWith, @SubWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %SubWith.impl_witness: <witness> = impl_witness imports.%SubWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %SubWith.facet: %SubWith.type.a89 = facet_value Core.IntLiteral, (%SubWith.impl_witness) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.7ed: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(Core.IntLiteral, %SubWith.facet) [concrete]
// CHECK:STDOUT:   %.24b: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.7ed, %SubWith.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.SubWith.impl.Op.type: type = fn_type @Core.IntLiteral.as.SubWith.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.SubWith.impl.Op: %Core.IntLiteral.as.SubWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.SubWith.impl.Op.bound: <bound method> = bound_method %int_3, %Core.IntLiteral.as.SubWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %DivWith.type.fc9: type = generic_interface_type @DivWith [concrete]
// CHECK:STDOUT:   %DivWith.generic: %DivWith.type.fc9 = struct_value () [concrete]
// CHECK:STDOUT:   %DivWith.type.234: type = facet_type <@DivWith, @DivWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %DivWith.impl_witness: <witness> = impl_witness imports.%DivWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %DivWith.facet: %DivWith.type.234 = facet_value Core.IntLiteral, (%DivWith.impl_witness) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.04d: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(Core.IntLiteral, %DivWith.facet) [concrete]
// CHECK:STDOUT:   %.d00: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.04d, %DivWith.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.DivWith.impl.Op.type: type = fn_type @Core.IntLiteral.as.DivWith.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.DivWith.impl.Op: %Core.IntLiteral.as.DivWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.DivWith.impl.Op.bound: <bound method> = bound_method %int_4, %Core.IntLiteral.as.DivWith.impl.Op [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.358da = %struct.682> [concrete]
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
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc141, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.SubWith: %SubWith.type.378 = import_ref Core//prelude/operators/arithmetic, SubWith, loaded [concrete = constants.%SubWith.generic]
// CHECK:STDOUT:   %Core.import_ref.abd97d.1 = import_ref Core//prelude/operators/arithmetic, loc494, unloaded
// CHECK:STDOUT:   %Core.import_ref.d95: %Core.IntLiteral.as.SubWith.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc509, loaded [concrete = constants.%Core.IntLiteral.as.SubWith.impl.Op]
// CHECK:STDOUT:   %SubWith.impl_witness_table = impl_witness_table (%Core.import_ref.abd97d.1, %Core.import_ref.d95), @Core.IntLiteral.as.SubWith.impl [concrete]
// CHECK:STDOUT:   %Core.DivWith: %DivWith.type.fc9 = import_ref Core//prelude/operators/arithmetic, DivWith, loaded [concrete = constants.%DivWith.generic]
// CHECK:STDOUT:   %Core.import_ref.abd97d.2 = import_ref Core//prelude/operators/arithmetic, loc368, unloaded
// CHECK:STDOUT:   %Core.import_ref.93eb: %Core.IntLiteral.as.DivWith.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc383, loaded [concrete = constants.%Core.IntLiteral.as.DivWith.impl.Op]
// CHECK:STDOUT:   %DivWith.impl_witness_table = impl_witness_table (%Core.import_ref.abd97d.2, %Core.import_ref.93eb), @Core.IntLiteral.as.DivWith.impl [concrete]
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
// CHECK:STDOUT:     %.loc29.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc29.2: type = converted %.loc29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.1dc]
// CHECK:STDOUT:     %.Self.ref.loc33: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.1dc]
// CHECK:STDOUT:     %.Self.as_type.loc33: type = facet_access_type %.Self.ref.loc33 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc33: type = converted %.Self.ref.loc33, %.Self.as_type.loc33 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %X.ref.loc33: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.be0]
// CHECK:STDOUT:     %impl.elem0.loc33: %struct_type.a.b.fe2 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.358da]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2.loc48: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc49.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2.loc48) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:     %.loc50.1: %struct_type.a.b.aa4 = struct_literal (%true, %.loc49.1) [concrete = constants.%struct.e57]
// CHECK:STDOUT:     %impl.elem0.loc49.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc49.1: <bound method> = bound_method %int_1, %impl.elem0.loc49.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:     %specific_fn.loc49.1: <specific function> = specific_function %impl.elem0.loc49.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc49.2: <bound method> = bound_method %int_1, %specific_fn.loc49.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49.1: init %i32 = call %bound_method.loc49.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc49.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc49.3: %i32 = converted %int_1, %.loc49.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc49.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc49.3: <bound method> = bound_method %int_2.loc48, %impl.elem0.loc49.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:     %specific_fn.loc49.2: <specific function> = specific_function %impl.elem0.loc49.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc49.4: <bound method> = bound_method %int_2.loc48, %specific_fn.loc49.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49.2: init %i32 = call %bound_method.loc49.4(%int_2.loc48) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc49.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc49.5: %i32 = converted %int_2.loc48, %.loc49.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %tuple.loc49: %tuple.type.d07 = tuple_value (%.loc49.3, %.loc49.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %.loc50.2: %tuple.type.d07 = converted %.loc49.1, %tuple.loc49 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %struct.loc50: %struct_type.a.b.fe2 = struct_value (%true, %.loc50.2) [concrete = constants.%struct.682]
// CHECK:STDOUT:     %.loc50.3: %struct_type.a.b.fe2 = converted %.loc50.1, %struct.loc50 [concrete = constants.%struct.682]
// CHECK:STDOUT:     %.Self.ref.loc52: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.1dc]
// CHECK:STDOUT:     %.Self.as_type.loc52: type = facet_access_type %.Self.ref.loc52 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc52: type = converted %.Self.ref.loc52, %.Self.as_type.loc52 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %X.ref.loc52: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.be0]
// CHECK:STDOUT:     %impl.elem0.loc52: %struct_type.a.b.fe2 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.358da]
// CHECK:STDOUT:     %impl.elem0.subst: %struct_type.a.b.fe2 = impl_witness_access_substituted %impl.elem0.loc52, %.loc50.3 [concrete = constants.%struct.682]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %.loc59: bool = not %false [concrete = constants.%true]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %int_2.loc68: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem1.loc67: %.24b = impl_witness_access constants.%SubWith.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.SubWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc67: <bound method> = bound_method %int_3, %impl.elem1.loc67 [concrete = constants.%Core.IntLiteral.as.SubWith.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.SubWith.impl.Op.call: init Core.IntLiteral = call %bound_method.loc67(%int_3, %int_2.loc68) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %int_2.loc72: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem1.loc71: %.d00 = impl_witness_access constants.%DivWith.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.DivWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc71: <bound method> = bound_method %int_4, %impl.elem1.loc71 [concrete = constants.%Core.IntLiteral.as.DivWith.impl.Op.bound]
// CHECK:STDOUT:     %Core.IntLiteral.as.DivWith.impl.Op.call: init Core.IntLiteral = call %bound_method.loc71(%int_4, %int_2.loc72) [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc73.1: %tuple.type.f94 = tuple_literal (%Core.IntLiteral.as.SubWith.impl.Op.call, %Core.IntLiteral.as.DivWith.impl.Op.call) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:     %.loc74.1: %struct_type.a.b.aa4 = struct_literal (%.loc59, %.loc73.1) [concrete = constants.%struct.e57]
// CHECK:STDOUT:     %impl.elem0.loc73.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc73.1: <bound method> = bound_method %Core.IntLiteral.as.SubWith.impl.Op.call, %impl.elem0.loc73.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:     %specific_fn.loc73.1: <specific function> = specific_function %impl.elem0.loc73.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc73.2: <bound method> = bound_method %Core.IntLiteral.as.SubWith.impl.Op.call, %specific_fn.loc73.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:     %.loc67.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.SubWith.impl.Op.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc67.2: Core.IntLiteral = converted %Core.IntLiteral.as.SubWith.impl.Op.call, %.loc67.1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc73.1: init %i32 = call %bound_method.loc73.2(%.loc67.2) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc73.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc73.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc73.3: %i32 = converted %Core.IntLiteral.as.SubWith.impl.Op.call, %.loc73.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc73.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc73.3: <bound method> = bound_method %Core.IntLiteral.as.DivWith.impl.Op.call, %impl.elem0.loc73.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:     %specific_fn.loc73.2: <specific function> = specific_function %impl.elem0.loc73.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc73.4: <bound method> = bound_method %Core.IntLiteral.as.DivWith.impl.Op.call, %specific_fn.loc73.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:     %.loc71.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.DivWith.impl.Op.call [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc71.2: Core.IntLiteral = converted %Core.IntLiteral.as.DivWith.impl.Op.call, %.loc71.1 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc73.2: init %i32 = call %bound_method.loc73.4(%.loc71.2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc73.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc73.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc73.5: %i32 = converted %Core.IntLiteral.as.DivWith.impl.Op.call, %.loc73.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %tuple.loc73: %tuple.type.d07 = tuple_value (%.loc73.3, %.loc73.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %.loc74.2: %tuple.type.d07 = converted %.loc73.1, %tuple.loc73 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %struct.loc74: %struct_type.a.b.fe2 = struct_value (%.loc59, %.loc74.2) [concrete = constants.%struct.682]
// CHECK:STDOUT:     %.loc74.3: %struct_type.a.b.fe2 = converted %.loc74.1, %struct.loc74 [concrete = constants.%struct.682]
// CHECK:STDOUT:     %.loc32: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc33, %.loc50.3
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.subst, %.loc74.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %X: %struct_type.a.b.fe2 = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%X [concrete = constants.%assoc0.be0]
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
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc29.2 as %.loc32 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @empty_tuple.type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %struct_type.a.b.fe2 = impl_witness_assoc_constant constants.%struct.682 [concrete = constants.%struct.682]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.1dc) {
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
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.fe2: type = struct_type {.a: bool, .b: %tuple.type.d07} [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.be0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%X [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.358: %struct_type.a.b.fe2 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.aa4: type = struct_type {.a: bool, .b: %tuple.type.f94} [concrete]
// CHECK:STDOUT:   %struct.e57: %struct_type.a.b.aa4 = struct_value (%true, %tuple.ad8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %struct.682: %struct_type.a.b.fe2 = struct_value (%true, %tuple.21c) [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.302: %tuple.type.f94 = tuple_value (%int_3.1ba, %int_4.0c1) [concrete]
// CHECK:STDOUT:   %struct.1ec: %struct_type.a.b.aa4 = struct_value (%false, %tuple.302) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.ffd: %tuple.type.d07 = tuple_value (%int_3.822, %int_4.940) [concrete]
// CHECK:STDOUT:   %struct.68c: %struct_type.a.b.fe2 = struct_value (%false, %tuple.ffd) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc141, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %.loc29.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc29.2: type = converted %.loc29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc33: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc33: type = facet_access_type %.Self.ref.loc33 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc33: type = converted %.Self.ref.loc33, %.Self.as_type.loc33 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %X.ref.loc33: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.be0]
// CHECK:STDOUT:     %impl.elem0.loc33: %struct_type.a.b.fe2 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.358]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc49.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:     %.loc50.1: %struct_type.a.b.aa4 = struct_literal (%true, %.loc49.1) [concrete = constants.%struct.e57]
// CHECK:STDOUT:     %impl.elem0.loc49.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc49.1: <bound method> = bound_method %int_1, %impl.elem0.loc49.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:     %specific_fn.loc49.1: <specific function> = specific_function %impl.elem0.loc49.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc49.2: <bound method> = bound_method %int_1, %specific_fn.loc49.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49.1: init %i32 = call %bound_method.loc49.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc49.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc49.3: %i32 = converted %int_1, %.loc49.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc49.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc49.3: <bound method> = bound_method %int_2, %impl.elem0.loc49.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:     %specific_fn.loc49.2: <specific function> = specific_function %impl.elem0.loc49.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc49.4: <bound method> = bound_method %int_2, %specific_fn.loc49.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49.2: init %i32 = call %bound_method.loc49.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc49.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc49.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc49.5: %i32 = converted %int_2, %.loc49.4 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %tuple.loc49: %tuple.type.d07 = tuple_value (%.loc49.3, %.loc49.5) [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %.loc50.2: %tuple.type.d07 = converted %.loc49.1, %tuple.loc49 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:     %struct.loc50: %struct_type.a.b.fe2 = struct_value (%true, %.loc50.2) [concrete = constants.%struct.682]
// CHECK:STDOUT:     %.loc50.3: %struct_type.a.b.fe2 = converted %.loc50.1, %struct.loc50 [concrete = constants.%struct.682]
// CHECK:STDOUT:     %.Self.ref.loc52: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc52: type = facet_access_type %.Self.ref.loc52 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc52: type = converted %.Self.ref.loc52, %.Self.as_type.loc52 [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %X.ref.loc52: %I.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0.be0]
// CHECK:STDOUT:     %impl.elem0.loc52: %struct_type.a.b.fe2 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.358]
// CHECK:STDOUT:     %impl.elem0.subst: %struct_type.a.b.fe2 = impl_witness_access_substituted %impl.elem0.loc52, %.loc50.3 [concrete = constants.%struct.682]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %.loc68.1: %tuple.type.f94 = tuple_literal (%int_3, %int_4) [concrete = constants.%tuple.302]
// CHECK:STDOUT:     %.loc69.1: %struct_type.a.b.aa4 = struct_literal (%false, %.loc68.1) [concrete = constants.%struct.1ec]
// CHECK:STDOUT:     %impl.elem0.loc68.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc68.1: <bound method> = bound_method %int_3, %impl.elem0.loc68.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:     %specific_fn.loc68.1: <specific function> = specific_function %impl.elem0.loc68.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc68.2: <bound method> = bound_method %int_3, %specific_fn.loc68.1 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc68.1: init %i32 = call %bound_method.loc68.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc68.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc68.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc68.3: %i32 = converted %int_3, %.loc68.2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %impl.elem0.loc68.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc68.3: <bound method> = bound_method %int_4, %impl.elem0.loc68.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:     %specific_fn.loc68.2: <specific function> = specific_function %impl.elem0.loc68.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc68.4: <bound method> = bound_method %int_4, %specific_fn.loc68.2 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc68.2: init %i32 = call %bound_method.loc68.4(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc68.4: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc68.2 [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc68.5: %i32 = converted %int_4, %.loc68.4 [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %tuple.loc68: %tuple.type.d07 = tuple_value (%.loc68.3, %.loc68.5) [concrete = constants.%tuple.ffd]
// CHECK:STDOUT:     %.loc69.2: %tuple.type.d07 = converted %.loc68.1, %tuple.loc68 [concrete = constants.%tuple.ffd]
// CHECK:STDOUT:     %struct.loc69: %struct_type.a.b.fe2 = struct_value (%false, %.loc69.2) [concrete = constants.%struct.68c]
// CHECK:STDOUT:     %.loc69.3: %struct_type.a.b.fe2 = converted %.loc69.1, %struct.loc69 [concrete = constants.%struct.68c]
// CHECK:STDOUT:     %.loc32: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%I.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc33, %.loc50.3
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.subst, %.loc69.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %X: %struct_type.a.b.fe2 = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%X [concrete = constants.%assoc0.be0]
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
// CHECK:STDOUT: impl @empty_tuple.type.as.<error>.impl: %.loc29.2 as %.loc32 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/specialization_with_symbolic_rewrite.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/specialization_with_symbolic_rewrite.carbon

// --- final_specialized_symbolic_rewrite.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

class C {}

impl forall [T:! type, S:! type] T as Z(S) where .X = () {}

final impl forall [T:! type] T as Z(C) where .X = T {}

fn F(T:! Z(C), t: T) {
  // This should typecheck, the `final impl` should give the same `T`.
  let a: T.X = t;
}

// --- fail_nonfinal_specialized_symbolic_rewrite.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}
interface Y {}

class C {}

impl forall [T:! type, S:! type] T as Z(S) where .X = () {}

impl forall [T:! type] T as Z(C) where .X = T {}

fn F(T:! Z(C), t: T) {
  // CHECK:STDERR: fail_nonfinal_specialized_symbolic_rewrite.carbon:[[@LINE+7]]:16: error: cannot implicitly convert expression of type `T` to `T.(Z(C).X)` [ConversionFailure]
  // CHECK:STDERR:   let a: T.X = t;
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_nonfinal_specialized_symbolic_rewrite.carbon:[[@LINE+4]]:16: note: type `T` does not implement interface `Core.ImplicitAs(T.(Z(C).X))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a: T.X = t;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  let a: T.X = t;
}

// --- final_impl_rewrite_of_symbolic_through_impl_lookup.carbon
library "[[@TEST_NAME]]";

interface Ptr {
  let Type:! type;
}

final impl forall [U:! type] U as Ptr where .Type = U* {}

fn F[T:! type](var t: T) -> T.(Ptr.Type) {
  return &t;
}

// --- final_impl_rewrite_of_symbolic_through_facet_access_type.carbon
library "[[@TEST_NAME]]";

interface Ptr {
  let Type:! type;
}

final impl forall [U:! type] U as Ptr where .Type = U* {}

fn F[T:! Ptr](var t: T) -> T.Type {
  return &t;
}

// --- final_impl_rewrite_of_symbolic_through_facet_value.carbon
library "[[@TEST_NAME]]";

interface Ptr {
  let Type:! type;
}

final impl forall [U:! type] U as Ptr where .Type = U* {}

fn F[T:! Ptr](var t: T) -> T.(Ptr.Type) {
  return &t;
}

// CHECK:STDOUT: --- final_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.659: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.dd5: type = facet_type <@Z, @Z(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.cbbd: %Z.type.dd5 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.26e: type = assoc_entity_type @Z, @Z(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.1cd: %Z.assoc_type.26e = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = bind_symbolic_name S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.318: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.b4a: %Z.type.318 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %Self.a07: %Z.type.318 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0e7: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.abb: %Z.assoc_type.0e7 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.760: <witness> = require_complete_type %Z.type.318 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.d30: type = facet_access_type %.Self.b4a [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.e23: <witness> = lookup_impl_witness %.Self.b4a, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.abb: type = impl_witness_access %Z.lookup_impl_witness.e23, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.90e: type = facet_type <@Z, @Z(%S) where %impl.elem0.abb = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.dd6: <witness> = require_complete_type %Z_where.type.90e [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.a5c: <witness> = impl_witness file.%Z.impl_witness_table.loc9, @T.as.Z.impl.c87(%T.8b3, %S) [symbolic]
// CHECK:STDOUT:   %Z.type.a71: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.784: %Z.type.a71 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.cbb9: %Z.type.a71 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.as_type.16f: type = facet_access_type %.Self.784 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.62b: <witness> = lookup_impl_witness %.Self.784, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.dac: type = impl_witness_access %Z.lookup_impl_witness.62b, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.9b7: type = facet_type <@Z, @Z(%C) where %impl.elem0.dac = %T.8b3> [symbolic]
// CHECK:STDOUT:   %require_complete.eca: <witness> = require_complete_type %Z_where.type.9b7 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.e18: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.3bd(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.fcc: %Z.type.a71 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6dc: type = pattern_type %Z.type.a71 [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.fcc [symbolic]
// CHECK:STDOUT:   %pattern_type.3c4: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.08e: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Z_where.type.7a1: type = facet_type <@Z, @Z(%C) where %impl.elem0.dac = %T.as_type> [symbolic]
// CHECK:STDOUT:   %require_complete.685: <witness> = require_complete_type %Z_where.type.7a1 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.69d: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.3bd(%T.as_type) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Z.decl: %Z.type.9fb = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %T.loc3_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.c87 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc9_24.1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc9_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.318)]
// CHECK:STDOUT:     %.Self.1: @T.as.Z.impl.c87.%Z.type.loc9_42.2 (%Z.type.318) = bind_symbolic_name .Self [symbolic = %.Self.4 (constants.%.Self.b4a)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.c87.%Z.type.loc9_42.2 (%Z.type.318) = name_ref .Self, %.Self.1 [symbolic = %.Self.4 (constants.%.Self.b4a)]
// CHECK:STDOUT:     %.loc9_50.1: @T.as.Z.impl.c87.%Z.assoc_type (%Z.assoc_type.0e7) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.c87.%Z.assoc_type (%Z.assoc_type.0e7) = name_ref X, %.loc9_50.1 [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %.Self.as_type.loc9_50.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.d30)]
// CHECK:STDOUT:     %.loc9_50.2: type = converted %.Self.ref, %.Self.as_type.loc9_50.1 [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.d30)]
// CHECK:STDOUT:     %impl.elem0.loc9_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.e23, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.abb)]
// CHECK:STDOUT:     %.loc9_56.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_56.2: type = converted %.loc9_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.90e)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.318
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_50.1, %.loc9_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %T.loc9_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.Self.3: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %S.loc9_24.1: type = bind_symbolic_name S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc9 = impl_witness_table (%impl_witness_assoc_constant.loc9), @T.as.Z.impl.c87 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc9: <witness> = impl_witness %Z.impl_witness_table.loc9, @T.as.Z.impl.c87(constants.%T.8b3, constants.%S) [symbolic = @T.as.Z.impl.c87.%Z.impl_witness (constants.%Z.impl_witness.a5c)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc9: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.3bd [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc11_30: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     %.Self.1: %Z.type.a71 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.784]
// CHECK:STDOUT:     %.Self.ref: %Z.type.a71 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.784]
// CHECK:STDOUT:     %.loc11_46.1: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.252 = name_ref X, %.loc11_46.1 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.16f]
// CHECK:STDOUT:     %.loc11_46.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.16f]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.62b, element0 [symbolic_self = constants.%impl.elem0.dac]
// CHECK:STDOUT:     %T.ref.loc11_51: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc11_40: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.9b7)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.a71
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc11_51
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %T.loc11_20.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc11 = impl_witness_table (%impl_witness_assoc_constant.loc11), @T.as.Z.impl.3bd [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc11: <witness> = impl_witness %Z.impl_witness_table.loc11, @T.as.Z.impl.3bd(constants.%T.8b3) [symbolic = @T.as.Z.impl.3bd.%Z.impl_witness (constants.%Z.impl_witness.e18)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc11: type = impl_witness_assoc_constant constants.%T.8b3 [symbolic = @T.as.Z.impl.3bd.%T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.6dc = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type (%pattern_type.3c4) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type (%pattern_type.3c4) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_13: type = splice_block %Z.type [concrete = constants.%Z.type.a71] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:       %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_6.2: %Z.type.a71 = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.1 (constants.%T.fcc)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc13_19.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_19.1: type = splice_block %.loc13_19.2 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc13: %Z.type.a71 = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T.fcc)]
// CHECK:STDOUT:       %T.as_type.loc13_19.2: type = facet_access_type %T.ref.loc13 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc13_19.2: type = converted %T.ref.loc13, %T.as_type.loc13_19.2 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc13_19.1 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_13.2: type) {
// CHECK:STDOUT:   %T.loc3_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_13.1)> [symbolic = %Z.type (constants.%Z.type.dd5)]
// CHECK:STDOUT:   %Self.2: @Z.%Z.type (%Z.type.dd5) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.cbbd)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc3_13.1) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.26e)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.26e) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.1cd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Z.%Z.type (%Z.type.dd5) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.cbbd)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.26e) = assoc_entity element0, @Z.%X [symbolic = @Z.%assoc0 (constants.%assoc0.1cd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc3_13.2: type, @Z.%Self.1: @Z.%Z.type (%Z.type.dd5)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.c87(%T.loc9_14.1: type, %S.loc9_24.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %S.loc9_24.2: type = bind_symbolic_name S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc9_42.2: type = facet_type <@Z, @Z(%S.loc9_24.2)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.318)]
// CHECK:STDOUT:   %.Self.4: @T.as.Z.impl.c87.%Z.type.loc9_42.2 (%Z.type.318) = bind_symbolic_name .Self [symbolic = %.Self.4 (constants.%.Self.b4a)]
// CHECK:STDOUT:   %require_complete.loc9_50: <witness> = require_complete_type %Z.type.loc9_42.2 [symbolic = %require_complete.loc9_50 (constants.%require_complete.760)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc9_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.0e7)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.c87.%Z.assoc_type (%Z.assoc_type.0e7) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:   %.Self.as_type.loc9_50.2: type = facet_access_type %.Self.4 [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.d30)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @Z, @Z(%S.loc9_24.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.e23)]
// CHECK:STDOUT:   %impl.elem0.loc9_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.abb)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc9_24.2) where %impl.elem0.loc9_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.90e)]
// CHECK:STDOUT:   %require_complete.loc9_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc9_44 (constants.%require_complete.dd6)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc9, @T.as.Z.impl.c87(%T.loc9_14.2, %S.loc9_24.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.a5c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc9_44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.3bd(%T.loc11_20.1: type) {
// CHECK:STDOUT:   %T.loc11_20.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.dac = %T.loc11_20.2> [symbolic = %Z_where.type (constants.%Z_where.type.9b7)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.eca)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.3bd(%T.loc11_20.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.e18)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc11_30 as %.loc11_40 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.2: %Z.type.a71) {
// CHECK:STDOUT:   %T.loc13_6.1: %Z.type.a71 = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.1 (constants.%T.fcc)]
// CHECK:STDOUT:   %T.as_type.loc13_19.1: type = facet_access_type %T.loc13_6.1 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc13_19.1 [symbolic = %pattern_type (constants.%pattern_type.3c4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc13_19.1 [symbolic = %require_complete (constants.%require_complete.08e)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.3bd(%T.as_type.loc13_19.1) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.69d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc13_19.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type (%pattern_type.3c4) = binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc13_19.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc15_11.1: type = splice_block %impl.elem0 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc15: %Z.type.a71 = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T.fcc)]
// CHECK:STDOUT:       %.loc15_11.2: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.252 = name_ref X, %.loc15_11.2 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %T.as_type.loc15_11.1: type = facet_access_type %T.ref.loc15 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_11.3: type = converted %T.ref.loc15, %T.as_type.loc15_11.1 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %T.as_type.loc15_11.2: type = facet_access_type constants.%T.fcc [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_11.4: type = converted constants.%T.fcc, %T.as_type.loc15_11.2 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.69d, element0 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%T.as_type.loc13_19.1 (%T.as_type) = bind_name a, %t.ref
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.8b3, constants.%Self.cbbd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.318
// CHECK:STDOUT:   %Self.2 => constants.%Self.a07
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%.Self.b4a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.c87(constants.%T.8b3, constants.%S) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %S.loc9_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc9_42.2 => constants.%Z.type.318
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.b4a
// CHECK:STDOUT:   %require_complete.loc9_50 => constants.%require_complete.760
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT:   %.Self.as_type.loc9_50.2 => constants.%.Self.as_type.d30
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.e23
// CHECK:STDOUT:   %impl.elem0.loc9_50.2 => constants.%impl.elem0.abb
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.90e
// CHECK:STDOUT:   %require_complete.loc9_44 => constants.%require_complete.dd6
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.a5c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.a71
// CHECK:STDOUT:   %Self.2 => constants.%Self.cbb9
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%.Self.784) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.3bd(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.8b3
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.9b7
// CHECK:STDOUT:   %require_complete => constants.%require_complete.eca
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.e18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.fcc) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%T.fcc
// CHECK:STDOUT:   %T.as_type.loc13_19.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.3bd(constants.%T.as_type) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.as_type
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.7a1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.685
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.69d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%T.fcc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.659: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.dd5: type = facet_type <@Z, @Z(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.cbbd: %Z.type.dd5 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.26e: type = assoc_entity_type @Z, @Z(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.1cd: %Z.assoc_type.26e = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.045: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = bind_symbolic_name S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.318: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.b4a: %Z.type.318 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %Self.a07: %Z.type.318 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0e7: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.abb: %Z.assoc_type.0e7 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.760: <witness> = require_complete_type %Z.type.318 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.d30: type = facet_access_type %.Self.b4a [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.e23: <witness> = lookup_impl_witness %.Self.b4a, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.abb: type = impl_witness_access %Z.lookup_impl_witness.e23, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.90e: type = facet_type <@Z, @Z(%S) where %impl.elem0.abb = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.dd6: <witness> = require_complete_type %Z_where.type.90e [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.a5c: <witness> = impl_witness file.%Z.impl_witness_table.loc10, @T.as.Z.impl.c87(%T.8b3, %S) [symbolic]
// CHECK:STDOUT:   %Z.type.a71: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.784: %Z.type.a71 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.cbb9: %Z.type.a71 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.as_type.16f: type = facet_access_type %.Self.784 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.62b: <witness> = lookup_impl_witness %.Self.784, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.dac: type = impl_witness_access %Z.lookup_impl_witness.62b, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.9b7: type = facet_type <@Z, @Z(%C) where %impl.elem0.dac = %T.8b3> [symbolic]
// CHECK:STDOUT:   %require_complete.eca: <witness> = require_complete_type %Z_where.type.9b7 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.e18: <witness> = impl_witness file.%Z.impl_witness_table.loc12, @T.as.Z.impl.3bd(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.fcc: %Z.type.a71 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6dc: type = pattern_type %Z.type.a71 [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.fcc [symbolic]
// CHECK:STDOUT:   %pattern_type.3c4: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.08e: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.38e: <witness> = lookup_impl_witness %T.fcc, @Z, @Z(%C) [symbolic]
// CHECK:STDOUT:   %impl.elem0.407: type = impl_witness_access %Z.lookup_impl_witness.38e, element0 [symbolic]
// CHECK:STDOUT:   %require_complete.d9a: <witness> = require_complete_type %impl.elem0.407 [symbolic]
// CHECK:STDOUT:   %pattern_type.420: type = pattern_type %impl.elem0.407 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.275: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.42e: %ImplicitAs.Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d74: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.407)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.cd9: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.407) [symbolic]
// CHECK:STDOUT:   %assoc0.1bf: %ImplicitAs.assoc_type.cd9 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %require_complete.744: <witness> = require_complete_type %ImplicitAs.type.d74 [symbolic]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//prelude/parts/as, loc17_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.275) = import_ref Core//prelude/parts/as, loc17_35, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//prelude/parts/as, loc17_35, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Z.decl: %Z.type.9fb = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %T.loc3_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.c87 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc10_24.1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc10_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.318)]
// CHECK:STDOUT:     %.Self.1: @T.as.Z.impl.c87.%Z.type.loc10_42.2 (%Z.type.318) = bind_symbolic_name .Self [symbolic = %.Self.4 (constants.%.Self.b4a)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.c87.%Z.type.loc10_42.2 (%Z.type.318) = name_ref .Self, %.Self.1 [symbolic = %.Self.4 (constants.%.Self.b4a)]
// CHECK:STDOUT:     %.loc10_50.1: @T.as.Z.impl.c87.%Z.assoc_type (%Z.assoc_type.0e7) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.c87.%Z.assoc_type (%Z.assoc_type.0e7) = name_ref X, %.loc10_50.1 [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %.Self.as_type.loc10_50.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.d30)]
// CHECK:STDOUT:     %.loc10_50.2: type = converted %.Self.ref, %.Self.as_type.loc10_50.1 [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.d30)]
// CHECK:STDOUT:     %impl.elem0.loc10_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.e23, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.abb)]
// CHECK:STDOUT:     %.loc10_56.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_56.2: type = converted %.loc10_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.90e)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.318
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_50.1, %.loc10_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %T.loc10_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.Self.3: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %S.loc10_24.1: type = bind_symbolic_name S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc10 = impl_witness_table (%impl_witness_assoc_constant.loc10), @T.as.Z.impl.c87 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc10: <witness> = impl_witness %Z.impl_witness_table.loc10, @T.as.Z.impl.c87(constants.%T.8b3, constants.%S) [symbolic = @T.as.Z.impl.c87.%Z.impl_witness (constants.%Z.impl_witness.a5c)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc10: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.3bd [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_24: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     %.Self.1: %Z.type.a71 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.784]
// CHECK:STDOUT:     %.Self.ref: %Z.type.a71 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.784]
// CHECK:STDOUT:     %.loc12_40.1: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.252 = name_ref X, %.loc12_40.1 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.16f]
// CHECK:STDOUT:     %.loc12_40.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.16f]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.62b, element0 [symbolic_self = constants.%impl.elem0.dac]
// CHECK:STDOUT:     %T.ref.loc12_45: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc12_34: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.9b7)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.a71
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc12_45
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %T.loc12_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc12 = impl_witness_table (%impl_witness_assoc_constant.loc12), @T.as.Z.impl.3bd [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc12: <witness> = impl_witness %Z.impl_witness_table.loc12, @T.as.Z.impl.3bd(constants.%T.8b3) [symbolic = @T.as.Z.impl.3bd.%Z.impl_witness (constants.%Z.impl_witness.e18)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc12: type = impl_witness_assoc_constant constants.%T.8b3 [symbolic = @T.as.Z.impl.3bd.%T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.6dc = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc14 (%pattern_type.3c4) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc14 (%pattern_type.3c4) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_13: type = splice_block %Z.type [concrete = constants.%Z.type.a71] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:       %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_6.2: %Z.type.a71 = bind_symbolic_name T, 0 [symbolic = %T.loc14_6.1 (constants.%T.fcc)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc14_19.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc14: %Z.type.a71 = name_ref T, %T.loc14_6.2 [symbolic = %T.loc14_6.1 (constants.%T.fcc)]
// CHECK:STDOUT:       %T.as_type.loc14_19.2: type = facet_access_type %T.ref.loc14 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc14_19.2: type = converted %T.ref.loc14, %T.as_type.loc14_19.2 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc14_19.1 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_13.2: type) {
// CHECK:STDOUT:   %T.loc3_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_13.1)> [symbolic = %Z.type (constants.%Z.type.dd5)]
// CHECK:STDOUT:   %Self.2: @Z.%Z.type (%Z.type.dd5) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.cbbd)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc3_13.1) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.26e)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.26e) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.1cd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Z.%Z.type (%Z.type.dd5) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.cbbd)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.26e) = assoc_entity element0, @Z.%X [symbolic = @Z.%assoc0 (constants.%assoc0.1cd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.045]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc3_13.2: type, @Z.%Self.1: @Z.%Z.type (%Z.type.dd5)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.c87(%T.loc10_14.1: type, %S.loc10_24.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %S.loc10_24.2: type = bind_symbolic_name S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc10_42.2: type = facet_type <@Z, @Z(%S.loc10_24.2)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.318)]
// CHECK:STDOUT:   %.Self.4: @T.as.Z.impl.c87.%Z.type.loc10_42.2 (%Z.type.318) = bind_symbolic_name .Self [symbolic = %.Self.4 (constants.%.Self.b4a)]
// CHECK:STDOUT:   %require_complete.loc10_50: <witness> = require_complete_type %Z.type.loc10_42.2 [symbolic = %require_complete.loc10_50 (constants.%require_complete.760)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc10_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.0e7)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.c87.%Z.assoc_type (%Z.assoc_type.0e7) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:   %.Self.as_type.loc10_50.2: type = facet_access_type %.Self.4 [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.d30)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @Z, @Z(%S.loc10_24.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.e23)]
// CHECK:STDOUT:   %impl.elem0.loc10_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.abb)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc10_24.2) where %impl.elem0.loc10_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.90e)]
// CHECK:STDOUT:   %require_complete.loc10_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc10_44 (constants.%require_complete.dd6)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc10, @T.as.Z.impl.c87(%T.loc10_14.2, %S.loc10_24.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.a5c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc10_44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.3bd(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.dac = %T.loc12_14.2> [symbolic = %Z_where.type (constants.%Z_where.type.9b7)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.eca)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc12, @T.as.Z.impl.3bd(%T.loc12_14.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.e18)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc12_24 as %.loc12_34 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc14_6.2: %Z.type.a71) {
// CHECK:STDOUT:   %T.loc14_6.1: %Z.type.a71 = bind_symbolic_name T, 0 [symbolic = %T.loc14_6.1 (constants.%T.fcc)]
// CHECK:STDOUT:   %T.as_type.loc14_19.1: type = facet_access_type %T.loc14_6.1 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc14: type = pattern_type %T.as_type.loc14_19.1 [symbolic = %pattern_type.loc14 (constants.%pattern_type.3c4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.as_type.loc14_19.1 [symbolic = %require_complete.loc14 (constants.%require_complete.08e)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc14_6.1, @Z, @Z(constants.%C) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.38e)]
// CHECK:STDOUT:   %impl.elem0.loc22_11.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.407)]
// CHECK:STDOUT:   %require_complete.loc22_11: <witness> = require_complete_type %impl.elem0.loc22_11.2 [symbolic = %require_complete.loc22_11 (constants.%require_complete.d9a)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %impl.elem0.loc22_11.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.420)]
// CHECK:STDOUT:   %ImplicitAs.type.loc22_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2)> [symbolic = %ImplicitAs.type.loc22_16.2 (constants.%ImplicitAs.type.d74)]
// CHECK:STDOUT:   %require_complete.loc22_16: <witness> = require_complete_type %ImplicitAs.type.loc22_16.2 [symbolic = %require_complete.loc22_16 (constants.%require_complete.744)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.cd9)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.cd9) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.1bf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc14_19.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type.loc22 (%pattern_type.420) = binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc14_19.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %impl.elem0.loc22_11.1 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.407)] {
// CHECK:STDOUT:       %T.ref.loc22: %Z.type.a71 = name_ref T, %T.loc14_6.2 [symbolic = %T.loc14_6.1 (constants.%T.fcc)]
// CHECK:STDOUT:       %.loc22_11.2: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.252 = name_ref X, %.loc22_11.2 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_11.3: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc22_11.1: type = impl_witness_access constants.%Z.lookup_impl_witness.38e, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.407)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ImplicitAs.type.loc22_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%impl.elem0.407)> [symbolic = %ImplicitAs.type.loc22_16.2 (constants.%ImplicitAs.type.d74)]
// CHECK:STDOUT:     %.loc22_16.1: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.cd9) = specific_constant imports.%Core.import_ref.492, @ImplicitAs(constants.%impl.elem0.407) [symbolic = %assoc0 (constants.%assoc0.1bf)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.cd9) = name_ref Convert, %.loc22_16.1 [symbolic = %assoc0 (constants.%assoc0.1bf)]
// CHECK:STDOUT:     %.loc22_16.2: @F.%impl.elem0.loc22_11.2 (%impl.elem0.407) = converted %t.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %a: @F.%impl.elem0.loc22_11.2 (%impl.elem0.407) = bind_name a, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.8b3, constants.%Self.cbbd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.318
// CHECK:STDOUT:   %Self.2 => constants.%Self.a07
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%.Self.b4a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.c87(constants.%T.8b3, constants.%S) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %S.loc10_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc10_42.2 => constants.%Z.type.318
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.b4a
// CHECK:STDOUT:   %require_complete.loc10_50 => constants.%require_complete.760
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT:   %.Self.as_type.loc10_50.2 => constants.%.Self.as_type.d30
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.e23
// CHECK:STDOUT:   %impl.elem0.loc10_50.2 => constants.%impl.elem0.abb
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.90e
// CHECK:STDOUT:   %require_complete.loc10_44 => constants.%require_complete.dd6
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.a5c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.a71
// CHECK:STDOUT:   %Self.2 => constants.%Self.cbb9
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%.Self.784) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.3bd(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.9b7
// CHECK:STDOUT:   %require_complete => constants.%require_complete.eca
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.e18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.fcc) {
// CHECK:STDOUT:   %T.loc14_6.1 => constants.%T.fcc
// CHECK:STDOUT:   %T.as_type.loc14_19.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc14 => constants.%pattern_type.3c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%T.fcc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.a46: %Ptr.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.d28: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.659: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.690: %Ptr.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.690 [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.690, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b59: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %U.8b3 [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.4f051e.1: type = facet_type <@Ptr where %impl.elem0.b59 = %ptr.79f131.1> [symbolic]
// CHECK:STDOUT:   %require_complete.334099.1: <witness> = require_complete_type %Ptr_where.type.4f051e.1 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.da31d2.1: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.8b3) [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %ptr.79f131.2: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.4f051e.2: type = facet_type <@Ptr where %impl.elem0.b59 = %ptr.79f131.2> [symbolic]
// CHECK:STDOUT:   %require_complete.334099.2: <witness> = require_complete_type %Ptr_where.type.4f051e.2 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.da31d2.2: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Ptr.facet: %Ptr.type = facet_value %T.8b3, (%Ptr.impl_witness.da31d2.2) [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f131.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f131.2 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cb2: <witness> = lookup_impl_witness %ptr.79f131.2, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.79f131.2, (%Copy.lookup_impl_witness.cb2) [symbolic]
// CHECK:STDOUT:   %.1cc: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.751: %.1cc = impl_witness_access %Copy.lookup_impl_witness.cb2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f44: <specific function> = specific_impl_function %impl.elem0.751, @Copy.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Ptr = %Ptr.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Ptr.decl: type = interface_decl @Ptr [concrete = constants.%Ptr.type] {} {}
// CHECK:STDOUT:   impl_decl @U.as.Ptr.impl [concrete] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.690]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.690]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.b59]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.4f051e.1)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %U.loc7_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.8b3) [symbolic = @U.as.Ptr.impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.da31d2.1)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.79f131.1 [symbolic = @U.as.Ptr.impl.%ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_20 (%pattern_type.7dc) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_20 (%pattern_type.7dc) = ref_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_20 (%pattern_type.7dc) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_26 (%pattern_type.afe) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_26 (%pattern_type.afe) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_29: type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %Ptr.facet.loc9_30.2: %Ptr.type = facet_value constants.%T.8b3, (constants.%Ptr.impl_witness.da31d2.2) [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:     %.loc9: %Ptr.type = converted %T.ref.loc9_29, %Ptr.facet.loc9_30.2 [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.da31d2.2, element0 [symbolic = %ptr (constants.%ptr.79f131.2)]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %T.loc9_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %t.param: ref @F.%T.loc9_6.1 (%T.8b3) = ref_param call_param0
// CHECK:STDOUT:     %T.ref.loc9_23: type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %t: ref @F.%T.loc9_6.1 (%T.8b3) = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.79f131.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.79f131.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.a46]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Type = @Type.%assoc0
// CHECK:STDOUT:   witness = (%Type)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Type(@Ptr.%Self: %Ptr.type) {
// CHECK:STDOUT:   assoc_const Type:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc7_20.1: type) {
// CHECK:STDOUT:   %U.loc7_20.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.b59 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.4f051e.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.334099.1)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.da31d2.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Ptr.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.2: type) {
// CHECK:STDOUT:   %T.loc9_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.8b3)]
// CHECK:STDOUT:   %pattern_type.loc9_20: type = pattern_type %T.loc9_6.1 [symbolic = %pattern_type.loc9_20 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.loc9_6.1) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.da31d2.2)]
// CHECK:STDOUT:   %Ptr.facet.loc9_30.1: %Ptr.type = facet_value %T.loc9_6.1, (%Ptr.impl_witness) [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc9_6.1 [symbolic = %ptr (constants.%ptr.79f131.2)]
// CHECK:STDOUT:   %pattern_type.loc9_26: type = pattern_type %ptr [symbolic = %pattern_type.loc9_26 (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_26: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_26 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc9_16: <witness> = require_complete_type %T.loc9_6.1 [symbolic = %require_complete.loc9_16 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cb2)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.1cc)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.1cc) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.751)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.loc9_6.1 (%T.8b3)) -> @F.%ptr (%ptr.79f131.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc9_6.1 (%T.8b3) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.79f131.2) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.1cc) = impl_witness_access constants.%Copy.lookup_impl_witness.cb2, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.751)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.f44)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %.loc10_10.1: init @F.%ptr (%ptr.79f131.2) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %.loc10_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.a46) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.690) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.8b3
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.4f051e.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.334099.1
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.da31d2.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.8b3) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.8b3
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f131.2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.4f051e.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.334099.2
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.da31d2.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.8b3
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.da31d2.2
// CHECK:STDOUT:   %Ptr.facet.loc9_30.1 => constants.%Ptr.facet
// CHECK:STDOUT:   %ptr => constants.%ptr.79f131.2
// CHECK:STDOUT:   %pattern_type.loc9_26 => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.a46: %Ptr.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.d28: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.659: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.690: %Ptr.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.690 [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.690, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b59: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %U.8b3 [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.4f0: type = facet_type <@Ptr where %impl.elem0.b59 = %ptr.79f131.1> [symbolic]
// CHECK:STDOUT:   %require_complete.334: <witness> = require_complete_type %Ptr_where.type.4f0 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.da3: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.8b3) [symbolic]
// CHECK:STDOUT:   %T.a46: %Ptr.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3d8: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.as_type.dc4: type = facet_access_type %T.a46 [symbolic]
// CHECK:STDOUT:   %pattern_type.851: type = pattern_type %T.as_type.dc4 [symbolic]
// CHECK:STDOUT:   %ptr.ae5: type = ptr_type %T.as_type.dc4 [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.483: type = facet_type <@Ptr where %impl.elem0.b59 = %ptr.ae5> [symbolic]
// CHECK:STDOUT:   %require_complete.c7f: <witness> = require_complete_type %Ptr_where.type.483 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.530: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.as_type.dc4) [symbolic]
// CHECK:STDOUT:   %pattern_type.ec6: type = pattern_type %ptr.ae5 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.e69: <witness> = require_complete_type %ptr.ae5 [symbolic]
// CHECK:STDOUT:   %require_complete.43a: <witness> = require_complete_type %T.as_type.dc4 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.68e: <witness> = lookup_impl_witness %ptr.ae5, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.ae5, (%Copy.lookup_impl_witness.68e) [symbolic]
// CHECK:STDOUT:   %.19c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.6d5: %.19c = impl_witness_access %Copy.lookup_impl_witness.68e, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.8d6: <specific function> = specific_impl_function %impl.elem0.6d5, @Copy.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Ptr = %Ptr.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Ptr.decl: type = interface_decl @Ptr [concrete = constants.%Ptr.type] {} {}
// CHECK:STDOUT:   impl_decl @U.as.Ptr.impl [concrete] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.690]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.690]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.b59]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.4f0)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %U.loc7_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.8b3) [symbolic = @U.as.Ptr.impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.da3)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.79f131.1 [symbolic = @U.as.Ptr.impl.%ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3d8 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_19 (%pattern_type.851) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_19 (%pattern_type.851) = ref_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_19 (%pattern_type.851) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_25 (%pattern_type.ec6) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_25 (%pattern_type.ec6) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.a46)]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %T.as_type.loc9_29.1: type = facet_access_type %T.ref.loc9_28 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:     %.loc9_29.1: type = converted %T.ref.loc9_28, %T.as_type.loc9_29.1 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:     %T.as_type.loc9_29.2: type = facet_access_type constants.%T.a46 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:     %.loc9_29.2: type = converted constants.%T.a46, %T.as_type.loc9_29.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.530, element0 [symbolic = %ptr (constants.%ptr.ae5)]
// CHECK:STDOUT:     %.loc9_10: type = splice_block %Ptr.ref [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:       %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.a46)]
// CHECK:STDOUT:     %t.param: ref @F.%T.as_type.loc9_22.1 (%T.as_type.dc4) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.a46)]
// CHECK:STDOUT:       %T.as_type.loc9_22.2: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type.loc9_22.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.as_type.loc9_22.1 (%T.as_type.dc4) = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.ae5) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.ae5) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.a46]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Type = @Type.%assoc0
// CHECK:STDOUT:   witness = (%Type)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Type(@Ptr.%Self: %Ptr.type) {
// CHECK:STDOUT:   assoc_const Type:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc7_20.1: type) {
// CHECK:STDOUT:   %U.loc7_20.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.b59 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.4f0)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.334)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.da3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Ptr.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.2: %Ptr.type) {
// CHECK:STDOUT:   %T.loc9_6.1: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.a46)]
// CHECK:STDOUT:   %T.as_type.loc9_22.1: type = facet_access_type %T.loc9_6.1 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.as_type.loc9_22.1 [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.851)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.as_type.loc9_22.1) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.530)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_22.1 [symbolic = %ptr (constants.%ptr.ae5)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.ec6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_25 (constants.%require_complete.e69)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.as_type.loc9_22.1 [symbolic = %require_complete.loc9_15 (constants.%require_complete.43a)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.68e)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.19c)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.19c) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.6d5)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.8d6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc9_22.1 (%T.as_type.dc4)) -> @F.%ptr (%ptr.ae5) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc9_22.1 (%T.as_type.dc4) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.ae5) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.19c) = impl_witness_access constants.%Copy.lookup_impl_witness.68e, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.6d5)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.8d6)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %.loc10_10.1: init @F.%ptr (%ptr.ae5) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %.loc10_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.a46) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.690) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.8b3
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.4f0
// CHECK:STDOUT:   %require_complete => constants.%require_complete.334
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.as_type.dc4) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.as_type.dc4
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.ae5
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.483
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c7f
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.530
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%T.a46) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.a46) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.a46
// CHECK:STDOUT:   %T.as_type.loc9_22.1 => constants.%T.as_type.dc4
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.851
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.530
// CHECK:STDOUT:   %ptr => constants.%ptr.ae5
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.ec6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.a46: %Ptr.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.d28: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.659: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.690: %Ptr.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.690 [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.690, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b59: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %U.8b3 [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.4f0: type = facet_type <@Ptr where %impl.elem0.b59 = %ptr.79f131.1> [symbolic]
// CHECK:STDOUT:   %require_complete.334: <witness> = require_complete_type %Ptr_where.type.4f0 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.da3: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.8b3) [symbolic]
// CHECK:STDOUT:   %T.a46: %Ptr.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3d8: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.as_type.dc4: type = facet_access_type %T.a46 [symbolic]
// CHECK:STDOUT:   %pattern_type.851: type = pattern_type %T.as_type.dc4 [symbolic]
// CHECK:STDOUT:   %ptr.ae5: type = ptr_type %T.as_type.dc4 [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.483: type = facet_type <@Ptr where %impl.elem0.b59 = %ptr.ae5> [symbolic]
// CHECK:STDOUT:   %require_complete.c7f: <witness> = require_complete_type %Ptr_where.type.483 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.530: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.as_type.dc4) [symbolic]
// CHECK:STDOUT:   %pattern_type.ec6: type = pattern_type %ptr.ae5 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.e69: <witness> = require_complete_type %ptr.ae5 [symbolic]
// CHECK:STDOUT:   %require_complete.43a: <witness> = require_complete_type %T.as_type.dc4 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.68e: <witness> = lookup_impl_witness %ptr.ae5, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.ae5, (%Copy.lookup_impl_witness.68e) [symbolic]
// CHECK:STDOUT:   %.19c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.6d5: %.19c = impl_witness_access %Copy.lookup_impl_witness.68e, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.8d6: <specific function> = specific_impl_function %impl.elem0.6d5, @Copy.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Ptr = %Ptr.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Ptr.decl: type = interface_decl @Ptr [concrete = constants.%Ptr.type] {} {}
// CHECK:STDOUT:   impl_decl @U.as.Ptr.impl [concrete] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.690]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.690]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.b59]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.4f0)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:     %U.loc7_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.8b3) [symbolic = @U.as.Ptr.impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.da3)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.79f131.1 [symbolic = @U.as.Ptr.impl.%ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3d8 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_19 (%pattern_type.851) = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_19 (%pattern_type.851) = ref_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_19 (%pattern_type.851) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_25 (%pattern_type.ec6) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_25 (%pattern_type.ec6) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.a46)]
// CHECK:STDOUT:     %Ptr.ref.loc9_31: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %T.as_type.loc9_29: type = facet_access_type constants.%T.a46 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:     %.loc9_29: type = converted constants.%T.a46, %T.as_type.loc9_29 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.530, element0 [symbolic = %ptr (constants.%ptr.ae5)]
// CHECK:STDOUT:     %.loc9_10: type = splice_block %Ptr.ref.loc9_10 [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.659]
// CHECK:STDOUT:       %Ptr.ref.loc9_10: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.a46)]
// CHECK:STDOUT:     %t.param: ref @F.%T.as_type.loc9_22.1 (%T.as_type.dc4) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.a46)]
// CHECK:STDOUT:       %T.as_type.loc9_22.2: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type.loc9_22.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.as_type.loc9_22.1 (%T.as_type.dc4) = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.ae5) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.ae5) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.a46]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Type = @Type.%assoc0
// CHECK:STDOUT:   witness = (%Type)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Type(@Ptr.%Self: %Ptr.type) {
// CHECK:STDOUT:   assoc_const Type:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc7_20.1: type) {
// CHECK:STDOUT:   %U.loc7_20.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U.8b3)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.b59 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.4f0)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.334)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.da3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Ptr.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.2: %Ptr.type) {
// CHECK:STDOUT:   %T.loc9_6.1: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.1 (constants.%T.a46)]
// CHECK:STDOUT:   %T.as_type.loc9_22.1: type = facet_access_type %T.loc9_6.1 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.dc4)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.as_type.loc9_22.1 [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.851)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.as_type.loc9_22.1) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.530)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_22.1 [symbolic = %ptr (constants.%ptr.ae5)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.ec6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_25 (constants.%require_complete.e69)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.as_type.loc9_22.1 [symbolic = %require_complete.loc9_15 (constants.%require_complete.43a)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.68e)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.19c)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.19c) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.6d5)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.8d6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc9_22.1 (%T.as_type.dc4)) -> @F.%ptr (%ptr.ae5) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc9_22.1 (%T.as_type.dc4) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.ae5) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.19c) = impl_witness_access constants.%Copy.lookup_impl_witness.68e, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.6d5)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.8d6)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %.loc10_10.1: init @F.%ptr (%ptr.ae5) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %.loc10_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.a46) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.690) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.8b3) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.8b3
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.4f0
// CHECK:STDOUT:   %require_complete => constants.%require_complete.334
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.as_type.dc4) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.as_type.dc4
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.ae5
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.483
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c7f
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.530
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%T.a46) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.a46) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.a46
// CHECK:STDOUT:   %T.as_type.loc9_22.1 => constants.%T.as_type.dc4
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.851
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.530
// CHECK:STDOUT:   %ptr => constants.%ptr.ae5
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.ec6
// CHECK:STDOUT: }
// CHECK:STDOUT:

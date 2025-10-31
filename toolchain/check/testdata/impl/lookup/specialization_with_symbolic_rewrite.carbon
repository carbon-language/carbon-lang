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
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.4d7: type = facet_type <@Z, @Z(%T.d9f)> [symbolic]
// CHECK:STDOUT:   %Self.c41: %Z.type.4d7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.815: type = assoc_entity_type @Z, @Z(%T.d9f) [symbolic]
// CHECK:STDOUT:   %assoc0.6bf: %Z.assoc_type.815 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.c56: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.fe7: %Z.type.c56 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.af2: %Z.type.c56 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.fae: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.175: %Z.assoc_type.fae = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.de4: <witness> = require_complete_type %Z.type.c56 [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type.7e8: type = symbolic_binding_type .Self, %.Self.fe7 [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.c15: <witness> = lookup_impl_witness %.Self.fe7, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b31: type = impl_witness_access %Z.lookup_impl_witness.c15, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.d64: type = facet_type <@Z, @Z(%S) where %impl.elem0.b31 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.d2e: <witness> = require_complete_type %Z_where.type.d64 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.aba: <witness> = impl_witness file.%Z.impl_witness_table.loc9, @T.as.Z.impl.008(%T.d9f, %S) [symbolic]
// CHECK:STDOUT:   %Z.type.a71: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.7aa: %Z.type.a71 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.6ec: %Z.type.a71 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.78d: type = symbolic_binding_type .Self, %.Self.7aa [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.dee: <witness> = lookup_impl_witness %.Self.7aa, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.62f: type = impl_witness_access %Z.lookup_impl_witness.dee, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.d81: type = facet_type <@Z, @Z(%C) where %impl.elem0.62f = %T.d9f> [symbolic]
// CHECK:STDOUT:   %require_complete.909: <witness> = require_complete_type %Z_where.type.d81 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.b4b: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.030(%T.d9f) [symbolic]
// CHECK:STDOUT:   %T.38e: %Z.type.a71 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6dc: type = pattern_type %Z.type.a71 [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.38e [symbolic]
// CHECK:STDOUT:   %pattern_type.951: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9c0: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Z_where.type.593: type = facet_type <@Z, @Z(%C) where %impl.elem0.62f = %T.binding.as_type> [symbolic]
// CHECK:STDOUT:   %require_complete.bf7: <witness> = require_complete_type %Z_where.type.593 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.f1f: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.030(%T.binding.as_type) [symbolic]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc3_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.008 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc9_24.1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc9_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.c56)]
// CHECK:STDOUT:     %.Self.1: @T.as.Z.impl.008.%Z.type.loc9_42.2 (%Z.type.c56) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.fe7)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.008.%Z.type.loc9_42.2 (%Z.type.c56) = name_ref .Self, %.Self.1 [symbolic = %.Self.4 (constants.%.Self.fe7)]
// CHECK:STDOUT:     %.loc9_50.1: @T.as.Z.impl.008.%Z.assoc_type (%Z.assoc_type.fae) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.175)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.008.%Z.assoc_type (%Z.assoc_type.fae) = name_ref X, %.loc9_50.1 [symbolic = %assoc0 (constants.%assoc0.175)]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.7e8)]
// CHECK:STDOUT:     %.loc9_50.2: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.7e8)]
// CHECK:STDOUT:     %impl.elem0.loc9_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.c15, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.b31)]
// CHECK:STDOUT:     %.loc9_56.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_56.2: type = converted %.loc9_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.d64)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.c56
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_50.1, %.loc9_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc9_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:     %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %S.loc9_24.1: type = symbolic_binding S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc9 = impl_witness_table (%impl_witness_assoc_constant.loc9), @T.as.Z.impl.008 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc9: <witness> = impl_witness %Z.impl_witness_table.loc9, @T.as.Z.impl.008(constants.%T.d9f, constants.%S) [symbolic = @T.as.Z.impl.008.%Z.impl_witness (constants.%Z.impl_witness.aba)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc9: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.030 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc11_30: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.d9f)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     %.Self.1: %Z.type.a71 = symbolic_binding .Self [symbolic_self = constants.%.Self.7aa]
// CHECK:STDOUT:     %.Self.ref: %Z.type.a71 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.7aa]
// CHECK:STDOUT:     %.loc11_46.1: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.252 = name_ref X, %.loc11_46.1 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.78d]
// CHECK:STDOUT:     %.loc11_46.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.78d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.dee, element0 [symbolic_self = constants.%impl.elem0.62f]
// CHECK:STDOUT:     %T.ref.loc11_51: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.d9f)]
// CHECK:STDOUT:     %.loc11_40: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.d81)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.a71
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc11_51
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc11_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_20.2 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc11 = impl_witness_table (%impl_witness_assoc_constant.loc11), @T.as.Z.impl.030 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc11: <witness> = impl_witness %Z.impl_witness_table.loc11, @T.as.Z.impl.030(constants.%T.d9f) [symbolic = @T.as.Z.impl.030.%Z.impl_witness (constants.%Z.impl_witness.b4b)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc11: type = impl_witness_assoc_constant constants.%T.d9f [symbolic = @T.as.Z.impl.030.%T.loc11_20.2 (constants.%T.d9f)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.6dc = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type (%pattern_type.951) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type (%pattern_type.951) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_13: type = splice_block %Z.type [concrete = constants.%Z.type.a71] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_6.2: %Z.type.a71 = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T.38e)]
// CHECK:STDOUT:     %t.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_19.1: type = splice_block %.loc13_19.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc13: %Z.type.a71 = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T.38e)]
// CHECK:STDOUT:       %T.as_type.loc13: type = facet_access_type %T.ref.loc13 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc13_19.2: type = converted %T.ref.loc13, %T.as_type.loc13 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_13.2: type) {
// CHECK:STDOUT:   %T.loc3_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T.d9f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_13.1)> [symbolic = %Z.type (constants.%Z.type.4d7)]
// CHECK:STDOUT:   %Self.loc3_23.2: @Z.%Z.type (%Z.type.4d7) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.c41)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc3_13.1) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.815)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.815) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.6bf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @Z.%Z.type (%Z.type.4d7) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.c41)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.815) = assoc_entity element0, @Z.%X [symbolic = @Z.%assoc0 (constants.%assoc0.6bf)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc3_13.2: type, @Z.%Self.loc3_23.1: @Z.%Z.type (%Z.type.4d7)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.008(%T.loc9_14.1: type, %S.loc9_24.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:   %S.loc9_24.2: type = symbolic_binding S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc9_42.2: type = facet_type <@Z, @Z(%S.loc9_24.2)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.c56)]
// CHECK:STDOUT:   %.Self.4: @T.as.Z.impl.008.%Z.type.loc9_42.2 (%Z.type.c56) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.fe7)]
// CHECK:STDOUT:   %require_complete.loc9_50: <witness> = require_complete_type %Z.type.loc9_42.2 [symbolic = %require_complete.loc9_50 (constants.%require_complete.de4)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc9_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.fae)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.008.%Z.assoc_type (%Z.assoc_type.fae) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.175)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.4 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.7e8)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @Z, @Z(%S.loc9_24.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.c15)]
// CHECK:STDOUT:   %impl.elem0.loc9_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.b31)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc9_24.2) where %impl.elem0.loc9_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.d64)]
// CHECK:STDOUT:   %require_complete.loc9_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc9_44 (constants.%require_complete.d2e)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc9, @T.as.Z.impl.008(%T.loc9_14.2, %S.loc9_24.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.aba)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc9_44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.030(%T.loc11_20.1: type) {
// CHECK:STDOUT:   %T.loc11_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_20.2 (constants.%T.d9f)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.62f = %T.loc11_20.2> [symbolic = %Z_where.type (constants.%Z_where.type.d81)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.909)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.030(%T.loc11_20.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.b4b)]
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
// CHECK:STDOUT:   %T.loc13_6.1: %Z.type.a71 = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T.38e)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc13_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.951)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.9c0)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.030(%T.binding.as_type) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.f1f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type (%pattern_type.951) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc15_11.1: type = splice_block %impl.elem0 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc15: %Z.type.a71 = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T.38e)]
// CHECK:STDOUT:       %.loc15_11.2: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.252 = name_ref X, %.loc15_11.2 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc15_11.3: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.f1f, element0 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%T.binding.as_type (%T.binding.as_type) = value_binding a, %t.ref
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%T.d9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.d9f, constants.%Self.c41) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.c56
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.af2
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.fae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%.Self.fe7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.008(constants.%T.d9f, constants.%S) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.d9f
// CHECK:STDOUT:   %S.loc9_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc9_42.2 => constants.%Z.type.c56
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.fe7
// CHECK:STDOUT:   %require_complete.loc9_50 => constants.%require_complete.de4
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.fae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.175
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type.7e8
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.c15
// CHECK:STDOUT:   %impl.elem0.loc9_50.2 => constants.%impl.elem0.b31
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.d64
// CHECK:STDOUT:   %require_complete.loc9_44 => constants.%require_complete.d2e
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.aba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.a71
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.6ec
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%.Self.7aa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.030(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.d9f
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.d81
// CHECK:STDOUT:   %require_complete => constants.%require_complete.909
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.b4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.38e) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%T.38e
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.951
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.030(constants.%T.binding.as_type) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.593
// CHECK:STDOUT:   %require_complete => constants.%require_complete.bf7
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.f1f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%T.38e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.4d7: type = facet_type <@Z, @Z(%T.d9f)> [symbolic]
// CHECK:STDOUT:   %Self.c41: %Z.type.4d7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.815: type = assoc_entity_type @Z, @Z(%T.d9f) [symbolic]
// CHECK:STDOUT:   %assoc0.6bf: %Z.assoc_type.815 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.29b: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.c56: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.fe7: %Z.type.c56 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.af2: %Z.type.c56 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.fae: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.175: %Z.assoc_type.fae = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.de4: <witness> = require_complete_type %Z.type.c56 [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type.7e8: type = symbolic_binding_type .Self, %.Self.fe7 [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.c15: <witness> = lookup_impl_witness %.Self.fe7, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b31: type = impl_witness_access %Z.lookup_impl_witness.c15, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.d64: type = facet_type <@Z, @Z(%S) where %impl.elem0.b31 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.d2e: <witness> = require_complete_type %Z_where.type.d64 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.aba: <witness> = impl_witness file.%Z.impl_witness_table.loc10, @T.as.Z.impl.008(%T.d9f, %S) [symbolic]
// CHECK:STDOUT:   %Z.type.a71: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.7aa: %Z.type.a71 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.6ec: %Z.type.a71 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.78d: type = symbolic_binding_type .Self, %.Self.7aa [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.dee: <witness> = lookup_impl_witness %.Self.7aa, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.62f: type = impl_witness_access %Z.lookup_impl_witness.dee, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.d81: type = facet_type <@Z, @Z(%C) where %impl.elem0.62f = %T.d9f> [symbolic]
// CHECK:STDOUT:   %require_complete.909: <witness> = require_complete_type %Z_where.type.d81 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.b4b: <witness> = impl_witness file.%Z.impl_witness_table.loc12, @T.as.Z.impl.030(%T.d9f) [symbolic]
// CHECK:STDOUT:   %T.38e: %Z.type.a71 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6dc: type = pattern_type %Z.type.a71 [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.38e [symbolic]
// CHECK:STDOUT:   %pattern_type.951: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9c0: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.420: <witness> = lookup_impl_witness %T.38e, @Z, @Z(%C) [symbolic]
// CHECK:STDOUT:   %impl.elem0.99d: type = impl_witness_access %Z.lookup_impl_witness.420, element0 [symbolic]
// CHECK:STDOUT:   %require_complete.35b: <witness> = require_complete_type %impl.elem0.99d [symbolic]
// CHECK:STDOUT:   %pattern_type.3f9: type = pattern_type %impl.elem0.99d [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c8: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.e9f: %ImplicitAs.Convert.type.4c8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.8b5: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4d8: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.99d)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.0e5: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.99d) [symbolic]
// CHECK:STDOUT:   %assoc0.f67: %ImplicitAs.assoc_type.0e5 = assoc_entity element0, imports.%Core.import_ref.b91 [symbolic]
// CHECK:STDOUT:   %require_complete.a5d: <witness> = require_complete_type %ImplicitAs.type.4d8 [symbolic]
// CHECK:STDOUT:   %assoc0.7b6: %ImplicitAs.assoc_type.8b5 = assoc_entity element0, imports.%Core.import_ref.d11 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b9c: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8b5) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.7b6)]
// CHECK:STDOUT:   %Core.import_ref.b91: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.4c8) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.e9f)]
// CHECK:STDOUT:   %Core.import_ref.d11 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc3_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.008 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc10_24.1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc10_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.c56)]
// CHECK:STDOUT:     %.Self.1: @T.as.Z.impl.008.%Z.type.loc10_42.2 (%Z.type.c56) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.fe7)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.008.%Z.type.loc10_42.2 (%Z.type.c56) = name_ref .Self, %.Self.1 [symbolic = %.Self.4 (constants.%.Self.fe7)]
// CHECK:STDOUT:     %.loc10_50.1: @T.as.Z.impl.008.%Z.assoc_type (%Z.assoc_type.fae) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.175)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.008.%Z.assoc_type (%Z.assoc_type.fae) = name_ref X, %.loc10_50.1 [symbolic = %assoc0 (constants.%assoc0.175)]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.7e8)]
// CHECK:STDOUT:     %.loc10_50.2: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.7e8)]
// CHECK:STDOUT:     %impl.elem0.loc10_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.c15, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.b31)]
// CHECK:STDOUT:     %.loc10_56.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_56.2: type = converted %.loc10_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.d64)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.c56
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_50.1, %.loc10_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc10_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:     %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %S.loc10_24.1: type = symbolic_binding S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc10 = impl_witness_table (%impl_witness_assoc_constant.loc10), @T.as.Z.impl.008 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc10: <witness> = impl_witness %Z.impl_witness_table.loc10, @T.as.Z.impl.008(constants.%T.d9f, constants.%S) [symbolic = @T.as.Z.impl.008.%Z.impl_witness (constants.%Z.impl_witness.aba)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc10: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.030 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_24: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     %.Self.1: %Z.type.a71 = symbolic_binding .Self [symbolic_self = constants.%.Self.7aa]
// CHECK:STDOUT:     %.Self.ref: %Z.type.a71 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.7aa]
// CHECK:STDOUT:     %.loc12_40.1: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.252 = name_ref X, %.loc12_40.1 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.78d]
// CHECK:STDOUT:     %.loc12_40.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.78d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.dee, element0 [symbolic_self = constants.%impl.elem0.62f]
// CHECK:STDOUT:     %T.ref.loc12_45: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:     %.loc12_34: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.d81)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.a71
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc12_45
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc12_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc12 = impl_witness_table (%impl_witness_assoc_constant.loc12), @T.as.Z.impl.030 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc12: <witness> = impl_witness %Z.impl_witness_table.loc12, @T.as.Z.impl.030(constants.%T.d9f) [symbolic = @T.as.Z.impl.030.%Z.impl_witness (constants.%Z.impl_witness.b4b)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc12: type = impl_witness_assoc_constant constants.%T.d9f [symbolic = @T.as.Z.impl.030.%T.loc12_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.6dc = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc14 (%pattern_type.951) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc14 (%pattern_type.951) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_13: type = splice_block %Z.type [concrete = constants.%Z.type.a71] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_6.2: %Z.type.a71 = symbolic_binding T, 0 [symbolic = %T.loc14_6.1 (constants.%T.38e)]
// CHECK:STDOUT:     %t.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc14: %Z.type.a71 = name_ref T, %T.loc14_6.2 [symbolic = %T.loc14_6.1 (constants.%T.38e)]
// CHECK:STDOUT:       %T.as_type.loc14: type = facet_access_type %T.ref.loc14 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc14_19.2: type = converted %T.ref.loc14, %T.as_type.loc14 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_13.2: type) {
// CHECK:STDOUT:   %T.loc3_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T.d9f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_13.1)> [symbolic = %Z.type (constants.%Z.type.4d7)]
// CHECK:STDOUT:   %Self.loc3_23.2: @Z.%Z.type (%Z.type.4d7) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.c41)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc3_13.1) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.815)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.815) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.6bf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @Z.%Z.type (%Z.type.4d7) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.c41)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.815) = assoc_entity element0, @Z.%X [symbolic = @Z.%assoc0 (constants.%assoc0.6bf)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.29b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc3_13.2: type, @Z.%Self.loc3_23.1: @Z.%Z.type (%Z.type.4d7)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.008(%T.loc10_14.1: type, %S.loc10_24.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:   %S.loc10_24.2: type = symbolic_binding S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc10_42.2: type = facet_type <@Z, @Z(%S.loc10_24.2)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.c56)]
// CHECK:STDOUT:   %.Self.4: @T.as.Z.impl.008.%Z.type.loc10_42.2 (%Z.type.c56) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.fe7)]
// CHECK:STDOUT:   %require_complete.loc10_50: <witness> = require_complete_type %Z.type.loc10_42.2 [symbolic = %require_complete.loc10_50 (constants.%require_complete.de4)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc10_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.fae)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.008.%Z.assoc_type (%Z.assoc_type.fae) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.175)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.4 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.7e8)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @Z, @Z(%S.loc10_24.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.c15)]
// CHECK:STDOUT:   %impl.elem0.loc10_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.b31)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc10_24.2) where %impl.elem0.loc10_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.d64)]
// CHECK:STDOUT:   %require_complete.loc10_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc10_44 (constants.%require_complete.d2e)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc10, @T.as.Z.impl.008(%T.loc10_14.2, %S.loc10_24.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.aba)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc10_44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.030(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_14.2 (constants.%T.d9f)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.62f = %T.loc12_14.2> [symbolic = %Z_where.type (constants.%Z_where.type.d81)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.909)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc12, @T.as.Z.impl.030(%T.loc12_14.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.b4b)]
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
// CHECK:STDOUT:   %T.loc14_6.1: %Z.type.a71 = symbolic_binding T, 0 [symbolic = %T.loc14_6.1 (constants.%T.38e)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc14_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc14: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc14 (constants.%pattern_type.951)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc14 (constants.%require_complete.9c0)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc14_6.1, @Z, @Z(constants.%C) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.420)]
// CHECK:STDOUT:   %impl.elem0.loc22_11.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.99d)]
// CHECK:STDOUT:   %require_complete.loc22_11: <witness> = require_complete_type %impl.elem0.loc22_11.2 [symbolic = %require_complete.loc22_11 (constants.%require_complete.35b)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %impl.elem0.loc22_11.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.3f9)]
// CHECK:STDOUT:   %ImplicitAs.type.loc22_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2)> [symbolic = %ImplicitAs.type.loc22_16.2 (constants.%ImplicitAs.type.4d8)]
// CHECK:STDOUT:   %require_complete.loc22_16: <witness> = require_complete_type %ImplicitAs.type.loc22_16.2 [symbolic = %require_complete.loc22_16 (constants.%require_complete.a5d)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.0e5)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.0e5) = assoc_entity element0, imports.%Core.import_ref.b91 [symbolic = %assoc0 (constants.%assoc0.f67)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type.loc22 (%pattern_type.3f9) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %impl.elem0.loc22_11.1 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.99d)] {
// CHECK:STDOUT:       %T.ref.loc22: %Z.type.a71 = name_ref T, %T.loc14_6.2 [symbolic = %T.loc14_6.1 (constants.%T.38e)]
// CHECK:STDOUT:       %.loc22_11.2: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.252 = name_ref X, %.loc22_11.2 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc22_11.3: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc22_11.1: type = impl_witness_access constants.%Z.lookup_impl_witness.420, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.99d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ImplicitAs.type.loc22_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%impl.elem0.99d)> [symbolic = %ImplicitAs.type.loc22_16.2 (constants.%ImplicitAs.type.4d8)]
// CHECK:STDOUT:     %.loc22_16.1: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.0e5) = specific_constant imports.%Core.import_ref.b9c, @ImplicitAs(constants.%impl.elem0.99d) [symbolic = %assoc0 (constants.%assoc0.f67)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.0e5) = name_ref Convert, %.loc22_16.1 [symbolic = %assoc0 (constants.%assoc0.f67)]
// CHECK:STDOUT:     %.loc22_16.2: @F.%impl.elem0.loc22_11.2 (%impl.elem0.99d) = converted %t.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %a: @F.%impl.elem0.loc22_11.2 (%impl.elem0.99d) = value_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%T.d9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.d9f, constants.%Self.c41) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.c56
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.af2
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.fae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%.Self.fe7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.008(constants.%T.d9f, constants.%S) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T.d9f
// CHECK:STDOUT:   %S.loc10_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc10_42.2 => constants.%Z.type.c56
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.fe7
// CHECK:STDOUT:   %require_complete.loc10_50 => constants.%require_complete.de4
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.fae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.175
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type.7e8
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.c15
// CHECK:STDOUT:   %impl.elem0.loc10_50.2 => constants.%impl.elem0.b31
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.d64
// CHECK:STDOUT:   %require_complete.loc10_44 => constants.%require_complete.d2e
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.aba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.a71
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.6ec
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%.Self.7aa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.030(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.d9f
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.d81
// CHECK:STDOUT:   %require_complete => constants.%require_complete.909
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.b4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.38e) {
// CHECK:STDOUT:   %T.loc14_6.1 => constants.%T.38e
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc14 => constants.%pattern_type.951
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%T.38e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.717: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.d28: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.d9f: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.701: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.701 [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.701, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.00d: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.4f0b5c.1: type = ptr_type %U.d9f [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.0d1962.1: type = facet_type <@Ptr where %impl.elem0.00d = %ptr.4f0b5c.1> [symbolic]
// CHECK:STDOUT:   %require_complete.19c230.1: <witness> = require_complete_type %Ptr_where.type.0d1962.1 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.a48d1a.1: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.d9f) [symbolic]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %T.d9f [symbolic]
// CHECK:STDOUT:   %ptr.4f0b5c.2: type = ptr_type %T.d9f [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.0d1962.2: type = facet_type <@Ptr where %impl.elem0.00d = %ptr.4f0b5c.2> [symbolic]
// CHECK:STDOUT:   %require_complete.19c230.2: <witness> = require_complete_type %Ptr_where.type.0d1962.2 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.a48d1a.2: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %Ptr.facet: %Ptr.type = facet_value %T.d9f, (%Ptr.impl_witness.a48d1a.2) [symbolic]
// CHECK:STDOUT:   %pattern_type.a60: type = pattern_type %ptr.4f0b5c.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.482: <witness> = require_complete_type %ptr.4f0b5c.2 [symbolic]
// CHECK:STDOUT:   %require_complete.4b7: <witness> = require_complete_type %T.d9f [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c66: <witness> = lookup_impl_witness %ptr.4f0b5c.2, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.4f0b5c.2, (%Copy.lookup_impl_witness.c66) [symbolic]
// CHECK:STDOUT:   %.89d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.928: %.89d = impl_witness_access %Copy.lookup_impl_witness.c66, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6a1: <specific function> = specific_impl_function %impl.elem0.928, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.701]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.701]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.00d]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.0d1962.1)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %U.loc7_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.d9f) [symbolic = @U.as.Ptr.impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.a48d1a.1)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.4f0b5c.1 [symbolic = @U.as.Ptr.impl.%ptr.loc7_54.2 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_20 (%pattern_type.e68) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_20 (%pattern_type.e68) = ref_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_20 (%pattern_type.e68) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_26 (%pattern_type.a60) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_26 (%pattern_type.a60) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_29: type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.d9f)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %Ptr.facet.loc9_30.2: %Ptr.type = facet_value %T.ref.loc9_29, (constants.%Ptr.impl_witness.a48d1a.2) [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:     %.loc9: %Ptr.type = converted %T.ref.loc9_29, %Ptr.facet.loc9_30.2 [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.a48d1a.2, element0 [symbolic = %ptr (constants.%ptr.4f0b5c.2)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %T.loc9_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.d9f)]
// CHECK:STDOUT:     %t.param: ref @F.%T.loc9_6.1 (%T.d9f) = ref_param call_param0
// CHECK:STDOUT:     %T.ref.loc9_23: type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.d9f)]
// CHECK:STDOUT:     %t: ref @F.%T.loc9_6.1 (%T.d9f) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.4f0b5c.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.4f0b5c.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.717]
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
// CHECK:STDOUT:   %U.loc7_20.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.00d = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.0d1962.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.19c230.1)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.a48d1a.1)]
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
// CHECK:STDOUT:   %T.loc9_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.d9f)]
// CHECK:STDOUT:   %pattern_type.loc9_20: type = pattern_type %T.loc9_6.1 [symbolic = %pattern_type.loc9_20 (constants.%pattern_type.e68)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.loc9_6.1) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.a48d1a.2)]
// CHECK:STDOUT:   %Ptr.facet.loc9_30.1: %Ptr.type = facet_value %T.loc9_6.1, (%Ptr.impl_witness) [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc9_6.1 [symbolic = %ptr (constants.%ptr.4f0b5c.2)]
// CHECK:STDOUT:   %pattern_type.loc9_26: type = pattern_type %ptr [symbolic = %pattern_type.loc9_26 (constants.%pattern_type.a60)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_26: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_26 (constants.%require_complete.482)]
// CHECK:STDOUT:   %require_complete.loc9_16: <witness> = require_complete_type %T.loc9_6.1 [symbolic = %require_complete.loc9_16 (constants.%require_complete.4b7)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c66)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.89d)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.89d) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.loc9_6.1 (%T.d9f)) -> @F.%ptr (%ptr.4f0b5c.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc9_6.1 (%T.d9f) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.4f0b5c.2) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.89d) = impl_witness_access constants.%Copy.lookup_impl_witness.c66, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %.loc10_10.1: init @F.%ptr (%ptr.4f0b5c.2) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %.loc10_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.717) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.701) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.d9f) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.d9f
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.4f0b5c.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.0d1962.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.19c230.1
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.a48d1a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.d9f) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.d9f
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.4f0b5c.2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.0d1962.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.19c230.2
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.a48d1a.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.d9f
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.e68
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.a48d1a.2
// CHECK:STDOUT:   %Ptr.facet.loc9_30.1 => constants.%Ptr.facet
// CHECK:STDOUT:   %ptr => constants.%ptr.4f0b5c.2
// CHECK:STDOUT:   %pattern_type.loc9_26 => constants.%pattern_type.a60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.717: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.d28: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.d9f: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.701: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.701 [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.701, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.00d: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.4f0b5c.1: type = ptr_type %U.d9f [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.0d1: type = facet_type <@Ptr where %impl.elem0.00d = %ptr.4f0b5c.1> [symbolic]
// CHECK:STDOUT:   %require_complete.19c: <witness> = require_complete_type %Ptr_where.type.0d1 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.a48: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.d9f) [symbolic]
// CHECK:STDOUT:   %T.717: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3d8: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.28a: type = symbolic_binding_type T, 0, %T.717 [symbolic]
// CHECK:STDOUT:   %pattern_type.6f0: type = pattern_type %T.binding.as_type.28a [symbolic]
// CHECK:STDOUT:   %ptr.86c: type = ptr_type %T.binding.as_type.28a [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.1bd: type = facet_type <@Ptr where %impl.elem0.00d = %ptr.86c> [symbolic]
// CHECK:STDOUT:   %require_complete.ec0: <witness> = require_complete_type %Ptr_where.type.1bd [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.d27: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type.28a) [symbolic]
// CHECK:STDOUT:   %pattern_type.425: type = pattern_type %ptr.86c [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c88: <witness> = require_complete_type %ptr.86c [symbolic]
// CHECK:STDOUT:   %require_complete.b27: <witness> = require_complete_type %T.binding.as_type.28a [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c59: <witness> = lookup_impl_witness %ptr.86c, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.86c, (%Copy.lookup_impl_witness.c59) [symbolic]
// CHECK:STDOUT:   %.32f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1e1: %.32f = impl_witness_access %Copy.lookup_impl_witness.c59, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cac: <specific function> = specific_impl_function %impl.elem0.1e1, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.701]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.701]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.00d]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.0d1)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %U.loc7_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.d9f) [symbolic = @U.as.Ptr.impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.a48)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.4f0b5c.1 [symbolic = @U.as.Ptr.impl.%ptr.loc7_54.2 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3d8 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_19 (%pattern_type.6f0) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_19 (%pattern_type.6f0) = ref_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_19 (%pattern_type.6f0) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_25 (%pattern_type.425) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_25 (%pattern_type.425) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.717)]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %T.as_type.loc9_29: type = facet_access_type %T.ref.loc9_28 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.28a)]
// CHECK:STDOUT:     %.loc9_29: type = converted %T.ref.loc9_28, %T.as_type.loc9_29 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.28a)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.d27, element0 [symbolic = %ptr (constants.%ptr.86c)]
// CHECK:STDOUT:     %.loc9_10: type = splice_block %Ptr.ref [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.717)]
// CHECK:STDOUT:     %t.param: ref @F.%T.binding.as_type (%T.binding.as_type.28a) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.28a)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.717)]
// CHECK:STDOUT:       %T.as_type.loc9_22: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.28a)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type.loc9_22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.28a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.binding.as_type (%T.binding.as_type.28a) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.86c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.86c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.717]
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
// CHECK:STDOUT:   %U.loc7_20.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.00d = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.0d1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.19c)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.a48)]
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
// CHECK:STDOUT:   %T.loc9_6.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.717)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.28a)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.6f0)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.d27)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.binding.as_type [symbolic = %ptr (constants.%ptr.86c)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.425)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_25 (constants.%require_complete.c88)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9_15 (constants.%require_complete.b27)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c59)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.32f)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.32f) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.1e1)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.cac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type.28a)) -> @F.%ptr (%ptr.86c) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.binding.as_type (%T.binding.as_type.28a) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.86c) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.32f) = impl_witness_access constants.%Copy.lookup_impl_witness.c59, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.1e1)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.cac)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %.loc10_10.1: init @F.%ptr (%ptr.86c) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %.loc10_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.717) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.701) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.d9f) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.d9f
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.4f0b5c.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.0d1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.19c
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.a48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.binding.as_type.28a) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.binding.as_type.28a
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.86c
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.1bd
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ec0
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.d27
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%T.717) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.717) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.717
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.28a
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.6f0
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.d27
// CHECK:STDOUT:   %ptr => constants.%ptr.86c
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.425
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.717: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.d28: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.3b7: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.d9f: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.701: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.701 [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.701, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.00d: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.4f0b5c.1: type = ptr_type %U.d9f [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.0d1: type = facet_type <@Ptr where %impl.elem0.00d = %ptr.4f0b5c.1> [symbolic]
// CHECK:STDOUT:   %require_complete.19c: <witness> = require_complete_type %Ptr_where.type.0d1 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.a48: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.d9f) [symbolic]
// CHECK:STDOUT:   %T.717: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3d8: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.28a: type = symbolic_binding_type T, 0, %T.717 [symbolic]
// CHECK:STDOUT:   %pattern_type.6f0: type = pattern_type %T.binding.as_type.28a [symbolic]
// CHECK:STDOUT:   %ptr.86c: type = ptr_type %T.binding.as_type.28a [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.1bd: type = facet_type <@Ptr where %impl.elem0.00d = %ptr.86c> [symbolic]
// CHECK:STDOUT:   %require_complete.ec0: <witness> = require_complete_type %Ptr_where.type.1bd [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.d27: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type.28a) [symbolic]
// CHECK:STDOUT:   %pattern_type.425: type = pattern_type %ptr.86c [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c88: <witness> = require_complete_type %ptr.86c [symbolic]
// CHECK:STDOUT:   %require_complete.b27: <witness> = require_complete_type %T.binding.as_type.28a [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c59: <witness> = lookup_impl_witness %ptr.86c, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.86c, (%Copy.lookup_impl_witness.c59) [symbolic]
// CHECK:STDOUT:   %.32f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1e1: %.32f = impl_witness_access %Copy.lookup_impl_witness.c59, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cac: <specific function> = specific_impl_function %impl.elem0.1e1, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.701]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.701]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.00d]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.0d1)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:     %U.loc7_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.d9f) [symbolic = @U.as.Ptr.impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.a48)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.4f0b5c.1 [symbolic = @U.as.Ptr.impl.%ptr.loc7_54.2 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3d8 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_19 (%pattern_type.6f0) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_19 (%pattern_type.6f0) = ref_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_19 (%pattern_type.6f0) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_25 (%pattern_type.425) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_25 (%pattern_type.425) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.717)]
// CHECK:STDOUT:     %Ptr.ref.loc9_31: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.d27, element0 [symbolic = %ptr (constants.%ptr.86c)]
// CHECK:STDOUT:     %.loc9_10: type = splice_block %Ptr.ref.loc9_10 [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.3b7]
// CHECK:STDOUT:       %Ptr.ref.loc9_10: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.717)]
// CHECK:STDOUT:     %t.param: ref @F.%T.binding.as_type (%T.binding.as_type.28a) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.28a)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.717)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.28a)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.28a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.binding.as_type (%T.binding.as_type.28a) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.86c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.86c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.717]
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
// CHECK:STDOUT:   %U.loc7_20.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.d9f)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.4f0b5c.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.00d = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.0d1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.19c)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.a48)]
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
// CHECK:STDOUT:   %T.loc9_6.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.717)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.28a)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.6f0)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.d27)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.binding.as_type [symbolic = %ptr (constants.%ptr.86c)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.425)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_25 (constants.%require_complete.c88)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9_15 (constants.%require_complete.b27)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c59)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.32f)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.32f) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.1e1)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.cac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type.28a)) -> @F.%ptr (%ptr.86c) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.binding.as_type (%T.binding.as_type.28a) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.86c) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.32f) = impl_witness_access constants.%Copy.lookup_impl_witness.c59, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.1e1)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.cac)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %.loc10_10.1: init @F.%ptr (%ptr.86c) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %.loc10_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.717) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.701) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.d9f) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.d9f
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.4f0b5c.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.0d1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.19c
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.a48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.binding.as_type.28a) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.binding.as_type.28a
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.86c
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.1bd
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ec0
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.d27
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%T.717) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.717) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.717
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.28a
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.6f0
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.d27
// CHECK:STDOUT:   %ptr => constants.%ptr.86c
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.425
// CHECK:STDOUT: }
// CHECK:STDOUT:

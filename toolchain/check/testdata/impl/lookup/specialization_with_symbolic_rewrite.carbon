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
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.352: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.352 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.0ed: type = facet_type <@Z, @Z(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.822: %Z.type.0ed = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0bf: type = assoc_entity_type @Z, @Z(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.bb9: %Z.assoc_type.0bf = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.4d0: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.90f: %Z.type.4d0 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.427: %Z.type.4d0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.5b2: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.1d5: %Z.assoc_type.5b2 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.72d: <witness> = require_complete_type %Z.type.4d0 [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type.097: type = symbolic_binding_type .Self, %.Self.90f [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.7d6: <witness> = lookup_impl_witness %.Self.90f, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e5e: type = impl_witness_access %Z.lookup_impl_witness.7d6, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z_where.type.a76: type = facet_type <@Z, @Z(%S) where %impl.elem0.e5e = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.cfe: <witness> = impl_witness @T.as.Z.impl.34a.%Z.impl_witness_table, @T.as.Z.impl.34a(%T.67d, %S) [symbolic]
// CHECK:STDOUT:   %require_complete.1f8: <witness> = require_complete_type %Z_where.type.a76 [symbolic]
// CHECK:STDOUT:   %Z.type.be4: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.bf2: %Z.type.be4 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.af9: %Z.type.be4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.553: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.20f: %Z.assoc_type.553 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.8c6: type = symbolic_binding_type .Self, %.Self.bf2 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.c06: <witness> = lookup_impl_witness %.Self.bf2, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2cd: type = impl_witness_access %Z.lookup_impl_witness.c06, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.c3e: type = facet_type <@Z, @Z(%C) where %impl.elem0.2cd = %T.67d> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.9b0: <witness> = impl_witness @T.as.Z.impl.cf0.%Z.impl_witness_table, @T.as.Z.impl.cf0(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.864: <witness> = require_complete_type %Z_where.type.c3e [symbolic]
// CHECK:STDOUT:   %T.fe5: %Z.type.be4 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.53f: type = pattern_type %Z.type.be4 [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.fe5 [symbolic]
// CHECK:STDOUT:   %pattern_type.bb9: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.f3b: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Z_where.type.325: type = facet_type <@Z, @Z(%C) where %impl.elem0.2cd = %T.binding.as_type> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.5a3: <witness> = impl_witness @T.as.Z.impl.cf0.%Z.impl_witness_table, @T.as.Z.impl.cf0(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.267: <witness> = require_complete_type %Z_where.type.325 [symbolic]
// CHECK:STDOUT:   %.7b3: require_specific_def_type = require_specific_def @T.as.Z.impl.cf0(%T.binding.as_type) [symbolic]
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
// CHECK:STDOUT:   %Z.decl: %Z.type.352 = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %T.loc3_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.34a [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc9_24.1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc9_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.4d0)]
// CHECK:STDOUT:     %.Self.1: @T.as.Z.impl.34a.%Z.type.loc9_42.2 (%Z.type.4d0) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.34a.%Z.type.loc9_42.2 (%Z.type.4d0) = name_ref .Self, %.Self.1 [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:     %.loc9_50.1: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.1d5)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = name_ref X, %.loc9_50.1 [symbolic = %assoc0 (constants.%assoc0.1d5)]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:     %.loc9_50.2: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:     %impl.elem0.loc9_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.7d6, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.e5e)]
// CHECK:STDOUT:     %.loc9_56.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_56.2: type = converted %.loc9_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.a76)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.4d0
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_50.1, %.loc9_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %T.loc9_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %S.loc9_24.1: type = symbolic_binding S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.cf0 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc11_30: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.be4]
// CHECK:STDOUT:     %.Self.1: %Z.type.be4 = symbolic_binding .Self [symbolic_self = constants.%.Self.bf2]
// CHECK:STDOUT:     %.Self.ref: %Z.type.be4 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.bf2]
// CHECK:STDOUT:     %.loc11_46.1: %Z.assoc_type.553 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.20f]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.553 = name_ref X, %.loc11_46.1 [concrete = constants.%assoc0.20f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.8c6]
// CHECK:STDOUT:     %.loc11_46.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.8c6]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.c06, element0 [symbolic_self = constants.%impl.elem0.2cd]
// CHECK:STDOUT:     %T.ref.loc11_51: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc11_40: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.c3e)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.be4
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc11_51
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %T.loc11_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_20.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.53f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type (%pattern_type.bb9) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type (%pattern_type.bb9) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_13: type = splice_block %Z.type [concrete = constants.%Z.type.be4] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.be4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_6.2: %Z.type.be4 = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T.fe5)]
// CHECK:STDOUT:     %t.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_19.1: type = splice_block %.loc13_19.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc13: %Z.type.be4 = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T.fe5)]
// CHECK:STDOUT:       %T.as_type.loc13: type = facet_access_type %T.ref.loc13 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc13_19.2: type = converted %T.ref.loc13, %T.as_type.loc13 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_13.2: type) {
// CHECK:STDOUT:   %T.loc3_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_13.1)> [symbolic = %Z.type (constants.%Z.type.0ed)]
// CHECK:STDOUT:   %Self.loc3_23.2: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.822)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc3_13.1) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.0bf)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.0bf) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.bb9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.822)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.0bf) = assoc_entity element0, @Z.%X [symbolic = @Z.%assoc0 (constants.%assoc0.bb9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc3_13.2: type, @Z.%Self.loc3_23.1: @Z.%Z.type (%Z.type.0ed)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.34a(%T.loc9_14.1: type, %S.loc9_24.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   %S.loc9_24.2: type = symbolic_binding S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc9_42.2: type = facet_type <@Z, @Z(%S.loc9_24.2)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.4d0)]
// CHECK:STDOUT:   %.Self.4: @T.as.Z.impl.34a.%Z.type.loc9_42.2 (%Z.type.4d0) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:   %require_complete.loc9_50: <witness> = require_complete_type %Z.type.loc9_42.2 [symbolic = %require_complete.loc9_50 (constants.%require_complete.72d)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc9_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.5b2)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.1d5)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.4 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @Z, @Z(%S.loc9_24.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.7d6)]
// CHECK:STDOUT:   %impl.elem0.loc9_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.e5e)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc9_24.2) where %impl.elem0.loc9_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.a76)]
// CHECK:STDOUT:   %Z.impl_witness.loc9_58.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.34a(%T.loc9_14.2, %S.loc9_24.2) [symbolic = %Z.impl_witness.loc9_58.2 (constants.%Z.impl_witness.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc9_44 (constants.%require_complete.1f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc9_44 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.34a [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc9_58.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.34a(constants.%T.67d, constants.%S) [symbolic = %Z.impl_witness.loc9_58.2 (constants.%Z.impl_witness.cfe)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Z.impl_witness.loc9_58.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.cf0(%T.loc11_20.1: type) {
// CHECK:STDOUT:   %T.loc11_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_20.2 (constants.%T.67d)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.2cd = %T.loc11_20.2> [symbolic = %Z_where.type (constants.%Z_where.type.c3e)]
// CHECK:STDOUT:   %Z.impl_witness.loc11_53.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.cf0(%T.loc11_20.2) [symbolic = %Z.impl_witness.loc11_53.2 (constants.%Z.impl_witness.9b0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.864)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc11_30 as %.loc11_40 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.cf0 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc11_53.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.cf0(constants.%T.67d) [symbolic = %Z.impl_witness.loc11_53.2 (constants.%Z.impl_witness.9b0)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%T.67d [symbolic = %T.loc11_20.2 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Z.impl_witness.loc11_53.1
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
// CHECK:STDOUT: generic fn @F(%T.loc13_6.2: %Z.type.be4) {
// CHECK:STDOUT:   %T.loc13_6.1: %Z.type.be4 = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T.fe5)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc13_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.bb9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.f3b)]
// CHECK:STDOUT:   %.loc15_11.4: require_specific_def_type = require_specific_def @T.as.Z.impl.cf0(%T.binding.as_type) [symbolic = %.loc15_11.4 (constants.%.7b3)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @T.as.Z.impl.cf0.%Z.impl_witness_table, @T.as.Z.impl.cf0(%T.binding.as_type) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.5a3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type (%pattern_type.bb9) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc15_11.1: type = splice_block %impl.elem0 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc15: %Z.type.be4 = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T.fe5)]
// CHECK:STDOUT:       %.loc15_11.2: %Z.assoc_type.553 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.20f]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.553 = name_ref X, %.loc15_11.2 [concrete = constants.%assoc0.20f]
// CHECK:STDOUT:       %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc15_11.3: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.5a3, element0 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%T.binding.as_type (%T.binding.as_type) = value_binding a, %t.ref
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.67d, constants.%Self.822) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.4d0
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.427
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%.Self.90f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.34a(constants.%T.67d, constants.%S) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.67d
// CHECK:STDOUT:   %S.loc9_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc9_42.2 => constants.%Z.type.4d0
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.90f
// CHECK:STDOUT:   %require_complete.loc9_50 => constants.%require_complete.72d
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1d5
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type.097
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.7d6
// CHECK:STDOUT:   %impl.elem0.loc9_50.2 => constants.%impl.elem0.e5e
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.a76
// CHECK:STDOUT:   %Z.impl_witness.loc9_58.2 => constants.%Z.impl_witness.cfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.be4
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.af9
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.553
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.20f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%.Self.bf2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.cf0(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.67d
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.c3e
// CHECK:STDOUT:   %Z.impl_witness.loc11_53.2 => constants.%Z.impl_witness.9b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.fe5) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%T.fe5
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bb9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.cf0(constants.%T.binding.as_type) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.325
// CHECK:STDOUT:   %Z.impl_witness.loc11_53.2 => constants.%Z.impl_witness.5a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.267
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%T.fe5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.352: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.352 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.0ed: type = facet_type <@Z, @Z(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.822: %Z.type.0ed = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0bf: type = assoc_entity_type @Z, @Z(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.bb9: %Z.assoc_type.0bf = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.162: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.4d0: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.90f: %Z.type.4d0 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.427: %Z.type.4d0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.5b2: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.1d5: %Z.assoc_type.5b2 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.72d: <witness> = require_complete_type %Z.type.4d0 [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type.097: type = symbolic_binding_type .Self, %.Self.90f [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.7d6: <witness> = lookup_impl_witness %.Self.90f, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e5e: type = impl_witness_access %Z.lookup_impl_witness.7d6, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z_where.type.a76: type = facet_type <@Z, @Z(%S) where %impl.elem0.e5e = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.cfe: <witness> = impl_witness @T.as.Z.impl.34a.%Z.impl_witness_table, @T.as.Z.impl.34a(%T.67d, %S) [symbolic]
// CHECK:STDOUT:   %require_complete.1f8: <witness> = require_complete_type %Z_where.type.a76 [symbolic]
// CHECK:STDOUT:   %Z.type.be4: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.bf2: %Z.type.be4 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.af9: %Z.type.be4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.553: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.20f: %Z.assoc_type.553 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.8c6: type = symbolic_binding_type .Self, %.Self.bf2 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.c06: <witness> = lookup_impl_witness %.Self.bf2, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2cd: type = impl_witness_access %Z.lookup_impl_witness.c06, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.c3e: type = facet_type <@Z, @Z(%C) where %impl.elem0.2cd = %T.67d> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.9b0: <witness> = impl_witness @T.as.Z.impl.cf0.%Z.impl_witness_table, @T.as.Z.impl.cf0(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.864: <witness> = require_complete_type %Z_where.type.c3e [symbolic]
// CHECK:STDOUT:   %T.fe5: %Z.type.be4 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.53f: type = pattern_type %Z.type.be4 [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.fe5 [symbolic]
// CHECK:STDOUT:   %pattern_type.bb9: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.f3b: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.1cb: <witness> = lookup_impl_witness %T.fe5, @Z, @Z(%C) [symbolic]
// CHECK:STDOUT:   %impl.elem0.a5e: type = impl_witness_access %Z.lookup_impl_witness.1cb, element0 [symbolic]
// CHECK:STDOUT:   %require_complete.7cf: <witness> = require_complete_type %impl.elem0.a5e [symbolic]
// CHECK:STDOUT:   %pattern_type.7ae: type = pattern_type %impl.elem0.a5e [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6d3: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.50f: %ImplicitAs.Convert.type.6d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7e4: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.a5e)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ac6: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.a5e) [symbolic]
// CHECK:STDOUT:   %assoc0.43e: %ImplicitAs.assoc_type.ac6 = assoc_entity element0, imports.%Core.import_ref.218 [symbolic]
// CHECK:STDOUT:   %require_complete.591: <witness> = require_complete_type %ImplicitAs.type.7e4 [symbolic]
// CHECK:STDOUT:   %assoc0.0bc: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.fa2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a80: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.0bc)]
// CHECK:STDOUT:   %Core.import_ref.218: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.6d3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.50f)]
// CHECK:STDOUT:   %Core.import_ref.fa2 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:   %Z.decl: %Z.type.352 = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %T.loc3_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.34a [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc10_24.1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc10_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.4d0)]
// CHECK:STDOUT:     %.Self.1: @T.as.Z.impl.34a.%Z.type.loc10_42.2 (%Z.type.4d0) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.34a.%Z.type.loc10_42.2 (%Z.type.4d0) = name_ref .Self, %.Self.1 [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:     %.loc10_50.1: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.1d5)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = name_ref X, %.loc10_50.1 [symbolic = %assoc0 (constants.%assoc0.1d5)]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:     %.loc10_50.2: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:     %impl.elem0.loc10_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.7d6, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.e5e)]
// CHECK:STDOUT:     %.loc10_56.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_56.2: type = converted %.loc10_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.a76)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.4d0
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_50.1, %.loc10_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %T.loc10_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %S.loc10_24.1: type = symbolic_binding S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.cf0 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_24: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.be4]
// CHECK:STDOUT:     %.Self.1: %Z.type.be4 = symbolic_binding .Self [symbolic_self = constants.%.Self.bf2]
// CHECK:STDOUT:     %.Self.ref: %Z.type.be4 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.bf2]
// CHECK:STDOUT:     %.loc12_40.1: %Z.assoc_type.553 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.20f]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.553 = name_ref X, %.loc12_40.1 [concrete = constants.%assoc0.20f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.8c6]
// CHECK:STDOUT:     %.loc12_40.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.8c6]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.c06, element0 [symbolic_self = constants.%impl.elem0.2cd]
// CHECK:STDOUT:     %T.ref.loc12_45: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc12_34: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.c3e)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.be4
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc12_45
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %T.loc12_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.53f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc14 (%pattern_type.bb9) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc14 (%pattern_type.bb9) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_13: type = splice_block %Z.type [concrete = constants.%Z.type.be4] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.be4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_6.2: %Z.type.be4 = symbolic_binding T, 0 [symbolic = %T.loc14_6.1 (constants.%T.fe5)]
// CHECK:STDOUT:     %t.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc14: %Z.type.be4 = name_ref T, %T.loc14_6.2 [symbolic = %T.loc14_6.1 (constants.%T.fe5)]
// CHECK:STDOUT:       %T.as_type.loc14: type = facet_access_type %T.ref.loc14 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc14_19.2: type = converted %T.ref.loc14, %T.as_type.loc14 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_13.2: type) {
// CHECK:STDOUT:   %T.loc3_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_13.1)> [symbolic = %Z.type (constants.%Z.type.0ed)]
// CHECK:STDOUT:   %Self.loc3_23.2: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.822)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc3_13.1) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.0bf)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.0bf) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.bb9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.822)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.0bf) = assoc_entity element0, @Z.%X [symbolic = @Z.%assoc0 (constants.%assoc0.bb9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.162]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc3_13.2: type, @Z.%Self.loc3_23.1: @Z.%Z.type (%Z.type.0ed)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.34a(%T.loc10_14.1: type, %S.loc10_24.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   %S.loc10_24.2: type = symbolic_binding S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc10_42.2: type = facet_type <@Z, @Z(%S.loc10_24.2)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.4d0)]
// CHECK:STDOUT:   %.Self.4: @T.as.Z.impl.34a.%Z.type.loc10_42.2 (%Z.type.4d0) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:   %require_complete.loc10_50: <witness> = require_complete_type %Z.type.loc10_42.2 [symbolic = %require_complete.loc10_50 (constants.%require_complete.72d)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc10_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.5b2)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.1d5)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.4 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @Z, @Z(%S.loc10_24.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.7d6)]
// CHECK:STDOUT:   %impl.elem0.loc10_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.e5e)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc10_24.2) where %impl.elem0.loc10_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.a76)]
// CHECK:STDOUT:   %Z.impl_witness.loc10_58.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.34a(%T.loc10_14.2, %S.loc10_24.2) [symbolic = %Z.impl_witness.loc10_58.2 (constants.%Z.impl_witness.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc10_44 (constants.%require_complete.1f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc10_44 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.34a [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc10_58.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.34a(constants.%T.67d, constants.%S) [symbolic = %Z.impl_witness.loc10_58.2 (constants.%Z.impl_witness.cfe)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Z.impl_witness.loc10_58.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.cf0(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.2cd = %T.loc12_14.2> [symbolic = %Z_where.type (constants.%Z_where.type.c3e)]
// CHECK:STDOUT:   %Z.impl_witness.loc12_47.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.cf0(%T.loc12_14.2) [symbolic = %Z.impl_witness.loc12_47.2 (constants.%Z.impl_witness.9b0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.864)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc12_24 as %.loc12_34 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.cf0 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc12_47.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.cf0(constants.%T.67d) [symbolic = %Z.impl_witness.loc12_47.2 (constants.%Z.impl_witness.9b0)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%T.67d [symbolic = %T.loc12_14.2 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Z.impl_witness.loc12_47.1
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
// CHECK:STDOUT: generic fn @F(%T.loc14_6.2: %Z.type.be4) {
// CHECK:STDOUT:   %T.loc14_6.1: %Z.type.be4 = symbolic_binding T, 0 [symbolic = %T.loc14_6.1 (constants.%T.fe5)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc14_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc14: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc14 (constants.%pattern_type.bb9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc14 (constants.%require_complete.f3b)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc14_6.1, @Z, @Z(constants.%C) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.1cb)]
// CHECK:STDOUT:   %impl.elem0.loc22_11.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.a5e)]
// CHECK:STDOUT:   %require_complete.loc22_11: <witness> = require_complete_type %impl.elem0.loc22_11.2 [symbolic = %require_complete.loc22_11 (constants.%require_complete.7cf)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %impl.elem0.loc22_11.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.7ae)]
// CHECK:STDOUT:   %ImplicitAs.type.loc22_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2)> [symbolic = %ImplicitAs.type.loc22_16.2 (constants.%ImplicitAs.type.7e4)]
// CHECK:STDOUT:   %require_complete.loc22_16: <witness> = require_complete_type %ImplicitAs.type.loc22_16.2 [symbolic = %require_complete.loc22_16 (constants.%require_complete.591)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ac6)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ac6) = assoc_entity element0, imports.%Core.import_ref.218 [symbolic = %assoc0 (constants.%assoc0.43e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type.loc22 (%pattern_type.7ae) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %impl.elem0.loc22_11.1 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.a5e)] {
// CHECK:STDOUT:       %T.ref.loc22: %Z.type.be4 = name_ref T, %T.loc14_6.2 [symbolic = %T.loc14_6.1 (constants.%T.fe5)]
// CHECK:STDOUT:       %.loc22_11.2: %Z.assoc_type.553 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.20f]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.553 = name_ref X, %.loc22_11.2 [concrete = constants.%assoc0.20f]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc22_11.3: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc22_11.1: type = impl_witness_access constants.%Z.lookup_impl_witness.1cb, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.a5e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ImplicitAs.type.loc22_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%impl.elem0.a5e)> [symbolic = %ImplicitAs.type.loc22_16.2 (constants.%ImplicitAs.type.7e4)]
// CHECK:STDOUT:     %.loc22_16.1: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ac6) = specific_constant imports.%Core.import_ref.a80, @ImplicitAs(constants.%impl.elem0.a5e) [symbolic = %assoc0 (constants.%assoc0.43e)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ac6) = name_ref Convert, %.loc22_16.1 [symbolic = %assoc0 (constants.%assoc0.43e)]
// CHECK:STDOUT:     %.loc22_16.2: @F.%impl.elem0.loc22_11.2 (%impl.elem0.a5e) = converted %t.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %a: @F.%impl.elem0.loc22_11.2 (%impl.elem0.a5e) = value_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.67d, constants.%Self.822) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.4d0
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.427
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%.Self.90f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.34a(constants.%T.67d, constants.%S) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T.67d
// CHECK:STDOUT:   %S.loc10_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc10_42.2 => constants.%Z.type.4d0
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.90f
// CHECK:STDOUT:   %require_complete.loc10_50 => constants.%require_complete.72d
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1d5
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type.097
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.7d6
// CHECK:STDOUT:   %impl.elem0.loc10_50.2 => constants.%impl.elem0.e5e
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.a76
// CHECK:STDOUT:   %Z.impl_witness.loc10_58.2 => constants.%Z.impl_witness.cfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.be4
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.af9
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.553
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.20f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%.Self.bf2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.cf0(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.67d
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.c3e
// CHECK:STDOUT:   %Z.impl_witness.loc12_47.2 => constants.%Z.impl_witness.9b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.fe5) {
// CHECK:STDOUT:   %T.loc14_6.1 => constants.%T.fe5
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc14 => constants.%pattern_type.bb9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%T.fe5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.0e0: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.516: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.47a: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.47a [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.47a, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ac: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.3963c5.1: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.117547.1: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.73ad1d.1: <witness> = require_complete_type %Ptr_where.type.3963c5.1 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.3963c5.2: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.e8f8f9.2> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.117547.2: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.73ad1d.2: <witness> = require_complete_type %Ptr_where.type.3963c5.2 [symbolic]
// CHECK:STDOUT:   %.656: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Ptr.facet: %Ptr.type = facet_value %T.67d, (%Ptr.impl_witness.117547.2) [symbolic]
// CHECK:STDOUT:   %.ba4: form = init_form %ptr.e8f8f9.2, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.2f2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f8f9.2, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f8f9.2, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %.b46: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.201: %.b46 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b84: <specific function> = specific_impl_function %impl.elem0.201, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.516]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.4ac]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.3963c5.1)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %U.loc7_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_20 (%pattern_type.51d) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_20 (%pattern_type.51d) = var_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_20 (%pattern_type.51d) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_26 (%pattern_type.4f4) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_26 (%pattern_type.4f4) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_29: type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.516]
// CHECK:STDOUT:     %Ptr.facet.loc9_30.2: %Ptr.type = facet_value %T.ref.loc9_29, (constants.%Ptr.impl_witness.117547.2) [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:     %.loc9_30.3: %Ptr.type = converted %T.ref.loc9_29, %Ptr.facet.loc9_30.2 [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.117547.2, element0 [symbolic = %ptr (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:     %.loc9_30.4: form = init_form %impl.elem0.loc9, call_param1 [symbolic = %.loc9_30.2 (constants.%.ba4)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %T.loc9_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.67d)]
// CHECK:STDOUT:     %t.param: ref @F.%T.loc9_6.1 (%T.67d) = ref_param call_param0
// CHECK:STDOUT:     %T.ref.loc9_23: type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.67d)]
// CHECK:STDOUT:     %t: ref @F.%T.loc9_6.1 (%T.67d) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.e8f8f9.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.e8f8f9.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.0e0]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0.516]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Type = @Type.%assoc0
// CHECK:STDOUT:   witness = (%Type)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Type(@Ptr.%Self: %Ptr.type) {
// CHECK:STDOUT:   assoc_const Type:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc7_20.1: type) {
// CHECK:STDOUT:   %U.loc7_20.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.4ac = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.3963c5.1)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.117547.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.73ad1d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc7_56.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.117547.1)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc7_56.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.2: type) {
// CHECK:STDOUT:   %T.loc9_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.67d)]
// CHECK:STDOUT:   %pattern_type.loc9_20: type = pattern_type %T.loc9_6.1 [symbolic = %pattern_type.loc9_20 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %.loc9_30.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.loc9_6.1) [symbolic = %.loc9_30.1 (constants.%.656)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.loc9_6.1) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.117547.2)]
// CHECK:STDOUT:   %Ptr.facet.loc9_30.1: %Ptr.type = facet_value %T.loc9_6.1, (%Ptr.impl_witness) [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc9_6.1 [symbolic = %ptr (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %.loc9_30.2: form = init_form %ptr, call_param1 [symbolic = %.loc9_30.2 (constants.%.ba4)]
// CHECK:STDOUT:   %pattern_type.loc9_26: type = pattern_type %ptr [symbolic = %pattern_type.loc9_26 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_30: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_30 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %require_complete.loc9_16: <witness> = require_complete_type %T.loc9_6.1 [symbolic = %require_complete.loc9_16 (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc10_10.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc9_6.1) [symbolic = %.loc10_10.1 (constants.%.2f2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.b46)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.b46) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.201)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.b84)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.loc9_6.1 (%T.67d)) -> @F.%ptr (%ptr.e8f8f9.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc9_6.1 (%T.67d) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.e8f8f9.2) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.b46) = impl_witness_access constants.%Copy.lookup_impl_witness.2e6, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.201)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.b84)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %Copy.Op.call: init @F.%ptr (%ptr.e8f8f9.2) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %Copy.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.0e0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.47a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.3963c5.1
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.117547.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.67d) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.3963c5.2
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.117547.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.73ad1d.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.67d
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.51d
// CHECK:STDOUT:   %.loc9_30.1 => constants.%.656
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.117547.2
// CHECK:STDOUT:   %Ptr.facet.loc9_30.1 => constants.%Ptr.facet
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %.loc9_30.2 => constants.%.ba4
// CHECK:STDOUT:   %pattern_type.loc9_26 => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.0e0: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.516: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.47a: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.47a [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.47a, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ac: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.396: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.117: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.73a: <witness> = require_complete_type %Ptr_where.type.396 [symbolic]
// CHECK:STDOUT:   %T.ffa: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6d2: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.3dc: type = symbolic_binding_type T, 0, %T.ffa [symbolic]
// CHECK:STDOUT:   %pattern_type.82a: type = pattern_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %ptr.a86: type = ptr_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.540: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.a86> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.373: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %require_complete.2f9: <witness> = require_complete_type %Ptr_where.type.540 [symbolic]
// CHECK:STDOUT:   %.0a6: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %.5ca: form = init_form %ptr.a86, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.804: type = pattern_type %ptr.a86 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c1c: <witness> = require_complete_type %ptr.a86 [symbolic]
// CHECK:STDOUT:   %require_complete.552: <witness> = require_complete_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.54b: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3c7: <witness> = lookup_impl_witness %ptr.a86, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.a86, (%Copy.lookup_impl_witness.3c7) [symbolic]
// CHECK:STDOUT:   %.4b6: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.8b5: %.4b6 = impl_witness_access %Copy.lookup_impl_witness.3c7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4b3: <specific function> = specific_impl_function %impl.elem0.8b5, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.516]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.4ac]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.396)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %U.loc7_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.6d2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_19 (%pattern_type.82a) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_19 (%pattern_type.82a) = var_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_19 (%pattern_type.82a) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_25 (%pattern_type.804) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_25 (%pattern_type.804) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.ffa)]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.516]
// CHECK:STDOUT:     %T.as_type.loc9_29: type = facet_access_type %T.ref.loc9_28 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:     %.loc9_29.3: type = converted %T.ref.loc9_28, %T.as_type.loc9_29 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.373, element0 [symbolic = %ptr (constants.%ptr.a86)]
// CHECK:STDOUT:     %.loc9_29.4: form = init_form %impl.elem0.loc9, call_param1 [symbolic = %.loc9_29.2 (constants.%.5ca)]
// CHECK:STDOUT:     %.loc9_10: type = splice_block %Ptr.ref [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.ffa)]
// CHECK:STDOUT:     %t.param: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.ffa)]
// CHECK:STDOUT:       %T.as_type.loc9_22: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type.loc9_22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.a86) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.a86) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.0e0]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0.516]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Type = @Type.%assoc0
// CHECK:STDOUT:   witness = (%Type)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Type(@Ptr.%Self: %Ptr.type) {
// CHECK:STDOUT:   assoc_const Type:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc7_20.1: type) {
// CHECK:STDOUT:   %U.loc7_20.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.4ac = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.396)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.117)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.73a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc7_56.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.117)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc7_56.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.2: %Ptr.type) {
// CHECK:STDOUT:   %T.loc9_6.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.ffa)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.82a)]
// CHECK:STDOUT:   %.loc9_29.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %.loc9_29.1 (constants.%.0a6)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.373)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.binding.as_type [symbolic = %ptr (constants.%ptr.a86)]
// CHECK:STDOUT:   %.loc9_29.2: form = init_form %ptr, call_param1 [symbolic = %.loc9_29.2 (constants.%.5ca)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.804)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_29: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_29 (constants.%require_complete.c1c)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9_15 (constants.%require_complete.552)]
// CHECK:STDOUT:   %.loc10_10.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type) [symbolic = %.loc10_10.1 (constants.%.54b)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3c7)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.4b6)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.4b6) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.8b5)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.4b3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type.3dc)) -> @F.%ptr (%ptr.a86) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.a86) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.4b6) = impl_witness_access constants.%Copy.lookup_impl_witness.3c7, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.8b5)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.4b3)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %Copy.Op.call: init @F.%ptr (%ptr.a86) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %Copy.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.0e0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.47a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.396
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.binding.as_type.3dc) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.binding.as_type.3dc
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.a86
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.540
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.373
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%T.ffa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.ffa) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.ffa
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.3dc
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.82a
// CHECK:STDOUT:   %.loc9_29.1 => constants.%.0a6
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.373
// CHECK:STDOUT:   %ptr => constants.%ptr.a86
// CHECK:STDOUT:   %.loc9_29.2 => constants.%.5ca
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.804
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.0e0: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.516: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.47a: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.47a [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.47a, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ac: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.396: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.117: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.73a: <witness> = require_complete_type %Ptr_where.type.396 [symbolic]
// CHECK:STDOUT:   %T.ffa: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6d2: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.3dc: type = symbolic_binding_type T, 0, %T.ffa [symbolic]
// CHECK:STDOUT:   %pattern_type.82a: type = pattern_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %ptr.a86: type = ptr_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.540: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.a86> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.373: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %require_complete.2f9: <witness> = require_complete_type %Ptr_where.type.540 [symbolic]
// CHECK:STDOUT:   %.0a6: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %.5ca: form = init_form %ptr.a86, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.804: type = pattern_type %ptr.a86 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c1c: <witness> = require_complete_type %ptr.a86 [symbolic]
// CHECK:STDOUT:   %require_complete.552: <witness> = require_complete_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.54b: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3c7: <witness> = lookup_impl_witness %ptr.a86, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.a86, (%Copy.lookup_impl_witness.3c7) [symbolic]
// CHECK:STDOUT:   %.4b6: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.8b5: %.4b6 = impl_witness_access %Copy.lookup_impl_witness.3c7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4b3: <specific function> = specific_impl_function %impl.elem0.8b5, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.516]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.4ac]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.396)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:     %U.loc7_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.6d2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_19 (%pattern_type.82a) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_19 (%pattern_type.82a) = var_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_19 (%pattern_type.82a) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_25 (%pattern_type.804) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_25 (%pattern_type.804) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.ffa)]
// CHECK:STDOUT:     %Ptr.ref.loc9_31: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.516]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.373, element0 [symbolic = %ptr (constants.%ptr.a86)]
// CHECK:STDOUT:     %.loc9_29.3: form = init_form %impl.elem0.loc9, call_param1 [symbolic = %.loc9_29.2 (constants.%.5ca)]
// CHECK:STDOUT:     %.loc9_10: type = splice_block %Ptr.ref.loc9_10 [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Ptr.ref.loc9_10: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.ffa)]
// CHECK:STDOUT:     %t.param: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.ffa)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.a86) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.a86) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.0e0]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0.516]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Type = @Type.%assoc0
// CHECK:STDOUT:   witness = (%Type)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Type(@Ptr.%Self: %Ptr.type) {
// CHECK:STDOUT:   assoc_const Type:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc7_20.1: type) {
// CHECK:STDOUT:   %U.loc7_20.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.4ac = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.396)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.117)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.73a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc7_56.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.117)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc7_56.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.2: %Ptr.type) {
// CHECK:STDOUT:   %T.loc9_6.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.ffa)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.82a)]
// CHECK:STDOUT:   %.loc9_29.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %.loc9_29.1 (constants.%.0a6)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.373)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.binding.as_type [symbolic = %ptr (constants.%ptr.a86)]
// CHECK:STDOUT:   %.loc9_29.2: form = init_form %ptr, call_param1 [symbolic = %.loc9_29.2 (constants.%.5ca)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.804)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_29: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_29 (constants.%require_complete.c1c)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9_15 (constants.%require_complete.552)]
// CHECK:STDOUT:   %.loc10_10.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type) [symbolic = %.loc10_10.1 (constants.%.54b)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3c7)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.4b6)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.4b6) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.8b5)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.4b3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type.3dc)) -> @F.%ptr (%ptr.a86) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.a86) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.4b6) = impl_witness_access constants.%Copy.lookup_impl_witness.3c7, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.8b5)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.4b3)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %Copy.Op.call: init @F.%ptr (%ptr.a86) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %Copy.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.0e0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.47a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.396
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.binding.as_type.3dc) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.binding.as_type.3dc
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.a86
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.540
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.373
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%T.ffa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.ffa) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.ffa
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.3dc
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.82a
// CHECK:STDOUT:   %.loc9_29.1 => constants.%.0a6
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.373
// CHECK:STDOUT:   %ptr => constants.%ptr.a86
// CHECK:STDOUT:   %.loc9_29.2 => constants.%.5ca
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.804
// CHECK:STDOUT: }
// CHECK:STDOUT:

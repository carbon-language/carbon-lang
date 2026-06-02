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
  let unused a: T.X = t;
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
  // CHECK:STDERR: fail_nonfinal_specialized_symbolic_rewrite.carbon:[[@LINE+7]]:23: error: cannot implicitly convert expression of type `T` to `T.(Z(C).X)` [ConversionFailure]
  // CHECK:STDERR:   let unused a: T.X = t;
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: fail_nonfinal_specialized_symbolic_rewrite.carbon:[[@LINE+4]]:23: note: type `T` does not implement interface `Core.ImplicitAs(T.(Z(C).X))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let unused a: T.X = t;
  // CHECK:STDERR:                       ^
  // CHECK:STDERR:
  let unused a: T.X = t;
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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Z.type.8a6: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.8a6 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.284: type = facet_type <@Z, @Z(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.f76: %Z.type.284 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.e20: type = assoc_entity_type @Z, @Z(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.5be: %Z.assoc_type.e20 = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.6ef: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.d86: %Z.type.6ef = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.d23: %Z.type.6ef = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.795: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.281: %Z.assoc_type.795 = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %require_complete.c80: <witness> = require_complete_type %Z.type.6ef [symbolic]
// CHECK:STDOUT:   %.Self.as_type.1a7: type = facet_access_type %.Self.d86 [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.18f: <witness> = lookup_impl_witness %.Self.d86, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.6c0: type = impl_witness_access %Z.lookup_impl_witness.18f, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z_where.type.d8a: type = facet_type <@Z, @Z(%S) where %impl.elem0.6c0 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.62a: <witness> = impl_witness @T.as.Z.impl.e64.%Z.impl_witness_table, @T.as.Z.impl.e64(%T.67d, %S) [symbolic]
// CHECK:STDOUT:   %require_complete.310: <witness> = require_complete_type %Z_where.type.d8a [symbolic]
// CHECK:STDOUT:   %Z.facet.59a: %Z.type.6ef = facet_value %T.67d, (%Z.impl_witness.62a) [symbolic]
// CHECK:STDOUT:   %Z.type.2d0: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.e6c: %Z.type.2d0 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.f16: %Z.type.2d0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.8db: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.f02: %Z.assoc_type.8db = assoc_entity element0, @Z.WithSelf.%X [concrete]
// CHECK:STDOUT:   %.Self.as_type.e4e: type = facet_access_type %.Self.e6c [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.c21: <witness> = lookup_impl_witness %.Self.e6c, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fe4: type = impl_witness_access %Z.lookup_impl_witness.c21, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.346: type = facet_type <@Z, @Z(%C) where %impl.elem0.fe4 = %T.67d> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.61a: <witness> = impl_witness @T.as.Z.impl.b90.%Z.impl_witness_table, @T.as.Z.impl.b90(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.3d6: <witness> = require_complete_type %Z_where.type.346 [symbolic]
// CHECK:STDOUT:   %Z.facet.587: %Z.type.2d0 = facet_value %T.67d, (%Z.impl_witness.61a) [symbolic]
// CHECK:STDOUT:   %pattern_type.59a: type = pattern_type %Z.type.2d0 [concrete]
// CHECK:STDOUT:   %T.196: %Z.type.2d0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.196 [symbolic]
// CHECK:STDOUT:   %pattern_type.56b: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c09: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Z_where.type.c40: type = facet_type <@Z, @Z(%C) where %impl.elem0.fe4 = %T.as_type> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.f75: <witness> = impl_witness @T.as.Z.impl.b90.%Z.impl_witness_table, @T.as.Z.impl.b90(%T.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.386: <witness> = require_complete_type %Z_where.type.c40 [symbolic]
// CHECK:STDOUT:   %.639: require_specific_def_type = require_specific_def @T.as.Z.impl.b90(%T.as_type) [symbolic]
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
// CHECK:STDOUT:   %Z.decl: %Z.type.8a6 = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_17.1: type = splice_block %.loc3_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc3_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.e64 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_15.1 [symbolic = %T.loc9_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.8a6 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc9_25.1 [symbolic = %S.loc9_25.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc9_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.6ef)]
// CHECK:STDOUT:     %.Self.loc9_44.1: @T.as.Z.impl.e64.%Z.type.loc9_42.2 (%Z.type.6ef) = symbolic_binding .Self [symbolic = %.Self.loc9_44.2 (constants.%.Self.d86)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.e64.%Z.type.loc9_42.2 (%Z.type.6ef) = name_ref .Self, %.Self.loc9_44.1 [symbolic = %.Self.loc9_44.2 (constants.%.Self.d86)]
// CHECK:STDOUT:     %.Self.as_type.loc9_50.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.1a7)]
// CHECK:STDOUT:     %.loc9_50.1: type = converted %.Self.ref, %.Self.as_type.loc9_50.1 [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.1a7)]
// CHECK:STDOUT:     %.loc9_50.2: @T.as.Z.impl.e64.%Z.assoc_type (%Z.assoc_type.795) = specific_constant @X.%assoc0, @Z.WithSelf(constants.%S, constants.%.Self.d86) [symbolic = %assoc0 (constants.%assoc0.281)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.e64.%Z.assoc_type (%Z.assoc_type.795) = name_ref X, %.loc9_50.2 [symbolic = %assoc0 (constants.%assoc0.281)]
// CHECK:STDOUT:     %impl.elem0.loc9_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.18f, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.6c0)]
// CHECK:STDOUT:     %.loc9_56.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_56.2: type = converted %.loc9_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_44: type = where_expr [symbolic = %Z_where.type (constants.%Z_where.type.d8a)] {
// CHECK:STDOUT:       requirement_base_facet_type %Z.type.loc9_42.1
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_50.1, %.loc9_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc9_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc9_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc9_28.1: type = splice_block %.loc9_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc9_25: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc9_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc9_25.1: type = symbolic_binding S, 1 [symbolic = %S.loc9_25.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.b90 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc11_30: type = name_ref T, %T.loc11_21.1 [symbolic = %T.loc11_21.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.8a6 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.2d0]
// CHECK:STDOUT:     %.Self.loc11_40: %Z.type.2d0 = symbolic_binding .Self [symbolic_self = constants.%.Self.e6c]
// CHECK:STDOUT:     %.Self.ref: %Z.type.2d0 = name_ref .Self, %.Self.loc11_40 [symbolic_self = constants.%.Self.e6c]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.e4e]
// CHECK:STDOUT:     %.loc11_46.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.e4e]
// CHECK:STDOUT:     %.loc11_46.2: %Z.assoc_type.8db = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%.Self.e6c) [concrete = constants.%assoc0.f02]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.8db = name_ref X, %.loc11_46.2 [concrete = constants.%assoc0.f02]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.c21, element0 [symbolic_self = constants.%impl.elem0.fe4]
// CHECK:STDOUT:     %T.ref.loc11_51: type = name_ref T, %T.loc11_21.1 [symbolic = %T.loc11_21.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc11_40: type = where_expr [symbolic = %Z_where.type (constants.%Z_where.type.346)] {
// CHECK:STDOUT:       requirement_base_facet_type %Z.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc11_51
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc11_24.1: type = splice_block %.loc11_24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc11_21: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc11_24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_21.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_21.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.59a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type (%pattern_type.56b) = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type (%pattern_type.56b) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_13: type = splice_block %Z.type [concrete = constants.%Z.type.2d0] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Z.ref: %Z.type.8a6 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.2d0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_7.2: %Z.type.2d0 = symbolic_binding T, 0 [symbolic = %T.loc13_7.1 (constants.%T.196)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc13_19.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_19.1: type = splice_block %.loc13_19.2 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc13: %Z.type.2d0 = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T.196)]
// CHECK:STDOUT:       %T.as_type.loc13_19.2: type = facet_access_type %T.ref.loc13 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc13_19.2: type = converted %T.ref.loc13, %T.as_type.loc13_19.2 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc13_19.1 (%T.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_14.2: type) {
// CHECK:STDOUT:   %T.loc3_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_14.1)> [symbolic = %Z.type (constants.%Z.type.284)]
// CHECK:STDOUT:   %Self.loc3_23.2: @Z.%Z.type (%Z.type.284) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.f76)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @Z.%Z.type (%Z.type.284) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.f76)]
// CHECK:STDOUT:     %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.WithSelf.%Z.assoc_type (%Z.assoc_type.e20) = assoc_entity element0, @Z.WithSelf.%X [symbolic = @Z.WithSelf.%assoc0 (constants.%assoc0.5be)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (@Z.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.e64(%T.loc9_15.1: type, %S.loc9_25.1: type) {
// CHECK:STDOUT:   %T.loc9_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   %S.loc9_25.2: type = symbolic_binding S, 1 [symbolic = %S.loc9_25.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc9_42.2: type = facet_type <@Z, @Z(%S.loc9_25.2)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.6ef)]
// CHECK:STDOUT:   %.Self.loc9_44.2: @T.as.Z.impl.e64.%Z.type.loc9_42.2 (%Z.type.6ef) = symbolic_binding .Self [symbolic = %.Self.loc9_44.2 (constants.%.Self.d86)]
// CHECK:STDOUT:   %require_complete.loc9_50: <witness> = require_complete_type %Z.type.loc9_42.2 [symbolic = %require_complete.loc9_50 (constants.%require_complete.c80)]
// CHECK:STDOUT:   %.Self.as_type.loc9_50.2: type = facet_access_type %.Self.loc9_44.2 [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.1a7)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc9_25.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.795)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.e64.%Z.assoc_type (%Z.assoc_type.795) = assoc_entity element0, @Z.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.281)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.loc9_44.2, @Z, @Z(%S.loc9_25.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.18f)]
// CHECK:STDOUT:   %impl.elem0.loc9_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.6c0)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc9_25.2) where %impl.elem0.loc9_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.d8a)]
// CHECK:STDOUT:   %Z.impl_witness.loc9_58.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.e64(%T.loc9_15.2, %S.loc9_25.2) [symbolic = %Z.impl_witness.loc9_58.2 (constants.%Z.impl_witness.62a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc9_44 (constants.%require_complete.310)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc9_44 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.e64 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc9_58.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.e64(constants.%T.67d, constants.%S) [symbolic = %Z.impl_witness.loc9_58.2 (constants.%Z.impl_witness.62a)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %.loc9_44
// CHECK:STDOUT:     witness = %Z.impl_witness.loc9_58.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.b90(%T.loc11_21.1: type) {
// CHECK:STDOUT:   %T.loc11_21.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_21.2 (constants.%T.67d)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.fe4 = %T.loc11_21.2> [symbolic = %Z_where.type (constants.%Z_where.type.346)]
// CHECK:STDOUT:   %Z.impl_witness.loc11_53.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.b90(%T.loc11_21.2) [symbolic = %Z.impl_witness.loc11_53.2 (constants.%Z.impl_witness.61a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.3d6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc11_30 as %.loc11_40 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.b90 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc11_53.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.b90(constants.%T.67d) [symbolic = %Z.impl_witness.loc11_53.2 (constants.%Z.impl_witness.61a)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%T.67d [symbolic = %T.loc11_21.2 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %.loc11_40
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
// CHECK:STDOUT: generic fn @F(%T.loc13_7.2: %Z.type.2d0) {
// CHECK:STDOUT:   %T.loc13_7.1: %Z.type.2d0 = symbolic_binding T, 0 [symbolic = %T.loc13_7.1 (constants.%T.196)]
// CHECK:STDOUT:   %T.as_type.loc13_19.1: type = facet_access_type %T.loc13_7.1 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc13_19.1 [symbolic = %pattern_type (constants.%pattern_type.56b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc13_19.1 [symbolic = %require_complete (constants.%require_complete.c09)]
// CHECK:STDOUT:   %.loc15_18.4: require_specific_def_type = require_specific_def @T.as.Z.impl.b90(%T.as_type.loc13_19.1) [symbolic = %.loc15_18.4 (constants.%.639)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc13_19.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type (%pattern_type.56b) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc13_19.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc15_18.1: type = splice_block %impl.elem0 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc15: %Z.type.2d0 = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T.196)]
// CHECK:STDOUT:       %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_18.2: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_18.3: %Z.assoc_type.8db = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%T.196) [concrete = constants.%assoc0.f02]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.8db = name_ref X, %.loc15_18.3 [concrete = constants.%assoc0.f02]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.f75, element0 [symbolic = %T.as_type.loc13_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%T.as_type.loc13_19.1 (%T.as_type) = value_binding a, %t.ref
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T.67d, constants.%Self.f76) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.6ef
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.d23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Self.f76) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.795
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%.Self.d86) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.795
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.e64(constants.%T.67d, constants.%S) {
// CHECK:STDOUT:   %T.loc9_15.2 => constants.%T.67d
// CHECK:STDOUT:   %S.loc9_25.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc9_42.2 => constants.%Z.type.6ef
// CHECK:STDOUT:   %.Self.loc9_44.2 => constants.%.Self.d86
// CHECK:STDOUT:   %require_complete.loc9_50 => constants.%require_complete.c80
// CHECK:STDOUT:   %.Self.as_type.loc9_50.2 => constants.%.Self.as_type.1a7
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.795
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.281
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.18f
// CHECK:STDOUT:   %impl.elem0.loc9_50.2 => constants.%impl.elem0.6c0
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.d8a
// CHECK:STDOUT:   %Z.impl_witness.loc9_58.2 => constants.%Z.impl_witness.62a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Z.facet.59a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.795
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.2d0
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.f16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Self.f76) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.8db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%.Self.e6c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.8db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.b90(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc11_21.2 => constants.%T.67d
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.346
// CHECK:STDOUT:   %Z.impl_witness.loc11_53.2 => constants.%Z.impl_witness.61a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Z.facet.587) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.8db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.196) {
// CHECK:STDOUT:   %T.loc13_7.1 => constants.%T.196
// CHECK:STDOUT:   %T.as_type.loc13_19.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.56b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%T.196) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.8db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.b90(constants.%T.as_type) {
// CHECK:STDOUT:   %T.loc11_21.2 => constants.%T.as_type
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.c40
// CHECK:STDOUT:   %Z.impl_witness.loc11_53.2 => constants.%Z.impl_witness.f75
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.386
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Z.type.8a6: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.8a6 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.284: type = facet_type <@Z, @Z(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.f76: %Z.type.284 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.e20: type = assoc_entity_type @Z, @Z(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.5be: %Z.assoc_type.e20 = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.d9e: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.6ef: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.d86: %Z.type.6ef = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.d23: %Z.type.6ef = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.795: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.281: %Z.assoc_type.795 = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %require_complete.c80: <witness> = require_complete_type %Z.type.6ef [symbolic]
// CHECK:STDOUT:   %.Self.as_type.1a7: type = facet_access_type %.Self.d86 [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.18f: <witness> = lookup_impl_witness %.Self.d86, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.6c0: type = impl_witness_access %Z.lookup_impl_witness.18f, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z_where.type.d8a: type = facet_type <@Z, @Z(%S) where %impl.elem0.6c0 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.62a: <witness> = impl_witness @T.as.Z.impl.e64.%Z.impl_witness_table, @T.as.Z.impl.e64(%T.67d, %S) [symbolic]
// CHECK:STDOUT:   %require_complete.310: <witness> = require_complete_type %Z_where.type.d8a [symbolic]
// CHECK:STDOUT:   %Z.facet.59a: %Z.type.6ef = facet_value %T.67d, (%Z.impl_witness.62a) [symbolic]
// CHECK:STDOUT:   %Z.type.2d0: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.e6c: %Z.type.2d0 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.f16: %Z.type.2d0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.8db: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.f02: %Z.assoc_type.8db = assoc_entity element0, @Z.WithSelf.%X [concrete]
// CHECK:STDOUT:   %.Self.as_type.e4e: type = facet_access_type %.Self.e6c [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.c21: <witness> = lookup_impl_witness %.Self.e6c, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fe4: type = impl_witness_access %Z.lookup_impl_witness.c21, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.346: type = facet_type <@Z, @Z(%C) where %impl.elem0.fe4 = %T.67d> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.61a: <witness> = impl_witness @T.as.Z.impl.b90.%Z.impl_witness_table, @T.as.Z.impl.b90(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.3d6: <witness> = require_complete_type %Z_where.type.346 [symbolic]
// CHECK:STDOUT:   %Z.facet.587: %Z.type.2d0 = facet_value %T.67d, (%Z.impl_witness.61a) [symbolic]
// CHECK:STDOUT:   %pattern_type.59a: type = pattern_type %Z.type.2d0 [concrete]
// CHECK:STDOUT:   %T.196: %Z.type.2d0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.196 [symbolic]
// CHECK:STDOUT:   %pattern_type.56b: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c09: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.34b: <witness> = lookup_impl_witness %T.196, @Z, @Z(%C) [symbolic]
// CHECK:STDOUT:   %impl.elem0.aaf: type = impl_witness_access %Z.lookup_impl_witness.34b, element0 [symbolic]
// CHECK:STDOUT:   %require_complete.11b: <witness> = require_complete_type %impl.elem0.aaf [symbolic]
// CHECK:STDOUT:   %pattern_type.87f: type = pattern_type %impl.elem0.aaf [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.754: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.754 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6e6: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b7e: %ImplicitAs.type.6e6 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.bf1: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c63: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.b7e) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.cba: %ImplicitAs.WithSelf.Convert.type.c63 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.ad3: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.aaf)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.09c: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.aaf) [symbolic]
// CHECK:STDOUT:   %assoc0.59a: %ImplicitAs.assoc_type.09c = assoc_entity element0, imports.%Core.import_ref.569 [symbolic]
// CHECK:STDOUT:   %require_complete.d79: <witness> = require_complete_type %ImplicitAs.type.ad3 [symbolic]
// CHECK:STDOUT:   %assoc0.bd2: %ImplicitAs.assoc_type.bf1 = assoc_entity element0, imports.%Core.import_ref.cd5 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.754 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.195: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.bf1) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.bd2)]
// CHECK:STDOUT:   %Core.import_ref.569: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.c63) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.cba)]
// CHECK:STDOUT:   %Core.import_ref.cd5 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:   %Z.decl: %Z.type.8a6 = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_17.1: type = splice_block %.loc3_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc3_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.e64 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_15.1 [symbolic = %T.loc10_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.8a6 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc10_25.1 [symbolic = %S.loc10_25.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc10_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.6ef)]
// CHECK:STDOUT:     %.Self.loc10_44.1: @T.as.Z.impl.e64.%Z.type.loc10_42.2 (%Z.type.6ef) = symbolic_binding .Self [symbolic = %.Self.loc10_44.2 (constants.%.Self.d86)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.e64.%Z.type.loc10_42.2 (%Z.type.6ef) = name_ref .Self, %.Self.loc10_44.1 [symbolic = %.Self.loc10_44.2 (constants.%.Self.d86)]
// CHECK:STDOUT:     %.Self.as_type.loc10_50.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.1a7)]
// CHECK:STDOUT:     %.loc10_50.1: type = converted %.Self.ref, %.Self.as_type.loc10_50.1 [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.1a7)]
// CHECK:STDOUT:     %.loc10_50.2: @T.as.Z.impl.e64.%Z.assoc_type (%Z.assoc_type.795) = specific_constant @X.%assoc0, @Z.WithSelf(constants.%S, constants.%.Self.d86) [symbolic = %assoc0 (constants.%assoc0.281)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.e64.%Z.assoc_type (%Z.assoc_type.795) = name_ref X, %.loc10_50.2 [symbolic = %assoc0 (constants.%assoc0.281)]
// CHECK:STDOUT:     %impl.elem0.loc10_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.18f, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.6c0)]
// CHECK:STDOUT:     %.loc10_56.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_56.2: type = converted %.loc10_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_44: type = where_expr [symbolic = %Z_where.type (constants.%Z_where.type.d8a)] {
// CHECK:STDOUT:       requirement_base_facet_type %Z.type.loc10_42.1
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_50.1, %.loc10_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10_18.1: type = splice_block %.loc10_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc10_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc10_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc10_28.1: type = splice_block %.loc10_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc10_25: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc10_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc10_25.1: type = symbolic_binding S, 1 [symbolic = %S.loc10_25.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.b90 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_24: type = name_ref T, %T.loc12_15.1 [symbolic = %T.loc12_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.8a6 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.2d0]
// CHECK:STDOUT:     %.Self.loc12_34: %Z.type.2d0 = symbolic_binding .Self [symbolic_self = constants.%.Self.e6c]
// CHECK:STDOUT:     %.Self.ref: %Z.type.2d0 = name_ref .Self, %.Self.loc12_34 [symbolic_self = constants.%.Self.e6c]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.e4e]
// CHECK:STDOUT:     %.loc12_40.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.e4e]
// CHECK:STDOUT:     %.loc12_40.2: %Z.assoc_type.8db = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%.Self.e6c) [concrete = constants.%assoc0.f02]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.8db = name_ref X, %.loc12_40.2 [concrete = constants.%assoc0.f02]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.c21, element0 [symbolic_self = constants.%impl.elem0.fe4]
// CHECK:STDOUT:     %T.ref.loc12_45: type = name_ref T, %T.loc12_15.1 [symbolic = %T.loc12_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc12_34: type = where_expr [symbolic = %Z_where.type (constants.%Z_where.type.346)] {
// CHECK:STDOUT:       requirement_base_facet_type %Z.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc12_45
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc12_18.1: type = splice_block %.loc12_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc12_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc12_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.59a = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc14 (%pattern_type.56b) = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc14 (%pattern_type.56b) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_13: type = splice_block %Z.type [concrete = constants.%Z.type.2d0] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Z.ref: %Z.type.8a6 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.2d0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_7.2: %Z.type.2d0 = symbolic_binding T, 0 [symbolic = %T.loc14_7.1 (constants.%T.196)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc14_19.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc14: %Z.type.2d0 = name_ref T, %T.loc14_7.2 [symbolic = %T.loc14_7.1 (constants.%T.196)]
// CHECK:STDOUT:       %T.as_type.loc14_19.2: type = facet_access_type %T.ref.loc14 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc14_19.2: type = converted %T.ref.loc14, %T.as_type.loc14_19.2 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc14_19.1 (%T.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_14.2: type) {
// CHECK:STDOUT:   %T.loc3_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_14.1)> [symbolic = %Z.type (constants.%Z.type.284)]
// CHECK:STDOUT:   %Self.loc3_23.2: @Z.%Z.type (%Z.type.284) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.f76)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @Z.%Z.type (%Z.type.284) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.f76)]
// CHECK:STDOUT:     %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.WithSelf.%Z.assoc_type (%Z.assoc_type.e20) = assoc_entity element0, @Z.WithSelf.%X [symbolic = @Z.WithSelf.%assoc0 (constants.%assoc0.5be)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (@Z.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d9e]
// CHECK:STDOUT:   %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.e64(%T.loc10_15.1: type, %S.loc10_25.1: type) {
// CHECK:STDOUT:   %T.loc10_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   %S.loc10_25.2: type = symbolic_binding S, 1 [symbolic = %S.loc10_25.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc10_42.2: type = facet_type <@Z, @Z(%S.loc10_25.2)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.6ef)]
// CHECK:STDOUT:   %.Self.loc10_44.2: @T.as.Z.impl.e64.%Z.type.loc10_42.2 (%Z.type.6ef) = symbolic_binding .Self [symbolic = %.Self.loc10_44.2 (constants.%.Self.d86)]
// CHECK:STDOUT:   %require_complete.loc10_50: <witness> = require_complete_type %Z.type.loc10_42.2 [symbolic = %require_complete.loc10_50 (constants.%require_complete.c80)]
// CHECK:STDOUT:   %.Self.as_type.loc10_50.2: type = facet_access_type %.Self.loc10_44.2 [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.1a7)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc10_25.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.795)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.e64.%Z.assoc_type (%Z.assoc_type.795) = assoc_entity element0, @Z.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.281)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.loc10_44.2, @Z, @Z(%S.loc10_25.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.18f)]
// CHECK:STDOUT:   %impl.elem0.loc10_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.6c0)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc10_25.2) where %impl.elem0.loc10_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.d8a)]
// CHECK:STDOUT:   %Z.impl_witness.loc10_58.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.e64(%T.loc10_15.2, %S.loc10_25.2) [symbolic = %Z.impl_witness.loc10_58.2 (constants.%Z.impl_witness.62a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc10_44 (constants.%require_complete.310)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc10_44 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.e64 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc10_58.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.e64(constants.%T.67d, constants.%S) [symbolic = %Z.impl_witness.loc10_58.2 (constants.%Z.impl_witness.62a)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %.loc10_44
// CHECK:STDOUT:     witness = %Z.impl_witness.loc10_58.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.b90(%T.loc12_15.1: type) {
// CHECK:STDOUT:   %T.loc12_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.fe4 = %T.loc12_15.2> [symbolic = %Z_where.type (constants.%Z_where.type.346)]
// CHECK:STDOUT:   %Z.impl_witness.loc12_47.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.b90(%T.loc12_15.2) [symbolic = %Z.impl_witness.loc12_47.2 (constants.%Z.impl_witness.61a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.3d6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc12_24 as %.loc12_34 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.b90 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc12_47.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.b90(constants.%T.67d) [symbolic = %Z.impl_witness.loc12_47.2 (constants.%Z.impl_witness.61a)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%T.67d [symbolic = %T.loc12_15.2 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %.loc12_34
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
// CHECK:STDOUT: generic fn @F(%T.loc14_7.2: %Z.type.2d0) {
// CHECK:STDOUT:   %T.loc14_7.1: %Z.type.2d0 = symbolic_binding T, 0 [symbolic = %T.loc14_7.1 (constants.%T.196)]
// CHECK:STDOUT:   %T.as_type.loc14_19.1: type = facet_access_type %T.loc14_7.1 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc14: type = pattern_type %T.as_type.loc14_19.1 [symbolic = %pattern_type.loc14 (constants.%pattern_type.56b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.as_type.loc14_19.1 [symbolic = %require_complete.loc14 (constants.%require_complete.c09)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc14_7.1, @Z, @Z(constants.%C) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.34b)]
// CHECK:STDOUT:   %impl.elem0.loc22_18.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_18.2 (constants.%impl.elem0.aaf)]
// CHECK:STDOUT:   %require_complete.loc22_18: <witness> = require_complete_type %impl.elem0.loc22_18.2 [symbolic = %require_complete.loc22_18 (constants.%require_complete.11b)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %impl.elem0.loc22_18.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.87f)]
// CHECK:STDOUT:   %ImplicitAs.type.loc22_23.2: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.loc22_18.2)> [symbolic = %ImplicitAs.type.loc22_23.2 (constants.%ImplicitAs.type.ad3)]
// CHECK:STDOUT:   %require_complete.loc22_23: <witness> = require_complete_type %ImplicitAs.type.loc22_23.2 [symbolic = %require_complete.loc22_23 (constants.%require_complete.d79)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.loc22_18.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.09c)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.09c) = assoc_entity element0, imports.%Core.import_ref.569 [symbolic = %assoc0 (constants.%assoc0.59a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc14_19.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type.loc22 (%pattern_type.87f) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc14_19.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc22_18.1: type = splice_block %impl.elem0.loc22_18.1 [symbolic = %impl.elem0.loc22_18.2 (constants.%impl.elem0.aaf)] {
// CHECK:STDOUT:       %T.ref.loc22: %Z.type.2d0 = name_ref T, %T.loc14_7.2 [symbolic = %T.loc14_7.1 (constants.%T.196)]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_18.2: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.as_type.loc14_19.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_18.3: %Z.assoc_type.8db = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%T.196) [concrete = constants.%assoc0.f02]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.8db = name_ref X, %.loc22_18.3 [concrete = constants.%assoc0.f02]
// CHECK:STDOUT:       %impl.elem0.loc22_18.1: type = impl_witness_access constants.%Z.lookup_impl_witness.34b, element0 [symbolic = %impl.elem0.loc22_18.2 (constants.%impl.elem0.aaf)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ImplicitAs.type.loc22_23.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%impl.elem0.aaf)> [symbolic = %ImplicitAs.type.loc22_23.2 (constants.%ImplicitAs.type.ad3)]
// CHECK:STDOUT:     %.loc22_23.1: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.09c) = specific_constant imports.%Core.import_ref.195, @ImplicitAs.WithSelf(constants.%impl.elem0.aaf, constants.%Self.b7e) [symbolic = %assoc0 (constants.%assoc0.59a)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.09c) = name_ref Convert, %.loc22_23.1 [symbolic = %assoc0 (constants.%assoc0.59a)]
// CHECK:STDOUT:     %.loc22_23.2: @F.%impl.elem0.loc22_18.2 (%impl.elem0.aaf) = converted %t.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %a: @F.%impl.elem0.loc22_18.2 (%impl.elem0.aaf) = value_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T.67d, constants.%Self.f76) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.d9e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.6ef
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.d23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Self.f76) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.795
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%.Self.d86) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.795
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.e64(constants.%T.67d, constants.%S) {
// CHECK:STDOUT:   %T.loc10_15.2 => constants.%T.67d
// CHECK:STDOUT:   %S.loc10_25.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc10_42.2 => constants.%Z.type.6ef
// CHECK:STDOUT:   %.Self.loc10_44.2 => constants.%.Self.d86
// CHECK:STDOUT:   %require_complete.loc10_50 => constants.%require_complete.c80
// CHECK:STDOUT:   %.Self.as_type.loc10_50.2 => constants.%.Self.as_type.1a7
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.795
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.281
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.18f
// CHECK:STDOUT:   %impl.elem0.loc10_50.2 => constants.%impl.elem0.6c0
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.d8a
// CHECK:STDOUT:   %Z.impl_witness.loc10_58.2 => constants.%Z.impl_witness.62a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Z.facet.59a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.795
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.281
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.2d0
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.f16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Self.f76) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.8db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%.Self.e6c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.8db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.b90(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc12_15.2 => constants.%T.67d
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.346
// CHECK:STDOUT:   %Z.impl_witness.loc12_47.2 => constants.%Z.impl_witness.61a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Z.facet.587) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.8db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.196) {
// CHECK:STDOUT:   %T.loc14_7.1 => constants.%T.196
// CHECK:STDOUT:   %T.as_type.loc14_19.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc14 => constants.%pattern_type.56b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%T.196) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.8db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.ca4: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.a3d: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.332: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.332 [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.332, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.bd8: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.bb99f8.1: type = facet_type <@Ptr where %impl.elem0.bd8 = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.1f9b1e.1: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.425ff2.1: <witness> = require_complete_type %Ptr_where.type.bb99f8.1 [symbolic]
// CHECK:STDOUT:   %Ptr.facet.d63c8c.1: %Ptr.type = facet_value %U.67d, (%Ptr.impl_witness.1f9b1e.1) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.bb99f8.2: type = facet_type <@Ptr where %impl.elem0.bd8 = %ptr.e8f8f9.2> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.1f9b1e.2: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.425ff2.2: <witness> = require_complete_type %Ptr_where.type.bb99f8.2 [symbolic]
// CHECK:STDOUT:   %.aed: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Ptr.facet.d63c8c.2: %Ptr.type = facet_value %T.67d, (%Ptr.impl_witness.1f9b1e.2) [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2bf: <witness> = lookup_impl_witness %ptr.e8f8f9.2, @Copy [symbolic]
// CHECK:STDOUT:   %.b6a: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f8f9.2, (%Copy.lookup_impl_witness.2bf) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.72b: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.39b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.72b, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.8a3: %.39b = impl_witness_access %Copy.lookup_impl_witness.2bf, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a13: <specific function> = specific_impl_function %impl.elem0.8a3, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.loc7_39: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.332]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.loc7_39 [symbolic_self = constants.%.Self.332]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.a3d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.bd8]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bb99f8.1)] {
// CHECK:STDOUT:       requirement_base_facet_type %Ptr.ref
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_24.1: type = splice_block %.loc7_24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc7_21: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc7_24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_21.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_21 (%pattern_type.51d) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_21 (%pattern_type.51d) = var_param_pattern %t.patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_30 (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_30 (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %impl.elem0.loc9 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_29: type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.a3d]
// CHECK:STDOUT:     %Ptr.facet.loc9_30.2: %Ptr.type = facet_value %T.ref.loc9_29, (constants.%Ptr.impl_witness.1f9b1e.2) [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet.d63c8c.2)]
// CHECK:STDOUT:     %.loc9_30.3: %Ptr.type = converted %T.ref.loc9_29, %Ptr.facet.loc9_30.2 [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet.d63c8c.2)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.1f9b1e.2, element0 [symbolic = %ptr (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:     %.loc9_30.4: Core.Form = init_form %impl.elem0.loc9 [symbolic = %.loc9_30.2 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc9_10.1: type = splice_block %.loc9_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc9_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %t.param: ref @F.%T.loc9_7.1 (%T.67d) = ref_param call_param0
// CHECK:STDOUT:     %T.ref.loc9_23: type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %t: ref @F.%T.loc9_7.1 (%T.67d) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.e8f8f9.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.e8f8f9.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ca4]
// CHECK:STDOUT:   %Ptr.WithSelf.decl = interface_with_self_decl @Ptr [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete = constants.%assoc0.a3d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Type = @Type.%assoc0
// CHECK:STDOUT:   witness = (@Ptr.WithSelf.%Type)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc7_21.1: type) {
// CHECK:STDOUT:   %U.loc7_21.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_21.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.bd8 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bb99f8.1)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_21.2) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.1f9b1e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.425ff2.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc7_56.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.1f9b1e.1)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %.loc7_39
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc7_56.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_7.2: type) {
// CHECK:STDOUT:   %T.loc9_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.67d)]
// CHECK:STDOUT:   %pattern_type.loc9_21: type = pattern_type %T.loc9_7.1 [symbolic = %pattern_type.loc9_21 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %.loc9_30.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.loc9_7.1) [symbolic = %.loc9_30.1 (constants.%.aed)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.loc9_7.1) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.1f9b1e.2)]
// CHECK:STDOUT:   %Ptr.facet.loc9_30.1: %Ptr.type = facet_value %T.loc9_7.1, (%Ptr.impl_witness) [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet.d63c8c.2)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc9_7.1 [symbolic = %ptr (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %.loc9_30.2: Core.Form = init_form %ptr [symbolic = %.loc9_30.2 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type.loc9_30: type = pattern_type %ptr [symbolic = %pattern_type.loc9_30 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_16: <witness> = require_complete_type %T.loc9_7.1 [symbolic = %require_complete.loc9_16 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc9_26: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_26 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %.loc10_10.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc9_7.1) [symbolic = %.loc10_10.1 (constants.%.b6a)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2bf)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.72b)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.39b)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.39b) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.8a3)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.a13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.loc9_7.1 (%T.67d)) -> out %return.param: @F.%ptr (%ptr.e8f8f9.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc9_7.1 (%T.67d) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.e8f8f9.2) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.39b) = impl_witness_access constants.%Copy.lookup_impl_witness.2bf, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.8a3)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.a13)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr (%ptr.e8f8f9.2) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Self.ca4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%.Self.332) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.bb99f8.1
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.1f9b1e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet.d63c8c.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.67d) {
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.bb99f8.2
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.1f9b1e.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.425ff2.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet.d63c8c.2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T.67d
// CHECK:STDOUT:   %pattern_type.loc9_21 => constants.%pattern_type.51d
// CHECK:STDOUT:   %.loc9_30.1 => constants.%.aed
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.1f9b1e.2
// CHECK:STDOUT:   %Ptr.facet.loc9_30.1 => constants.%Ptr.facet.d63c8c.2
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %.loc9_30.2 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type.loc9_30 => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.ca4: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.a3d: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.332: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.332 [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.332, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.bd8: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.bb9: type = facet_type <@Ptr where %impl.elem0.bd8 = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.1f9: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.425: <witness> = require_complete_type %Ptr_where.type.bb9 [symbolic]
// CHECK:STDOUT:   %Ptr.facet: %Ptr.type = facet_value %U.67d, (%Ptr.impl_witness.1f9) [symbolic]
// CHECK:STDOUT:   %pattern_type.51e: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.7bc: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.6c8: type = facet_access_type %T.7bc [symbolic]
// CHECK:STDOUT:   %pattern_type.256: type = pattern_type %T.as_type.6c8 [symbolic]
// CHECK:STDOUT:   %ptr.3c2: type = ptr_type %T.as_type.6c8 [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.b07: type = facet_type <@Ptr where %impl.elem0.bd8 = %ptr.3c2> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.140: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.as_type.6c8) [symbolic]
// CHECK:STDOUT:   %require_complete.76a: <witness> = require_complete_type %Ptr_where.type.b07 [symbolic]
// CHECK:STDOUT:   %.dab: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.as_type.6c8) [symbolic]
// CHECK:STDOUT:   %.ed7: Core.Form = init_form %ptr.3c2 [symbolic]
// CHECK:STDOUT:   %pattern_type.438: type = pattern_type %ptr.3c2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.233: <witness> = require_complete_type %T.as_type.6c8 [symbolic]
// CHECK:STDOUT:   %require_complete.b3d: <witness> = require_complete_type %ptr.3c2 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3b7: <witness> = lookup_impl_witness %ptr.3c2, @Copy [symbolic]
// CHECK:STDOUT:   %.4fe: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.6c8) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.3c2, (%Copy.lookup_impl_witness.3b7) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.666: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.1ab: type = fn_type_with_self_type %Copy.WithSelf.Op.type.666, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.077: %.1ab = impl_witness_access %Copy.lookup_impl_witness.3b7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9fb: <specific function> = specific_impl_function %impl.elem0.077, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.loc7_39: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.332]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.loc7_39 [symbolic_self = constants.%.Self.332]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.a3d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.bd8]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bb9)] {
// CHECK:STDOUT:       requirement_base_facet_type %Ptr.ref
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_24.1: type = splice_block %.loc7_24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc7_21: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc7_24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_21.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.51e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_20 (%pattern_type.256) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_20 (%pattern_type.256) = var_param_pattern %t.patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_29 (%pattern_type.438) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_29 (%pattern_type.438) = return_slot_pattern %return.param_patt, %impl.elem0.loc9 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.7bc)]
// CHECK:STDOUT:     %T.as_type.loc9_29: type = facet_access_type %T.ref.loc9_28 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.6c8)]
// CHECK:STDOUT:     %.loc9_29.3: type = converted %T.ref.loc9_28, %T.as_type.loc9_29 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.6c8)]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.a3d]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.140, element0 [symbolic = %ptr (constants.%ptr.3c2)]
// CHECK:STDOUT:     %.loc9_29.4: Core.Form = init_form %impl.elem0.loc9 [symbolic = %.loc9_29.2 (constants.%.ed7)]
// CHECK:STDOUT:     %.loc9_10: type = splice_block %Ptr.ref [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_7.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.7bc)]
// CHECK:STDOUT:     %t.param: ref @F.%T.as_type.loc9_22.1 (%T.as_type.6c8) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.6c8)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.7bc)]
// CHECK:STDOUT:       %T.as_type.loc9_22.2: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.6c8)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type.loc9_22.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.6c8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.as_type.loc9_22.1 (%T.as_type.6c8) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.3c2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.3c2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ca4]
// CHECK:STDOUT:   %Ptr.WithSelf.decl = interface_with_self_decl @Ptr [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete = constants.%assoc0.a3d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Type = @Type.%assoc0
// CHECK:STDOUT:   witness = (@Ptr.WithSelf.%Type)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc7_21.1: type) {
// CHECK:STDOUT:   %U.loc7_21.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_21.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.bd8 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bb9)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_21.2) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.1f9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.425)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc7_56.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.1f9)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %.loc7_39
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc7_56.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_7.2: %Ptr.type) {
// CHECK:STDOUT:   %T.loc9_7.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.7bc)]
// CHECK:STDOUT:   %T.as_type.loc9_22.1: type = facet_access_type %T.loc9_7.1 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.6c8)]
// CHECK:STDOUT:   %pattern_type.loc9_20: type = pattern_type %T.as_type.loc9_22.1 [symbolic = %pattern_type.loc9_20 (constants.%pattern_type.256)]
// CHECK:STDOUT:   %.loc9_29.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.as_type.loc9_22.1) [symbolic = %.loc9_29.1 (constants.%.dab)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_22.1 [symbolic = %ptr (constants.%ptr.3c2)]
// CHECK:STDOUT:   %.loc9_29.2: Core.Form = init_form %ptr [symbolic = %.loc9_29.2 (constants.%.ed7)]
// CHECK:STDOUT:   %pattern_type.loc9_29: type = pattern_type %ptr [symbolic = %pattern_type.loc9_29 (constants.%pattern_type.438)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.as_type.loc9_22.1 [symbolic = %require_complete.loc9_15 (constants.%require_complete.233)]
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_25 (constants.%require_complete.b3d)]
// CHECK:STDOUT:   %.loc10_10.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc9_22.1) [symbolic = %.loc10_10.1 (constants.%.4fe)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3b7)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.666)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.1ab)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.1ab) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.077)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.9fb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.as_type.loc9_22.1 (%T.as_type.6c8)) -> out %return.param: @F.%ptr (%ptr.3c2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc9_22.1 (%T.as_type.6c8) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.3c2) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.1ab) = impl_witness_access constants.%Copy.lookup_impl_witness.3b7, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.077)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.9fb)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr (%ptr.3c2) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Self.ca4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%.Self.332) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.bb9
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.1f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%T.7bc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.as_type.6c8) {
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%T.as_type.6c8
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.3c2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.b07
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.140
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.76a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.7bc) {
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T.7bc
// CHECK:STDOUT:   %T.as_type.loc9_22.1 => constants.%T.as_type.6c8
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.256
// CHECK:STDOUT:   %.loc9_29.1 => constants.%.dab
// CHECK:STDOUT:   %ptr => constants.%ptr.3c2
// CHECK:STDOUT:   %.loc9_29.2 => constants.%.ed7
// CHECK:STDOUT:   %pattern_type.loc9_29 => constants.%pattern_type.438
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.ca4: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.a3d: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.332: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.332 [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.332, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.bd8: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.bb9: type = facet_type <@Ptr where %impl.elem0.bd8 = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.1f9: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.425: <witness> = require_complete_type %Ptr_where.type.bb9 [symbolic]
// CHECK:STDOUT:   %Ptr.facet: %Ptr.type = facet_value %U.67d, (%Ptr.impl_witness.1f9) [symbolic]
// CHECK:STDOUT:   %pattern_type.51e: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.7bc: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.6c8: type = facet_access_type %T.7bc [symbolic]
// CHECK:STDOUT:   %pattern_type.256: type = pattern_type %T.as_type.6c8 [symbolic]
// CHECK:STDOUT:   %ptr.3c2: type = ptr_type %T.as_type.6c8 [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.b07: type = facet_type <@Ptr where %impl.elem0.bd8 = %ptr.3c2> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.140: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.as_type.6c8) [symbolic]
// CHECK:STDOUT:   %require_complete.76a: <witness> = require_complete_type %Ptr_where.type.b07 [symbolic]
// CHECK:STDOUT:   %.dab: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.as_type.6c8) [symbolic]
// CHECK:STDOUT:   %.ed7: Core.Form = init_form %ptr.3c2 [symbolic]
// CHECK:STDOUT:   %pattern_type.438: type = pattern_type %ptr.3c2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.233: <witness> = require_complete_type %T.as_type.6c8 [symbolic]
// CHECK:STDOUT:   %require_complete.b3d: <witness> = require_complete_type %ptr.3c2 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3b7: <witness> = lookup_impl_witness %ptr.3c2, @Copy [symbolic]
// CHECK:STDOUT:   %.4fe: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.6c8) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.3c2, (%Copy.lookup_impl_witness.3b7) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.666: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.1ab: type = fn_type_with_self_type %Copy.WithSelf.Op.type.666, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.077: %.1ab = impl_witness_access %Copy.lookup_impl_witness.3b7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9fb: <specific function> = specific_impl_function %impl.elem0.077, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.loc7_39: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.332]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.loc7_39 [symbolic_self = constants.%.Self.332]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.a3d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.bd8]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bb9)] {
// CHECK:STDOUT:       requirement_base_facet_type %Ptr.ref
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_24.1: type = splice_block %.loc7_24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc7_21: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc7_24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_21.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.51e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_20 (%pattern_type.256) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_20 (%pattern_type.256) = var_param_pattern %t.patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_29 (%pattern_type.438) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_29 (%pattern_type.438) = return_slot_pattern %return.param_patt, %impl.elem0.loc9 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.7bc)]
// CHECK:STDOUT:     %Ptr.ref.loc9_31: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.a3d]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.140, element0 [symbolic = %ptr (constants.%ptr.3c2)]
// CHECK:STDOUT:     %.loc9_29.3: Core.Form = init_form %impl.elem0.loc9 [symbolic = %.loc9_29.2 (constants.%.ed7)]
// CHECK:STDOUT:     %.loc9_10: type = splice_block %Ptr.ref.loc9_10 [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Ptr.ref.loc9_10: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_7.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.7bc)]
// CHECK:STDOUT:     %t.param: ref @F.%T.as_type.loc9_22.1 (%T.as_type.6c8) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.6c8)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.7bc)]
// CHECK:STDOUT:       %T.as_type.loc9_22.2: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.6c8)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type.loc9_22.2 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.6c8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.as_type.loc9_22.1 (%T.as_type.6c8) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.3c2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.3c2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ca4]
// CHECK:STDOUT:   %Ptr.WithSelf.decl = interface_with_self_decl @Ptr [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete = constants.%assoc0.a3d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Type = @Type.%assoc0
// CHECK:STDOUT:   witness = (@Ptr.WithSelf.%Type)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc7_21.1: type) {
// CHECK:STDOUT:   %U.loc7_21.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_21.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.bd8 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bb9)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_21.2) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.1f9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.425)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc7_56.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc7_56.2 (constants.%Ptr.impl_witness.1f9)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %.loc7_39
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc7_56.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_7.2: %Ptr.type) {
// CHECK:STDOUT:   %T.loc9_7.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.7bc)]
// CHECK:STDOUT:   %T.as_type.loc9_22.1: type = facet_access_type %T.loc9_7.1 [symbolic = %T.as_type.loc9_22.1 (constants.%T.as_type.6c8)]
// CHECK:STDOUT:   %pattern_type.loc9_20: type = pattern_type %T.as_type.loc9_22.1 [symbolic = %pattern_type.loc9_20 (constants.%pattern_type.256)]
// CHECK:STDOUT:   %.loc9_29.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.as_type.loc9_22.1) [symbolic = %.loc9_29.1 (constants.%.dab)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_22.1 [symbolic = %ptr (constants.%ptr.3c2)]
// CHECK:STDOUT:   %.loc9_29.2: Core.Form = init_form %ptr [symbolic = %.loc9_29.2 (constants.%.ed7)]
// CHECK:STDOUT:   %pattern_type.loc9_29: type = pattern_type %ptr [symbolic = %pattern_type.loc9_29 (constants.%pattern_type.438)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.as_type.loc9_22.1 [symbolic = %require_complete.loc9_15 (constants.%require_complete.233)]
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_25 (constants.%require_complete.b3d)]
// CHECK:STDOUT:   %.loc10_10.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc9_22.1) [symbolic = %.loc10_10.1 (constants.%.4fe)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3b7)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.666)]
// CHECK:STDOUT:   %.loc10_10.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc10_10.2 (constants.%.1ab)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.2 (%.1ab) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.077)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.9fb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.as_type.loc9_22.1 (%T.as_type.6c8)) -> out %return.param: @F.%ptr (%ptr.3c2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc9_22.1 (%T.as_type.6c8) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.3c2) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.2 (%.1ab) = impl_witness_access constants.%Copy.lookup_impl_witness.3b7, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.077)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.9fb)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr (%ptr.3c2) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Self.ca4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%.Self.332) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.bb9
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.1f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.as_type.6c8) {
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%T.as_type.6c8
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.3c2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.b07
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_56.2 => constants.%Ptr.impl_witness.140
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.76a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%T.7bc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.7bc) {
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T.7bc
// CHECK:STDOUT:   %T.as_type.loc9_22.1 => constants.%T.as_type.6c8
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.256
// CHECK:STDOUT:   %.loc9_29.1 => constants.%.dab
// CHECK:STDOUT:   %ptr => constants.%ptr.3c2
// CHECK:STDOUT:   %.loc9_29.2 => constants.%.ed7
// CHECK:STDOUT:   %pattern_type.loc9_29 => constants.%pattern_type.438
// CHECK:STDOUT: }
// CHECK:STDOUT:

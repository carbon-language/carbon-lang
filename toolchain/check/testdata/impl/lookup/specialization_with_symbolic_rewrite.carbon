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
// CHECK:STDOUT:   %.Self.c97: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.d21: type = facet_type <@Z, @Z(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.1ab: %Z.type.d21 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.993: type = assoc_entity_type @Z, @Z(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.788: %Z.assoc_type.993 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.7ee: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.d69: %Z.type.7ee = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.030: %Z.type.7ee = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.4f2: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.df8: %Z.assoc_type.4f2 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.bf0: <witness> = require_complete_type %Z.type.7ee [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type.a4e: type = symbolic_binding_type .Self, %.Self.d69 [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.165: <witness> = lookup_impl_witness %.Self.d69, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.4fb: type = impl_witness_access %Z.lookup_impl_witness.165, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z_where.type.949: type = facet_type <@Z, @Z(%S) where %impl.elem0.4fb = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.a66: <witness> = require_complete_type %Z_where.type.949 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.20c: <witness> = impl_witness file.%Z.impl_witness_table.loc9, @T.as.Z.impl.506(%T.67d, %S) [symbolic]
// CHECK:STDOUT:   %Z.type.a71: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.dc9: %Z.type.a71 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.44a: %Z.type.a71 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.58d: type = symbolic_binding_type .Self, %.Self.dc9 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.309: <witness> = lookup_impl_witness %.Self.dc9, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.dd6: type = impl_witness_access %Z.lookup_impl_witness.309, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.3a7: type = facet_type <@Z, @Z(%C) where %impl.elem0.dd6 = %T.67d> [symbolic]
// CHECK:STDOUT:   %require_complete.4bc: <witness> = require_complete_type %Z_where.type.3a7 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.ae6: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.365(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.5ba: %Z.type.a71 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6dc: type = pattern_type %Z.type.a71 [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.5ba [symbolic]
// CHECK:STDOUT:   %pattern_type.e10: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.244: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Z_where.type.6de: type = facet_type <@Z, @Z(%C) where %impl.elem0.dd6 = %T.binding.as_type> [symbolic]
// CHECK:STDOUT:   %require_complete.435: <witness> = require_complete_type %Z_where.type.6de [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.c63: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.365(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %.69b: require_specific_def_type = require_specific_def @T.as.Z.impl.365(%T.binding.as_type) [symbolic]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %T.loc3_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.506 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc9_24.1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc9_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.7ee)]
// CHECK:STDOUT:     %.Self.1: @T.as.Z.impl.506.%Z.type.loc9_42.2 (%Z.type.7ee) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.d69)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.506.%Z.type.loc9_42.2 (%Z.type.7ee) = name_ref .Self, %.Self.1 [symbolic = %.Self.4 (constants.%.Self.d69)]
// CHECK:STDOUT:     %.loc9_50.1: @T.as.Z.impl.506.%Z.assoc_type (%Z.assoc_type.4f2) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.df8)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.506.%Z.assoc_type (%Z.assoc_type.4f2) = name_ref X, %.loc9_50.1 [symbolic = %assoc0 (constants.%assoc0.df8)]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.a4e)]
// CHECK:STDOUT:     %.loc9_50.2: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.a4e)]
// CHECK:STDOUT:     %impl.elem0.loc9_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.165, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.4fb)]
// CHECK:STDOUT:     %.loc9_56.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_56.2: type = converted %.loc9_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.949)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.7ee
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_50.1, %.loc9_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %T.loc9_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %S.loc9_24.1: type = symbolic_binding S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc9 = impl_witness_table (%impl_witness_assoc_constant.loc9), @T.as.Z.impl.506 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc9: <witness> = impl_witness %Z.impl_witness_table.loc9, @T.as.Z.impl.506(constants.%T.67d, constants.%S) [symbolic = @T.as.Z.impl.506.%Z.impl_witness (constants.%Z.impl_witness.20c)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc9: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.365 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc11_30: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     %.Self.1: %Z.type.a71 = symbolic_binding .Self [symbolic_self = constants.%.Self.dc9]
// CHECK:STDOUT:     %.Self.ref: %Z.type.a71 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.dc9]
// CHECK:STDOUT:     %.loc11_46.1: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.252 = name_ref X, %.loc11_46.1 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.58d]
// CHECK:STDOUT:     %.loc11_46.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.58d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.309, element0 [symbolic_self = constants.%impl.elem0.dd6]
// CHECK:STDOUT:     %T.ref.loc11_51: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc11_40: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.3a7)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.a71
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc11_51
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %T.loc11_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_20.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc11 = impl_witness_table (%impl_witness_assoc_constant.loc11), @T.as.Z.impl.365 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc11: <witness> = impl_witness %Z.impl_witness_table.loc11, @T.as.Z.impl.365(constants.%T.67d) [symbolic = @T.as.Z.impl.365.%Z.impl_witness (constants.%Z.impl_witness.ae6)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc11: type = impl_witness_assoc_constant constants.%T.67d [symbolic = @T.as.Z.impl.365.%T.loc11_20.2 (constants.%T.67d)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.6dc = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type (%pattern_type.e10) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type (%pattern_type.e10) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_13: type = splice_block %Z.type [concrete = constants.%Z.type.a71] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:       %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_6.2: %Z.type.a71 = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T.5ba)]
// CHECK:STDOUT:     %t.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_19.1: type = splice_block %.loc13_19.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc13: %Z.type.a71 = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T.5ba)]
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
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_13.1)> [symbolic = %Z.type (constants.%Z.type.d21)]
// CHECK:STDOUT:   %Self.loc3_23.2: @Z.%Z.type (%Z.type.d21) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.1ab)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc3_13.1) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.993)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.993) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.788)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @Z.%Z.type (%Z.type.d21) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.1ab)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.993) = assoc_entity element0, @Z.%X [symbolic = @Z.%assoc0 (constants.%assoc0.788)]
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
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc3_13.2: type, @Z.%Self.loc3_23.1: @Z.%Z.type (%Z.type.d21)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.506(%T.loc9_14.1: type, %S.loc9_24.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   %S.loc9_24.2: type = symbolic_binding S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc9_42.2: type = facet_type <@Z, @Z(%S.loc9_24.2)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.7ee)]
// CHECK:STDOUT:   %.Self.4: @T.as.Z.impl.506.%Z.type.loc9_42.2 (%Z.type.7ee) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.d69)]
// CHECK:STDOUT:   %require_complete.loc9_50: <witness> = require_complete_type %Z.type.loc9_42.2 [symbolic = %require_complete.loc9_50 (constants.%require_complete.bf0)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc9_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.4f2)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.506.%Z.assoc_type (%Z.assoc_type.4f2) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.df8)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.4 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.a4e)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @Z, @Z(%S.loc9_24.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.165)]
// CHECK:STDOUT:   %impl.elem0.loc9_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.4fb)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc9_24.2) where %impl.elem0.loc9_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.949)]
// CHECK:STDOUT:   %require_complete.loc9_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc9_44 (constants.%require_complete.a66)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc9, @T.as.Z.impl.506(%T.loc9_14.2, %S.loc9_24.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.20c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc9_44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.365(%T.loc11_20.1: type) {
// CHECK:STDOUT:   %T.loc11_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_20.2 (constants.%T.67d)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.dd6 = %T.loc11_20.2> [symbolic = %Z_where.type (constants.%Z_where.type.3a7)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.4bc)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.365(%T.loc11_20.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.ae6)]
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
// CHECK:STDOUT:   %T.loc13_6.1: %Z.type.a71 = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T.5ba)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc13_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.e10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.244)]
// CHECK:STDOUT:   %.loc15_11.4: require_specific_def_type = require_specific_def @T.as.Z.impl.365(%T.binding.as_type) [symbolic = %.loc15_11.4 (constants.%.69b)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @T.as.Z.impl.365(%T.binding.as_type) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.c63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type (%pattern_type.e10) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc15_11.1: type = splice_block %impl.elem0 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc15: %Z.type.a71 = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T.5ba)]
// CHECK:STDOUT:       %.loc15_11.2: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.252 = name_ref X, %.loc15_11.2 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc15_11.3: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.c63, element0 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
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
// CHECK:STDOUT: specific @X(constants.%T.67d, constants.%Self.1ab) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.7ee
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.030
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.4f2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.df8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%.Self.d69) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.506(constants.%T.67d, constants.%S) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.67d
// CHECK:STDOUT:   %S.loc9_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc9_42.2 => constants.%Z.type.7ee
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.d69
// CHECK:STDOUT:   %require_complete.loc9_50 => constants.%require_complete.bf0
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.4f2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.df8
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type.a4e
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.165
// CHECK:STDOUT:   %impl.elem0.loc9_50.2 => constants.%impl.elem0.4fb
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.949
// CHECK:STDOUT:   %require_complete.loc9_44 => constants.%require_complete.a66
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.20c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.a71
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.44a
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%.Self.dc9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.365(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.67d
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.3a7
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4bc
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.ae6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.5ba) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%T.5ba
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.365(constants.%T.binding.as_type) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.6de
// CHECK:STDOUT:   %require_complete => constants.%require_complete.435
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.c63
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%T.5ba) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c97: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.d21: type = facet_type <@Z, @Z(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.1ab: %Z.type.d21 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.993: type = assoc_entity_type @Z, @Z(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.788: %Z.assoc_type.993 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.c8d: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.7ee: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.d69: %Z.type.7ee = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.030: %Z.type.7ee = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.4f2: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.df8: %Z.assoc_type.4f2 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.bf0: <witness> = require_complete_type %Z.type.7ee [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type.a4e: type = symbolic_binding_type .Self, %.Self.d69 [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.165: <witness> = lookup_impl_witness %.Self.d69, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.4fb: type = impl_witness_access %Z.lookup_impl_witness.165, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z_where.type.949: type = facet_type <@Z, @Z(%S) where %impl.elem0.4fb = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.a66: <witness> = require_complete_type %Z_where.type.949 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.20c: <witness> = impl_witness file.%Z.impl_witness_table.loc10, @T.as.Z.impl.506(%T.67d, %S) [symbolic]
// CHECK:STDOUT:   %Z.type.a71: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.dc9: %Z.type.a71 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.44a: %Z.type.a71 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.58d: type = symbolic_binding_type .Self, %.Self.dc9 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.309: <witness> = lookup_impl_witness %.Self.dc9, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.dd6: type = impl_witness_access %Z.lookup_impl_witness.309, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.3a7: type = facet_type <@Z, @Z(%C) where %impl.elem0.dd6 = %T.67d> [symbolic]
// CHECK:STDOUT:   %require_complete.4bc: <witness> = require_complete_type %Z_where.type.3a7 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.ae6: <witness> = impl_witness file.%Z.impl_witness_table.loc12, @T.as.Z.impl.365(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.5ba: %Z.type.a71 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.6dc: type = pattern_type %Z.type.a71 [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.5ba [symbolic]
// CHECK:STDOUT:   %pattern_type.e10: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.244: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.544: <witness> = lookup_impl_witness %T.5ba, @Z, @Z(%C) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b53: type = impl_witness_access %Z.lookup_impl_witness.544, element0 [symbolic]
// CHECK:STDOUT:   %require_complete.ea7: <witness> = require_complete_type %impl.elem0.b53 [symbolic]
// CHECK:STDOUT:   %pattern_type.33b: type = pattern_type %impl.elem0.b53 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6d3: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.50f: %ImplicitAs.Convert.type.6d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.fb1: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.b53)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.8b4: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.b53) [symbolic]
// CHECK:STDOUT:   %assoc0.a40: %ImplicitAs.assoc_type.8b4 = assoc_entity element0, imports.%Core.import_ref.d3c [symbolic]
// CHECK:STDOUT:   %require_complete.1d2: <witness> = require_complete_type %ImplicitAs.type.fb1 [symbolic]
// CHECK:STDOUT:   %assoc0.df2: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.a08 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.c18: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.df2)]
// CHECK:STDOUT:   %Core.import_ref.d3c: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.6d3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.50f)]
// CHECK:STDOUT:   %Core.import_ref.a08 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %T.loc3_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.506 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc10_24.1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc10_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.7ee)]
// CHECK:STDOUT:     %.Self.1: @T.as.Z.impl.506.%Z.type.loc10_42.2 (%Z.type.7ee) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.d69)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.506.%Z.type.loc10_42.2 (%Z.type.7ee) = name_ref .Self, %.Self.1 [symbolic = %.Self.4 (constants.%.Self.d69)]
// CHECK:STDOUT:     %.loc10_50.1: @T.as.Z.impl.506.%Z.assoc_type (%Z.assoc_type.4f2) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.df8)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.506.%Z.assoc_type (%Z.assoc_type.4f2) = name_ref X, %.loc10_50.1 [symbolic = %assoc0 (constants.%assoc0.df8)]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.a4e)]
// CHECK:STDOUT:     %.loc10_50.2: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.a4e)]
// CHECK:STDOUT:     %impl.elem0.loc10_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.165, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.4fb)]
// CHECK:STDOUT:     %.loc10_56.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_56.2: type = converted %.loc10_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.949)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.7ee
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_50.1, %.loc10_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %T.loc10_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %S.loc10_24.1: type = symbolic_binding S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc10 = impl_witness_table (%impl_witness_assoc_constant.loc10), @T.as.Z.impl.506 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc10: <witness> = impl_witness %Z.impl_witness_table.loc10, @T.as.Z.impl.506(constants.%T.67d, constants.%S) [symbolic = @T.as.Z.impl.506.%Z.impl_witness (constants.%Z.impl_witness.20c)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc10: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.365 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_24: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     %.Self.1: %Z.type.a71 = symbolic_binding .Self [symbolic_self = constants.%.Self.dc9]
// CHECK:STDOUT:     %.Self.ref: %Z.type.a71 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.dc9]
// CHECK:STDOUT:     %.loc12_40.1: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.252 = name_ref X, %.loc12_40.1 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.58d]
// CHECK:STDOUT:     %.loc12_40.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.58d]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.309, element0 [symbolic_self = constants.%impl.elem0.dd6]
// CHECK:STDOUT:     %T.ref.loc12_45: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc12_34: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.3a7)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.a71
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc12_45
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %T.loc12_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc12 = impl_witness_table (%impl_witness_assoc_constant.loc12), @T.as.Z.impl.365 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc12: <witness> = impl_witness %Z.impl_witness_table.loc12, @T.as.Z.impl.365(constants.%T.67d) [symbolic = @T.as.Z.impl.365.%Z.impl_witness (constants.%Z.impl_witness.ae6)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc12: type = impl_witness_assoc_constant constants.%T.67d [symbolic = @T.as.Z.impl.365.%T.loc12_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.6dc = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc14 (%pattern_type.e10) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc14 (%pattern_type.e10) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_13: type = splice_block %Z.type [concrete = constants.%Z.type.a71] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:       %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.a71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_6.2: %Z.type.a71 = symbolic_binding T, 0 [symbolic = %T.loc14_6.1 (constants.%T.5ba)]
// CHECK:STDOUT:     %t.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc14: %Z.type.a71 = name_ref T, %T.loc14_6.2 [symbolic = %T.loc14_6.1 (constants.%T.5ba)]
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
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_13.1)> [symbolic = %Z.type (constants.%Z.type.d21)]
// CHECK:STDOUT:   %Self.loc3_23.2: @Z.%Z.type (%Z.type.d21) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.1ab)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc3_13.1) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.993)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.993) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.788)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @Z.%Z.type (%Z.type.d21) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.1ab)]
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.993) = assoc_entity element0, @Z.%X [symbolic = @Z.%assoc0 (constants.%assoc0.788)]
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
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c8d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc3_13.2: type, @Z.%Self.loc3_23.1: @Z.%Z.type (%Z.type.d21)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.506(%T.loc10_14.1: type, %S.loc10_24.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   %S.loc10_24.2: type = symbolic_binding S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc10_42.2: type = facet_type <@Z, @Z(%S.loc10_24.2)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.7ee)]
// CHECK:STDOUT:   %.Self.4: @T.as.Z.impl.506.%Z.type.loc10_42.2 (%Z.type.7ee) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.d69)]
// CHECK:STDOUT:   %require_complete.loc10_50: <witness> = require_complete_type %Z.type.loc10_42.2 [symbolic = %require_complete.loc10_50 (constants.%require_complete.bf0)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc10_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.4f2)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.506.%Z.assoc_type (%Z.assoc_type.4f2) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.df8)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.4 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.a4e)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @Z, @Z(%S.loc10_24.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.165)]
// CHECK:STDOUT:   %impl.elem0.loc10_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.4fb)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc10_24.2) where %impl.elem0.loc10_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.949)]
// CHECK:STDOUT:   %require_complete.loc10_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc10_44 (constants.%require_complete.a66)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc10, @T.as.Z.impl.506(%T.loc10_14.2, %S.loc10_24.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.20c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc10_44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.365(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_14.2 (constants.%T.67d)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.dd6 = %T.loc12_14.2> [symbolic = %Z_where.type (constants.%Z_where.type.3a7)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.4bc)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc12, @T.as.Z.impl.365(%T.loc12_14.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.ae6)]
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
// CHECK:STDOUT:   %T.loc14_6.1: %Z.type.a71 = symbolic_binding T, 0 [symbolic = %T.loc14_6.1 (constants.%T.5ba)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc14_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc14: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc14 (constants.%pattern_type.e10)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc14 (constants.%require_complete.244)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc14_6.1, @Z, @Z(constants.%C) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.544)]
// CHECK:STDOUT:   %impl.elem0.loc22_11.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.b53)]
// CHECK:STDOUT:   %require_complete.loc22_11: <witness> = require_complete_type %impl.elem0.loc22_11.2 [symbolic = %require_complete.loc22_11 (constants.%require_complete.ea7)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %impl.elem0.loc22_11.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.33b)]
// CHECK:STDOUT:   %ImplicitAs.type.loc22_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2)> [symbolic = %ImplicitAs.type.loc22_16.2 (constants.%ImplicitAs.type.fb1)]
// CHECK:STDOUT:   %require_complete.loc22_16: <witness> = require_complete_type %ImplicitAs.type.loc22_16.2 [symbolic = %require_complete.loc22_16 (constants.%require_complete.1d2)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.8b4)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8b4) = assoc_entity element0, imports.%Core.import_ref.d3c [symbolic = %assoc0 (constants.%assoc0.a40)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type.loc22 (%pattern_type.33b) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %impl.elem0.loc22_11.1 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.b53)] {
// CHECK:STDOUT:       %T.ref.loc22: %Z.type.a71 = name_ref T, %T.loc14_6.2 [symbolic = %T.loc14_6.1 (constants.%T.5ba)]
// CHECK:STDOUT:       %.loc22_11.2: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.252 = name_ref X, %.loc22_11.2 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc22_11.3: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc22_11.1: type = impl_witness_access constants.%Z.lookup_impl_witness.544, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.b53)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ImplicitAs.type.loc22_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%impl.elem0.b53)> [symbolic = %ImplicitAs.type.loc22_16.2 (constants.%ImplicitAs.type.fb1)]
// CHECK:STDOUT:     %.loc22_16.1: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8b4) = specific_constant imports.%Core.import_ref.c18, @ImplicitAs(constants.%impl.elem0.b53) [symbolic = %assoc0 (constants.%assoc0.a40)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8b4) = name_ref Convert, %.loc22_16.1 [symbolic = %assoc0 (constants.%assoc0.a40)]
// CHECK:STDOUT:     %.loc22_16.2: @F.%impl.elem0.loc22_11.2 (%impl.elem0.b53) = converted %t.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %a: @F.%impl.elem0.loc22_11.2 (%impl.elem0.b53) = value_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.67d, constants.%Self.1ab) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.7ee
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.030
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.4f2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.df8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%.Self.d69) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.506(constants.%T.67d, constants.%S) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T.67d
// CHECK:STDOUT:   %S.loc10_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc10_42.2 => constants.%Z.type.7ee
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.d69
// CHECK:STDOUT:   %require_complete.loc10_50 => constants.%require_complete.bf0
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.4f2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.df8
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type.a4e
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.165
// CHECK:STDOUT:   %impl.elem0.loc10_50.2 => constants.%impl.elem0.4fb
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.949
// CHECK:STDOUT:   %require_complete.loc10_44 => constants.%require_complete.a66
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.20c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.a71
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.44a
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%.Self.dc9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.365(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.67d
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.3a7
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4bc
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.ae6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.5ba) {
// CHECK:STDOUT:   %T.loc14_6.1 => constants.%T.5ba
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc14 => constants.%pattern_type.e10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%T.5ba) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.aed: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.d28: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c97: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.70c: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.70c [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.70c, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5f5: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.300a2d.1: type = facet_type <@Ptr where %impl.elem0.5f5 = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %require_complete.49f764.1: <witness> = require_complete_type %Ptr_where.type.300a2d.1 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.31e287.1: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.300a2d.2: type = facet_type <@Ptr where %impl.elem0.5f5 = %ptr.e8f8f9.2> [symbolic]
// CHECK:STDOUT:   %require_complete.49f764.2: <witness> = require_complete_type %Ptr_where.type.300a2d.2 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.31e287.2: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %.45d: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Ptr.facet: %Ptr.type = facet_value %T.67d, (%Ptr.impl_witness.31e287.2) [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.841: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f8f9.2, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f8f9.2, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %.cb5: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.771: %.cb5 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b85: <specific function> = specific_impl_function %impl.elem0.771, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.70c]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.70c]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.5f5]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.300a2d.1)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %U.loc7_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = @U.as.Ptr.impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.31e287.1)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = @U.as.Ptr.impl.%ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
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
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %Ptr.facet.loc9_30.2: %Ptr.type = facet_value %T.ref.loc9_29, (constants.%Ptr.impl_witness.31e287.2) [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:     %.loc9_30.2: %Ptr.type = converted %T.ref.loc9_29, %Ptr.facet.loc9_30.2 [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.31e287.2, element0 [symbolic = %ptr (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
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
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.aed]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0.d28]
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
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.5f5 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.300a2d.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.49f764.1)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.31e287.1)]
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
// CHECK:STDOUT:   %T.loc9_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.67d)]
// CHECK:STDOUT:   %pattern_type.loc9_20: type = pattern_type %T.loc9_6.1 [symbolic = %pattern_type.loc9_20 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %.loc9_30.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.loc9_6.1) [symbolic = %.loc9_30.1 (constants.%.45d)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.loc9_6.1) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.31e287.2)]
// CHECK:STDOUT:   %Ptr.facet.loc9_30.1: %Ptr.type = facet_value %T.loc9_6.1, (%Ptr.impl_witness) [symbolic = %Ptr.facet.loc9_30.1 (constants.%Ptr.facet)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc9_6.1 [symbolic = %ptr (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %pattern_type.loc9_26: type = pattern_type %ptr [symbolic = %pattern_type.loc9_26 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_30: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_30 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %require_complete.loc9_16: <witness> = require_complete_type %T.loc9_6.1 [symbolic = %require_complete.loc9_16 (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc10_10.2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc9_6.1) [symbolic = %.loc10_10.2 (constants.%.841)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.3: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.3 (constants.%.cb5)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.3 (%.cb5) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.771)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.b85)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.loc9_6.1 (%T.67d)) -> @F.%ptr (%ptr.e8f8f9.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc9_6.1 (%T.67d) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.e8f8f9.2) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.3 (%.cb5) = impl_witness_access constants.%Copy.lookup_impl_witness.2e6, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.771)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.b85)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %.loc10_10.1: init @F.%ptr (%ptr.e8f8f9.2) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %.loc10_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.aed) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.70c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.300a2d.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.49f764.1
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.31e287.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.67d) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.300a2d.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.49f764.2
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.31e287.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.67d
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.51d
// CHECK:STDOUT:   %.loc9_30.1 => constants.%.45d
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.31e287.2
// CHECK:STDOUT:   %Ptr.facet.loc9_30.1 => constants.%Ptr.facet
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %pattern_type.loc9_26 => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.aed: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.d28: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c97: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.70c: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.70c [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.70c, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5f5: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.300: type = facet_type <@Ptr where %impl.elem0.5f5 = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %require_complete.49f: <witness> = require_complete_type %Ptr_where.type.300 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.31e: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %T.52c: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3d8: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.cdd: type = symbolic_binding_type T, 0, %T.52c [symbolic]
// CHECK:STDOUT:   %pattern_type.e55: type = pattern_type %T.binding.as_type.cdd [symbolic]
// CHECK:STDOUT:   %ptr.586: type = ptr_type %T.binding.as_type.cdd [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.793: type = facet_type <@Ptr where %impl.elem0.5f5 = %ptr.586> [symbolic]
// CHECK:STDOUT:   %require_complete.cc6: <witness> = require_complete_type %Ptr_where.type.793 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.682: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type.cdd) [symbolic]
// CHECK:STDOUT:   %.672: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type.cdd) [symbolic]
// CHECK:STDOUT:   %pattern_type.cd0: type = pattern_type %ptr.586 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.861: <witness> = require_complete_type %ptr.586 [symbolic]
// CHECK:STDOUT:   %require_complete.46d: <witness> = require_complete_type %T.binding.as_type.cdd [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.763: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type.cdd) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.b92: <witness> = lookup_impl_witness %ptr.586, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.586, (%Copy.lookup_impl_witness.b92) [symbolic]
// CHECK:STDOUT:   %.16e: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.bf9: %.16e = impl_witness_access %Copy.lookup_impl_witness.b92, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.51f: <specific function> = specific_impl_function %impl.elem0.bf9, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.70c]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.70c]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.5f5]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.300)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %U.loc7_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = @U.as.Ptr.impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.31e)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = @U.as.Ptr.impl.%ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3d8 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_19 (%pattern_type.e55) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_19 (%pattern_type.e55) = var_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_19 (%pattern_type.e55) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_25 (%pattern_type.cd0) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_25 (%pattern_type.cd0) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.52c)]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %T.as_type.loc9_29: type = facet_access_type %T.ref.loc9_28 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.cdd)]
// CHECK:STDOUT:     %.loc9_29.2: type = converted %T.ref.loc9_28, %T.as_type.loc9_29 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.cdd)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.682, element0 [symbolic = %ptr (constants.%ptr.586)]
// CHECK:STDOUT:     %.loc9_10: type = splice_block %Ptr.ref [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:       %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.52c)]
// CHECK:STDOUT:     %t.param: ref @F.%T.binding.as_type (%T.binding.as_type.cdd) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.cdd)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.52c)]
// CHECK:STDOUT:       %T.as_type.loc9_22: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.cdd)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type.loc9_22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.cdd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.binding.as_type (%T.binding.as_type.cdd) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.586) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.586) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.aed]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0.d28]
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
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.5f5 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.300)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.49f)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.31e)]
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
// CHECK:STDOUT:   %T.loc9_6.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.52c)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.cdd)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.e55)]
// CHECK:STDOUT:   %.loc9_29.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %.loc9_29.1 (constants.%.672)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.682)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.binding.as_type [symbolic = %ptr (constants.%ptr.586)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.cd0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_29: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_29 (constants.%require_complete.861)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9_15 (constants.%require_complete.46d)]
// CHECK:STDOUT:   %.loc10_10.2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type) [symbolic = %.loc10_10.2 (constants.%.763)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.b92)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.3: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.3 (constants.%.16e)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.3 (%.16e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.bf9)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.51f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type.cdd)) -> @F.%ptr (%ptr.586) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.binding.as_type (%T.binding.as_type.cdd) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.586) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.3 (%.16e) = impl_witness_access constants.%Copy.lookup_impl_witness.b92, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.bf9)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.51f)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %.loc10_10.1: init @F.%ptr (%ptr.586) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %.loc10_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.aed) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.70c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.300
// CHECK:STDOUT:   %require_complete => constants.%require_complete.49f
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.31e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.binding.as_type.cdd) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.binding.as_type.cdd
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.586
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.793
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cc6
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.682
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%T.52c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.52c) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.52c
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.cdd
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.e55
// CHECK:STDOUT:   %.loc9_29.1 => constants.%.672
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.682
// CHECK:STDOUT:   %ptr => constants.%ptr.586
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.cd0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.aed: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.d28: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c97: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self.70c: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.70c [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.70c, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5f5: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.300: type = facet_type <@Ptr where %impl.elem0.5f5 = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %require_complete.49f: <witness> = require_complete_type %Ptr_where.type.300 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.31e: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %T.52c: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3d8: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.cdd: type = symbolic_binding_type T, 0, %T.52c [symbolic]
// CHECK:STDOUT:   %pattern_type.e55: type = pattern_type %T.binding.as_type.cdd [symbolic]
// CHECK:STDOUT:   %ptr.586: type = ptr_type %T.binding.as_type.cdd [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.793: type = facet_type <@Ptr where %impl.elem0.5f5 = %ptr.586> [symbolic]
// CHECK:STDOUT:   %require_complete.cc6: <witness> = require_complete_type %Ptr_where.type.793 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.682: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type.cdd) [symbolic]
// CHECK:STDOUT:   %.672: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type.cdd) [symbolic]
// CHECK:STDOUT:   %pattern_type.cd0: type = pattern_type %ptr.586 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.861: <witness> = require_complete_type %ptr.586 [symbolic]
// CHECK:STDOUT:   %require_complete.46d: <witness> = require_complete_type %T.binding.as_type.cdd [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %.763: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type.cdd) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.b92: <witness> = lookup_impl_witness %ptr.586, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.586, (%Copy.lookup_impl_witness.b92) [symbolic]
// CHECK:STDOUT:   %.16e: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.bf9: %.16e = impl_witness_access %Copy.lookup_impl_witness.b92, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.51f: <specific function> = specific_impl_function %impl.elem0.bf9, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.70c]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.70c]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.5f5]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.300)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:     %U.loc7_20.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_20.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = @U.as.Ptr.impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.31e)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = @U.as.Ptr.impl.%ptr.loc7_54.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3d8 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_19 (%pattern_type.e55) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_19 (%pattern_type.e55) = var_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.var_patt: @F.%pattern_type.loc9_19 (%pattern_type.e55) = var_pattern %t.param_patt [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_25 (%pattern_type.cd0) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_25 (%pattern_type.cd0) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.52c)]
// CHECK:STDOUT:     %Ptr.ref.loc9_31: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.682, element0 [symbolic = %ptr (constants.%ptr.586)]
// CHECK:STDOUT:     %.loc9_10: type = splice_block %Ptr.ref.loc9_10 [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c97]
// CHECK:STDOUT:       %Ptr.ref.loc9_10: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.52c)]
// CHECK:STDOUT:     %t.param: ref @F.%T.binding.as_type (%T.binding.as_type.cdd) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.cdd)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.52c)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.cdd)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.cdd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.binding.as_type (%T.binding.as_type.cdd) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.586) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.586) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.aed]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0.d28]
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
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.5f5 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.300)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.49f)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.31e)]
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
// CHECK:STDOUT:   %T.loc9_6.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.52c)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.cdd)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.e55)]
// CHECK:STDOUT:   %.loc9_29: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %.loc9_29 (constants.%.672)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.682)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.binding.as_type [symbolic = %ptr (constants.%ptr.586)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.cd0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_29: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_29 (constants.%require_complete.861)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9_15 (constants.%require_complete.46d)]
// CHECK:STDOUT:   %.loc10_10.2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type) [symbolic = %.loc10_10.2 (constants.%.763)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.b92)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc10_10.3: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_10.3 (constants.%.16e)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.3 (%.16e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.bf9)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.51f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type.cdd)) -> @F.%ptr (%ptr.586) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.binding.as_type (%T.binding.as_type.cdd) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.586) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.3 (%.16e) = impl_witness_access constants.%Copy.lookup_impl_witness.b92, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.bf9)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.51f)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %.loc10_10.1: init @F.%ptr (%ptr.586) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %.loc10_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.aed) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%.Self.70c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.300
// CHECK:STDOUT:   %require_complete => constants.%require_complete.49f
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.31e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.binding.as_type.cdd) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.binding.as_type.cdd
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.586
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.793
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cc6
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.682
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%T.52c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.52c) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.52c
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.cdd
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.e55
// CHECK:STDOUT:   %.loc9_29 => constants.%.672
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.682
// CHECK:STDOUT:   %ptr => constants.%ptr.586
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.cd0
// CHECK:STDOUT: }
// CHECK:STDOUT:

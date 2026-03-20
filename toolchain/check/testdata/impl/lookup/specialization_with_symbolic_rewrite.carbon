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
// CHECK:STDOUT:   %Z.type.352: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.352 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.0ed: type = facet_type <@Z, @Z(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.822: %Z.type.0ed = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0bf: type = assoc_entity_type @Z, @Z(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.f33: %Z.assoc_type.0bf = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.4d0: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.90f: %Z.type.4d0 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.427: %Z.type.4d0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.5b2: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.8bc: %Z.assoc_type.5b2 = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %require_complete.72d: <witness> = require_complete_type %Z.type.4d0 [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type.097: type = symbolic_binding_type .Self, %.Self.90f [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.7d6: <witness> = lookup_impl_witness %.Self.90f, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e5e: type = impl_witness_access %Z.lookup_impl_witness.7d6, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z_where.type.a76: type = facet_type <@Z, @Z(%S) where %impl.elem0.e5e = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.cfe: <witness> = impl_witness @T.as.Z.impl.34a.%Z.impl_witness_table, @T.as.Z.impl.34a(%T.67d, %S) [symbolic]
// CHECK:STDOUT:   %require_complete.1f8: <witness> = require_complete_type %Z_where.type.a76 [symbolic]
// CHECK:STDOUT:   %Z.facet.100: %Z.type.4d0 = facet_value %T.67d, (%Z.impl_witness.cfe) [symbolic]
// CHECK:STDOUT:   %Z.type.be4: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.bf2: %Z.type.be4 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.af9: %Z.type.be4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.553: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.1e0: %Z.assoc_type.553 = assoc_entity element0, @Z.WithSelf.%X [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.8c6: type = symbolic_binding_type .Self, %.Self.bf2 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.c06: <witness> = lookup_impl_witness %.Self.bf2, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2cd: type = impl_witness_access %Z.lookup_impl_witness.c06, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.c3e: type = facet_type <@Z, @Z(%C) where %impl.elem0.2cd = %T.67d> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.9b0: <witness> = impl_witness @T.as.Z.impl.cf0.%Z.impl_witness_table, @T.as.Z.impl.cf0(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.864: <witness> = require_complete_type %Z_where.type.c3e [symbolic]
// CHECK:STDOUT:   %Z.facet.738: %Z.type.be4 = facet_value %T.67d, (%Z.impl_witness.9b0) [symbolic]
// CHECK:STDOUT:   %pattern_type.53f: type = pattern_type %Z.type.be4 [concrete]
// CHECK:STDOUT:   %T.fe5: %Z.type.be4 = symbolic_binding T, 0 [symbolic]
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
// CHECK:STDOUT:     %.loc9.1: type = splice_block %.loc9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.34a [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc25.1 [symbolic = %T.loc25.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc29.1 [symbolic = %S.loc29.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc38.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc38.2 (constants.%Z.type.4d0)]
// CHECK:STDOUT:     %.Self.1: @T.as.Z.impl.34a.%Z.type.loc38.2 (%Z.type.4d0) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.34a.%Z.type.loc38.2 (%Z.type.4d0) = name_ref .Self, %.Self.1 [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:     %.loc40.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:     %.loc40.2: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = specific_constant @X.%assoc0, @Z.WithSelf(constants.%S, constants.%.Self.90f) [symbolic = %assoc0 (constants.%assoc0.8bc)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = name_ref X, %.loc40.2 [symbolic = %assoc0 (constants.%assoc0.8bc)]
// CHECK:STDOUT:     %impl.elem0.loc40.1: type = impl_witness_access constants.%Z.lookup_impl_witness.7d6, element0 [symbolic = %impl.elem0.loc40.2 (constants.%impl.elem0.e5e)]
// CHECK:STDOUT:     %.loc44.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc44.2: type = converted %.loc44.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc39: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.a76)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.4d0
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc40.1, %.loc44.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc27.1: type = splice_block %.loc27.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc27.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc25.1: type = symbolic_binding T, 0 [symbolic = %T.loc25.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc31.1: type = splice_block %.loc31.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc31.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc29.1: type = symbolic_binding S, 1 [symbolic = %S.loc29.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.cf0 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc55: type = name_ref T, %T.loc51.1 [symbolic = %T.loc51.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.be4]
// CHECK:STDOUT:     %.Self.1: %Z.type.be4 = symbolic_binding .Self [symbolic_self = constants.%.Self.bf2]
// CHECK:STDOUT:     %.Self.ref: %Z.type.be4 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.bf2]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.8c6]
// CHECK:STDOUT:     %.loc62.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.8c6]
// CHECK:STDOUT:     %.loc62.2: %Z.assoc_type.553 = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%.Self.bf2) [concrete = constants.%assoc0.1e0]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.553 = name_ref X, %.loc62.2 [concrete = constants.%assoc0.1e0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.c06, element0 [symbolic_self = constants.%impl.elem0.2cd]
// CHECK:STDOUT:     %T.ref.loc65: type = name_ref T, %T.loc51.1 [symbolic = %T.loc51.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc61: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.c3e)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.be4
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc65
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc53.1: type = splice_block %.loc53.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc53.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc51.1: type = symbolic_binding T, 0 [symbolic = %T.loc51.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.53f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type (%pattern_type.bb9) = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type (%pattern_type.bb9) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc76: type = splice_block %Z.type [concrete = constants.%Z.type.be4] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.be4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc71.2: %Z.type.be4 = symbolic_binding T, 0 [symbolic = %T.loc71.1 (constants.%T.fe5)]
// CHECK:STDOUT:     %t.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc80.1: type = splice_block %.loc80.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc80: %Z.type.be4 = name_ref T, %T.loc71.2 [symbolic = %T.loc71.1 (constants.%T.fe5)]
// CHECK:STDOUT:       %T.as_type.loc80: type = facet_access_type %T.ref.loc80 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc80.2: type = converted %T.ref.loc80, %T.as_type.loc80 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc7.2: type) {
// CHECK:STDOUT:   %T.loc7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc7.1)> [symbolic = %Z.type (constants.%Z.type.0ed)]
// CHECK:STDOUT:   %Self.loc11.2: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc11.2 (constants.%Self.822)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11.1: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc11.2 (constants.%Self.822)]
// CHECK:STDOUT:     %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.WithSelf.%Z.assoc_type (%Z.assoc_type.0bf) = assoc_entity element0, @Z.WithSelf.%X [symbolic = @Z.WithSelf.%assoc0 (constants.%assoc0.f33)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (@Z.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.34a(%T.loc25.1: type, %S.loc29.1: type) {
// CHECK:STDOUT:   %T.loc25.2: type = symbolic_binding T, 0 [symbolic = %T.loc25.2 (constants.%T.67d)]
// CHECK:STDOUT:   %S.loc29.2: type = symbolic_binding S, 1 [symbolic = %S.loc29.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc38.2: type = facet_type <@Z, @Z(%S.loc29.2)> [symbolic = %Z.type.loc38.2 (constants.%Z.type.4d0)]
// CHECK:STDOUT:   %.Self.4: @T.as.Z.impl.34a.%Z.type.loc38.2 (%Z.type.4d0) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:   %require_complete.loc40: <witness> = require_complete_type %Z.type.loc38.2 [symbolic = %require_complete.loc40 (constants.%require_complete.72d)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.4 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc29.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.5b2)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = assoc_entity element0, @Z.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.8bc)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @Z, @Z(%S.loc29.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.7d6)]
// CHECK:STDOUT:   %impl.elem0.loc40.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc40.2 (constants.%impl.elem0.e5e)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc29.2) where %impl.elem0.loc40.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.a76)]
// CHECK:STDOUT:   %Z.impl_witness.loc45.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.34a(%T.loc25.2, %S.loc29.2) [symbolic = %Z.impl_witness.loc45.2 (constants.%Z.impl_witness.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc39: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc39 (constants.%require_complete.1f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc39 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.34a [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc45.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.34a(constants.%T.67d, constants.%S) [symbolic = %Z.impl_witness.loc45.2 (constants.%Z.impl_witness.cfe)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Z.impl_witness.loc45.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.cf0(%T.loc51.1: type) {
// CHECK:STDOUT:   %T.loc51.2: type = symbolic_binding T, 0 [symbolic = %T.loc51.2 (constants.%T.67d)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.2cd = %T.loc51.2> [symbolic = %Z_where.type (constants.%Z_where.type.c3e)]
// CHECK:STDOUT:   %Z.impl_witness.loc66.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.cf0(%T.loc51.2) [symbolic = %Z.impl_witness.loc66.2 (constants.%Z.impl_witness.9b0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.864)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc55 as %.loc61 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.cf0 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc66.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.cf0(constants.%T.67d) [symbolic = %Z.impl_witness.loc66.2 (constants.%Z.impl_witness.9b0)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%T.67d [symbolic = %T.loc51.2 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Z.impl_witness.loc66.1
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
// CHECK:STDOUT: generic fn @F(%T.loc71.2: %Z.type.be4) {
// CHECK:STDOUT:   %T.loc71.1: %Z.type.be4 = symbolic_binding T, 0 [symbolic = %T.loc71.1 (constants.%T.fe5)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc71.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.bb9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.f3b)]
// CHECK:STDOUT:   %.loc88.4: require_specific_def_type = require_specific_def @T.as.Z.impl.cf0(%T.binding.as_type) [symbolic = %.loc88.4 (constants.%.7b3)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @T.as.Z.impl.cf0.%Z.impl_witness_table, @T.as.Z.impl.cf0(%T.binding.as_type) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.5a3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type (%pattern_type.bb9) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc88.1: type = splice_block %impl.elem0 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc87: %Z.type.be4 = name_ref T, %T.loc71.2 [symbolic = %T.loc71.1 (constants.%T.fe5)]
// CHECK:STDOUT:       %T.as_type.loc88: type = facet_access_type %T.ref.loc87 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc88.2: type = converted %T.ref.loc87, %T.as_type.loc88 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc88.3: %Z.assoc_type.553 = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%T.fe5) [concrete = constants.%assoc0.1e0]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.553 = name_ref X, %.loc88.3 [concrete = constants.%assoc0.1e0]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.5a3, element0 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%T.binding.as_type (%T.binding.as_type) = value_binding a, %t.ref
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T.67d, constants.%Self.822) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.4d0
// CHECK:STDOUT:   %Self.loc11.2 => constants.%Self.427
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Self.822) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%.Self.90f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.34a(constants.%T.67d, constants.%S) {
// CHECK:STDOUT:   %T.loc25.2 => constants.%T.67d
// CHECK:STDOUT:   %S.loc29.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc38.2 => constants.%Z.type.4d0
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.90f
// CHECK:STDOUT:   %require_complete.loc40 => constants.%require_complete.72d
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type.097
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8bc
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.7d6
// CHECK:STDOUT:   %impl.elem0.loc40.2 => constants.%impl.elem0.e5e
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.a76
// CHECK:STDOUT:   %Z.impl_witness.loc45.2 => constants.%Z.impl_witness.cfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Z.facet.100) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.be4
// CHECK:STDOUT:   %Self.loc11.2 => constants.%Self.af9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Self.822) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.553
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%.Self.bf2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.553
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.cf0(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc51.2 => constants.%T.67d
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.c3e
// CHECK:STDOUT:   %Z.impl_witness.loc66.2 => constants.%Z.impl_witness.9b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Z.facet.738) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.553
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.fe5) {
// CHECK:STDOUT:   %T.loc71.1 => constants.%T.fe5
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bb9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%T.fe5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.553
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.cf0(constants.%T.binding.as_type) {
// CHECK:STDOUT:   %T.loc51.2 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.325
// CHECK:STDOUT:   %Z.impl_witness.loc66.2 => constants.%Z.impl_witness.5a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.267
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Z.type.352: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.352 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.0ed: type = facet_type <@Z, @Z(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.822: %Z.type.0ed = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0bf: type = assoc_entity_type @Z, @Z(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.f33: %Z.assoc_type.0bf = assoc_entity element0, @Z.WithSelf.%X [symbolic]
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
// CHECK:STDOUT:   %assoc0.8bc: %Z.assoc_type.5b2 = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %require_complete.72d: <witness> = require_complete_type %Z.type.4d0 [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type.097: type = symbolic_binding_type .Self, %.Self.90f [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.7d6: <witness> = lookup_impl_witness %.Self.90f, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e5e: type = impl_witness_access %Z.lookup_impl_witness.7d6, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Z_where.type.a76: type = facet_type <@Z, @Z(%S) where %impl.elem0.e5e = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.cfe: <witness> = impl_witness @T.as.Z.impl.34a.%Z.impl_witness_table, @T.as.Z.impl.34a(%T.67d, %S) [symbolic]
// CHECK:STDOUT:   %require_complete.1f8: <witness> = require_complete_type %Z_where.type.a76 [symbolic]
// CHECK:STDOUT:   %Z.facet.100: %Z.type.4d0 = facet_value %T.67d, (%Z.impl_witness.cfe) [symbolic]
// CHECK:STDOUT:   %Z.type.be4: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.bf2: %Z.type.be4 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.af9: %Z.type.be4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.553: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.1e0: %Z.assoc_type.553 = assoc_entity element0, @Z.WithSelf.%X [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.8c6: type = symbolic_binding_type .Self, %.Self.bf2 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.c06: <witness> = lookup_impl_witness %.Self.bf2, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2cd: type = impl_witness_access %Z.lookup_impl_witness.c06, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.c3e: type = facet_type <@Z, @Z(%C) where %impl.elem0.2cd = %T.67d> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.9b0: <witness> = impl_witness @T.as.Z.impl.cf0.%Z.impl_witness_table, @T.as.Z.impl.cf0(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.864: <witness> = require_complete_type %Z_where.type.c3e [symbolic]
// CHECK:STDOUT:   %Z.facet.738: %Z.type.be4 = facet_value %T.67d, (%Z.impl_witness.9b0) [symbolic]
// CHECK:STDOUT:   %pattern_type.53f: type = pattern_type %Z.type.be4 [concrete]
// CHECK:STDOUT:   %T.fe5: %Z.type.be4 = symbolic_binding T, 0 [symbolic]
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
// CHECK:STDOUT:   %ImplicitAs.type.031: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.738: %ImplicitAs.type.031 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b3a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.738) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.1de: %ImplicitAs.WithSelf.Convert.type.b3a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7e4: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.a5e)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ac6: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.a5e) [symbolic]
// CHECK:STDOUT:   %assoc0.ae8: %ImplicitAs.assoc_type.ac6 = assoc_entity element0, imports.%Core.import_ref.201 [symbolic]
// CHECK:STDOUT:   %require_complete.591: <witness> = require_complete_type %ImplicitAs.type.7e4 [symbolic]
// CHECK:STDOUT:   %assoc0.843: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.cc1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.178: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//prelude/parts/as, loc82, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.843)]
// CHECK:STDOUT:   %Core.import_ref.201: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.b3a) = import_ref Core//prelude/parts/as, loc82, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.1de)]
// CHECK:STDOUT:   %Core.import_ref.cc1 = import_ref Core//prelude/parts/as, loc82, unloaded
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
// CHECK:STDOUT:     %.loc9.1: type = splice_block %.loc9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.34a [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc29.1 [symbolic = %T.loc29.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc33.1 [symbolic = %S.loc33.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc42.2 (constants.%Z.type.4d0)]
// CHECK:STDOUT:     %.Self.1: @T.as.Z.impl.34a.%Z.type.loc42.2 (%Z.type.4d0) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.34a.%Z.type.loc42.2 (%Z.type.4d0) = name_ref .Self, %.Self.1 [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:     %.loc44.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:     %.loc44.2: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = specific_constant @X.%assoc0, @Z.WithSelf(constants.%S, constants.%.Self.90f) [symbolic = %assoc0 (constants.%assoc0.8bc)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = name_ref X, %.loc44.2 [symbolic = %assoc0 (constants.%assoc0.8bc)]
// CHECK:STDOUT:     %impl.elem0.loc44.1: type = impl_witness_access constants.%Z.lookup_impl_witness.7d6, element0 [symbolic = %impl.elem0.loc44.2 (constants.%impl.elem0.e5e)]
// CHECK:STDOUT:     %.loc48.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc48.2: type = converted %.loc48.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc43: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.a76)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.4d0
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc44.1, %.loc48.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc31.1: type = splice_block %.loc31.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc31.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc29.1: type = symbolic_binding T, 0 [symbolic = %T.loc29.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc35.1: type = splice_block %.loc35.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc35.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc33.1: type = symbolic_binding S, 1 [symbolic = %S.loc33.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.cf0 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc58: type = name_ref T, %T.loc54.1 [symbolic = %T.loc54.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.be4]
// CHECK:STDOUT:     %.Self.1: %Z.type.be4 = symbolic_binding .Self [symbolic_self = constants.%.Self.bf2]
// CHECK:STDOUT:     %.Self.ref: %Z.type.be4 = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.bf2]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type.8c6]
// CHECK:STDOUT:     %.loc65.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type.8c6]
// CHECK:STDOUT:     %.loc65.2: %Z.assoc_type.553 = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%.Self.bf2) [concrete = constants.%assoc0.1e0]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.553 = name_ref X, %.loc65.2 [concrete = constants.%assoc0.1e0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.c06, element0 [symbolic_self = constants.%impl.elem0.2cd]
// CHECK:STDOUT:     %T.ref.loc68: type = name_ref T, %T.loc54.1 [symbolic = %T.loc54.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc64: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.c3e)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Z.type.be4
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc68
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc56.1: type = splice_block %.loc56.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc56.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc54.1: type = symbolic_binding T, 0 [symbolic = %T.loc54.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.53f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc82 (%pattern_type.bb9) = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc82 (%pattern_type.bb9) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc79: type = splice_block %Z.type [concrete = constants.%Z.type.be4] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Z.ref: %Z.type.352 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.be4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc74.2: %Z.type.be4 = symbolic_binding T, 0 [symbolic = %T.loc74.1 (constants.%T.fe5)]
// CHECK:STDOUT:     %t.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc83.1: type = splice_block %.loc83.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc83: %Z.type.be4 = name_ref T, %T.loc74.2 [symbolic = %T.loc74.1 (constants.%T.fe5)]
// CHECK:STDOUT:       %T.as_type.loc83: type = facet_access_type %T.ref.loc83 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc83.2: type = converted %T.ref.loc83, %T.as_type.loc83 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc7.2: type) {
// CHECK:STDOUT:   %T.loc7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc7.1)> [symbolic = %Z.type (constants.%Z.type.0ed)]
// CHECK:STDOUT:   %Self.loc11.2: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc11.2 (constants.%Self.822)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11.1: @Z.%Z.type (%Z.type.0ed) = symbolic_binding Self, 1 [symbolic = %Self.loc11.2 (constants.%Self.822)]
// CHECK:STDOUT:     %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.WithSelf.%Z.assoc_type (%Z.assoc_type.0bf) = assoc_entity element0, @Z.WithSelf.%X [symbolic = @Z.WithSelf.%assoc0 (constants.%assoc0.f33)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (@Z.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.162]
// CHECK:STDOUT:   %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.34a(%T.loc29.1: type, %S.loc33.1: type) {
// CHECK:STDOUT:   %T.loc29.2: type = symbolic_binding T, 0 [symbolic = %T.loc29.2 (constants.%T.67d)]
// CHECK:STDOUT:   %S.loc33.2: type = symbolic_binding S, 1 [symbolic = %S.loc33.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc42.2: type = facet_type <@Z, @Z(%S.loc33.2)> [symbolic = %Z.type.loc42.2 (constants.%Z.type.4d0)]
// CHECK:STDOUT:   %.Self.4: @T.as.Z.impl.34a.%Z.type.loc42.2 (%Z.type.4d0) = symbolic_binding .Self [symbolic = %.Self.4 (constants.%.Self.90f)]
// CHECK:STDOUT:   %require_complete.loc44: <witness> = require_complete_type %Z.type.loc42.2 [symbolic = %require_complete.loc44 (constants.%require_complete.72d)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.4 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.097)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc33.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.5b2)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.34a.%Z.assoc_type (%Z.assoc_type.5b2) = assoc_entity element0, @Z.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.8bc)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.4, @Z, @Z(%S.loc33.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.7d6)]
// CHECK:STDOUT:   %impl.elem0.loc44.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc44.2 (constants.%impl.elem0.e5e)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc33.2) where %impl.elem0.loc44.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.a76)]
// CHECK:STDOUT:   %Z.impl_witness.loc49.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.34a(%T.loc29.2, %S.loc33.2) [symbolic = %Z.impl_witness.loc49.2 (constants.%Z.impl_witness.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc43: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc43 (constants.%require_complete.1f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc43 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.34a [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc49.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.34a(constants.%T.67d, constants.%S) [symbolic = %Z.impl_witness.loc49.2 (constants.%Z.impl_witness.cfe)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Z.impl_witness.loc49.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.cf0(%T.loc54.1: type) {
// CHECK:STDOUT:   %T.loc54.2: type = symbolic_binding T, 0 [symbolic = %T.loc54.2 (constants.%T.67d)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.2cd = %T.loc54.2> [symbolic = %Z_where.type (constants.%Z_where.type.c3e)]
// CHECK:STDOUT:   %Z.impl_witness.loc69.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.cf0(%T.loc54.2) [symbolic = %Z.impl_witness.loc69.2 (constants.%Z.impl_witness.9b0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.864)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc58 as %.loc64 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.cf0 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc69.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.cf0(constants.%T.67d) [symbolic = %Z.impl_witness.loc69.2 (constants.%Z.impl_witness.9b0)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%T.67d [symbolic = %T.loc54.2 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Z.impl_witness.loc69.1
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
// CHECK:STDOUT: generic fn @F(%T.loc74.2: %Z.type.be4) {
// CHECK:STDOUT:   %T.loc74.1: %Z.type.be4 = symbolic_binding T, 0 [symbolic = %T.loc74.1 (constants.%T.fe5)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc74.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc82: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc82 (constants.%pattern_type.bb9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc82: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc82 (constants.%require_complete.f3b)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc74.1, @Z, @Z(constants.%C) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.1cb)]
// CHECK:STDOUT:   %impl.elem0.loc91.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc91.2 (constants.%impl.elem0.a5e)]
// CHECK:STDOUT:   %require_complete.loc91: <witness> = require_complete_type %impl.elem0.loc91.2 [symbolic = %require_complete.loc91 (constants.%require_complete.7cf)]
// CHECK:STDOUT:   %pattern_type.loc88: type = pattern_type %impl.elem0.loc91.2 [symbolic = %pattern_type.loc88 (constants.%pattern_type.7ae)]
// CHECK:STDOUT:   %ImplicitAs.type.loc94.2: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.loc91.2)> [symbolic = %ImplicitAs.type.loc94.2 (constants.%ImplicitAs.type.7e4)]
// CHECK:STDOUT:   %require_complete.loc94: <witness> = require_complete_type %ImplicitAs.type.loc94.2 [symbolic = %require_complete.loc94 (constants.%require_complete.591)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.loc91.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ac6)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ac6) = assoc_entity element0, imports.%Core.import_ref.201 [symbolic = %assoc0 (constants.%assoc0.ae8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type.loc88 (%pattern_type.7ae) = value_binding_pattern a [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc91.1: type = splice_block %impl.elem0.loc91.1 [symbolic = %impl.elem0.loc91.2 (constants.%impl.elem0.a5e)] {
// CHECK:STDOUT:       %T.ref.loc90: %Z.type.be4 = name_ref T, %T.loc74.2 [symbolic = %T.loc74.1 (constants.%T.fe5)]
// CHECK:STDOUT:       %T.as_type.loc91: type = facet_access_type %T.ref.loc90 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc91.2: type = converted %T.ref.loc90, %T.as_type.loc91 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc91.3: %Z.assoc_type.553 = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%T.fe5) [concrete = constants.%assoc0.1e0]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.553 = name_ref X, %.loc91.3 [concrete = constants.%assoc0.1e0]
// CHECK:STDOUT:       %impl.elem0.loc91.1: type = impl_witness_access constants.%Z.lookup_impl_witness.1cb, element0 [symbolic = %impl.elem0.loc91.2 (constants.%impl.elem0.a5e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ImplicitAs.type.loc94.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%impl.elem0.a5e)> [symbolic = %ImplicitAs.type.loc94.2 (constants.%ImplicitAs.type.7e4)]
// CHECK:STDOUT:     %.loc94.1: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ac6) = specific_constant imports.%Core.import_ref.178, @ImplicitAs.WithSelf(constants.%impl.elem0.a5e, constants.%Self.738) [symbolic = %assoc0 (constants.%assoc0.ae8)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ac6) = name_ref Convert, %.loc94.1 [symbolic = %assoc0 (constants.%assoc0.ae8)]
// CHECK:STDOUT:     %.loc94.2: @F.%impl.elem0.loc91.2 (%impl.elem0.a5e) = converted %t.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %a: @F.%impl.elem0.loc91.2 (%impl.elem0.a5e) = value_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T.67d, constants.%Self.822) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.162) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.4d0
// CHECK:STDOUT:   %Self.loc11.2 => constants.%Self.427
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Self.822) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%.Self.90f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.34a(constants.%T.67d, constants.%S) {
// CHECK:STDOUT:   %T.loc29.2 => constants.%T.67d
// CHECK:STDOUT:   %S.loc33.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc42.2 => constants.%Z.type.4d0
// CHECK:STDOUT:   %.Self.4 => constants.%.Self.90f
// CHECK:STDOUT:   %require_complete.loc44 => constants.%require_complete.72d
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type.097
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8bc
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.7d6
// CHECK:STDOUT:   %impl.elem0.loc44.2 => constants.%impl.elem0.e5e
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.a76
// CHECK:STDOUT:   %Z.impl_witness.loc49.2 => constants.%Z.impl_witness.cfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Z.facet.100) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.5b2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.be4
// CHECK:STDOUT:   %Self.loc11.2 => constants.%Self.af9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Self.822) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.553
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%.Self.bf2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.553
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.cf0(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc54.2 => constants.%T.67d
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.c3e
// CHECK:STDOUT:   %Z.impl_witness.loc69.2 => constants.%Z.impl_witness.9b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Z.facet.738) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.553
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.fe5) {
// CHECK:STDOUT:   %T.loc74.1 => constants.%T.fe5
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc82 => constants.%pattern_type.bb9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%T.fe5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.553
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.0e0: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.6bc: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.47a: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.47a [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.47a, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ac: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.3963c5.1: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.117547.1: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.73ad1d.1: <witness> = require_complete_type %Ptr_where.type.3963c5.1 [symbolic]
// CHECK:STDOUT:   %Ptr.facet.5445c3.1: %Ptr.type = facet_value %U.67d, (%Ptr.impl_witness.117547.1) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.3963c5.2: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.e8f8f9.2> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.117547.2: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.73ad1d.2: <witness> = require_complete_type %Ptr_where.type.3963c5.2 [symbolic]
// CHECK:STDOUT:   %.656: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Ptr.facet.5445c3.2: %Ptr.type = facet_value %T.67d, (%Ptr.impl_witness.117547.2) [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %.2f2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f8f9.2, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f8f9.2, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d82: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.299: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d82, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1c7: %.299 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.366: <specific function> = specific_impl_function %impl.elem0.1c7, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc21: type = name_ref U, %U.loc17.1 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc25: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.6bc]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.4ac]
// CHECK:STDOUT:     %U.ref.loc28: type = name_ref U, %U.loc17.1 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc29.1: type = ptr_type %U.ref.loc28 [symbolic = %ptr.loc29.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc24: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.3963c5.1)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc29.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc19.1: type = splice_block %.loc19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc17.1: type = symbolic_binding U, 0 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc41 (%pattern_type.51d) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc41 (%pattern_type.51d) = var_param_pattern %t.patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc47 (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc47 (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %impl.elem0.loc47 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc46: type = name_ref T, %T.loc35.2 [symbolic = %T.loc35.1 (constants.%T.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.6bc]
// CHECK:STDOUT:     %Ptr.facet.loc47.2: %Ptr.type = facet_value %T.ref.loc46, (constants.%Ptr.impl_witness.117547.2) [symbolic = %Ptr.facet.loc47.1 (constants.%Ptr.facet.5445c3.2)]
// CHECK:STDOUT:     %.loc47.3: %Ptr.type = converted %T.ref.loc46, %Ptr.facet.loc47.2 [symbolic = %Ptr.facet.loc47.1 (constants.%Ptr.facet.5445c3.2)]
// CHECK:STDOUT:     %impl.elem0.loc47: type = impl_witness_access constants.%Ptr.impl_witness.117547.2, element0 [symbolic = %ptr (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:     %.loc47.4: Core.Form = init_form %impl.elem0.loc47 [symbolic = %.loc47.2 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc37.1: type = splice_block %.loc37.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc37.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc35.2: type = symbolic_binding T, 0 [symbolic = %T.loc35.1 (constants.%T.67d)]
// CHECK:STDOUT:     %t.param: ref @F.%T.loc35.1 (%T.67d) = ref_param call_param0
// CHECK:STDOUT:     %T.ref.loc43: type = name_ref T, %T.loc35.2 [symbolic = %T.loc35.1 (constants.%T.67d)]
// CHECK:STDOUT:     %t: ref @F.%T.loc35.1 (%T.67d) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.e8f8f9.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.e8f8f9.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.0e0]
// CHECK:STDOUT:   %Ptr.WithSelf.decl = interface_with_self_decl @Ptr [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete = constants.%assoc0.6bc]
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
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc17.1: type) {
// CHECK:STDOUT:   %U.loc17.2: type = symbolic_binding U, 0 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc29.2: type = ptr_type %U.loc17.2 [symbolic = %ptr.loc29.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.4ac = %ptr.loc29.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.3963c5.1)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc30.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc17.2) [symbolic = %Ptr.impl_witness.loc30.2 (constants.%Ptr.impl_witness.117547.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.73ad1d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc21 as %.loc24 {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc30.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc30.2 (constants.%Ptr.impl_witness.117547.1)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc29.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc35.2: type) {
// CHECK:STDOUT:   %T.loc35.1: type = symbolic_binding T, 0 [symbolic = %T.loc35.1 (constants.%T.67d)]
// CHECK:STDOUT:   %pattern_type.loc41: type = pattern_type %T.loc35.1 [symbolic = %pattern_type.loc41 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %.loc47.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.loc35.1) [symbolic = %.loc47.1 (constants.%.656)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.loc35.1) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.117547.2)]
// CHECK:STDOUT:   %Ptr.facet.loc47.1: %Ptr.type = facet_value %T.loc35.1, (%Ptr.impl_witness) [symbolic = %Ptr.facet.loc47.1 (constants.%Ptr.facet.5445c3.2)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc35.1 [symbolic = %ptr (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %.loc47.2: Core.Form = init_form %ptr [symbolic = %.loc47.2 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type.loc47: type = pattern_type %ptr [symbolic = %pattern_type.loc47 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40: <witness> = require_complete_type %T.loc35.1 [symbolic = %require_complete.loc40 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc47: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc47 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %.loc55.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc35.1) [symbolic = %.loc55.1 (constants.%.2f2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d82)]
// CHECK:STDOUT:   %.loc55.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc55.2 (constants.%.299)]
// CHECK:STDOUT:   %impl.elem0.loc55.2: @F.%.loc55.2 (%.299) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc55.2 (constants.%impl.elem0.1c7)]
// CHECK:STDOUT:   %specific_impl_fn.loc55.2: <specific function> = specific_impl_function %impl.elem0.loc55.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc55.2 (constants.%specific_impl_fn.366)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.loc35.1 (%T.67d)) -> out %return.param: @F.%ptr (%ptr.e8f8f9.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc35.1 (%T.67d) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.e8f8f9.2) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc55.1: @F.%.loc55.2 (%.299) = impl_witness_access constants.%Copy.lookup_impl_witness.2e6, element0 [symbolic = %impl.elem0.loc55.2 (constants.%impl.elem0.1c7)]
// CHECK:STDOUT:     %bound_method.loc55.1: <bound method> = bound_method %addr, %impl.elem0.loc55.1
// CHECK:STDOUT:     %specific_impl_fn.loc55.1: <specific function> = specific_impl_function %impl.elem0.loc55.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc55.2 (constants.%specific_impl_fn.366)]
// CHECK:STDOUT:     %bound_method.loc55.2: <bound method> = bound_method %addr, %specific_impl_fn.loc55.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr (%ptr.e8f8f9.2) = call %bound_method.loc55.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Self.0e0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%.Self.47a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc17.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc29.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.3963c5.1
// CHECK:STDOUT:   %Ptr.impl_witness.loc30.2 => constants.%Ptr.impl_witness.117547.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet.5445c3.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.67d) {
// CHECK:STDOUT:   %U.loc17.2 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc29.2 => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.3963c5.2
// CHECK:STDOUT:   %Ptr.impl_witness.loc30.2 => constants.%Ptr.impl_witness.117547.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.73ad1d.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet.5445c3.2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc35.1 => constants.%T.67d
// CHECK:STDOUT:   %pattern_type.loc41 => constants.%pattern_type.51d
// CHECK:STDOUT:   %.loc47.1 => constants.%.656
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.117547.2
// CHECK:STDOUT:   %Ptr.facet.loc47.1 => constants.%Ptr.facet.5445c3.2
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %.loc47.2 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type.loc47 => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.0e0: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.6bc: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.47a: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.47a [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.47a, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ac: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.396: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.117: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.73a: <witness> = require_complete_type %Ptr_where.type.396 [symbolic]
// CHECK:STDOUT:   %Ptr.facet: %Ptr.type = facet_value %U.67d, (%Ptr.impl_witness.117) [symbolic]
// CHECK:STDOUT:   %pattern_type.6d2: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.ffa: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.3dc: type = symbolic_binding_type T, 0, %T.ffa [symbolic]
// CHECK:STDOUT:   %pattern_type.82a: type = pattern_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %ptr.a86: type = ptr_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.540: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.a86> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.373: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %require_complete.2f9: <witness> = require_complete_type %Ptr_where.type.540 [symbolic]
// CHECK:STDOUT:   %.0a6: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %.321: Core.Form = init_form %ptr.a86 [symbolic]
// CHECK:STDOUT:   %pattern_type.804: type = pattern_type %ptr.a86 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.552: <witness> = require_complete_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %require_complete.c1c: <witness> = require_complete_type %ptr.a86 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %.54b: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3c7: <witness> = lookup_impl_witness %ptr.a86, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.a86, (%Copy.lookup_impl_witness.3c7) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.207: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.382: type = fn_type_with_self_type %Copy.WithSelf.Op.type.207, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.c19: %.382 = impl_witness_access %Copy.lookup_impl_witness.3c7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.25f: <specific function> = specific_impl_function %impl.elem0.c19, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc21: type = name_ref U, %U.loc17.1 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc25: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.6bc]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.4ac]
// CHECK:STDOUT:     %U.ref.loc28: type = name_ref U, %U.loc17.1 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc29.1: type = ptr_type %U.ref.loc28 [symbolic = %ptr.loc29.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc24: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.396)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc29.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc19.1: type = splice_block %.loc19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc17.1: type = symbolic_binding U, 0 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.6d2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc41 (%pattern_type.82a) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc41 (%pattern_type.82a) = var_param_pattern %t.patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc47 (%pattern_type.804) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc47 (%pattern_type.804) = return_slot_pattern %return.param_patt, %impl.elem0.loc47 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc46: %Ptr.type = name_ref T, %T.loc35.2 [symbolic = %T.loc35.1 (constants.%T.ffa)]
// CHECK:STDOUT:     %T.as_type.loc47: type = facet_access_type %T.ref.loc46 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:     %.loc47.3: type = converted %T.ref.loc46, %T.as_type.loc47 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.6bc]
// CHECK:STDOUT:     %impl.elem0.loc47: type = impl_witness_access constants.%Ptr.impl_witness.373, element0 [symbolic = %ptr (constants.%ptr.a86)]
// CHECK:STDOUT:     %.loc47.4: Core.Form = init_form %impl.elem0.loc47 [symbolic = %.loc47.2 (constants.%.321)]
// CHECK:STDOUT:     %.loc37: type = splice_block %Ptr.ref [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc35.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc35.1 (constants.%T.ffa)]
// CHECK:STDOUT:     %t.param: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = ref_param call_param0
// CHECK:STDOUT:     %.loc43.1: type = splice_block %.loc43.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)] {
// CHECK:STDOUT:       %T.ref.loc43: %Ptr.type = name_ref T, %T.loc35.2 [symbolic = %T.loc35.1 (constants.%T.ffa)]
// CHECK:STDOUT:       %T.as_type.loc43: type = facet_access_type %T.ref.loc43 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:       %.loc43.2: type = converted %T.ref.loc43, %T.as_type.loc43 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.a86) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.a86) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.0e0]
// CHECK:STDOUT:   %Ptr.WithSelf.decl = interface_with_self_decl @Ptr [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete = constants.%assoc0.6bc]
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
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc17.1: type) {
// CHECK:STDOUT:   %U.loc17.2: type = symbolic_binding U, 0 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc29.2: type = ptr_type %U.loc17.2 [symbolic = %ptr.loc29.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.4ac = %ptr.loc29.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.396)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc30.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc17.2) [symbolic = %Ptr.impl_witness.loc30.2 (constants.%Ptr.impl_witness.117)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.73a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc21 as %.loc24 {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc30.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc30.2 (constants.%Ptr.impl_witness.117)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc29.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc35.2: %Ptr.type) {
// CHECK:STDOUT:   %T.loc35.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc35.1 (constants.%T.ffa)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc35.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:   %pattern_type.loc41: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc41 (constants.%pattern_type.82a)]
// CHECK:STDOUT:   %.loc47.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %.loc47.1 (constants.%.0a6)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.373)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.binding.as_type [symbolic = %ptr (constants.%ptr.a86)]
// CHECK:STDOUT:   %.loc47.2: Core.Form = init_form %ptr [symbolic = %.loc47.2 (constants.%.321)]
// CHECK:STDOUT:   %pattern_type.loc47: type = pattern_type %ptr [symbolic = %pattern_type.loc47 (constants.%pattern_type.804)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc40 (constants.%require_complete.552)]
// CHECK:STDOUT:   %require_complete.loc47: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc47 (constants.%require_complete.c1c)]
// CHECK:STDOUT:   %.loc51.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type) [symbolic = %.loc51.1 (constants.%.54b)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3c7)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.207)]
// CHECK:STDOUT:   %.loc51.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc51.2 (constants.%.382)]
// CHECK:STDOUT:   %impl.elem0.loc51.2: @F.%.loc51.2 (%.382) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc51.2 (constants.%impl.elem0.c19)]
// CHECK:STDOUT:   %specific_impl_fn.loc51.2: <specific function> = specific_impl_function %impl.elem0.loc51.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc51.2 (constants.%specific_impl_fn.25f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.binding.as_type (%T.binding.as_type.3dc)) -> out %return.param: @F.%ptr (%ptr.a86) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.a86) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc51.1: @F.%.loc51.2 (%.382) = impl_witness_access constants.%Copy.lookup_impl_witness.3c7, element0 [symbolic = %impl.elem0.loc51.2 (constants.%impl.elem0.c19)]
// CHECK:STDOUT:     %bound_method.loc51.1: <bound method> = bound_method %addr, %impl.elem0.loc51.1
// CHECK:STDOUT:     %specific_impl_fn.loc51.1: <specific function> = specific_impl_function %impl.elem0.loc51.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc51.2 (constants.%specific_impl_fn.25f)]
// CHECK:STDOUT:     %bound_method.loc51.2: <bound method> = bound_method %addr, %specific_impl_fn.loc51.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr (%ptr.a86) = call %bound_method.loc51.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Self.0e0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%.Self.47a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc17.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc29.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.396
// CHECK:STDOUT:   %Ptr.impl_witness.loc30.2 => constants.%Ptr.impl_witness.117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%T.ffa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.binding.as_type.3dc) {
// CHECK:STDOUT:   %U.loc17.2 => constants.%T.binding.as_type.3dc
// CHECK:STDOUT:   %ptr.loc29.2 => constants.%ptr.a86
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.540
// CHECK:STDOUT:   %Ptr.impl_witness.loc30.2 => constants.%Ptr.impl_witness.373
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.ffa) {
// CHECK:STDOUT:   %T.loc35.1 => constants.%T.ffa
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.3dc
// CHECK:STDOUT:   %pattern_type.loc41 => constants.%pattern_type.82a
// CHECK:STDOUT:   %.loc47.1 => constants.%.0a6
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.373
// CHECK:STDOUT:   %ptr => constants.%ptr.a86
// CHECK:STDOUT:   %.loc47.2 => constants.%.321
// CHECK:STDOUT:   %pattern_type.loc47 => constants.%pattern_type.804
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.0e0: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.6bc: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.47a: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.47a [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.47a, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ac: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.396: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.117: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.73a: <witness> = require_complete_type %Ptr_where.type.396 [symbolic]
// CHECK:STDOUT:   %Ptr.facet: %Ptr.type = facet_value %U.67d, (%Ptr.impl_witness.117) [symbolic]
// CHECK:STDOUT:   %pattern_type.6d2: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.ffa: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.3dc: type = symbolic_binding_type T, 0, %T.ffa [symbolic]
// CHECK:STDOUT:   %pattern_type.82a: type = pattern_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %ptr.a86: type = ptr_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.540: type = facet_type <@Ptr where %impl.elem0.4ac = %ptr.a86> [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.373: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %require_complete.2f9: <witness> = require_complete_type %Ptr_where.type.540 [symbolic]
// CHECK:STDOUT:   %.0a6: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %.321: Core.Form = init_form %ptr.a86 [symbolic]
// CHECK:STDOUT:   %pattern_type.804: type = pattern_type %ptr.a86 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.552: <witness> = require_complete_type %T.binding.as_type.3dc [symbolic]
// CHECK:STDOUT:   %require_complete.c1c: <witness> = require_complete_type %ptr.a86 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %.54b: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type.3dc) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3c7: <witness> = lookup_impl_witness %ptr.a86, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.a86, (%Copy.lookup_impl_witness.3c7) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.207: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.382: type = fn_type_with_self_type %Copy.WithSelf.Op.type.207, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.c19: %.382 = impl_witness_access %Copy.lookup_impl_witness.3c7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.25f: <specific function> = specific_impl_function %impl.elem0.c19, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.ref.loc21: type = name_ref U, %U.loc17.1 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.1: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.47a]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc25: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.6bc]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0.4ac]
// CHECK:STDOUT:     %U.ref.loc28: type = name_ref U, %U.loc17.1 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc29.1: type = ptr_type %U.ref.loc28 [symbolic = %ptr.loc29.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc24: type = where_expr %.Self.1 [symbolic = %Ptr_where.type (constants.%Ptr_where.type.396)] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%Ptr.type
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc29.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc19.1: type = splice_block %.loc19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.loc19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc17.1: type = symbolic_binding U, 0 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.6d2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc41 (%pattern_type.82a) = ref_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc41 (%pattern_type.82a) = var_param_pattern %t.patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc47 (%pattern_type.804) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc47 (%pattern_type.804) = return_slot_pattern %return.param_patt, %impl.elem0.loc47 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc46: %Ptr.type = name_ref T, %T.loc35.2 [symbolic = %T.loc35.1 (constants.%T.ffa)]
// CHECK:STDOUT:     %Ptr.ref.loc49: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.6bc]
// CHECK:STDOUT:     %impl.elem0.loc47: type = impl_witness_access constants.%Ptr.impl_witness.373, element0 [symbolic = %ptr (constants.%ptr.a86)]
// CHECK:STDOUT:     %.loc47.3: Core.Form = init_form %impl.elem0.loc47 [symbolic = %.loc47.2 (constants.%.321)]
// CHECK:STDOUT:     %.loc37: type = splice_block %Ptr.ref.loc37 [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Ptr.ref.loc37: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc35.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc35.1 (constants.%T.ffa)]
// CHECK:STDOUT:     %t.param: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = ref_param call_param0
// CHECK:STDOUT:     %.loc43.1: type = splice_block %.loc43.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)] {
// CHECK:STDOUT:       %T.ref.loc43: %Ptr.type = name_ref T, %T.loc35.2 [symbolic = %T.loc35.1 (constants.%T.ffa)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref.loc43 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:       %.loc43.2: type = converted %T.ref.loc43, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = ref_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.a86) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.a86) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.0e0]
// CHECK:STDOUT:   %Ptr.WithSelf.decl = interface_with_self_decl @Ptr [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete = constants.%assoc0.6bc]
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
// CHECK:STDOUT: generic impl @U.as.Ptr.impl(%U.loc17.1: type) {
// CHECK:STDOUT:   %U.loc17.2: type = symbolic_binding U, 0 [symbolic = %U.loc17.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc29.2: type = ptr_type %U.loc17.2 [symbolic = %ptr.loc29.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.4ac = %ptr.loc29.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.396)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc30.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc17.2) [symbolic = %Ptr.impl_witness.loc30.2 (constants.%Ptr.impl_witness.117)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.73a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc21 as %.loc24 {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc30.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc30.2 (constants.%Ptr.impl_witness.117)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc29.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc35.2: %Ptr.type) {
// CHECK:STDOUT:   %T.loc35.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc35.1 (constants.%T.ffa)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc35.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.3dc)]
// CHECK:STDOUT:   %pattern_type.loc41: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc41 (constants.%pattern_type.82a)]
// CHECK:STDOUT:   %.loc47.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %.loc47.1 (constants.%.0a6)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.binding.as_type) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.373)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.binding.as_type [symbolic = %ptr (constants.%ptr.a86)]
// CHECK:STDOUT:   %.loc47.2: Core.Form = init_form %ptr [symbolic = %.loc47.2 (constants.%.321)]
// CHECK:STDOUT:   %pattern_type.loc47: type = pattern_type %ptr [symbolic = %pattern_type.loc47 (constants.%pattern_type.804)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc40 (constants.%require_complete.552)]
// CHECK:STDOUT:   %require_complete.loc47: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc47 (constants.%require_complete.c1c)]
// CHECK:STDOUT:   %.loc55.1: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.binding.as_type) [symbolic = %.loc55.1 (constants.%.54b)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3c7)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.207)]
// CHECK:STDOUT:   %.loc55.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc55.2 (constants.%.382)]
// CHECK:STDOUT:   %impl.elem0.loc55.2: @F.%.loc55.2 (%.382) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc55.2 (constants.%impl.elem0.c19)]
// CHECK:STDOUT:   %specific_impl_fn.loc55.2: <specific function> = specific_impl_function %impl.elem0.loc55.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc55.2 (constants.%specific_impl_fn.25f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.binding.as_type (%T.binding.as_type.3dc)) -> out %return.param: @F.%ptr (%ptr.a86) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.binding.as_type (%T.binding.as_type.3dc) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.a86) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc55.1: @F.%.loc55.2 (%.382) = impl_witness_access constants.%Copy.lookup_impl_witness.3c7, element0 [symbolic = %impl.elem0.loc55.2 (constants.%impl.elem0.c19)]
// CHECK:STDOUT:     %bound_method.loc55.1: <bound method> = bound_method %addr, %impl.elem0.loc55.1
// CHECK:STDOUT:     %specific_impl_fn.loc55.1: <specific function> = specific_impl_function %impl.elem0.loc55.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc55.2 (constants.%specific_impl_fn.25f)]
// CHECK:STDOUT:     %bound_method.loc55.2: <bound method> = bound_method %addr, %specific_impl_fn.loc55.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr (%ptr.a86) = call %bound_method.loc55.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Self.0e0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%.Self.47a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.loc17.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc29.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.396
// CHECK:STDOUT:   %Ptr.impl_witness.loc30.2 => constants.%Ptr.impl_witness.117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.binding.as_type.3dc) {
// CHECK:STDOUT:   %U.loc17.2 => constants.%T.binding.as_type.3dc
// CHECK:STDOUT:   %ptr.loc29.2 => constants.%ptr.a86
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.540
// CHECK:STDOUT:   %Ptr.impl_witness.loc30.2 => constants.%Ptr.impl_witness.373
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%T.ffa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.ffa) {
// CHECK:STDOUT:   %T.loc35.1 => constants.%T.ffa
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.3dc
// CHECK:STDOUT:   %pattern_type.loc41 => constants.%pattern_type.82a
// CHECK:STDOUT:   %.loc47.1 => constants.%.0a6
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.373
// CHECK:STDOUT:   %ptr => constants.%ptr.a86
// CHECK:STDOUT:   %.loc47.2 => constants.%.321
// CHECK:STDOUT:   %pattern_type.loc47 => constants.%pattern_type.804
// CHECK:STDOUT: }
// CHECK:STDOUT:

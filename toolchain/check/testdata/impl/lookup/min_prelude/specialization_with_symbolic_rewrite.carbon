// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/specialization_with_symbolic_rewrite.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/specialization_with_symbolic_rewrite.carbon

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
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.a61: type = facet_type <@Z, @Z(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.2bc: %Z.type.a61 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.26e: type = assoc_entity_type @Z, @Z(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.1cd: %Z.assoc_type.26e = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = bind_symbolic_name S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.8e9: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.eb1: %Z.type.8e9 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %Self.831: %Z.type.8e9 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0e7: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.abb: %Z.assoc_type.0e7 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.b87: <witness> = require_complete_type %Z.type.8e9 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.ffd: type = facet_access_type %.Self.eb1 [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.597: <witness> = lookup_impl_witness %.Self.eb1, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %Z.facet.4f7: %Z.type.8e9 = facet_value %.Self.as_type.ffd, (%Z.lookup_impl_witness.597) [symbolic]
// CHECK:STDOUT:   %impl.elem0.240: type = impl_witness_access %Z.lookup_impl_witness.597, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.80d: type = facet_type <@Z, @Z(%S) where %impl.elem0.240 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.9ea: <witness> = require_complete_type %Z_where.type.80d [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.d59: <witness> = impl_witness file.%Z.impl_witness_table.loc9, @impl.119(%T.8b3, %S) [symbolic]
// CHECK:STDOUT:   %Z.type.049: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.3c5: %Z.type.049 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.a38: %Z.type.049 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.as_type.c3c: type = facet_access_type %.Self.3c5 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.9cc: <witness> = lookup_impl_witness %.Self.3c5, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %Z.facet.eb4: %Z.type.049 = facet_value %.Self.as_type.c3c, (%Z.lookup_impl_witness.9cc) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e29: type = impl_witness_access %Z.lookup_impl_witness.9cc, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.692: type = facet_type <@Z, @Z(%C) where %impl.elem0.e29 = %T.8b3> [symbolic]
// CHECK:STDOUT:   %require_complete.202: <witness> = require_complete_type %Z_where.type.692 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.093: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @impl.62c(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.62d: %Z.type.049 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c1f: type = pattern_type %Z.type.049 [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.62d [symbolic]
// CHECK:STDOUT:   %pattern_type.d8b: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.2b1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Z_where.type.272: type = facet_type <@Z, @Z(%C) where %impl.elem0.e29 = %T.as_type> [symbolic]
// CHECK:STDOUT:   %require_complete.915: <witness> = require_complete_type %Z_where.type.272 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.7c2: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @impl.62c(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Z.facet.2aa: %Z.type.049 = facet_value %T.as_type, (%Z.impl_witness.7c2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.119 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc9_24.1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc9_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.8e9)]
// CHECK:STDOUT:     %.Self.1: @impl.119.%Z.type.loc9_42.2 (%Z.type.8e9) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:     %.Self.ref: @impl.119.%Z.type.loc9_42.2 (%Z.type.8e9) = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:     %.loc9_50.1: @impl.119.%Z.assoc_type (%Z.assoc_type.0e7) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %X.ref: @impl.119.%Z.assoc_type (%Z.assoc_type.0e7) = name_ref X, %.loc9_50.1 [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %.Self.as_type.loc9_50.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:     %.loc9_50.2: type = converted %.Self.ref, %.Self.as_type.loc9_50.1 [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:     %impl.elem0.loc9_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.597, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.240)]
// CHECK:STDOUT:     %.loc9_56.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_56.2: type = converted %.loc9_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.80d)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_50.1, %.loc9_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %S.loc9_24.1: type = bind_symbolic_name S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc9 = impl_witness_table (%impl_witness_assoc_constant.loc9), @impl.119 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc9: <witness> = impl_witness %Z.impl_witness_table.loc9, @impl.119(constants.%T.8b3, constants.%S) [symbolic = @impl.119.%Z.impl_witness (constants.%Z.impl_witness.d59)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc9: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   impl_decl @impl.62c [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc11_30: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.049]
// CHECK:STDOUT:     %.Self: %Z.type.049 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.3c5]
// CHECK:STDOUT:     %.Self.ref: %Z.type.049 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.3c5]
// CHECK:STDOUT:     %.loc11_46.1: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.252 = name_ref X, %.loc11_46.1 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.c3c]
// CHECK:STDOUT:     %.loc11_46.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.c3c]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.9cc, element0 [symbolic_self = constants.%impl.elem0.e29]
// CHECK:STDOUT:     %T.ref.loc11_51: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc11_40: type = where_expr %.Self [symbolic = %Z_where.type (constants.%Z_where.type.692)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc11_51
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_20.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc11 = impl_witness_table (%impl_witness_assoc_constant.loc11), @impl.62c [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc11: <witness> = impl_witness %Z.impl_witness_table.loc11, @impl.62c(constants.%T.8b3) [symbolic = @impl.62c.%Z.impl_witness (constants.%Z.impl_witness.093)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc11: type = impl_witness_assoc_constant constants.%T.8b3 [symbolic = @impl.62c.%T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.c1f = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %t.patt: @F.%pattern_type (%pattern_type.d8b) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type (%pattern_type.d8b) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_13: type = splice_block %Z.type [concrete = constants.%Z.type.049] {
// CHECK:STDOUT:       %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.049]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_6.1: %Z.type.049 = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T.62d)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc13_19.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_19.1: type = splice_block %.loc13_19.2 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc13: %Z.type.049 = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T.62d)]
// CHECK:STDOUT:       %T.as_type.loc13_19.1: type = facet_access_type %T.ref.loc13 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc13_19.2: type = converted %T.ref.loc13, %T.as_type.loc13_19.1 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc13_19.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_13.1: type) {
// CHECK:STDOUT:   %T.loc3_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_13.2)> [symbolic = %Z.type (constants.%Z.type.a61)]
// CHECK:STDOUT:   %Self.2: @Z.%Z.type (%Z.type.a61) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2bc)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc3_13.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.26e)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.26e) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.1cd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Z.%Z.type (%Z.type.a61) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2bc)]
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
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc3_13.1: type, @Z.%Self.1: @Z.%Z.type (%Z.type.a61)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.119(%T.loc9_14.1: type, %S.loc9_24.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %S.loc9_24.2: type = bind_symbolic_name S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc9_42.2: type = facet_type <@Z, @Z(%S.loc9_24.2)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.8e9)]
// CHECK:STDOUT:   %.Self.2: @impl.119.%Z.type.loc9_42.2 (%Z.type.8e9) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:   %require_complete.loc9_50: <witness> = require_complete_type %Z.type.loc9_42.2 [symbolic = %require_complete.loc9_50 (constants.%require_complete.b87)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc9_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.0e7)]
// CHECK:STDOUT:   %assoc0: @impl.119.%Z.assoc_type (%Z.assoc_type.0e7) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:   %.Self.as_type.loc9_50.2: type = facet_access_type %.Self.2 [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.2, @Z, @Z(%S.loc9_24.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.597)]
// CHECK:STDOUT:   %impl.elem0.loc9_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.240)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc9_24.2) where %impl.elem0.loc9_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.80d)]
// CHECK:STDOUT:   %require_complete.loc9_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc9_44 (constants.%require_complete.9ea)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc9, @impl.119(%T.loc9_14.2, %S.loc9_24.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.d59)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc9_44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.62c(%T.loc11_20.1: type) {
// CHECK:STDOUT:   %T.loc11_20.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.e29 = %T.loc11_20.2> [symbolic = %Z_where.type (constants.%Z_where.type.692)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.202)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @impl.62c(%T.loc11_20.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.093)]
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.1: %Z.type.049) {
// CHECK:STDOUT:   %T.loc13_6.2: %Z.type.049 = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T.62d)]
// CHECK:STDOUT:   %T.as_type.loc13_19.2: type = facet_access_type %T.loc13_6.2 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc13_19.2 [symbolic = %pattern_type (constants.%pattern_type.d8b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc13_19.2 [symbolic = %require_complete (constants.%require_complete.2b1)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc11, @impl.62c(%T.as_type.loc13_19.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.7c2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc13_19.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type (%pattern_type.d8b) = binding_pattern a
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc13_19.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc15_11.1: type = splice_block %impl.elem0 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc15: %Z.type.049 = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T.62d)]
// CHECK:STDOUT:       %.loc15_11.2: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.252 = name_ref X, %.loc15_11.2 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %T.as_type.loc15_11.1: type = facet_access_type %T.ref.loc15 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_11.3: type = converted %T.ref.loc15, %T.as_type.loc15_11.1 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %T.as_type.loc15_11.2: type = facet_access_type constants.%T.62d [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_11.4: type = converted constants.%T.62d, %T.as_type.loc15_11.2 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.7c2, element0 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%T.as_type.loc13_19.2 (%T.as_type) = bind_name a, %t.ref
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.8b3, constants.%Self.2bc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(%T.loc3_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.8e9
// CHECK:STDOUT:   %Self.2 => constants.%Self.831
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%Z.facet.4f7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.119(constants.%T.8b3, constants.%S) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %S.loc9_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc9_42.2 => constants.%Z.type.8e9
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.eb1
// CHECK:STDOUT:   %require_complete.loc9_50 => constants.%require_complete.b87
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT:   %.Self.as_type.loc9_50.2 => constants.%.Self.as_type.ffd
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.597
// CHECK:STDOUT:   %impl.elem0.loc9_50.2 => constants.%impl.elem0.240
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.80d
// CHECK:STDOUT:   %require_complete.loc9_44 => constants.%require_complete.9ea
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.d59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(@impl.119.%S.loc9_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.119(%T.loc9_14.2, %S.loc9_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.049
// CHECK:STDOUT:   %Self.2 => constants.%Self.a38
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%Z.facet.eb4) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.62c(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.8b3
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.692
// CHECK:STDOUT:   %require_complete => constants.%require_complete.202
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.093
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.62c(%T.loc11_20.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.62d) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T.62d
// CHECK:STDOUT:   %T.as_type.loc13_19.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d8b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.62c(constants.%T.as_type) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.as_type
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.272
// CHECK:STDOUT:   %require_complete => constants.%require_complete.915
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.7c2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%Z.facet.2aa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.62c(@F.%T.as_type.loc13_19.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Z.type.9fb: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.a61: type = facet_type <@Z, @Z(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.2bc: %Z.type.a61 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.26e: type = assoc_entity_type @Z, @Z(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.1cd: %Z.assoc_type.26e = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.b29: %Y.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S: type = bind_symbolic_name S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.8e9: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.eb1: %Z.type.8e9 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %Self.831: %Z.type.8e9 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0e7: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.abb: %Z.assoc_type.0e7 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.b87: <witness> = require_complete_type %Z.type.8e9 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.ffd: type = facet_access_type %.Self.eb1 [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.597: <witness> = lookup_impl_witness %.Self.eb1, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %Z.facet.4f7: %Z.type.8e9 = facet_value %.Self.as_type.ffd, (%Z.lookup_impl_witness.597) [symbolic]
// CHECK:STDOUT:   %impl.elem0.240: type = impl_witness_access %Z.lookup_impl_witness.597, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.80d: type = facet_type <@Z, @Z(%S) where %impl.elem0.240 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.9ea: <witness> = require_complete_type %Z_where.type.80d [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.d59: <witness> = impl_witness file.%Z.impl_witness_table.loc10, @impl.119(%T.8b3, %S) [symbolic]
// CHECK:STDOUT:   %Z.type.049: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.3c5: %Z.type.049 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.a38: %Z.type.049 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.as_type.c3c: type = facet_access_type %.Self.3c5 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.9cc: <witness> = lookup_impl_witness %.Self.3c5, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %Z.facet.eb4: %Z.type.049 = facet_value %.Self.as_type.c3c, (%Z.lookup_impl_witness.9cc) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e29: type = impl_witness_access %Z.lookup_impl_witness.9cc, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.692: type = facet_type <@Z, @Z(%C) where %impl.elem0.e29 = %T.8b3> [symbolic]
// CHECK:STDOUT:   %require_complete.202: <witness> = require_complete_type %Z_where.type.692 [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.093: <witness> = impl_witness file.%Z.impl_witness_table.loc12, @impl.62c(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.62d: %Z.type.049 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c1f: type = pattern_type %Z.type.049 [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.62d [symbolic]
// CHECK:STDOUT:   %pattern_type.d8b: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.2b1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.103: <witness> = lookup_impl_witness %T.62d, @Z, @Z(%C) [symbolic]
// CHECK:STDOUT:   %Z.facet.5a7: %Z.type.049 = facet_value %T.as_type, (%Z.lookup_impl_witness.103) [symbolic]
// CHECK:STDOUT:   %impl.elem0.284: type = impl_witness_access %Z.lookup_impl_witness.103, element0 [symbolic]
// CHECK:STDOUT:   %require_complete.3ef: <witness> = require_complete_type %impl.elem0.284 [symbolic]
// CHECK:STDOUT:   %pattern_type.722: type = pattern_type %impl.elem0.284 [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %pattern_type.f3e: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.9f5: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.199: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.284)> [symbolic]
// CHECK:STDOUT:   %Self.b97: %ImplicitAs.type.199 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.6a2: type = fn_type @Convert, @ImplicitAs(%impl.elem0.284) [symbolic]
// CHECK:STDOUT:   %Convert.946: %Convert.type.6a2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.bd9: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.284) [symbolic]
// CHECK:STDOUT:   %assoc0.428: %ImplicitAs.assoc_type.bd9 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %require_complete.3ca: <witness> = require_complete_type %ImplicitAs.type.199 [symbolic]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (imports.%Core.import_ref.bd4), @impl.f92 [concrete]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness %BitAnd.impl_witness_table, @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.2: type = pattern_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//prelude, inst69 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//prelude, inst69 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst107 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.02e = import_ref Core//prelude, loc18_41, unloaded
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.287 = import_ref Core//prelude, loc21_36, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.3: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.4: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.119 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc10_24.1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc10_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.8e9)]
// CHECK:STDOUT:     %.Self.1: @impl.119.%Z.type.loc10_42.2 (%Z.type.8e9) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:     %.Self.ref: @impl.119.%Z.type.loc10_42.2 (%Z.type.8e9) = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:     %.loc10_50.1: @impl.119.%Z.assoc_type (%Z.assoc_type.0e7) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %X.ref: @impl.119.%Z.assoc_type (%Z.assoc_type.0e7) = name_ref X, %.loc10_50.1 [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %.Self.as_type.loc10_50.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:     %.loc10_50.2: type = converted %.Self.ref, %.Self.as_type.loc10_50.1 [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:     %impl.elem0.loc10_50.1: type = impl_witness_access constants.%Z.lookup_impl_witness.597, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.240)]
// CHECK:STDOUT:     %.loc10_56.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_56.2: type = converted %.loc10_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.80d)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_50.1, %.loc10_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %S.loc10_24.1: type = bind_symbolic_name S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc10 = impl_witness_table (%impl_witness_assoc_constant.loc10), @impl.119 [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc10: <witness> = impl_witness %Z.impl_witness_table.loc10, @impl.119(constants.%T.8b3, constants.%S) [symbolic = @impl.119.%Z.impl_witness (constants.%Z.impl_witness.d59)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc10: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   impl_decl @impl.62c [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_24: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.049]
// CHECK:STDOUT:     %.Self: %Z.type.049 = bind_symbolic_name .Self [symbolic_self = constants.%.Self.3c5]
// CHECK:STDOUT:     %.Self.ref: %Z.type.049 = name_ref .Self, %.Self [symbolic_self = constants.%.Self.3c5]
// CHECK:STDOUT:     %.loc12_40.1: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.252 = name_ref X, %.loc12_40.1 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type.c3c]
// CHECK:STDOUT:     %.loc12_40.2: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type.c3c]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Z.lookup_impl_witness.9cc, element0 [symbolic_self = constants.%impl.elem0.e29]
// CHECK:STDOUT:     %T.ref.loc12_45: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc12_34: type = where_expr %.Self [symbolic = %Z_where.type (constants.%Z_where.type.692)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc12_45
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.impl_witness_table.loc12 = impl_witness_table (%impl_witness_assoc_constant.loc12), @impl.62c [concrete]
// CHECK:STDOUT:   %Z.impl_witness.loc12: <witness> = impl_witness %Z.impl_witness_table.loc12, @impl.62c(constants.%T.8b3) [symbolic = @impl.62c.%Z.impl_witness (constants.%Z.impl_witness.093)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc12: type = impl_witness_assoc_constant constants.%T.8b3 [symbolic = @impl.62c.%T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.c1f = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc14 (%pattern_type.d8b) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc14 (%pattern_type.d8b) = value_param_pattern %t.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_13: type = splice_block %Z.type [concrete = constants.%Z.type.049] {
// CHECK:STDOUT:       %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.049]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_6.1: %Z.type.049 = bind_symbolic_name T, 0 [symbolic = %T.loc14_6.2 (constants.%T.62d)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc14_19.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %T.as_type.loc14_19.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc14: %Z.type.049 = name_ref T, %T.loc14_6.1 [symbolic = %T.loc14_6.2 (constants.%T.62d)]
// CHECK:STDOUT:       %T.as_type.loc14_19.1: type = facet_access_type %T.ref.loc14 [symbolic = %T.as_type.loc14_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc14_19.2: type = converted %T.ref.loc14, %T.as_type.loc14_19.1 [symbolic = %T.as_type.loc14_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc14_19.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_13.1: type) {
// CHECK:STDOUT:   %T.loc3_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_13.2)> [symbolic = %Z.type (constants.%Z.type.a61)]
// CHECK:STDOUT:   %Self.2: @Z.%Z.type (%Z.type.a61) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2bc)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%T.loc3_13.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.26e)]
// CHECK:STDOUT:   %assoc0: @Z.%Z.assoc_type (%Z.assoc_type.26e) = assoc_entity element0, %X [symbolic = %assoc0 (constants.%assoc0.1cd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Z.%Z.type (%Z.type.a61) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2bc)]
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
// CHECK:STDOUT:   %Self: %Y.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.b29]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.275)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.275) = struct_value () [symbolic = %Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ca0)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.492
// CHECK:STDOUT:     witness = (imports.%Core.Convert)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.ad0
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.02e
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@Z.%T.loc3_13.1: type, @Z.%Self.1: @Z.%Z.type (%Z.type.a61)) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.119(%T.loc10_14.1: type, %S.loc10_24.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %S.loc10_24.2: type = bind_symbolic_name S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc10_42.2: type = facet_type <@Z, @Z(%S.loc10_24.2)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.8e9)]
// CHECK:STDOUT:   %.Self.2: @impl.119.%Z.type.loc10_42.2 (%Z.type.8e9) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:   %require_complete.loc10_50: <witness> = require_complete_type %Z.type.loc10_42.2 [symbolic = %require_complete.loc10_50 (constants.%require_complete.b87)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc10_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.0e7)]
// CHECK:STDOUT:   %assoc0: @impl.119.%Z.assoc_type (%Z.assoc_type.0e7) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:   %.Self.as_type.loc10_50.2: type = facet_access_type %.Self.2 [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.2, @Z, @Z(%S.loc10_24.2) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.597)]
// CHECK:STDOUT:   %impl.elem0.loc10_50.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.240)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc10_24.2) where %impl.elem0.loc10_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.80d)]
// CHECK:STDOUT:   %require_complete.loc10_44: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete.loc10_44 (constants.%require_complete.9ea)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc10, @impl.119(%T.loc10_14.2, %S.loc10_24.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.d59)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc10_44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.62c(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.e29 = %T.loc12_14.2> [symbolic = %Z_where.type (constants.%Z_where.type.692)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z_where.type [symbolic = %require_complete (constants.%require_complete.202)]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness file.%Z.impl_witness_table.loc12, @impl.62c(%T.loc12_14.2) [symbolic = %Z.impl_witness (constants.%Z.impl_witness.093)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc12_24 as %.loc12_34 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Z.impl_witness.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.3: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness constants.%BitAnd.impl_witness_table, @impl.f92(%T) [symbolic = %BitAnd.impl_witness (constants.%BitAnd.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type) = struct_value () [symbolic = %Op (constants.%Op)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.287
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc14_6.1: %Z.type.049) {
// CHECK:STDOUT:   %T.loc14_6.2: %Z.type.049 = bind_symbolic_name T, 0 [symbolic = %T.loc14_6.2 (constants.%T.62d)]
// CHECK:STDOUT:   %T.as_type.loc14_19.2: type = facet_access_type %T.loc14_6.2 [symbolic = %T.as_type.loc14_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc14: type = pattern_type %T.as_type.loc14_19.2 [symbolic = %pattern_type.loc14 (constants.%pattern_type.d8b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.as_type.loc14_19.2 [symbolic = %require_complete.loc14 (constants.%require_complete.2b1)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc14_6.2, @Z, @Z(constants.%C) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.103)]
// CHECK:STDOUT:   %impl.elem0.loc22_11.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.284)]
// CHECK:STDOUT:   %require_complete.loc22_11: <witness> = require_complete_type %impl.elem0.loc22_11.2 [symbolic = %require_complete.loc22_11 (constants.%require_complete.3ef)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %impl.elem0.loc22_11.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.722)]
// CHECK:STDOUT:   %ImplicitAs.type.loc22_16.2: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2)> [symbolic = %ImplicitAs.type.loc22_16.2 (constants.%ImplicitAs.type.199)]
// CHECK:STDOUT:   %require_complete.loc22_16: <witness> = require_complete_type %ImplicitAs.type.loc22_16.2 [symbolic = %require_complete.loc22_16 (constants.%require_complete.3ca)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.bd9)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.bd9) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.428)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc14_19.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type.loc22 (%pattern_type.722) = binding_pattern a
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc14_19.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %impl.elem0.loc22_11.1 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.284)] {
// CHECK:STDOUT:       %T.ref.loc22: %Z.type.049 = name_ref T, %T.loc14_6.1 [symbolic = %T.loc14_6.2 (constants.%T.62d)]
// CHECK:STDOUT:       %.loc22_11.2: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.252 = name_ref X, %.loc22_11.2 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.as_type.loc14_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_11.3: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.as_type.loc14_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc22_11.1: type = impl_witness_access constants.%Z.lookup_impl_witness.103, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.284)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ImplicitAs.type.loc22_16.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%impl.elem0.284)> [symbolic = %ImplicitAs.type.loc22_16.2 (constants.%ImplicitAs.type.199)]
// CHECK:STDOUT:     %.loc22_16.1: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.bd9) = specific_constant imports.%Core.import_ref.492, @ImplicitAs(constants.%impl.elem0.284) [symbolic = %assoc0 (constants.%assoc0.428)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.bd9) = name_ref Convert, %.loc22_16.1 [symbolic = %assoc0 (constants.%assoc0.428)]
// CHECK:STDOUT:     %.loc22_16.2: @F.%impl.elem0.loc22_11.2 (%impl.elem0.284) = converted %t.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %a: @F.%impl.elem0.loc22_11.2 (%impl.elem0.284) = bind_name a, <error>
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @Convert.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.f3e)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %Dest [symbolic = %pattern_type.2 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(imports.%Core.import_ref.5ab3ec.4: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.8b3, constants.%Self.2bc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(%T.loc3_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.8e9
// CHECK:STDOUT:   %Self.2 => constants.%Self.831
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%Z.facet.4f7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.119(constants.%T.8b3, constants.%S) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %S.loc10_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc10_42.2 => constants.%Z.type.8e9
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.eb1
// CHECK:STDOUT:   %require_complete.loc10_50 => constants.%require_complete.b87
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT:   %.Self.as_type.loc10_50.2 => constants.%.Self.as_type.ffd
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.597
// CHECK:STDOUT:   %impl.elem0.loc10_50.2 => constants.%impl.elem0.240
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.80d
// CHECK:STDOUT:   %require_complete.loc10_44 => constants.%require_complete.9ea
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.d59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(@impl.119.%S.loc10_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.119(%T.loc10_14.2, %S.loc10_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.049
// CHECK:STDOUT:   %Self.2 => constants.%Self.a38
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%Z.facet.eb4) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.62c(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.692
// CHECK:STDOUT:   %require_complete => constants.%require_complete.202
// CHECK:STDOUT:   %Z.impl_witness => constants.%Z.impl_witness.093
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.62c(%T.loc12_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.62d) {
// CHECK:STDOUT:   %T.loc14_6.2 => constants.%T.62d
// CHECK:STDOUT:   %T.as_type.loc14_19.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc14 => constants.%pattern_type.d8b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%Z.facet.5a7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.519) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.f3e
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%impl.elem0.284) {
// CHECK:STDOUT:   %Dest => constants.%impl.elem0.284
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.199
// CHECK:STDOUT:   %Self => constants.%Self.b97
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.6a2
// CHECK:STDOUT:   %Convert => constants.%Convert.946
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.bd9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.428
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %BitAnd.impl_witness => constants.%BitAnd.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@F.%impl.elem0.loc22_11.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @Ptr [symbolic_self]
// CHECK:STDOUT:   %Ptr.facet.b6e: %Ptr.type = facet_value %.Self.as_type, (%Ptr.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %U [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.bd48cc.1: type = facet_type <@Ptr where %impl.elem0 = %ptr.79f131.1> [symbolic]
// CHECK:STDOUT:   %require_complete.465bad.1: <witness> = require_complete_type %Ptr_where.type.bd48cc.1 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.e8ce81.1: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%U) [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %ptr.79f131.2: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.bd48cc.2: type = facet_type <@Ptr where %impl.elem0 = %ptr.79f131.2> [symbolic]
// CHECK:STDOUT:   %require_complete.465bad.2: <witness> = require_complete_type %Ptr_where.type.bd48cc.2 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.e8ce81.2: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Ptr.facet.bbf: %Ptr.type = facet_value %T, (%Ptr.impl_witness.e8ce81.2) [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f131.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f131.2 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bd48cc.1)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @impl(constants.%U) [symbolic = @impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.e8ce81.1)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.79f131.1 [symbolic = @impl.%ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_20 (%pattern_type.7dc) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_20 (%pattern_type.7dc) = ref_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %.loc9_16: @F.%pattern_type.loc9_20 (%pattern_type.7dc) = var_pattern %t.param_patt
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_26 (%pattern_type.afe) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_26 (%pattern_type.afe) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_29: type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %Ptr.facet.loc9_30.1: %Ptr.type = facet_value constants.%T, (constants.%Ptr.impl_witness.e8ce81.2) [symbolic = %Ptr.facet.loc9_30.2 (constants.%Ptr.facet.bbf)]
// CHECK:STDOUT:     %.loc9_30: %Ptr.type = converted %T.ref.loc9_29, %Ptr.facet.loc9_30.1 [symbolic = %Ptr.facet.loc9_30.2 (constants.%Ptr.facet.bbf)]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.impl_witness.e8ce81.2, element0 [symbolic = %ptr (constants.%ptr.79f131.2)]
// CHECK:STDOUT:     %T.loc9_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: ref @F.%T.loc9_6.2 (%T) = ref_param call_param0
// CHECK:STDOUT:     %T.ref.loc9_23: type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %t: ref @F.%T.loc9_6.2 (%T) = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.79f131.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.79f131.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0]
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
// CHECK:STDOUT: generic impl @impl(%U.loc7_20.1: type) {
// CHECK:STDOUT:   %U.loc7_20.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bd48cc.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.465bad.1)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.e8ce81.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Ptr.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: type) {
// CHECK:STDOUT:   %T.loc9_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:   %pattern_type.loc9_20: type = pattern_type %T.loc9_6.2 [symbolic = %pattern_type.loc9_20 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%T.loc9_6.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.e8ce81.2)]
// CHECK:STDOUT:   %Ptr.facet.loc9_30.2: %Ptr.type = facet_value %T.loc9_6.2, (%Ptr.impl_witness) [symbolic = %Ptr.facet.loc9_30.2 (constants.%Ptr.facet.bbf)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc9_6.2 [symbolic = %ptr (constants.%ptr.79f131.2)]
// CHECK:STDOUT:   %pattern_type.loc9_26: type = pattern_type %ptr [symbolic = %pattern_type.loc9_26 (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_26: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_26 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc9_16: <witness> = require_complete_type %T.loc9_6.2 [symbolic = %require_complete.loc9_16 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.loc9_6.2 (%T)) -> @F.%ptr (%ptr.79f131.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc9_6.2 (%T) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.79f131.2) = addr_of %t.ref
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.b6e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%U) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.bd48cc.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.465bad.1
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.e8ce81.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%U.loc7_20.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f131.2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.bd48cc.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.465bad.2
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.e8ce81.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.bbf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.7dc
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.e8ce81.2
// CHECK:STDOUT:   %Ptr.facet.loc9_30.2 => constants.%Ptr.facet.bbf
// CHECK:STDOUT:   %ptr => constants.%ptr.79f131.2
// CHECK:STDOUT:   %pattern_type.loc9_26 => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(@F.%T.loc9_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @Ptr [symbolic_self]
// CHECK:STDOUT:   %Ptr.facet.b6e: %Ptr.type = facet_value %.Self.as_type, (%Ptr.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %U [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.bd4: type = facet_type <@Ptr where %impl.elem0 = %ptr.79f> [symbolic]
// CHECK:STDOUT:   %require_complete.465: <witness> = require_complete_type %Ptr_where.type.bd4 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.e8c: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%U) [symbolic]
// CHECK:STDOUT:   %T: %Ptr.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9bb: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.87e: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %ptr.900: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.0d6: type = facet_type <@Ptr where %impl.elem0 = %ptr.900> [symbolic]
// CHECK:STDOUT:   %require_complete.d01: <witness> = require_complete_type %Ptr_where.type.0d6 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.5aa: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Ptr.facet.db8: %Ptr.type = facet_value %T.as_type, (%Ptr.impl_witness.5aa) [symbolic]
// CHECK:STDOUT:   %pattern_type.a95: type = pattern_type %ptr.900 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.df6: <witness> = require_complete_type %ptr.900 [symbolic]
// CHECK:STDOUT:   %require_complete.020: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bd4)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @impl(constants.%U) [symbolic = @impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.e8c)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.79f [symbolic = @impl.%ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9bb = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_19 (%pattern_type.87e) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_19 (%pattern_type.87e) = ref_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %.loc9_15: @F.%pattern_type.loc9_19 (%pattern_type.87e) = var_pattern %t.param_patt
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_25 (%pattern_type.a95) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_25 (%pattern_type.a95) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9_29.1: type = facet_access_type %T.ref.loc9_28 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_29.1: type = converted %T.ref.loc9_28, %T.as_type.loc9_29.1 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_type.loc9_29.2: type = facet_access_type constants.%T [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_29.2: type = converted constants.%T, %T.as_type.loc9_29.2 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.impl_witness.5aa, element0 [symbolic = %ptr (constants.%ptr.900)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %T.loc9_6.1: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: ref @F.%T.as_type.loc9_22.2 (%T.as_type) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc9_22.1: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type.loc9_22.1 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.as_type.loc9_22.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.900) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.900) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0]
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
// CHECK:STDOUT: generic impl @impl(%U.loc7_20.1: type) {
// CHECK:STDOUT:   %U.loc7_20.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bd4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.465)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.e8c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Ptr.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: %Ptr.type) {
// CHECK:STDOUT:   %T.loc9_6.2: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc9_22.2: type = facet_access_type %T.loc9_6.2 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.as_type.loc9_22.2 [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.87e)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%T.as_type.loc9_22.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.5aa)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_22.2 [symbolic = %ptr (constants.%ptr.900)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.a95)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_25 (constants.%require_complete.df6)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.as_type.loc9_22.2 [symbolic = %require_complete.loc9_15 (constants.%require_complete.020)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc9_22.2 (%T.as_type)) -> @F.%ptr (%ptr.900) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc9_22.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.900) = addr_of %t.ref
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.b6e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%U) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.bd4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.465
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.e8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%U.loc7_20.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.as_type) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.900
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.0d6
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d01
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.5aa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.db8) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc9_22.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.87e
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.5aa
// CHECK:STDOUT:   %ptr => constants.%ptr.900
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.a95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(@F.%T.as_type.loc9_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %Ptr.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @Ptr [symbolic_self]
// CHECK:STDOUT:   %Ptr.facet.b6e: %Ptr.type = facet_value %.Self.as_type, (%Ptr.lookup_impl_witness) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %Ptr.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %U [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.bd4: type = facet_type <@Ptr where %impl.elem0 = %ptr.79f> [symbolic]
// CHECK:STDOUT:   %require_complete.465: <witness> = require_complete_type %Ptr_where.type.bd4 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.e8c: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%U) [symbolic]
// CHECK:STDOUT:   %T: %Ptr.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9bb: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.87e: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %ptr.900: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.0d6: type = facet_type <@Ptr where %impl.elem0 = %ptr.900> [symbolic]
// CHECK:STDOUT:   %require_complete.d01: <witness> = require_complete_type %Ptr_where.type.0d6 [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.5aa: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Ptr.facet.db8: %Ptr.type = facet_value %T.as_type, (%Ptr.impl_witness.5aa) [symbolic]
// CHECK:STDOUT:   %pattern_type.a95: type = pattern_type %ptr.900 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.df6: <witness> = require_complete_type %ptr.900 [symbolic]
// CHECK:STDOUT:   %require_complete.020: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bd4)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @impl [concrete]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness %Ptr.impl_witness_table, @impl(constants.%U) [symbolic = @impl.%Ptr.impl_witness (constants.%Ptr.impl_witness.e8c)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.79f [symbolic = @impl.%ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9bb = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc9_19 (%pattern_type.87e) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc9_19 (%pattern_type.87e) = ref_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %.loc9_15: @F.%pattern_type.loc9_19 (%pattern_type.87e) = var_pattern %t.param_patt
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc9_25 (%pattern_type.a95) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc9_25 (%pattern_type.a95) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %Ptr.ref.loc9_31: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9_29: type = facet_access_type constants.%T [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_29: type = converted constants.%T, %T.as_type.loc9_29 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%Ptr.impl_witness.5aa, element0 [symbolic = %ptr (constants.%ptr.900)]
// CHECK:STDOUT:     %Ptr.ref.loc9_10: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %T.loc9_6.1: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %t.param: ref @F.%T.as_type.loc9_22.2 (%T.as_type) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc9_22.1: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type.loc9_22.1 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.as_type.loc9_22.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.900) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.900) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete = constants.%assoc0]
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
// CHECK:STDOUT: generic impl @impl(%U.loc7_20.1: type) {
// CHECK:STDOUT:   %U.loc7_20.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.bd4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Ptr_where.type [symbolic = %require_complete (constants.%require_complete.465)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%U.loc7_20.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.e8c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%Ptr.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: %Ptr.type) {
// CHECK:STDOUT:   %T.loc9_6.2: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc9_22.2: type = facet_access_type %T.loc9_6.2 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.as_type.loc9_22.2 [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.87e)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness file.%Ptr.impl_witness_table, @impl(%T.as_type.loc9_22.2) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.5aa)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_22.2 [symbolic = %ptr (constants.%ptr.900)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.a95)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_25 (constants.%require_complete.df6)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.as_type.loc9_22.2 [symbolic = %require_complete.loc9_15 (constants.%require_complete.020)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc9_22.2 (%T.as_type)) -> @F.%ptr (%ptr.900) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc9_22.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.900) = addr_of %t.ref
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.b6e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%U) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.bd4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.465
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.e8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%U.loc7_20.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.as_type) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.900
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.0d6
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d01
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.5aa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.db8) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc9_22.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.87e
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.5aa
// CHECK:STDOUT:   %ptr => constants.%ptr.900
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.a95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(@F.%T.as_type.loc9_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %Self.as_type.7f0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %Dest [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.1d5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %pattern_type.a93: type = pattern_type %Self.as_type.419 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.8f8: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %pattern_type.a80: type = pattern_type %Self.as_type.560 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type @BitAnd [concrete]
// CHECK:STDOUT:   %assoc0.220: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness file.%BitAnd.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.2: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, (%BitAnd.impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc9_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt: %pattern_type.98f = symbolic_binding_pattern Dest, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc13_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc21_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.impl_witness_table = impl_witness_table (@impl.%Op.decl), @impl [concrete]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness %BitAnd.impl_witness_table, @impl(constants.%T) [symbolic = @impl.%BitAnd.impl_witness (constants.%BitAnd.impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc9_14.1: type) {
// CHECK:STDOUT:   %Dest.loc9_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest.loc9_14.2)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%Dest.loc9_14.2) [symbolic = %Convert.type (constants.%Convert.type.ad1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.ad1) = struct_value () [symbolic = %Convert (constants.%Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%Dest.loc9_14.2) [symbolic = %As.assoc_type (constants.%As.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc10_35.2: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.1d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.ad1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%pattern_type.loc10_14 (%pattern_type.947) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%pattern_type.loc10_14 (%pattern_type.947) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%pattern_type.loc10_28 (%pattern_type.7dcd0a.1) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%pattern_type.loc10_28 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @As.%Dest.loc9_14.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc10_20.1: type = splice_block %.loc10_20.3 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc10_20.2: @Convert.1.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%Dest) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8ba) = name_ref Self, %.loc10_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc10_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc10_20.3: type = converted %Self.ref, %Self.as_type.loc10_20.2 [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_35.1: @As.%As.assoc_type (%As.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.1d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc10_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%Dest.loc13_22.1: type) {
// CHECK:STDOUT:   %Dest.loc13_22.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest.loc13_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %Convert.type (constants.%Convert.type.4cf)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.4cf) = struct_value () [symbolic = %Convert (constants.%Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest.loc13_22.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc14_35.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.4cf) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.147)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%pattern_type.loc14_14 (%pattern_type.a93) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%pattern_type.loc14_14 (%pattern_type.a93) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%pattern_type.loc14_28 (%pattern_type.7dcd0a.1) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%pattern_type.loc14_28 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Dest.ref: type = name_ref Dest, @ImplicitAs.%Dest.loc13_22.1 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc14_20.1: type = splice_block %.loc14_20.3 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc14_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%Dest) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc14_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc14_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc14_20.3: type = converted %Self.ref, %Self.as_type.loc14_20.2 [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%Dest (%Dest) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%Dest (%Dest) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_35.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc14_35.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Dest = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc14_35.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitAnd {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.e44]
// CHECK:STDOUT:   %Op.decl: %Op.type.613 = fn_decl @Op.1 [concrete = constants.%Op.d98] {
// CHECK:STDOUT:     %self.patt: @Op.1.%pattern_type (%pattern_type.a80) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%pattern_type (%pattern_type.a80) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%pattern_type (%pattern_type.a80) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%pattern_type (%pattern_type.a80) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc18_37: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:     %Self.as_type.loc18_37: type = facet_access_type %Self.ref.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %.loc18_37: type = converted %Self.ref.loc18_37, %Self.as_type.loc18_37 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param0
// CHECK:STDOUT:     %.loc18_15.1: type = splice_block %.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_15: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_15.2: type = facet_access_type %Self.ref.loc18_15 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_15.2: type = converted %Self.ref.loc18_15, %Self.as_type.loc18_15.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param call_param1
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.2 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)] {
// CHECK:STDOUT:       %Self.ref.loc18_28: %BitAnd.type = name_ref Self, @BitAnd.%Self [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:       %Self.as_type.loc18_28: type = facet_access_type %Self.ref.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:       %.loc18_28.2: type = converted %Self.ref.loc18_28, %Self.as_type.loc18_28 [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BitAnd.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0.220]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T.loc21_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   %BitAnd.impl_witness: <witness> = impl_witness file.%BitAnd.impl_witness_table, @impl(%T.loc21_14.2) [symbolic = %BitAnd.impl_witness (constants.%BitAnd.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc21_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc21_14.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %other.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = binding_pattern other
// CHECK:STDOUT:       %other.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:       %return.patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Op.2.%pattern_type (%pattern_type.7dcd0a.2) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref.loc22_37: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref.loc22_15: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @Op.2.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %Self.ref.loc22_28: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %other: @Op.2.%T (%T) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @Op.2.%T (%T) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Op.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%BitAnd.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc9_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: @Convert.1.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc10_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:   %pattern_type.loc10_14: type = pattern_type %Self.as_type.loc10_20.1 [symbolic = %pattern_type.loc10_14 (constants.%pattern_type.947)]
// CHECK:STDOUT:   %pattern_type.loc10_28: type = pattern_type %Dest [symbolic = %pattern_type.loc10_28 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0)) -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc13_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:   %pattern_type.loc14_14: type = pattern_type %Self.as_type.loc14_20.1 [symbolic = %pattern_type.loc14_14 (constants.%pattern_type.a93)]
// CHECK:STDOUT:   %pattern_type.loc14_28: type = pattern_type %Dest [symbolic = %pattern_type.loc14_28 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)) -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc18_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc18_15.1 [symbolic = %pattern_type (constants.%pattern_type.a80)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560), %other.param: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.2.%T (%T), %other.param: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc9_14.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc10_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT:   %pattern_type.loc10_14 => constants.%pattern_type.947
// CHECK:STDOUT:   %pattern_type.loc10_28 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc13_22.2 => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT:   %pattern_type.loc14_14 => constants.%pattern_type.a93
// CHECK:STDOUT:   %pattern_type.loc14_28 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc13_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %BitAnd.impl_witness => constants.%BitAnd.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc21_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

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

// --- final_impl_rewrite_of_symbolic_through_impl_witness.carbon
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

// --- fail_todo_final_impl_rewrite_of_symbolic_through_facet_access_witness.carbon
library "[[@TEST_NAME]]";

interface Ptr {
  let Type:! type;
}

final impl forall [U:! type] U as Ptr where .Type = U* {}

fn F[T:! Ptr](var t: T) -> T.(Ptr.Type) {
  // CHECK:STDERR: fail_todo_final_impl_rewrite_of_symbolic_through_facet_access_witness.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `T*` to `T.(Ptr.Type)` [ConversionFailure]
  // CHECK:STDERR:   return &t;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_final_impl_rewrite_of_symbolic_through_facet_access_witness.carbon:[[@LINE+4]]:3: note: type `T*` does not implement interface `Core.ImplicitAs(T.(Ptr.Type))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return &t;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  return &t;
}

// CHECK:STDOUT: --- final_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
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
// CHECK:STDOUT:   %S.patt: type = symbolic_binding_pattern S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.8e9: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.eb1: %Z.type.8e9 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %Self.831: %Z.type.8e9 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0e7: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.abb: %Z.assoc_type.0e7 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.b87: <witness> = require_complete_type %Z.type.8e9 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.ffd: type = facet_access_type %.Self.eb1 [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.iface0.178: <witness> = facet_access_witness %.Self.eb1, element0 [symbolic]
// CHECK:STDOUT:   %Z.facet.868: %Z.type.8e9 = facet_value %.Self.as_type.ffd, (%.Self.as_wit.iface0.178) [symbolic]
// CHECK:STDOUT:   %impl.elem0.a53: type = impl_witness_access %.Self.as_wit.iface0.178, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.93a: type = facet_type <@Z, @Z(%S) where %impl.elem0.a53 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.619: <witness> = require_complete_type %Z_where.type.93a [symbolic]
// CHECK:STDOUT:   %impl_witness.dba: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc9), @impl.76d(%T.8b3, %S) [symbolic]
// CHECK:STDOUT:   %Z.type.049: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.3c5: %Z.type.049 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.a38: %Z.type.049 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.as_type.c3c: type = facet_access_type %.Self.3c5 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0.72c: <witness> = facet_access_witness %.Self.3c5, element0 [symbolic_self]
// CHECK:STDOUT:   %Z.facet.375: %Z.type.049 = facet_value %.Self.as_type.c3c, (%.Self.as_wit.iface0.72c) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.802: type = impl_witness_access %.Self.as_wit.iface0.72c, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.bee: type = facet_type <@Z, @Z(%C) where %impl.elem0.802 = %T.8b3> [symbolic]
// CHECK:STDOUT:   %require_complete.13f: <witness> = require_complete_type %Z_where.type.bee [symbolic]
// CHECK:STDOUT:   %impl_witness.a9c: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc11), @impl.44b(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.62d: %Z.type.049 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.897: %Z.type.049 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.62d [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.2b1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Z_where.type.c29: type = facet_type <@Z, @Z(%C) where %impl.elem0.802 = %T.as_type> [symbolic]
// CHECK:STDOUT:   %require_complete.c8a: <witness> = require_complete_type %Z_where.type.c29 [symbolic]
// CHECK:STDOUT:   %impl_witness.7ef: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc11), @impl.44b(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Z.facet.ffe: %Z.type.049 = facet_value %T.as_type, (%impl_witness.7ef) [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc3_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_13.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.76d [concrete] {
// CHECK:STDOUT:     %T.patt.loc9_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %S.patt.loc9_24.1: type = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc9_24.2 (constants.%S.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc9_24.1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc9_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.8e9)]
// CHECK:STDOUT:     %.Self.1: @impl.76d.%Z.type.loc9_42.2 (%Z.type.8e9) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:     %.Self.ref: @impl.76d.%Z.type.loc9_42.2 (%Z.type.8e9) = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:     %.loc9_50.1: @impl.76d.%Z.assoc_type (%Z.assoc_type.0e7) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %X.ref: @impl.76d.%Z.assoc_type (%Z.assoc_type.0e7) = name_ref X, %.loc9_50.1 [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %.Self.as_type.loc9_50.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:     %.loc9_50.2: type = converted %.Self.ref, %.Self.as_type.loc9_50.1 [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc9_50.1: <witness> = facet_access_witness %.Self.ref, element0 [symbolic = %.Self.as_wit.iface0.loc9_50.2 (constants.%.Self.as_wit.iface0.178)]
// CHECK:STDOUT:     %impl.elem0.loc9_50.1: type = impl_witness_access %.Self.as_wit.iface0.loc9_50.1, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.a53)]
// CHECK:STDOUT:     %.loc9_56.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_56.2: type = converted %.loc9_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.93a)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_50.1, %.loc9_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %S.loc9_24.1: type = bind_symbolic_name S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc9: <witness> = impl_witness (%impl_witness_assoc_constant.loc9), @impl.76d(constants.%T.8b3, constants.%S) [symbolic = @impl.76d.%impl_witness (constants.%impl_witness.dba)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc9: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   impl_decl @impl.44b [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_20.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_20.2 (constants.%T.patt.e01)]
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
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0.72c]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.802]
// CHECK:STDOUT:     %T.ref.loc11_51: type = name_ref T, %T.loc11_20.1 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc11_40: type = where_expr %.Self [symbolic = %Z_where.type (constants.%Z_where.type.bee)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc11_51
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_20.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc11: <witness> = impl_witness (%impl_witness_assoc_constant.loc11), @impl.44b(constants.%T.8b3) [symbolic = @impl.44b.%impl_witness (constants.%impl_witness.a9c)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc11: type = impl_witness_assoc_constant constants.%T.8b3 [symbolic = @impl.44b.%T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc13_6.1: %Z.type.049 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt.897)]
// CHECK:STDOUT:     %t.patt: @F.%T.as_type.loc13_19.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%T.as_type.loc13_19.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
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
// CHECK:STDOUT:   %T.patt.loc3_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_13.2 (constants.%T.patt.e01)]
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
// CHECK:STDOUT: generic impl @impl.76d(%T.loc9_14.1: type, %S.loc9_24.1: type) {
// CHECK:STDOUT:   %T.loc9_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc9_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %S.loc9_24.2: type = bind_symbolic_name S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   %S.patt.loc9_24.2: type = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc9_24.2 (constants.%S.patt)]
// CHECK:STDOUT:   %Z.type.loc9_42.2: type = facet_type <@Z, @Z(%S.loc9_24.2)> [symbolic = %Z.type.loc9_42.2 (constants.%Z.type.8e9)]
// CHECK:STDOUT:   %.Self.2: @impl.76d.%Z.type.loc9_42.2 (%Z.type.8e9) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:   %require_complete.loc9_50: <witness> = require_complete_type @impl.76d.%Z.type.loc9_42.2 (%Z.type.8e9) [symbolic = %require_complete.loc9_50 (constants.%require_complete.b87)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc9_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.0e7)]
// CHECK:STDOUT:   %assoc0: @impl.76d.%Z.assoc_type (%Z.assoc_type.0e7) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:   %.Self.as_type.loc9_50.2: type = facet_access_type %.Self.2 [symbolic = %.Self.as_type.loc9_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:   %.Self.as_wit.iface0.loc9_50.2: <witness> = facet_access_witness %.Self.2, element0 [symbolic = %.Self.as_wit.iface0.loc9_50.2 (constants.%.Self.as_wit.iface0.178)]
// CHECK:STDOUT:   %impl.elem0.loc9_50.2: type = impl_witness_access %.Self.as_wit.iface0.loc9_50.2, element0 [symbolic = %impl.elem0.loc9_50.2 (constants.%impl.elem0.a53)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc9_24.2) where %impl.elem0.loc9_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.93a)]
// CHECK:STDOUT:   %require_complete.loc9_44: <witness> = require_complete_type @impl.76d.%Z_where.type (%Z_where.type.93a) [symbolic = %require_complete.loc9_44 (constants.%require_complete.619)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc9), @impl.76d(%T.loc9_14.2, %S.loc9_24.2) [symbolic = %impl_witness (constants.%impl_witness.dba)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc9_44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.44b(%T.loc11_20.1: type) {
// CHECK:STDOUT:   %T.loc11_20.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_20.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc11_20.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_20.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.802 = %T.loc11_20.2> [symbolic = %Z_where.type (constants.%Z_where.type.bee)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.44b.%Z_where.type (%Z_where.type.bee) [symbolic = %require_complete (constants.%require_complete.13f)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc11), @impl.44b(%T.loc11_20.2) [symbolic = %impl_witness (constants.%impl_witness.a9c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc11_30 as %.loc11_40 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.1: %Z.type.049) {
// CHECK:STDOUT:   %T.loc13_6.2: %Z.type.049 = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T.62d)]
// CHECK:STDOUT:   %T.patt.loc13_6.2: %Z.type.049 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt.897)]
// CHECK:STDOUT:   %T.as_type.loc13_19.2: type = facet_access_type %T.loc13_6.2 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T.as_type.loc13_19.2 (%T.as_type) [symbolic = %require_complete (constants.%require_complete.2b1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc13_6.1: %Z.type.049, %t.param_patt: @F.%T.as_type.loc13_19.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%T.as_type.loc13_19.2 (%T.as_type) = binding_pattern a
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
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%impl_witness.7ef, element0 [symbolic = %T.as_type.loc13_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%T.as_type.loc13_19.2 (%T.as_type) = bind_name a, %t.ref
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc3_13.2 => constants.%T.patt.e01
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.8b3, constants.%Self.2bc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(%T.loc3_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%S
// CHECK:STDOUT:   %T.patt.loc3_13.2 => constants.%T.patt.e01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.8e9
// CHECK:STDOUT:   %Self.2 => constants.%Self.831
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%Z.facet.868) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.76d(constants.%T.8b3, constants.%S) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc9_14.2 => constants.%T.patt.e01
// CHECK:STDOUT:   %S.loc9_24.2 => constants.%S
// CHECK:STDOUT:   %S.patt.loc9_24.2 => constants.%S.patt
// CHECK:STDOUT:   %Z.type.loc9_42.2 => constants.%Z.type.8e9
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.eb1
// CHECK:STDOUT:   %require_complete.loc9_50 => constants.%require_complete.b87
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT:   %.Self.as_type.loc9_50.2 => constants.%.Self.as_type.ffd
// CHECK:STDOUT:   %.Self.as_wit.iface0.loc9_50.2 => constants.%.Self.as_wit.iface0.178
// CHECK:STDOUT:   %impl.elem0.loc9_50.2 => constants.%impl.elem0.a53
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.93a
// CHECK:STDOUT:   %require_complete.loc9_44 => constants.%require_complete.619
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.dba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(@impl.76d.%S.loc9_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.76d(%T.loc9_14.2, %S.loc9_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc3_13.2 => constants.%T.patt.e01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.049
// CHECK:STDOUT:   %Self.2 => constants.%Self.a38
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%Z.facet.375) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.44b(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc11_20.2 => constants.%T.patt.e01
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.bee
// CHECK:STDOUT:   %require_complete => constants.%require_complete.13f
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.a9c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.44b(%T.loc11_20.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.62d) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T.62d
// CHECK:STDOUT:   %T.patt.loc13_6.2 => constants.%T.patt.897
// CHECK:STDOUT:   %T.as_type.loc13_19.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.44b(constants.%T.as_type) {
// CHECK:STDOUT:   %T.loc11_20.2 => constants.%T.as_type
// CHECK:STDOUT:   %T.patt.loc11_20.2 => constants.%T.patt.e01
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.c29
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c8a
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.7ef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%Z.facet.ffe) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
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
// CHECK:STDOUT:   %S.patt: type = symbolic_binding_pattern S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.8e9: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.eb1: %Z.type.8e9 = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %Self.831: %Z.type.8e9 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.0e7: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.abb: %Z.assoc_type.0e7 = assoc_entity element0, @Z.%X [symbolic]
// CHECK:STDOUT:   %require_complete.b87: <witness> = require_complete_type %Z.type.8e9 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.ffd: type = facet_access_type %.Self.eb1 [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.iface0.178: <witness> = facet_access_witness %.Self.eb1, element0 [symbolic]
// CHECK:STDOUT:   %Z.facet.868: %Z.type.8e9 = facet_value %.Self.as_type.ffd, (%.Self.as_wit.iface0.178) [symbolic]
// CHECK:STDOUT:   %impl.elem0.a53: type = impl_witness_access %.Self.as_wit.iface0.178, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.93a: type = facet_type <@Z, @Z(%S) where %impl.elem0.a53 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %require_complete.619: <witness> = require_complete_type %Z_where.type.93a [symbolic]
// CHECK:STDOUT:   %impl_witness.dba: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc10), @impl.76d(%T.8b3, %S) [symbolic]
// CHECK:STDOUT:   %Z.type.049: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.3c5: %Z.type.049 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.a38: %Z.type.049 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.252: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.ea4: %Z.assoc_type.252 = assoc_entity element0, @Z.%X [concrete]
// CHECK:STDOUT:   %.Self.as_type.c3c: type = facet_access_type %.Self.3c5 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0.72c: <witness> = facet_access_witness %.Self.3c5, element0 [symbolic_self]
// CHECK:STDOUT:   %Z.facet.375: %Z.type.049 = facet_value %.Self.as_type.c3c, (%.Self.as_wit.iface0.72c) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.802: type = impl_witness_access %.Self.as_wit.iface0.72c, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.bee: type = facet_type <@Z, @Z(%C) where %impl.elem0.802 = %T.8b3> [symbolic]
// CHECK:STDOUT:   %require_complete.13f: <witness> = require_complete_type %Z_where.type.bee [symbolic]
// CHECK:STDOUT:   %impl_witness.a9c: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc12), @impl.44b(%T.8b3) [symbolic]
// CHECK:STDOUT:   %T.62d: %Z.type.049 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.897: %Z.type.049 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.62d [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.2b1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0: <witness> = facet_access_witness %T.62d, element0 [symbolic]
// CHECK:STDOUT:   %Z.facet.c2f: %Z.type.049 = facet_value %T.as_type, (%T.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.44a: type = impl_witness_access %T.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %require_complete.b00: <witness> = require_complete_type %impl.elem0.44a [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.9f5: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.fb2: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.44a)> [symbolic]
// CHECK:STDOUT:   %Self.643: %ImplicitAs.type.fb2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.6ea: type = fn_type @Convert, @ImplicitAs(%impl.elem0.44a) [symbolic]
// CHECK:STDOUT:   %Convert.d0b: %Convert.type.6ea = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.40d: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.44a) [symbolic]
// CHECK:STDOUT:   %assoc0.763: %ImplicitAs.assoc_type.40d = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %require_complete.462: <witness> = require_complete_type %ImplicitAs.type.fb2 [symbolic]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//prelude, inst67 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//prelude, inst67 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst102 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.02e = import_ref Core//prelude, loc18_41, unloaded
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.f80 = import_ref Core//prelude, loc21_36, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.3: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude, loc22_42, unloaded
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
// CHECK:STDOUT:     %T.patt.loc3_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_13.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_13.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.76d [concrete] {
// CHECK:STDOUT:     %T.patt.loc10_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %S.patt.loc10_24.1: type = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc10_24.2 (constants.%S.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %Z.ref: %Z.type.9fb = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc10_24.1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc10_42.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.8e9)]
// CHECK:STDOUT:     %.Self.1: @impl.76d.%Z.type.loc10_42.2 (%Z.type.8e9) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:     %.Self.ref: @impl.76d.%Z.type.loc10_42.2 (%Z.type.8e9) = name_ref .Self, %.Self.1 [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:     %.loc10_50.1: @impl.76d.%Z.assoc_type (%Z.assoc_type.0e7) = specific_constant @X.%assoc0, @Z(constants.%S) [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %X.ref: @impl.76d.%Z.assoc_type (%Z.assoc_type.0e7) = name_ref X, %.loc10_50.1 [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:     %.Self.as_type.loc10_50.1: type = facet_access_type %.Self.ref [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:     %.loc10_50.2: type = converted %.Self.ref, %.Self.as_type.loc10_50.1 [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc10_50.1: <witness> = facet_access_witness %.Self.ref, element0 [symbolic = %.Self.as_wit.iface0.loc10_50.2 (constants.%.Self.as_wit.iface0.178)]
// CHECK:STDOUT:     %impl.elem0.loc10_50.1: type = impl_witness_access %.Self.as_wit.iface0.loc10_50.1, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.a53)]
// CHECK:STDOUT:     %.loc10_56.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_56.2: type = converted %.loc10_56.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_44: type = where_expr %.Self.1 [symbolic = %Z_where.type (constants.%Z_where.type.93a)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_50.1, %.loc10_56.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %S.loc10_24.1: type = bind_symbolic_name S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc10: <witness> = impl_witness (%impl_witness_assoc_constant.loc10), @impl.76d(constants.%T.8b3, constants.%S) [symbolic = @impl.76d.%impl_witness (constants.%impl_witness.dba)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc10: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   impl_decl @impl.44b [concrete] {
// CHECK:STDOUT:     %T.patt.loc12_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.e01)]
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
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0.72c]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.802]
// CHECK:STDOUT:     %T.ref.loc12_45: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc12_34: type = where_expr %.Self [symbolic = %Z_where.type (constants.%Z_where.type.bee)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %T.ref.loc12_45
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc12: <witness> = impl_witness (%impl_witness_assoc_constant.loc12), @impl.44b(constants.%T.8b3) [symbolic = @impl.44b.%impl_witness (constants.%impl_witness.a9c)]
// CHECK:STDOUT:   %impl_witness_assoc_constant.loc12: type = impl_witness_assoc_constant constants.%T.8b3 [symbolic = @impl.44b.%T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc14_6.1: %Z.type.049 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_6.2 (constants.%T.patt.897)]
// CHECK:STDOUT:     %t.patt: @F.%T.as_type.loc14_19.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%T.as_type.loc14_19.2 (%T.as_type) = value_param_pattern %t.patt, call_param0
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
// CHECK:STDOUT:   %T.patt.loc3_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_13.2 (constants.%T.patt.e01)]
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
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
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
// CHECK:STDOUT: generic impl @impl.76d(%T.loc10_14.1: type, %S.loc10_24.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc10_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %S.loc10_24.2: type = bind_symbolic_name S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   %S.patt.loc10_24.2: type = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc10_24.2 (constants.%S.patt)]
// CHECK:STDOUT:   %Z.type.loc10_42.2: type = facet_type <@Z, @Z(%S.loc10_24.2)> [symbolic = %Z.type.loc10_42.2 (constants.%Z.type.8e9)]
// CHECK:STDOUT:   %.Self.2: @impl.76d.%Z.type.loc10_42.2 (%Z.type.8e9) = bind_symbolic_name .Self [symbolic = %.Self.2 (constants.%.Self.eb1)]
// CHECK:STDOUT:   %require_complete.loc10_50: <witness> = require_complete_type @impl.76d.%Z.type.loc10_42.2 (%Z.type.8e9) [symbolic = %require_complete.loc10_50 (constants.%require_complete.b87)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc10_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.0e7)]
// CHECK:STDOUT:   %assoc0: @impl.76d.%Z.assoc_type (%Z.assoc_type.0e7) = assoc_entity element0, @Z.%X [symbolic = %assoc0 (constants.%assoc0.abb)]
// CHECK:STDOUT:   %.Self.as_type.loc10_50.2: type = facet_access_type %.Self.2 [symbolic = %.Self.as_type.loc10_50.2 (constants.%.Self.as_type.ffd)]
// CHECK:STDOUT:   %.Self.as_wit.iface0.loc10_50.2: <witness> = facet_access_witness %.Self.2, element0 [symbolic = %.Self.as_wit.iface0.loc10_50.2 (constants.%.Self.as_wit.iface0.178)]
// CHECK:STDOUT:   %impl.elem0.loc10_50.2: type = impl_witness_access %.Self.as_wit.iface0.loc10_50.2, element0 [symbolic = %impl.elem0.loc10_50.2 (constants.%impl.elem0.a53)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc10_24.2) where %impl.elem0.loc10_50.2 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.93a)]
// CHECK:STDOUT:   %require_complete.loc10_44: <witness> = require_complete_type @impl.76d.%Z_where.type (%Z_where.type.93a) [symbolic = %require_complete.loc10_44 (constants.%require_complete.619)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc10), @impl.76d(%T.loc10_14.2, %S.loc10_24.2) [symbolic = %impl_witness (constants.%impl_witness.dba)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc10_44 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.44b(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc12_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.802 = %T.loc12_14.2> [symbolic = %Z_where.type (constants.%Z_where.type.bee)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.44b.%Z_where.type (%Z_where.type.bee) [symbolic = %require_complete (constants.%require_complete.13f)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (file.%impl_witness_assoc_constant.loc12), @impl.44b(%T.loc12_14.2) [symbolic = %impl_witness (constants.%impl_witness.a9c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc12_24 as %.loc12_34 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.3: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e01)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type) = struct_value () [symbolic = %Op (constants.%Op)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T.8b3) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.f80
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc14_6.1: %Z.type.049) {
// CHECK:STDOUT:   %T.loc14_6.2: %Z.type.049 = bind_symbolic_name T, 0 [symbolic = %T.loc14_6.2 (constants.%T.62d)]
// CHECK:STDOUT:   %T.patt.loc14_6.2: %Z.type.049 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_6.2 (constants.%T.patt.897)]
// CHECK:STDOUT:   %T.as_type.loc14_19.2: type = facet_access_type %T.loc14_6.2 [symbolic = %T.as_type.loc14_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type @F.%T.as_type.loc14_19.2 (%T.as_type) [symbolic = %require_complete.loc14 (constants.%require_complete.2b1)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc22_11.2: <witness> = facet_access_witness %T.loc14_6.2, element0 [symbolic = %T.as_wit.iface0.loc22_11.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc22_11.2: type = impl_witness_access %T.as_wit.iface0.loc22_11.2, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.44a)]
// CHECK:STDOUT:   %require_complete.loc22_11: <witness> = require_complete_type @F.%impl.elem0.loc22_11.2 (%impl.elem0.44a) [symbolic = %require_complete.loc22_11 (constants.%require_complete.b00)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.loc22_11.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.fb2)]
// CHECK:STDOUT:   %require_complete.loc22_16: <witness> = require_complete_type @F.%ImplicitAs.type (%ImplicitAs.type.fb2) [symbolic = %require_complete.loc22_16 (constants.%require_complete.462)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc14_6.1: %Z.type.049, %t.param_patt: @F.%T.as_type.loc14_19.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%impl.elem0.loc22_11.2 (%impl.elem0.44a) = binding_pattern a
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc14_19.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %impl.elem0.loc22_11.1 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.44a)] {
// CHECK:STDOUT:       %T.ref.loc22: %Z.type.049 = name_ref T, %T.loc14_6.1 [symbolic = %T.loc14_6.2 (constants.%T.62d)]
// CHECK:STDOUT:       %.loc22_11.2: %Z.assoc_type.252 = specific_constant @X.%assoc0, @Z(constants.%C) [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.252 = name_ref X, %.loc22_11.2 [concrete = constants.%assoc0.ea4]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.as_type.loc14_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_11.3: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.as_type.loc14_19.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %T.as_wit.iface0.loc22_11.1: <witness> = facet_access_witness %T.ref.loc22, element0 [symbolic = %T.as_wit.iface0.loc22_11.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:       %impl.elem0.loc22_11.1: type = impl_witness_access %T.as_wit.iface0.loc22_11.1, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.44a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc22_16: @F.%impl.elem0.loc22_11.2 (%impl.elem0.44a) = converted %t.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %a: @F.%impl.elem0.loc22_11.2 (%impl.elem0.44a) = bind_name a, <error>
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @Convert.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(imports.%Core.import_ref.5ab3ec.4: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.%T (%T.8b3)](%other.param_patt: @Op.%T (%T.8b3)) -> @Op.%T (%T.8b3) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc3_13.2 => constants.%T.patt.e01
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T.8b3, constants.%Self.2bc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(%T.loc3_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%S
// CHECK:STDOUT:   %T.patt.loc3_13.2 => constants.%T.patt.e01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.8e9
// CHECK:STDOUT:   %Self.2 => constants.%Self.831
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%S, constants.%Z.facet.868) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.76d(constants.%T.8b3, constants.%S) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc10_14.2 => constants.%T.patt.e01
// CHECK:STDOUT:   %S.loc10_24.2 => constants.%S
// CHECK:STDOUT:   %S.patt.loc10_24.2 => constants.%S.patt
// CHECK:STDOUT:   %Z.type.loc10_42.2 => constants.%Z.type.8e9
// CHECK:STDOUT:   %.Self.2 => constants.%.Self.eb1
// CHECK:STDOUT:   %require_complete.loc10_50 => constants.%require_complete.b87
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.0e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abb
// CHECK:STDOUT:   %.Self.as_type.loc10_50.2 => constants.%.Self.as_type.ffd
// CHECK:STDOUT:   %.Self.as_wit.iface0.loc10_50.2 => constants.%.Self.as_wit.iface0.178
// CHECK:STDOUT:   %impl.elem0.loc10_50.2 => constants.%impl.elem0.a53
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.93a
// CHECK:STDOUT:   %require_complete.loc10_44 => constants.%require_complete.619
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.dba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(@impl.76d.%S.loc10_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.76d(%T.loc10_14.2, %S.loc10_24.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.loc3_13.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc3_13.2 => constants.%T.patt.e01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.049
// CHECK:STDOUT:   %Self.2 => constants.%Self.a38
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.252
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ea4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%Z.facet.375) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.44b(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%T.patt.e01
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.bee
// CHECK:STDOUT:   %require_complete => constants.%require_complete.13f
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.a9c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.44b(%T.loc12_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.62d) {
// CHECK:STDOUT:   %T.loc14_6.2 => constants.%T.62d
// CHECK:STDOUT:   %T.patt.loc14_6.2 => constants.%T.patt.897
// CHECK:STDOUT:   %T.as_type.loc14_19.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%C, constants.%Z.facet.c2f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest.patt
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%impl.elem0.44a) {
// CHECK:STDOUT:   %Dest => constants.%impl.elem0.44a
// CHECK:STDOUT:   %Dest.patt => constants.%Dest.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.fb2
// CHECK:STDOUT:   %Self => constants.%Self.643
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.6ea
// CHECK:STDOUT:   %Convert => constants.%Convert.d0b
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.40d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.763
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %T.patt => constants.%T.patt.e01
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@F.%impl.elem0.loc22_11.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_impl_witness.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %Ptr.facet.879: %Ptr.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %U [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.d906c9.1: type = facet_type <@Ptr where %impl.elem0 = %ptr.79f131.1> [symbolic]
// CHECK:STDOUT:   %require_complete.0dd5ec.1: <witness> = require_complete_type %Ptr_where.type.d906c9.1 [symbolic]
// CHECK:STDOUT:   %impl_witness.4de5dc.1: <witness> = impl_witness (file.%impl_witness_assoc_constant), @impl(%U) [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f131.2: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.d906c9.2: type = facet_type <@Ptr where %impl.elem0 = %ptr.79f131.2> [symbolic]
// CHECK:STDOUT:   %require_complete.0dd5ec.2: <witness> = require_complete_type %Ptr_where.type.d906c9.2 [symbolic]
// CHECK:STDOUT:   %impl_witness.4de5dc.2: <witness> = impl_witness (file.%impl_witness_assoc_constant), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Ptr.facet.55b: %Ptr.type = facet_value %T, (%impl_witness.4de5dc.2) [symbolic]
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
// CHECK:STDOUT:     %U.patt.loc7_20.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_20.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self [symbolic = %Ptr_where.type (constants.%Ptr_where.type.d906c9.1)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%impl_witness_assoc_constant), @impl(constants.%U) [symbolic = @impl.%impl_witness (constants.%impl_witness.4de5dc.1)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.79f131.1 [symbolic = @impl.%ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: @F.%T.loc9_6.2 (%T) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%T.loc9_6.2 (%T) = ref_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %.loc9_16: @F.%T.loc9_6.2 (%T) = var_pattern %t.param_patt
// CHECK:STDOUT:     %return.patt: @F.%ptr (%ptr.79f131.2) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%ptr (%ptr.79f131.2) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_29: type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %Ptr.facet.loc9_30.1: %Ptr.type = facet_value constants.%T, (constants.%impl_witness.4de5dc.2) [symbolic = %Ptr.facet.loc9_30.2 (constants.%Ptr.facet.55b)]
// CHECK:STDOUT:     %.loc9_30: %Ptr.type = converted %T.ref.loc9_29, %Ptr.facet.loc9_30.1 [symbolic = %Ptr.facet.loc9_30.2 (constants.%Ptr.facet.55b)]
// CHECK:STDOUT:     %as_wit.iface0: <witness> = facet_access_witness %.loc9_30, element0 [symbolic = %impl_witness (constants.%impl_witness.4de5dc.2)]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %as_wit.iface0, element0 [symbolic = %ptr (constants.%ptr.79f131.2)]
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
// CHECK:STDOUT:   %U.patt.loc7_20.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_20.2 (constants.%U.patt)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.d906c9.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%Ptr_where.type (%Ptr_where.type.d906c9.1) [symbolic = %require_complete (constants.%require_complete.0dd5ec.1)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (file.%impl_witness_assoc_constant), @impl(%U.loc7_20.2) [symbolic = %impl_witness (constants.%impl_witness.4de5dc.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: type) {
// CHECK:STDOUT:   %T.loc9_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc9_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (file.%impl_witness_assoc_constant), @impl(%T.loc9_6.2) [symbolic = %impl_witness (constants.%impl_witness.4de5dc.2)]
// CHECK:STDOUT:   %Ptr.facet.loc9_30.2: %Ptr.type = facet_value %T.loc9_6.2, (%impl_witness) [symbolic = %Ptr.facet.loc9_30.2 (constants.%Ptr.facet.55b)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc9_6.2 [symbolic = %ptr (constants.%ptr.79f131.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_26: <witness> = require_complete_type @F.%ptr (%ptr.79f131.2) [symbolic = %require_complete.loc9_26 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc9_16: <witness> = require_complete_type @F.%T.loc9_6.2 (%T) [symbolic = %require_complete.loc9_16 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc9_6.1: type](%.loc9_16: @F.%T.loc9_6.2 (%T)) -> @F.%ptr (%ptr.79f131.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc9_6.2 (%T) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.79f131.2) = addr_of %t.ref
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.879) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%U) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc7_20.2 => constants.%U.patt
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.d906c9.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0dd5ec.1
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.4de5dc.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%U.loc7_20.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T
// CHECK:STDOUT:   %U.patt.loc7_20.2 => constants.%U.patt
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f131.2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.d906c9.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0dd5ec.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.4de5dc.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.55b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc9_6.2 => constants.%T.patt
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.4de5dc.2
// CHECK:STDOUT:   %Ptr.facet.loc9_30.2 => constants.%Ptr.facet.55b
// CHECK:STDOUT:   %ptr => constants.%ptr.79f131.2
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
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %Ptr.facet.879: %Ptr.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %U [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.d90: type = facet_type <@Ptr where %impl.elem0 = %ptr.79f> [symbolic]
// CHECK:STDOUT:   %require_complete.0dd: <witness> = require_complete_type %Ptr_where.type.d90 [symbolic]
// CHECK:STDOUT:   %impl_witness.4de: <witness> = impl_witness (file.%impl_witness_assoc_constant), @impl(%U) [symbolic]
// CHECK:STDOUT:   %T: %Ptr.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %Ptr.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %ptr.900: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.a9b: type = facet_type <@Ptr where %impl.elem0 = %ptr.900> [symbolic]
// CHECK:STDOUT:   %require_complete.a17: <witness> = require_complete_type %Ptr_where.type.a9b [symbolic]
// CHECK:STDOUT:   %impl_witness.780: <witness> = impl_witness (file.%impl_witness_assoc_constant), @impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Ptr.facet.7fd: %Ptr.type = facet_value %T.as_type, (%impl_witness.780) [symbolic]
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
// CHECK:STDOUT:     %U.patt.loc7_20.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_20.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self [symbolic = %Ptr_where.type (constants.%Ptr_where.type.d90)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%impl_witness_assoc_constant), @impl(constants.%U) [symbolic = @impl.%impl_witness (constants.%impl_witness.4de)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.79f [symbolic = @impl.%ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_6.1: %Ptr.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.patt: @F.%T.as_type.loc9_22.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%T.as_type.loc9_22.2 (%T.as_type) = ref_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %.loc9_15: @F.%T.as_type.loc9_22.2 (%T.as_type) = var_pattern %t.param_patt
// CHECK:STDOUT:     %return.patt: @F.%ptr (%ptr.900) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%ptr (%ptr.900) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9_29.1: type = facet_access_type %T.ref.loc9_28 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_29.1: type = converted %T.ref.loc9_28, %T.as_type.loc9_29.1 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %T.as_type.loc9_29.2: type = facet_access_type constants.%T [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_29.2: type = converted constants.%T, %T.as_type.loc9_29.2 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%impl_witness.780, element0 [symbolic = %ptr (constants.%ptr.900)]
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
// CHECK:STDOUT:   %U.patt.loc7_20.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_20.2 (constants.%U.patt)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0 = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.d90)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%Ptr_where.type (%Ptr_where.type.d90) [symbolic = %require_complete (constants.%require_complete.0dd)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (file.%impl_witness_assoc_constant), @impl(%U.loc7_20.2) [symbolic = %impl_witness (constants.%impl_witness.4de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: %Ptr.type) {
// CHECK:STDOUT:   %T.loc9_6.2: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc9_6.2: %Ptr.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc9_22.2: type = facet_access_type %T.loc9_6.2 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_22.2 [symbolic = %ptr (constants.%ptr.900)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type @F.%ptr (%ptr.900) [symbolic = %require_complete.loc9_25 (constants.%require_complete.df6)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type @F.%T.as_type.loc9_22.2 (%T.as_type) [symbolic = %require_complete.loc9_15 (constants.%require_complete.020)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc9_6.1: %Ptr.type](%.loc9_15: @F.%T.as_type.loc9_22.2 (%T.as_type)) -> @F.%ptr (%ptr.900) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc9_22.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.900) = addr_of %t.ref
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.879) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%U) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc7_20.2 => constants.%U.patt
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.d90
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0dd
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.4de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%U.loc7_20.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.as_type) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%T.as_type
// CHECK:STDOUT:   %U.patt.loc7_20.2 => constants.%U.patt
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.900
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.a9b
// CHECK:STDOUT:   %require_complete => constants.%require_complete.a17
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.780
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.7fd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc9_6.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc9_22.2 => constants.%T.as_type
// CHECK:STDOUT:   %ptr => constants.%ptr.900
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_final_impl_rewrite_of_symbolic_through_facet_access_witness.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.972: %Ptr.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.d28: %Ptr.assoc_type = assoc_entity element0, @Ptr.%Type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %Ptr.facet.879: %Ptr.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.dff: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %U [symbolic]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where %impl.elem0.dff = %ptr.79f> [symbolic]
// CHECK:STDOUT:   %require_complete.0dd: <witness> = require_complete_type %Ptr_where.type [symbolic]
// CHECK:STDOUT:   %impl_witness.4de: <witness> = impl_witness (file.%impl_witness_assoc_constant), @impl.a8e(%U) [symbolic]
// CHECK:STDOUT:   %T.972: %Ptr.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.0a1: %Ptr.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.972 [symbolic]
// CHECK:STDOUT:   %T.as_wit.iface0: <witness> = facet_access_witness %T.972, element0 [symbolic]
// CHECK:STDOUT:   %Ptr.facet.7ad: %Ptr.type = facet_value %T.as_type, (%T.as_wit.iface0) [symbolic]
// CHECK:STDOUT:   %impl.elem0.9f2: type = impl_witness_access %T.as_wit.iface0, element0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.357: <witness> = require_complete_type %impl.elem0.9f2 [symbolic]
// CHECK:STDOUT:   %require_complete.020: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %ptr.900: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.9f5: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.73f: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.9f2)> [symbolic]
// CHECK:STDOUT:   %Self.20d: %ImplicitAs.type.73f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.c52: type = fn_type @Convert, @ImplicitAs(%impl.elem0.9f2) [symbolic]
// CHECK:STDOUT:   %Convert.c34: %Convert.type.c52 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.3d1: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.9f2) [symbolic]
// CHECK:STDOUT:   %assoc0.802: %ImplicitAs.assoc_type.3d1 = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %require_complete.5f0: <witness> = require_complete_type %ImplicitAs.type.73f [symbolic]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.b81: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//prelude, inst67 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.Convert = import_ref Core//prelude, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc13_22, loaded [symbolic = @ImplicitAs.%Dest (constants.%Dest)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//prelude, inst67 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//prelude, loc14_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.ad0 = import_ref Core//prelude, inst102 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.02e = import_ref Core//prelude, loc18_41, unloaded
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.f80 = import_ref Core//prelude, loc21_36, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.3: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc21_24, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Core.import_ref.9c1: type = import_ref Core//prelude, loc21_29, loaded [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude, loc22_42, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.4: type = import_ref Core//prelude, loc21_14, loaded [symbolic = @impl.f92.%T (constants.%T.8b3)]
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
// CHECK:STDOUT:   impl_decl @impl.a8e [concrete] {
// CHECK:STDOUT:     %U.patt.loc7_20.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_20.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_30: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self: %Ptr.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_45: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0.dff]
// CHECK:STDOUT:     %U.ref.loc7_53: type = name_ref U, %U.loc7_20.1 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:     %ptr.loc7_54.1: type = ptr_type %U.ref.loc7_53 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %.loc7_39: type = where_expr %.Self [symbolic = %Ptr_where.type (constants.%Ptr_where.type)] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ptr.loc7_54.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_20.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%impl_witness_assoc_constant), @impl.a8e(constants.%U) [symbolic = @impl.a8e.%impl_witness (constants.%impl_witness.4de)]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.79f [symbolic = @impl.a8e.%ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_6.1: %Ptr.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.0a1)]
// CHECK:STDOUT:     %t.patt: @F.%T.as_type.loc9_22.2 (%T.as_type) = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: @F.%T.as_type.loc9_22.2 (%T.as_type) = ref_param_pattern %t.patt, call_param0
// CHECK:STDOUT:     %.loc9_15: @F.%T.as_type.loc9_22.2 (%T.as_type) = var_pattern %t.param_patt
// CHECK:STDOUT:     %return.patt: @F.%impl.elem0.loc9_29.2 (%impl.elem0.9f2) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%impl.elem0.loc9_29.2 (%impl.elem0.9f2) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: %Ptr.type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T.972)]
// CHECK:STDOUT:     %Ptr.ref.loc9_31: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.d28]
// CHECK:STDOUT:     %T.as_wit.iface0.loc9_29.1: <witness> = facet_access_witness %T.ref.loc9_28, element0 [symbolic = %T.as_wit.iface0.loc9_29.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:     %impl.elem0.loc9_29.1: type = impl_witness_access %T.as_wit.iface0.loc9_29.1, element0 [symbolic = %impl.elem0.loc9_29.2 (constants.%impl.elem0.9f2)]
// CHECK:STDOUT:     %Ptr.ref.loc9_10: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %T.loc9_6.1: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.972)]
// CHECK:STDOUT:     %t.param: ref @F.%T.as_type.loc9_22.2 (%T.as_type) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_22: %Ptr.type = name_ref T, %T.loc9_6.1 [symbolic = %T.loc9_6.2 (constants.%T.972)]
// CHECK:STDOUT:       %T.as_type.loc9_22.1: type = facet_access_type %T.ref.loc9_22 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_22.2: type = converted %T.ref.loc9_22, %T.as_type.loc9_22.1 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.as_type.loc9_22.2 (%T.as_type) = bind_name t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%impl.elem0.loc9_29.2 (%impl.elem0.9f2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%impl.elem0.loc9_29.2 (%impl.elem0.9f2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.972]
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
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
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
// CHECK:STDOUT: generic assoc_const @Type(@Ptr.%Self: %Ptr.type) {
// CHECK:STDOUT:   assoc_const Type:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.a8e(%U.loc7_20.1: type) {
// CHECK:STDOUT:   %U.loc7_20.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_20.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc7_20.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_20.2 (constants.%U.patt)]
// CHECK:STDOUT:   %ptr.loc7_54.2: type = ptr_type %U.loc7_20.2 [symbolic = %ptr.loc7_54.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.dff = %ptr.loc7_54.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.a8e.%Ptr_where.type (%Ptr_where.type) [symbolic = %require_complete (constants.%require_complete.0dd)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (file.%impl_witness_assoc_constant), @impl.a8e(%U.loc7_20.2) [symbolic = %impl_witness (constants.%impl_witness.4de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %U.ref.loc7_30 as %.loc7_39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.f92(imports.%Core.import_ref.5ab3ec.3: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e01)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Core.import_ref.bd4), @impl.f92(%T) [symbolic = %impl_witness (constants.%impl_witness.b81)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op, @impl.f92(%T) [symbolic = %Op.type (constants.%Op.type)]
// CHECK:STDOUT:   %Op: @impl.f92.%Op.type (%Op.type) = struct_value () [symbolic = %Op (constants.%Op)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.f92.%T (%T.8b3) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.9c1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.f80
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_6.1: %Ptr.type) {
// CHECK:STDOUT:   %T.loc9_6.2: %Ptr.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_6.2 (constants.%T.972)]
// CHECK:STDOUT:   %T.patt.loc9_6.2: %Ptr.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_6.2 (constants.%T.patt.0a1)]
// CHECK:STDOUT:   %T.as_type.loc9_22.2: type = facet_access_type %T.loc9_6.2 [symbolic = %T.as_type.loc9_22.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %T.as_wit.iface0.loc9_29.2: <witness> = facet_access_witness %T.loc9_6.2, element0 [symbolic = %T.as_wit.iface0.loc9_29.2 (constants.%T.as_wit.iface0)]
// CHECK:STDOUT:   %impl.elem0.loc9_29.2: type = impl_witness_access %T.as_wit.iface0.loc9_29.2, element0 [symbolic = %impl.elem0.loc9_29.2 (constants.%impl.elem0.9f2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type @F.%impl.elem0.loc9_29.2 (%impl.elem0.9f2) [symbolic = %require_complete.loc9_25 (constants.%require_complete.357)]
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type @F.%T.as_type.loc9_22.2 (%T.as_type) [symbolic = %require_complete.loc9_15 (constants.%require_complete.020)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_22.2 [symbolic = %ptr (constants.%ptr.900)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.loc9_29.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.73f)]
// CHECK:STDOUT:   %require_complete.loc17: <witness> = require_complete_type @F.%ImplicitAs.type (%ImplicitAs.type.73f) [symbolic = %require_complete.loc17 (constants.%require_complete.5f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc9_6.1: %Ptr.type](%.loc9_15: @F.%T.as_type.loc9_22.2 (%T.as_type)) -> @F.%impl.elem0.loc9_29.2 (%impl.elem0.9f2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc9_22.2 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.900) = addr_of %t.ref
// CHECK:STDOUT:     %.loc17: @F.%impl.elem0.loc9_29.2 (%impl.elem0.9f2) = converted %addr, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @Convert.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.%Self.as_type (%Self.as_type)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op(imports.%Core.import_ref.5ab3ec.4: type) [from "include_files/facet_types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.%T (%T.8b3)](%other.param_patt: @Op.%T (%T.8b3)) -> @Op.%T (%T.8b3) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Self.972) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.879) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.a8e(constants.%U) {
// CHECK:STDOUT:   %U.loc7_20.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc7_20.2 => constants.%U.patt
// CHECK:STDOUT:   %ptr.loc7_54.2 => constants.%ptr.79f
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0dd
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.4de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.a8e(%U.loc7_20.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Type(constants.%Ptr.facet.7ad) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.972) {
// CHECK:STDOUT:   %T.loc9_6.2 => constants.%T.972
// CHECK:STDOUT:   %T.patt.loc9_6.2 => constants.%T.patt.0a1
// CHECK:STDOUT:   %T.as_type.loc9_22.2 => constants.%T.as_type
// CHECK:STDOUT:   %T.as_wit.iface0.loc9_29.2 => constants.%T.as_wit.iface0
// CHECK:STDOUT:   %impl.elem0.loc9_29.2 => constants.%impl.elem0.9f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest.patt
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%impl.elem0.9f2) {
// CHECK:STDOUT:   %Dest => constants.%impl.elem0.9f2
// CHECK:STDOUT:   %Dest.patt => constants.%Dest.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.73f
// CHECK:STDOUT:   %Self => constants.%Self.20d
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.c52
// CHECK:STDOUT:   %Convert => constants.%Convert.c34
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.3d1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.802
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT:   %T.patt => constants.%T.patt.e01
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.f92(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@F.%impl.elem0.loc9_29.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/facet_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %Convert.type.ad1: type = fn_type @Convert.1, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.0ed: %Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.1d5: %As.assoc_type = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %Convert.type.4cf: type = fn_type @Convert.2, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.147: %Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %assoc0.8f8: %ImplicitAs.assoc_type = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %BitAnd.type: type = facet_type <@BitAnd> [concrete]
// CHECK:STDOUT:   %Self.e44: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.560: type = facet_access_type %Self.e44 [symbolic]
// CHECK:STDOUT:   %Op.type.613: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.d98: %Op.type.613 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAnd.assoc_type: type = assoc_entity_type @BitAnd [concrete]
// CHECK:STDOUT:   %assoc0.220: %BitAnd.assoc_type = assoc_entity element0, @BitAnd.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.28d: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.902: %Op.type.28d = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %BitAnd.facet: %BitAnd.type = facet_value %T, (%impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:     .BitAnd = %BitAnd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %Dest.patt.loc9_14.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc9_14.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc9_14.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %Dest.patt.loc13_22.1: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc13_22.2 (constants.%Dest.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Dest.loc13_22.1: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc13_22.2 (constants.%Dest)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitAnd.decl: type = interface_decl @BitAnd [concrete = constants.%BitAnd.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc21_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %BitAnd.ref: type = name_ref BitAnd, file.%BitAnd.decl [concrete = constants.%BitAnd.type]
// CHECK:STDOUT:     %T.loc21_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Op.decl), @impl(constants.%T) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%Dest.loc9_14.1: type) {
// CHECK:STDOUT:   %Dest.loc9_14.2: type = bind_symbolic_name Dest, 0 [symbolic = %Dest.loc9_14.2 (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt.loc9_14.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc9_14.2 (constants.%Dest.patt)]
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
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:   %Dest.patt.loc13_22.2: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt.loc13_22.2 (constants.%Dest.patt)]
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
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%Dest (%Dest) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%Dest (%Dest) = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560) = out_param_pattern %return.patt, call_param2
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
// CHECK:STDOUT:   %T.patt.loc21_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Op.decl), @impl(%T.loc21_14.2) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc21_14.2) [symbolic = %Op.type (constants.%Op.type.28d)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.28d) = struct_value () [symbolic = %Op (constants.%Op.902)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%T.loc21_14.2 (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %BitAnd.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.28d) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.902)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %other.patt: @Op.2.%T (%T) = binding_pattern other
// CHECK:STDOUT:       %other.param_patt: @Op.2.%T (%T) = value_param_pattern %other.patt, call_param1
// CHECK:STDOUT:       %return.patt: @Op.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Op.2.%T (%T) = out_param_pattern %return.patt, call_param2
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
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%Dest.loc9_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: @Convert.1.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc10_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc10_20.1 (%Self.as_type.7f0)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%Dest.loc13_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc14_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc14_20.1 (%Self.as_type.419)]() -> @Convert.2.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@BitAnd.%Self: %BitAnd.type) {
// CHECK:STDOUT:   %Self: %BitAnd.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.e44)]
// CHECK:STDOUT:   %Self.as_type.loc18_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_15.1 (constants.%Self.as_type.560)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)](%other.param_patt: @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560)) -> @Op.1.%Self.as_type.loc18_15.1 (%Self.as_type.560);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.2.%T (%T)](%other.param_patt: @Op.2.%T (%T)) -> @Op.2.%T (%T) = "type.and";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc9_14.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc9_14.2 => constants.%Dest.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.b4e) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc10_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%Dest.loc9_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest.loc13_22.2 => constants.%Dest
// CHECK:STDOUT:   %Dest.patt.loc13_22.2 => constants.%Dest.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%Dest, constants.%Self.0f3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc14_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%Dest) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%Dest.loc13_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.e44) {
// CHECK:STDOUT:   %Self => constants.%Self.e44
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%Self.as_type.560
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc21_14.2 => constants.%T.patt
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc21_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%BitAnd.facet) {
// CHECK:STDOUT:   %Self => constants.%BitAnd.facet
// CHECK:STDOUT:   %Self.as_type.loc18_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:

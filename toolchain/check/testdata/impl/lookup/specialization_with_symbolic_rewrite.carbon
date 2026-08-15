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

interface Z(T: type) {
  let X: type;
}

class C {}

impl forall [T: type, S: type] T as Z(S) where .X = () {}

final impl forall [T: type] T as Z(C) where .X = T {}

fn F(generic T: Z(C), t: T) {
  // This should typecheck, the `final impl` should give the same `T`.
  let unused a: T.X = t;
}

// --- fail_nonfinal_specialized_symbolic_rewrite.carbon
library "[[@TEST_NAME]]";

interface Z(T: type) {
  let X: type;
}
interface Y {}

class C {}

impl forall [T: type, S: type] T as Z(S) where .X = () {}

impl forall [T: type] T as Z(C) where .X = T {}

fn F(generic T: Z(C), t: T) {
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
  let Type: type;
}

final impl forall [U: type] U as Ptr where .Type = U* {}

fn F[T: type](var t: T) -> T.(Ptr.Type) {
  return &t;
}

// --- final_impl_rewrite_of_symbolic_through_facet_access_type.carbon
library "[[@TEST_NAME]]";

interface Ptr {
  let Type: type;
}

final impl forall [U: type] U as Ptr where .Type = U* {}

fn F[T: Ptr](var t: T) -> T.Type {
  return &t;
}

// --- final_impl_rewrite_of_symbolic_through_facet_value.carbon
library "[[@TEST_NAME]]";

interface Ptr {
  let Type: type;
}

final impl forall [U: type] U as Ptr where .Type = U* {}

fn F[T: Ptr](var t: T) -> T.(Ptr.Type) {
  return &t;
}


// CHECK:STDOUT: --- final_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Z.type.c39: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.c39 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.924: type = facet_type <@Z, @Z(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.21f: %Z.type.924 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.496: type = assoc_entity_type @Z, @Z(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.36e: %Z.assoc_type.496 = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [symbolic]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.78b: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.e6b: %Z.type.78b = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.1a9: %Z.type.78b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.083: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.c53: %Z.assoc_type.083 = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %require_complete.291: <witness> = require_complete_type %Z.type.78b [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.d9d: type = facet_access_type %.Self.frozen.e6b [symbolic]
// CHECK:STDOUT:   %.948: <witness> = impl_self_witness %.Self.frozen.e6b, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b8f: type = impl_witness_access %.948, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.6d3: %Z.type.78b = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %.29c: <witness> = impl_self_witness %.Self.6d3, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.6e9: type = impl_witness_access %.29c, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.28d: type = facet_type <@Z, @Z(%S) where %impl.elem0.6e9 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %.18b: <witness> = impl_self_witness %T.67d, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.e1a: <witness> = impl_witness @T.as.Z.impl.3c5.%Z.impl_witness_table, @T.as.Z.impl.3c5(%T.67d, %S) [symbolic]
// CHECK:STDOUT:   %Z.facet.192: %Z.type.78b = facet_value %T.67d, (%Z.impl_witness.e1a) [symbolic]
// CHECK:STDOUT:   %Z.type.963: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.frozen.16f: %Z.type.963 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.ae2: %Z.type.963 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.f4d: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.169: %Z.assoc_type.f4d = assoc_entity element0, @Z.WithSelf.%X [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.16b: type = facet_access_type %.Self.frozen.16f [symbolic_self]
// CHECK:STDOUT:   %.7fc: <witness> = impl_self_witness %.Self.frozen.16f, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.412: type = impl_witness_access %.7fc, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.e3e: %Z.type.963 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.286: <witness> = impl_self_witness %.Self.e3e, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f68: type = impl_witness_access %.286, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.26f: type = facet_type <@Z, @Z(%C) where %impl.elem0.f68 = %T.67d> [symbolic]
// CHECK:STDOUT:   %.c1d: <witness> = impl_self_witness %T.67d, @Z, @Z(%C) [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.12f: <witness> = impl_witness @T.as.Z.impl.dc2.%Z.impl_witness_table, @T.as.Z.impl.dc2(%T.67d) [symbolic]
// CHECK:STDOUT:   %Z.facet.fe0: %Z.type.963 = facet_value %T.67d, (%Z.impl_witness.12f) [symbolic]
// CHECK:STDOUT:   %pattern_type.d7e: type = pattern_type %Z.type.963 [concrete]
// CHECK:STDOUT:   %T.patt.4cc: %pattern_type.d7e = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.4b3: %Z.type.963 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.4b3 [symbolic]
// CHECK:STDOUT:   %pattern_type.d37: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.d37 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.d37 = at_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.df7: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Z_where.type.147: type = facet_type <@Z, @Z(%C) where %impl.elem0.f68 = %T.as_type> [symbolic]
// CHECK:STDOUT:   %.eef: <witness> = impl_self_witness %T.as_type, @Z, @Z(%C) [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.4cd: <witness> = impl_witness @T.as.Z.impl.dc2.%Z.impl_witness_table, @T.as.Z.impl.dc2(%T.as_type) [symbolic]
// CHECK:STDOUT:   %.ac3: require_specific_def_type = require_specific_def @T.as.Z.impl.dc2(%T.as_type) [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.d37 = value_binding_pattern a [symbolic]
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
// CHECK:STDOUT:   %Z.decl: %Z.type.c39 = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt.loc3_14.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_14.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_16.1: type = splice_block %.loc3_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc3_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.3c5 [concrete] {
// CHECK:STDOUT:     %T.patt.loc9_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:     %S.patt.loc9_24.1: %pattern_type.98f = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc9_24.2 (constants.%S.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_15.1 [symbolic = %T.loc9_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.c39 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc9_24.1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc9_40.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc9_40.2 (constants.%Z.type.78b)]
// CHECK:STDOUT:     %.Self.frozen.loc9_42.1: @T.as.Z.impl.3c5.%Z.type.loc9_40.2 (%Z.type.78b) = symbolic_binding .Self [symbolic = %.Self.frozen.loc9_42.2 (constants.%.Self.frozen.e6b)]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Z.type.loc9_40.1 [concrete]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.3c5.%Z.type.loc9_40.2 (%Z.type.78b) = name_ref .Self, %.Self.frozen.loc9_42.1 [symbolic = %.Self.frozen.loc9_42.2 (constants.%.Self.frozen.e6b)]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.d9d)]
// CHECK:STDOUT:     %.loc9_48.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.d9d)]
// CHECK:STDOUT:     %.loc9_48.2: @T.as.Z.impl.3c5.%Z.assoc_type (%Z.assoc_type.083) = specific_constant @X.%assoc0, @Z.WithSelf(constants.%S, constants.%.Self.frozen.e6b) [symbolic = %assoc0 (constants.%assoc0.c53)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.3c5.%Z.assoc_type (%Z.assoc_type.083) = name_ref X, %.loc9_48.2 [symbolic = %assoc0 (constants.%assoc0.c53)]
// CHECK:STDOUT:     %impl.elem0.loc9_48.1: type = impl_witness_access constants.%.948, element0 [symbolic = %impl.elem0.loc9_48.3 (constants.%impl.elem0.b8f)]
// CHECK:STDOUT:     %.loc9_54.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_54.2: type = converted %.loc9_54.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %rewrite.loc9_51.1 = requirement_rewrite %impl.elem0.loc9_48.1, %.loc9_54.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc9_48.2: type = impl_witness_access constants.%.29c, element0 [symbolic = %impl.elem0.loc9_48.4 (constants.%impl.elem0.6e9)]
// CHECK:STDOUT:     %rewrite.loc9_51.2 = requirement_rewrite %impl.elem0.loc9_48.2, %.loc9_54.2 [concrete]
// CHECK:STDOUT:     %.loc9_42: type = where_expr [symbolic = %Z_where.type (constants.%Z_where.type.28d)] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.type.loc9_40.1 [concrete]
// CHECK:STDOUT:       %rewrite.loc9_51.2 = requirement_rewrite %impl.elem0.loc9_48.2, %.loc9_54.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc9_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc9_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc9_26.1: type = splice_block %.loc9_26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc9_24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc9_26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc9_24.1: type = symbolic_binding S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = impl_self_witness @T.as.Z.impl.3c5.%T.ref, @Z, @Z(constants.%S) [symbolic = @T.as.Z.impl.3c5.%.loc9_56 (constants.%.18b)]
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.dc2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_21.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_21.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc11_29: type = name_ref T, %T.loc11_21.1 [symbolic = %T.loc11_21.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.c39 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.963]
// CHECK:STDOUT:     %.Self.frozen.loc11_39: %Z.type.963 = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.16f]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Z.type [concrete]
// CHECK:STDOUT:     %.Self.ref: %Z.type.963 = name_ref .Self, %.Self.frozen.loc11_39 [symbolic_self = constants.%.Self.frozen.16f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type.16b]
// CHECK:STDOUT:     %.loc11_45.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type.16b]
// CHECK:STDOUT:     %.loc11_45.2: %Z.assoc_type.f4d = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%.Self.frozen.16f) [concrete = constants.%assoc0.169]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.f4d = name_ref X, %.loc11_45.2 [concrete = constants.%assoc0.169]
// CHECK:STDOUT:     %impl.elem0.loc11_45.1: type = impl_witness_access constants.%.7fc, element0 [symbolic_self = constants.%impl.elem0.412]
// CHECK:STDOUT:     %T.ref.loc11_50: type = name_ref T, %T.loc11_21.1 [symbolic = %T.loc11_21.2 (constants.%T.67d)]
// CHECK:STDOUT:     %rewrite.loc11_48.1 = requirement_rewrite %impl.elem0.loc11_45.1, %T.ref.loc11_50 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc11_45.2: type = impl_witness_access constants.%.286, element0 [symbolic_self = constants.%impl.elem0.f68]
// CHECK:STDOUT:     %rewrite.loc11_48.2 = requirement_rewrite %impl.elem0.loc11_45.2, %T.ref.loc11_50 [concrete]
// CHECK:STDOUT:     %.loc11_39: type = where_expr [symbolic = %Z_where.type (constants.%Z_where.type.26f)] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.type [concrete]
// CHECK:STDOUT:       %rewrite.loc11_48.2 = requirement_rewrite %impl.elem0.loc11_45.2, %T.ref.loc11_50 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc11_23.1: type = splice_block %.loc11_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc11_21: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc11_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_21.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_21.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @T.as.Z.impl.dc2.%T.ref.loc11_29, @Z, @Z(constants.%C) [symbolic = @T.as.Z.impl.dc2.%.loc11_52 (constants.%.c1d)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc13_15.1: %pattern_type.d7e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt.4cc)]
// CHECK:STDOUT:     %t.param_patt.loc13_24.1: @F.%pattern_type (%pattern_type.d37) = value_param_pattern [symbolic = %t.param_patt.loc13_24.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc13_24.1: @F.%pattern_type (%pattern_type.d37) = at_binding_pattern t, %t.param_patt.loc13_24.1 [symbolic = %t.patt.loc13_24.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_20: type = splice_block %Z.type [concrete = constants.%Z.type.963] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %Z.ref: %Z.type.c39 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.963]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_15.2: %Z.type.963 = symbolic_binding T, 0 [symbolic = %T.loc13_15.1 (constants.%T.4b3)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc13_26.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc13_26.1: type = splice_block %.loc13_26.2 [symbolic = %T.as_type.loc13_26.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc13: %Z.type.963 = name_ref T, %T.loc13_15.2 [symbolic = %T.loc13_15.1 (constants.%T.4b3)]
// CHECK:STDOUT:       %T.as_type.loc13_26.2: type = facet_access_type %T.ref.loc13 [symbolic = %T.as_type.loc13_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc13_26.2: type = converted %T.ref.loc13, %T.as_type.loc13_26.2 [symbolic = %T.as_type.loc13_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc13_26.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc3_14.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_14.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc3_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_14.1)> [symbolic = %Z.type (constants.%Z.type.924)]
// CHECK:STDOUT:   %Self.loc3_22.2: @Z.%Z.type (%Z.type.924) = symbolic_binding Self, 1 [symbolic = %Self.loc3_22.2 (constants.%Self.21f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_22.1: @Z.%Z.type (%Z.type.924) = symbolic_binding Self, 1 [symbolic = %Self.loc3_22.2 (constants.%Self.21f)]
// CHECK:STDOUT:     %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.WithSelf.%Z.assoc_type (%Z.assoc_type.496) = assoc_entity element0, @Z.WithSelf.%X [symbolic = @Z.WithSelf.%assoc0 (constants.%assoc0.36e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_22.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (@Z.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.3c5(%T.loc9_15.1: type, %S.loc9_24.1: type) {
// CHECK:STDOUT:   %T.patt.loc9_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc9_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   %S.patt.loc9_24.2: %pattern_type.98f = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc9_24.2 (constants.%S.patt)]
// CHECK:STDOUT:   %S.loc9_24.2: type = symbolic_binding S, 1 [symbolic = %S.loc9_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc9_40.2: type = facet_type <@Z, @Z(%S.loc9_24.2)> [symbolic = %Z.type.loc9_40.2 (constants.%Z.type.78b)]
// CHECK:STDOUT:   %.Self.frozen.loc9_42.2: @T.as.Z.impl.3c5.%Z.type.loc9_40.2 (%Z.type.78b) = symbolic_binding .Self [symbolic = %.Self.frozen.loc9_42.2 (constants.%.Self.frozen.e6b)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z.type.loc9_40.2 [symbolic = %require_complete (constants.%require_complete.291)]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.loc9_42.2 [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.d9d)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc9_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.083)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.3c5.%Z.assoc_type (%Z.assoc_type.083) = assoc_entity element0, @Z.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.c53)]
// CHECK:STDOUT:   %.loc9_48.3: <witness> = impl_self_witness %.Self.frozen.loc9_42.2, @Z, @Z(%S.loc9_24.2) [symbolic = %.loc9_48.3 (constants.%.948)]
// CHECK:STDOUT:   %impl.elem0.loc9_48.3: type = impl_witness_access %.loc9_48.3, element0 [symbolic = %impl.elem0.loc9_48.3 (constants.%impl.elem0.b8f)]
// CHECK:STDOUT:   %.Self: @T.as.Z.impl.3c5.%Z.type.loc9_40.2 (%Z.type.78b) = symbolic_binding .Self [symbolic = %.Self (constants.%.Self.6d3)]
// CHECK:STDOUT:   %.loc9_48.4: <witness> = impl_self_witness %.Self, @Z, @Z(%S.loc9_24.2) [symbolic = %.loc9_48.4 (constants.%.29c)]
// CHECK:STDOUT:   %impl.elem0.loc9_48.4: type = impl_witness_access %.loc9_48.4, element0 [symbolic = %impl.elem0.loc9_48.4 (constants.%impl.elem0.6e9)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc9_24.2) where %impl.elem0.loc9_48.4 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.28d)]
// CHECK:STDOUT:   %.loc9_56: <witness> = impl_self_witness %T.loc9_15.2, @Z, @Z(%S.loc9_24.2) [symbolic = %.loc9_56 (constants.%.18b)]
// CHECK:STDOUT:   %Z.impl_witness.loc9_56.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.3c5(%T.loc9_15.2, %S.loc9_24.2) [symbolic = %Z.impl_witness.loc9_56.2 (constants.%Z.impl_witness.e1a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %Z.type.loc9_40.1 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.3c5 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc9_56.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.3c5(constants.%T.67d, constants.%S) [symbolic = %Z.impl_witness.loc9_56.2 (constants.%Z.impl_witness.e1a)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Z.type.loc9_40.1
// CHECK:STDOUT:     witness = %Z.impl_witness.loc9_56.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic final impl @T.as.Z.impl.dc2(%T.loc11_21.1: type) {
// CHECK:STDOUT:   %T.patt.loc11_21.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_21.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc11_21.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_21.2 (constants.%T.67d)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.f68 = %T.loc11_21.2> [symbolic = %Z_where.type (constants.%Z_where.type.26f)]
// CHECK:STDOUT:   %.loc11_52: <witness> = impl_self_witness %T.loc11_21.2, @Z, @Z(constants.%C) [symbolic = %.loc11_52 (constants.%.c1d)]
// CHECK:STDOUT:   %Z.impl_witness.loc11_52.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.dc2(%T.loc11_21.2) [symbolic = %Z.impl_witness.loc11_52.2 (constants.%Z.impl_witness.12f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   final impl: %T.ref.loc11_29 as %Z.type {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.dc2 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc11_52.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.dc2(constants.%T.67d) [symbolic = %Z.impl_witness.loc11_52.2 (constants.%Z.impl_witness.12f)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%T.67d [symbolic = %T.loc11_21.2 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Z.type
// CHECK:STDOUT:     witness = %Z.impl_witness.loc11_52.1
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
// CHECK:STDOUT: generic fn @F(%T.loc13_15.2: %Z.type.963) {
// CHECK:STDOUT:   %T.patt.loc13_15.2: %pattern_type.d7e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt.4cc)]
// CHECK:STDOUT:   %T.loc13_15.1: %Z.type.963 = symbolic_binding T, 0 [symbolic = %T.loc13_15.1 (constants.%T.4b3)]
// CHECK:STDOUT:   %T.as_type.loc13_26.1: type = facet_access_type %T.loc13_15.1 [symbolic = %T.as_type.loc13_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc13_26.1 [symbolic = %pattern_type (constants.%pattern_type.d37)]
// CHECK:STDOUT:   %t.param_patt.loc13_24.2: @F.%pattern_type (%pattern_type.d37) = value_param_pattern [symbolic = %t.param_patt.loc13_24.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc13_24.2: @F.%pattern_type (%pattern_type.d37) = at_binding_pattern t, %t.param_patt.loc13_24.2 [symbolic = %t.patt.loc13_24.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc13_26.1 [symbolic = %require_complete (constants.%require_complete.df7)]
// CHECK:STDOUT:   %.loc15_18.4: require_specific_def_type = require_specific_def @T.as.Z.impl.dc2(%T.as_type.loc13_26.1) [symbolic = %.loc15_18.4 (constants.%.ac3)]
// CHECK:STDOUT:   %a.patt.loc15_15.2: @F.%pattern_type (%pattern_type.d37) = value_binding_pattern a [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc13_26.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc13_26.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %.loc15_18.1: type = splice_block %impl.elem0 [symbolic = %T.as_type.loc13_26.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc15: %Z.type.963 = name_ref T, %T.loc13_15.2 [symbolic = %T.loc13_15.1 (constants.%T.4b3)]
// CHECK:STDOUT:       %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.as_type.loc13_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_18.2: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.as_type.loc13_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_18.3: %Z.assoc_type.f4d = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%T.4b3) [concrete = constants.%assoc0.169]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.f4d = name_ref X, %.loc15_18.3 [concrete = constants.%assoc0.169]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%Z.impl_witness.4cd, element0 [symbolic = %T.as_type.loc13_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%T.as_type.loc13_26.1 (%T.as_type) = wrapper_binding a, %t.ref
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc15_15.1: @F.%pattern_type (%pattern_type.d37) = value_binding_pattern a [symbolic = %a.patt.loc15_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc3_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T.67d, constants.%Self.21f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.patt.loc3_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.78b
// CHECK:STDOUT:   %Self.loc3_22.2 => constants.%Self.1a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Self.21f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%.Self.frozen.e6b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.3c5(constants.%T.67d, constants.%S) {
// CHECK:STDOUT:   %T.patt.loc9_15.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc9_15.2 => constants.%T.67d
// CHECK:STDOUT:   %S.patt.loc9_24.2 => constants.%S.patt
// CHECK:STDOUT:   %S.loc9_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc9_40.2 => constants.%Z.type.78b
// CHECK:STDOUT:   %.Self.frozen.loc9_42.2 => constants.%.Self.frozen.e6b
// CHECK:STDOUT:   %require_complete => constants.%require_complete.291
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.d9d
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c53
// CHECK:STDOUT:   %.loc9_48.3 => constants.%.948
// CHECK:STDOUT:   %impl.elem0.loc9_48.3 => constants.%impl.elem0.b8f
// CHECK:STDOUT:   %.Self => constants.%.Self.6d3
// CHECK:STDOUT:   %.loc9_48.4 => constants.%.29c
// CHECK:STDOUT:   %impl.elem0.loc9_48.4 => constants.%impl.elem0.6e9
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.28d
// CHECK:STDOUT:   %.loc9_56 => constants.%.18b
// CHECK:STDOUT:   %Z.impl_witness.loc9_56.2 => constants.%Z.impl_witness.e1a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Z.facet.192) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc3_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.963
// CHECK:STDOUT:   %Self.loc3_22.2 => constants.%Self.ae2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Self.21f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.f4d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%.Self.frozen.16f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.f4d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.dc2(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc11_21.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc11_21.2 => constants.%T.67d
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.26f
// CHECK:STDOUT:   %.loc11_52 => constants.%.c1d
// CHECK:STDOUT:   %Z.impl_witness.loc11_52.2 => constants.%Z.impl_witness.12f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Z.facet.fe0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.f4d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.4b3) {
// CHECK:STDOUT:   %T.patt.loc13_15.2 => constants.%T.patt.4cc
// CHECK:STDOUT:   %T.loc13_15.1 => constants.%T.4b3
// CHECK:STDOUT:   %T.as_type.loc13_26.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d37
// CHECK:STDOUT:   %t.param_patt.loc13_24.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc13_24.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%T.4b3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.f4d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.dc2(constants.%T.as_type) {
// CHECK:STDOUT:   %T.patt.loc11_21.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc11_21.2 => constants.%T.as_type
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.147
// CHECK:STDOUT:   %.loc11_52 => constants.%.eef
// CHECK:STDOUT:   %Z.impl_witness.loc11_52.2 => constants.%Z.impl_witness.4cd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_specialized_symbolic_rewrite.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Z.type.c39: type = generic_interface_type @Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Z.generic: %Z.type.c39 = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type.924: type = facet_type <@Z, @Z(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.21f: %Z.type.924 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.496: type = assoc_entity_type @Z, @Z(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.36e: %Z.assoc_type.496 = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %Y.type: type = facet_type <@Y> [concrete]
// CHECK:STDOUT:   %Self.765: %Y.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %S.patt: %pattern_type.98f = symbolic_binding_pattern S, 1 [symbolic]
// CHECK:STDOUT:   %S: type = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Z.type.78b: type = facet_type <@Z, @Z(%S)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.e6b: %Z.type.78b = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Self.1a9: %Z.type.78b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.083: type = assoc_entity_type @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %assoc0.c53: %Z.assoc_type.083 = assoc_entity element0, @Z.WithSelf.%X [symbolic]
// CHECK:STDOUT:   %require_complete.291: <witness> = require_complete_type %Z.type.78b [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.d9d: type = facet_access_type %.Self.frozen.e6b [symbolic]
// CHECK:STDOUT:   %.948: <witness> = impl_self_witness %.Self.frozen.e6b, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b8f: type = impl_witness_access %.948, element0 [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.6d3: %Z.type.78b = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %.29c: <witness> = impl_self_witness %.Self.6d3, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %impl.elem0.6e9: type = impl_witness_access %.29c, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.28d: type = facet_type <@Z, @Z(%S) where %impl.elem0.6e9 = %empty_tuple.type> [symbolic]
// CHECK:STDOUT:   %.18b: <witness> = impl_self_witness %T.67d, @Z, @Z(%S) [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.e1a: <witness> = impl_witness @T.as.Z.impl.3c5.%Z.impl_witness_table, @T.as.Z.impl.3c5(%T.67d, %S) [symbolic]
// CHECK:STDOUT:   %Z.facet.192: %Z.type.78b = facet_value %T.67d, (%Z.impl_witness.e1a) [symbolic]
// CHECK:STDOUT:   %Z.type.963: type = facet_type <@Z, @Z(%C)> [concrete]
// CHECK:STDOUT:   %.Self.frozen.16f: %Z.type.963 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Self.ae2: %Z.type.963 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.assoc_type.f4d: type = assoc_entity_type @Z, @Z(%C) [concrete]
// CHECK:STDOUT:   %assoc0.169: %Z.assoc_type.f4d = assoc_entity element0, @Z.WithSelf.%X [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.16b: type = facet_access_type %.Self.frozen.16f [symbolic_self]
// CHECK:STDOUT:   %.7fc: <witness> = impl_self_witness %.Self.frozen.16f, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.412: type = impl_witness_access %.7fc, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.e3e: %Z.type.963 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.286: <witness> = impl_self_witness %.Self.e3e, @Z, @Z(%C) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f68: type = impl_witness_access %.286, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.26f: type = facet_type <@Z, @Z(%C) where %impl.elem0.f68 = %T.67d> [symbolic]
// CHECK:STDOUT:   %.c1d: <witness> = impl_self_witness %T.67d, @Z, @Z(%C) [symbolic]
// CHECK:STDOUT:   %Z.impl_witness.12f: <witness> = impl_witness @T.as.Z.impl.dc2.%Z.impl_witness_table, @T.as.Z.impl.dc2(%T.67d) [symbolic]
// CHECK:STDOUT:   %Z.facet.fe0: %Z.type.963 = facet_value %T.67d, (%Z.impl_witness.12f) [symbolic]
// CHECK:STDOUT:   %pattern_type.d7e: type = pattern_type %Z.type.963 [concrete]
// CHECK:STDOUT:   %T.patt.4cc: %pattern_type.d7e = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.4b3: %Z.type.963 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.4b3 [symbolic]
// CHECK:STDOUT:   %pattern_type.d37: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.d37 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.d37 = at_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.df7: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.4b3, @Z, @Z(%C) [symbolic]
// CHECK:STDOUT:   %impl.elem0.aab: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %require_complete.d22: <witness> = require_complete_type %impl.elem0.aab [symbolic]
// CHECK:STDOUT:   %pattern_type.dc1: type = pattern_type %impl.elem0.aab [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.dc1 = value_binding_pattern a [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3aa: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.294: %ImplicitAs.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.fcc: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.97e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.945: %ImplicitAs.WithSelf.Convert.type.97e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.034: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.aab)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.aba: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.aab) [symbolic]
// CHECK:STDOUT:   %assoc0.7e2: %ImplicitAs.assoc_type.aba = assoc_entity element0, imports.%Core.import_ref.cb2 [symbolic]
// CHECK:STDOUT:   %require_complete.504: <witness> = require_complete_type %ImplicitAs.type.034 [symbolic]
// CHECK:STDOUT:   %assoc0.0ac: %ImplicitAs.assoc_type.fcc = assoc_entity element0, imports.%Core.import_ref.f9d [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.484: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.fcc) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.0ac)]
// CHECK:STDOUT:   %Core.import_ref.cb2: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.97e) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.945)]
// CHECK:STDOUT:   %Core.import_ref.f9d = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:   %Z.decl: %Z.type.c39 = interface_decl @Z [concrete = constants.%Z.generic] {
// CHECK:STDOUT:     %T.patt.loc3_14.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_14.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_16.1: type = splice_block %.loc3_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc3_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.decl: type = interface_decl @Y [concrete = constants.%Y.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.3c5 [concrete] {
// CHECK:STDOUT:     %T.patt.loc10_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:     %S.patt.loc10_24.1: %pattern_type.98f = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc10_24.2 (constants.%S.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_15.1 [symbolic = %T.loc10_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.c39 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %S.ref: type = name_ref S, %S.loc10_24.1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:     %Z.type.loc10_40.1: type = facet_type <@Z, @Z(constants.%S)> [symbolic = %Z.type.loc10_40.2 (constants.%Z.type.78b)]
// CHECK:STDOUT:     %.Self.frozen.loc10_42.1: @T.as.Z.impl.3c5.%Z.type.loc10_40.2 (%Z.type.78b) = symbolic_binding .Self [symbolic = %.Self.frozen.loc10_42.2 (constants.%.Self.frozen.e6b)]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Z.type.loc10_40.1 [concrete]
// CHECK:STDOUT:     %.Self.ref: @T.as.Z.impl.3c5.%Z.type.loc10_40.2 (%Z.type.78b) = name_ref .Self, %.Self.frozen.loc10_42.1 [symbolic = %.Self.frozen.loc10_42.2 (constants.%.Self.frozen.e6b)]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.d9d)]
// CHECK:STDOUT:     %.loc10_48.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.d9d)]
// CHECK:STDOUT:     %.loc10_48.2: @T.as.Z.impl.3c5.%Z.assoc_type (%Z.assoc_type.083) = specific_constant @X.%assoc0, @Z.WithSelf(constants.%S, constants.%.Self.frozen.e6b) [symbolic = %assoc0 (constants.%assoc0.c53)]
// CHECK:STDOUT:     %X.ref: @T.as.Z.impl.3c5.%Z.assoc_type (%Z.assoc_type.083) = name_ref X, %.loc10_48.2 [symbolic = %assoc0 (constants.%assoc0.c53)]
// CHECK:STDOUT:     %impl.elem0.loc10_48.1: type = impl_witness_access constants.%.948, element0 [symbolic = %impl.elem0.loc10_48.3 (constants.%impl.elem0.b8f)]
// CHECK:STDOUT:     %.loc10_54.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_54.2: type = converted %.loc10_54.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %rewrite.loc10_51.1 = requirement_rewrite %impl.elem0.loc10_48.1, %.loc10_54.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc10_48.2: type = impl_witness_access constants.%.29c, element0 [symbolic = %impl.elem0.loc10_48.4 (constants.%impl.elem0.6e9)]
// CHECK:STDOUT:     %rewrite.loc10_51.2 = requirement_rewrite %impl.elem0.loc10_48.2, %.loc10_54.2 [concrete]
// CHECK:STDOUT:     %.loc10_42: type = where_expr [symbolic = %Z_where.type (constants.%Z_where.type.28d)] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.type.loc10_40.1 [concrete]
// CHECK:STDOUT:       %rewrite.loc10_51.2 = requirement_rewrite %impl.elem0.loc10_48.2, %.loc10_54.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10_17.1: type = splice_block %.loc10_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc10_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc10_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc10_26.1: type = splice_block %.loc10_26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc10_24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc10_26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc10_24.1: type = symbolic_binding S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @T.as.Z.impl.3c5.%T.ref, @Z, @Z(constants.%S) [symbolic = @T.as.Z.impl.3c5.%.loc10_56 (constants.%.18b)]
// CHECK:STDOUT:   impl_decl @T.as.Z.impl.dc2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc12_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12_23: type = name_ref T, %T.loc12_15.1 [symbolic = %T.loc12_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %Z.ref: %Z.type.c39 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.963]
// CHECK:STDOUT:     %.Self.frozen.loc12_33: %Z.type.963 = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.16f]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Z.type [concrete]
// CHECK:STDOUT:     %.Self.ref: %Z.type.963 = name_ref .Self, %.Self.frozen.loc12_33 [symbolic_self = constants.%.Self.frozen.16f]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type.16b]
// CHECK:STDOUT:     %.loc12_39.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type.16b]
// CHECK:STDOUT:     %.loc12_39.2: %Z.assoc_type.f4d = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%.Self.frozen.16f) [concrete = constants.%assoc0.169]
// CHECK:STDOUT:     %X.ref: %Z.assoc_type.f4d = name_ref X, %.loc12_39.2 [concrete = constants.%assoc0.169]
// CHECK:STDOUT:     %impl.elem0.loc12_39.1: type = impl_witness_access constants.%.7fc, element0 [symbolic_self = constants.%impl.elem0.412]
// CHECK:STDOUT:     %T.ref.loc12_44: type = name_ref T, %T.loc12_15.1 [symbolic = %T.loc12_15.2 (constants.%T.67d)]
// CHECK:STDOUT:     %rewrite.loc12_42.1 = requirement_rewrite %impl.elem0.loc12_39.1, %T.ref.loc12_44 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc12_39.2: type = impl_witness_access constants.%.286, element0 [symbolic_self = constants.%impl.elem0.f68]
// CHECK:STDOUT:     %rewrite.loc12_42.2 = requirement_rewrite %impl.elem0.loc12_39.2, %T.ref.loc12_44 [concrete]
// CHECK:STDOUT:     %.loc12_33: type = where_expr [symbolic = %Z_where.type (constants.%Z_where.type.26f)] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.type [concrete]
// CHECK:STDOUT:       %rewrite.loc12_42.2 = requirement_rewrite %impl.elem0.loc12_39.2, %T.ref.loc12_44 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc12_17.1: type = splice_block %.loc12_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc12_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc12_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = impl_self_witness @T.as.Z.impl.dc2.%T.ref.loc12_23, @Z, @Z(constants.%C) [symbolic = @T.as.Z.impl.dc2.%.loc12_46 (constants.%.c1d)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc14_15.1: %pattern_type.d7e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt.4cc)]
// CHECK:STDOUT:     %t.param_patt.loc14_24.1: @F.%pattern_type.loc14 (%pattern_type.d37) = value_param_pattern [symbolic = %t.param_patt.loc14_24.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc14_24.1: @F.%pattern_type.loc14 (%pattern_type.d37) = at_binding_pattern t, %t.param_patt.loc14_24.1 [symbolic = %t.patt.loc14_24.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_20: type = splice_block %Z.type [concrete = constants.%Z.type.963] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %Z.ref: %Z.type.c39 = name_ref Z, file.%Z.decl [concrete = constants.%Z.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Z.type: type = facet_type <@Z, @Z(constants.%C)> [concrete = constants.%Z.type.963]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_15.2: %Z.type.963 = symbolic_binding T, 0 [symbolic = %T.loc14_15.1 (constants.%T.4b3)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc14_26.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_26.1: type = splice_block %.loc14_26.2 [symbolic = %T.as_type.loc14_26.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc14: %Z.type.963 = name_ref T, %T.loc14_15.2 [symbolic = %T.loc14_15.1 (constants.%T.4b3)]
// CHECK:STDOUT:       %T.as_type.loc14_26.2: type = facet_access_type %T.ref.loc14 [symbolic = %T.as_type.loc14_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc14_26.2: type = converted %T.ref.loc14, %T.as_type.loc14_26.2 [symbolic = %T.as_type.loc14_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc14_26.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Z(%T.loc3_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc3_14.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_14.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc3_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z, @Z(%T.loc3_14.1)> [symbolic = %Z.type (constants.%Z.type.924)]
// CHECK:STDOUT:   %Self.loc3_22.2: @Z.%Z.type (%Z.type.924) = symbolic_binding Self, 1 [symbolic = %Self.loc3_22.2 (constants.%Self.21f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_22.1: @Z.%Z.type (%Z.type.924) = symbolic_binding Self, 1 [symbolic = %Self.loc3_22.2 (constants.%Self.21f)]
// CHECK:STDOUT:     %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:       %assoc0: @Z.WithSelf.%Z.assoc_type (%Z.assoc_type.496) = assoc_entity element0, @Z.WithSelf.%X [symbolic = @Z.WithSelf.%assoc0 (constants.%assoc0.36e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_22.1
// CHECK:STDOUT:     .X = @X.%assoc0
// CHECK:STDOUT:     witness = (@Z.WithSelf.%X)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Y {
// CHECK:STDOUT:   %Self: %Y.type = symbolic_binding Self, 0 [symbolic = constants.%Self.765]
// CHECK:STDOUT:   %Y.WithSelf.decl = interface_with_self_decl @Y [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.3c5(%T.loc10_15.1: type, %S.loc10_24.1: type) {
// CHECK:STDOUT:   %T.patt.loc10_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc10_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   %S.patt.loc10_24.2: %pattern_type.98f = symbolic_binding_pattern S, 1 [symbolic = %S.patt.loc10_24.2 (constants.%S.patt)]
// CHECK:STDOUT:   %S.loc10_24.2: type = symbolic_binding S, 1 [symbolic = %S.loc10_24.2 (constants.%S)]
// CHECK:STDOUT:   %Z.type.loc10_40.2: type = facet_type <@Z, @Z(%S.loc10_24.2)> [symbolic = %Z.type.loc10_40.2 (constants.%Z.type.78b)]
// CHECK:STDOUT:   %.Self.frozen.loc10_42.2: @T.as.Z.impl.3c5.%Z.type.loc10_40.2 (%Z.type.78b) = symbolic_binding .Self [symbolic = %.Self.frozen.loc10_42.2 (constants.%.Self.frozen.e6b)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Z.type.loc10_40.2 [symbolic = %require_complete (constants.%require_complete.291)]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.loc10_42.2 [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.d9d)]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z, @Z(%S.loc10_24.2) [symbolic = %Z.assoc_type (constants.%Z.assoc_type.083)]
// CHECK:STDOUT:   %assoc0: @T.as.Z.impl.3c5.%Z.assoc_type (%Z.assoc_type.083) = assoc_entity element0, @Z.WithSelf.%X [symbolic = %assoc0 (constants.%assoc0.c53)]
// CHECK:STDOUT:   %.loc10_48.3: <witness> = impl_self_witness %.Self.frozen.loc10_42.2, @Z, @Z(%S.loc10_24.2) [symbolic = %.loc10_48.3 (constants.%.948)]
// CHECK:STDOUT:   %impl.elem0.loc10_48.3: type = impl_witness_access %.loc10_48.3, element0 [symbolic = %impl.elem0.loc10_48.3 (constants.%impl.elem0.b8f)]
// CHECK:STDOUT:   %.Self: @T.as.Z.impl.3c5.%Z.type.loc10_40.2 (%Z.type.78b) = symbolic_binding .Self [symbolic = %.Self (constants.%.Self.6d3)]
// CHECK:STDOUT:   %.loc10_48.4: <witness> = impl_self_witness %.Self, @Z, @Z(%S.loc10_24.2) [symbolic = %.loc10_48.4 (constants.%.29c)]
// CHECK:STDOUT:   %impl.elem0.loc10_48.4: type = impl_witness_access %.loc10_48.4, element0 [symbolic = %impl.elem0.loc10_48.4 (constants.%impl.elem0.6e9)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(%S.loc10_24.2) where %impl.elem0.loc10_48.4 = constants.%empty_tuple.type> [symbolic = %Z_where.type (constants.%Z_where.type.28d)]
// CHECK:STDOUT:   %.loc10_56: <witness> = impl_self_witness %T.loc10_15.2, @Z, @Z(%S.loc10_24.2) [symbolic = %.loc10_56 (constants.%.18b)]
// CHECK:STDOUT:   %Z.impl_witness.loc10_56.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.3c5(%T.loc10_15.2, %S.loc10_24.2) [symbolic = %Z.impl_witness.loc10_56.2 (constants.%Z.impl_witness.e1a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %Z.type.loc10_40.1 {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.3c5 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc10_56.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.3c5(constants.%T.67d, constants.%S) [symbolic = %Z.impl_witness.loc10_56.2 (constants.%Z.impl_witness.e1a)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Z.type.loc10_40.1
// CHECK:STDOUT:     witness = %Z.impl_witness.loc10_56.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.Z.impl.dc2(%T.loc12_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc12_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc12_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T.67d)]
// CHECK:STDOUT:   %Z_where.type: type = facet_type <@Z, @Z(constants.%C) where constants.%impl.elem0.f68 = %T.loc12_15.2> [symbolic = %Z_where.type (constants.%Z_where.type.26f)]
// CHECK:STDOUT:   %.loc12_46: <witness> = impl_self_witness %T.loc12_15.2, @Z, @Z(constants.%C) [symbolic = %.loc12_46 (constants.%.c1d)]
// CHECK:STDOUT:   %Z.impl_witness.loc12_46.2: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.dc2(%T.loc12_15.2) [symbolic = %Z.impl_witness.loc12_46.2 (constants.%Z.impl_witness.12f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref.loc12_23 as %Z.type {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @T.as.Z.impl.dc2 [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc12_46.1: <witness> = impl_witness %Z.impl_witness_table, @T.as.Z.impl.dc2(constants.%T.67d) [symbolic = %Z.impl_witness.loc12_46.2 (constants.%Z.impl_witness.12f)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%T.67d [symbolic = %T.loc12_15.2 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Z.type
// CHECK:STDOUT:     witness = %Z.impl_witness.loc12_46.1
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
// CHECK:STDOUT: generic fn @F(%T.loc14_15.2: %Z.type.963) {
// CHECK:STDOUT:   %T.patt.loc14_15.2: %pattern_type.d7e = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt.4cc)]
// CHECK:STDOUT:   %T.loc14_15.1: %Z.type.963 = symbolic_binding T, 0 [symbolic = %T.loc14_15.1 (constants.%T.4b3)]
// CHECK:STDOUT:   %T.as_type.loc14_26.1: type = facet_access_type %T.loc14_15.1 [symbolic = %T.as_type.loc14_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc14: type = pattern_type %T.as_type.loc14_26.1 [symbolic = %pattern_type.loc14 (constants.%pattern_type.d37)]
// CHECK:STDOUT:   %t.param_patt.loc14_24.2: @F.%pattern_type.loc14 (%pattern_type.d37) = value_param_pattern [symbolic = %t.param_patt.loc14_24.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc14_24.2: @F.%pattern_type.loc14 (%pattern_type.d37) = at_binding_pattern t, %t.param_patt.loc14_24.2 [symbolic = %t.patt.loc14_24.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.as_type.loc14_26.1 [symbolic = %require_complete.loc14 (constants.%require_complete.df7)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc14_15.1, @Z, @Z(constants.%C) [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc22_18.2: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_18.2 (constants.%impl.elem0.aab)]
// CHECK:STDOUT:   %require_complete.loc22_18: <witness> = require_complete_type %impl.elem0.loc22_18.2 [symbolic = %require_complete.loc22_18 (constants.%require_complete.d22)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %impl.elem0.loc22_18.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.dc1)]
// CHECK:STDOUT:   %a.patt.loc22_15.2: @F.%pattern_type.loc22 (%pattern_type.dc1) = value_binding_pattern a [symbolic = %a.patt.loc22_15.2 (constants.%a.patt)]
// CHECK:STDOUT:   %ImplicitAs.type.loc22_23.2: type = facet_type <@ImplicitAs, @ImplicitAs(%impl.elem0.loc22_18.2)> [symbolic = %ImplicitAs.type.loc22_23.2 (constants.%ImplicitAs.type.034)]
// CHECK:STDOUT:   %require_complete.loc22_23: <witness> = require_complete_type %ImplicitAs.type.loc22_23.2 [symbolic = %require_complete.loc22_23 (constants.%require_complete.504)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%impl.elem0.loc22_18.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.aba)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.aba) = assoc_entity element0, imports.%Core.import_ref.cb2 [symbolic = %assoc0 (constants.%assoc0.7e2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc14_26.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @F.%T.as_type.loc14_26.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %ImplicitAs.type.loc22_23.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%impl.elem0.aab)> [symbolic = %ImplicitAs.type.loc22_23.2 (constants.%ImplicitAs.type.034)]
// CHECK:STDOUT:     %.loc22_23.1: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.aba) = specific_constant imports.%Core.import_ref.484, @ImplicitAs.WithSelf(constants.%impl.elem0.aab, constants.%Self.294) [symbolic = %assoc0 (constants.%assoc0.7e2)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.aba) = name_ref Convert, %.loc22_23.1 [symbolic = %assoc0 (constants.%assoc0.7e2)]
// CHECK:STDOUT:     %.loc22_23.2: @F.%impl.elem0.loc22_18.2 (%impl.elem0.aab) = converted %t.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc22_18.1: type = splice_block %impl.elem0.loc22_18.1 [symbolic = %impl.elem0.loc22_18.2 (constants.%impl.elem0.aab)] {
// CHECK:STDOUT:       %T.ref.loc22: %Z.type.963 = name_ref T, %T.loc14_15.2 [symbolic = %T.loc14_15.1 (constants.%T.4b3)]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.as_type.loc14_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_18.2: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.as_type.loc14_26.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc22_18.3: %Z.assoc_type.f4d = specific_constant @X.%assoc0, @Z.WithSelf(constants.%C, constants.%T.4b3) [concrete = constants.%assoc0.169]
// CHECK:STDOUT:       %X.ref: %Z.assoc_type.f4d = name_ref X, %.loc22_18.3 [concrete = constants.%assoc0.169]
// CHECK:STDOUT:       %impl.elem0.loc22_18.1: type = impl_witness_access constants.%Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_18.2 (constants.%impl.elem0.aab)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%impl.elem0.loc22_18.2 (%impl.elem0.aab) = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc22_15.1: @F.%pattern_type.loc22 (%pattern_type.dc1) = value_binding_pattern a [symbolic = %a.patt.loc22_15.2 (constants.%a.patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc3_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%T.67d, constants.%Self.21f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y.WithSelf(constants.%Self.765) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%S) {
// CHECK:STDOUT:   %T.patt.loc3_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.78b
// CHECK:STDOUT:   %Self.loc3_22.2 => constants.%Self.1a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Self.21f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%.Self.frozen.e6b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.3c5(constants.%T.67d, constants.%S) {
// CHECK:STDOUT:   %T.patt.loc10_15.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc10_15.2 => constants.%T.67d
// CHECK:STDOUT:   %S.patt.loc10_24.2 => constants.%S.patt
// CHECK:STDOUT:   %S.loc10_24.2 => constants.%S
// CHECK:STDOUT:   %Z.type.loc10_40.2 => constants.%Z.type.78b
// CHECK:STDOUT:   %.Self.frozen.loc10_42.2 => constants.%.Self.frozen.e6b
// CHECK:STDOUT:   %require_complete => constants.%require_complete.291
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.d9d
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c53
// CHECK:STDOUT:   %.loc10_48.3 => constants.%.948
// CHECK:STDOUT:   %impl.elem0.loc10_48.3 => constants.%impl.elem0.b8f
// CHECK:STDOUT:   %.Self => constants.%.Self.6d3
// CHECK:STDOUT:   %.loc10_48.4 => constants.%.29c
// CHECK:STDOUT:   %impl.elem0.loc10_48.4 => constants.%impl.elem0.6e9
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.28d
// CHECK:STDOUT:   %.loc10_56 => constants.%.18b
// CHECK:STDOUT:   %Z.impl_witness.loc10_56.2 => constants.%Z.impl_witness.e1a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%S, constants.%Z.facet.192) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc3_14.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.type => constants.%Z.type.963
// CHECK:STDOUT:   %Self.loc3_22.2 => constants.%Self.ae2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Self.21f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.f4d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%.Self.frozen.16f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.f4d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.Z.impl.dc2(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc12_15.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc12_15.2 => constants.%T.67d
// CHECK:STDOUT:   %Z_where.type => constants.%Z_where.type.26f
// CHECK:STDOUT:   %.loc12_46 => constants.%.c1d
// CHECK:STDOUT:   %Z.impl_witness.loc12_46.2 => constants.%Z.impl_witness.12f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%Z.facet.fe0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.f4d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.4b3) {
// CHECK:STDOUT:   %T.patt.loc14_15.2 => constants.%T.patt.4cc
// CHECK:STDOUT:   %T.loc14_15.1 => constants.%T.4b3
// CHECK:STDOUT:   %T.as_type.loc14_26.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc14 => constants.%pattern_type.d37
// CHECK:STDOUT:   %t.param_patt.loc14_24.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc14_24.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%C, constants.%T.4b3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %Z.assoc_type => constants.%Z.assoc_type.f4d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_impl_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.99f: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.b1a: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt.d47: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.073: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.073 [symbolic_self]
// CHECK:STDOUT:   %.c6b: <witness> = impl_self_witness %.Self.frozen.073, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.949: type = impl_witness_access %.c6b, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %.Self: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.51f: <witness> = impl_self_witness %.Self, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.406: type = impl_witness_access %.51f, element0 [symbolic_self]
// CHECK:STDOUT:   %Ptr_where.type.774fa9.1: type = facet_type <@Ptr where %impl.elem0.406 = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %.f8fe26.1: <witness> = impl_self_witness %U.67d, @Ptr [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.548f1d.1: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %Ptr.facet.a0fe31.1: %Ptr.type = facet_value %U.67d, (%Ptr.impl_witness.548f1d.1) [symbolic]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.51d = ref_binding_pattern t [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.51d = var_param_pattern %t.patt [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.774fa9.2: type = facet_type <@Ptr where %impl.elem0.406 = %ptr.e8f8f9.2> [symbolic]
// CHECK:STDOUT:   %.f8fe26.2: <witness> = impl_self_witness %T.67d, @Ptr [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.548f1d.2: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %.ed1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Ptr.facet.a0fe31.2: %Ptr.type = facet_value %T.67d, (%Ptr.impl_witness.548f1d.2) [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %return.param_patt.27f: %pattern_type.4f4 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.d67: %pattern_type.4f4 = return_slot_pattern %return.param_patt.27f, %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr.e8f8f9.2, @Copy [symbolic]
// CHECK:STDOUT:   %.0e9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f8f9.2, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.patt.loc7_21.1: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_21.2 (constants.%U.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_29: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.frozen.loc7_38: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.073]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Ptr.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.frozen.loc7_38 [symbolic_self = constants.%.Self.frozen.073]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc7_44: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.b1a]
// CHECK:STDOUT:     %impl.elem0.loc7_44.1: type = impl_witness_access constants.%.c6b, element0 [symbolic_self = constants.%impl.elem0.949]
// CHECK:STDOUT:     %U.ref.loc7_52: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_53.1: type = ptr_type %U.ref.loc7_52 [symbolic = %ptr.loc7_53.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %rewrite.loc7_50.1 = requirement_rewrite %impl.elem0.loc7_44.1, %ptr.loc7_53.1 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc7_44.2: type = impl_witness_access constants.%.51f, element0 [symbolic_self = constants.%impl.elem0.406]
// CHECK:STDOUT:     %rewrite.loc7_50.2 = requirement_rewrite %impl.elem0.loc7_44.2, %ptr.loc7_53.1 [concrete]
// CHECK:STDOUT:     %.loc7_38: type = where_expr [symbolic = %Ptr_where.type (constants.%Ptr_where.type.774fa9.1)] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Ptr.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc7_50.2 = requirement_rewrite %impl.elem0.loc7_44.2, %ptr.loc7_53.1 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_23.1: type = splice_block %.loc7_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc7_21: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc7_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_21.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @U.as.Ptr.impl.%U.ref.loc7_29, @Ptr [symbolic = @U.as.Ptr.impl.%.loc7_55 (constants.%.f8fe26.1)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_7.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:     %t.patt.loc9_20.1: @F.%pattern_type.loc9_20 (%pattern_type.51d) = ref_binding_pattern t [symbolic = %t.patt.loc9_20.2 (constants.%t.patt)]
// CHECK:STDOUT:     %t.param_patt.loc9_15.1: @F.%pattern_type.loc9_20 (%pattern_type.51d) = var_param_pattern %t.patt.loc9_20.1 [symbolic = %t.param_patt.loc9_15.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %return.param_patt.loc9_29.1: @F.%pattern_type.loc9_29 (%pattern_type.4f4) = out_param_pattern [symbolic = %return.param_patt.loc9_29.2 (constants.%return.param_patt.27f)]
// CHECK:STDOUT:     %return.patt.loc9_25.1: @F.%pattern_type.loc9_29 (%pattern_type.4f4) = return_slot_pattern %return.param_patt.loc9_29.1, %impl.elem0.loc9 [symbolic = %return.patt.loc9_25.2 (constants.%return.patt.d67)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_28: type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.b1a]
// CHECK:STDOUT:     %Ptr.facet.loc9_29.2: %Ptr.type = facet_value %T.ref.loc9_28, (constants.%Ptr.impl_witness.548f1d.2) [symbolic = %Ptr.facet.loc9_29.1 (constants.%Ptr.facet.a0fe31.2)]
// CHECK:STDOUT:     %.loc9_29.3: %Ptr.type = converted %T.ref.loc9_28, %Ptr.facet.loc9_29.2 [symbolic = %Ptr.facet.loc9_29.1 (constants.%Ptr.facet.a0fe31.2)]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.548f1d.2, element0 [symbolic = %ptr (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:     %.loc9_29.4: Core.Form = init_form %impl.elem0.loc9 [symbolic = %.loc9_29.2 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc9_9.1: type = splice_block %.loc9_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc9_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %t.param: ref @F.%T.loc9_7.1 (%T.67d) = ref_param call_param0
// CHECK:STDOUT:     %T.ref.loc9_22: type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %t: ref @F.%T.loc9_7.1 (%T.67d) = wrapper_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.e8f8f9.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.e8f8f9.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.99f]
// CHECK:STDOUT:   %Ptr.WithSelf.decl = interface_with_self_decl @Ptr [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete = constants.%assoc0.b1a]
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
// CHECK:STDOUT: generic final impl @U.as.Ptr.impl(%U.loc7_21.1: type) {
// CHECK:STDOUT:   %U.patt.loc7_21.2: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_21.2 (constants.%U.patt.d47)]
// CHECK:STDOUT:   %U.loc7_21.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc7_53.2: type = ptr_type %U.loc7_21.2 [symbolic = %ptr.loc7_53.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.406 = %ptr.loc7_53.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.774fa9.1)]
// CHECK:STDOUT:   %.loc7_55: <witness> = impl_self_witness %U.loc7_21.2, @Ptr [symbolic = %.loc7_55 (constants.%.f8fe26.1)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_55.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_21.2) [symbolic = %Ptr.impl_witness.loc7_55.2 (constants.%Ptr.impl_witness.548f1d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   final impl: %U.ref.loc7_29 as %Ptr.ref {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc7_55.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc7_55.2 (constants.%Ptr.impl_witness.548f1d.1)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc7_53.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Ptr.ref
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc7_55.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_7.2: type) {
// CHECK:STDOUT:   %T.patt.loc9_7.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc9_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.67d)]
// CHECK:STDOUT:   %pattern_type.loc9_20: type = pattern_type %T.loc9_7.1 [symbolic = %pattern_type.loc9_20 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %t.patt.loc9_20.2: @F.%pattern_type.loc9_20 (%pattern_type.51d) = ref_binding_pattern t [symbolic = %t.patt.loc9_20.2 (constants.%t.patt)]
// CHECK:STDOUT:   %t.param_patt.loc9_15.2: @F.%pattern_type.loc9_20 (%pattern_type.51d) = var_param_pattern %t.patt.loc9_20.2 [symbolic = %t.param_patt.loc9_15.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %.loc9_29.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.loc9_7.1) [symbolic = %.loc9_29.1 (constants.%.ed1)]
// CHECK:STDOUT:   %Ptr.impl_witness: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.loc9_7.1) [symbolic = %Ptr.impl_witness (constants.%Ptr.impl_witness.548f1d.2)]
// CHECK:STDOUT:   %Ptr.facet.loc9_29.1: %Ptr.type = facet_value %T.loc9_7.1, (%Ptr.impl_witness) [symbolic = %Ptr.facet.loc9_29.1 (constants.%Ptr.facet.a0fe31.2)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc9_7.1 [symbolic = %ptr (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %.loc9_29.2: Core.Form = init_form %ptr [symbolic = %.loc9_29.2 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type.loc9_29: type = pattern_type %ptr [symbolic = %pattern_type.loc9_29 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:   %return.param_patt.loc9_29.2: @F.%pattern_type.loc9_29 (%pattern_type.4f4) = out_param_pattern [symbolic = %return.param_patt.loc9_29.2 (constants.%return.param_patt.27f)]
// CHECK:STDOUT:   %return.patt.loc9_25.2: @F.%pattern_type.loc9_29 (%pattern_type.4f4) = return_slot_pattern %return.param_patt.loc9_29.2, %ptr [symbolic = %return.patt.loc9_25.2 (constants.%return.patt.d67)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_15: <witness> = require_complete_type %T.loc9_7.1 [symbolic = %require_complete.loc9_15 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc9_25: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_25 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %.loc10_10.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc9_7.1) [symbolic = %.loc10_10.3 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc10_10.3: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc10_10.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.884)]
// CHECK:STDOUT:   %.loc10_10.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc10_10.3 [symbolic = %.loc10_10.4 (constants.%.be5)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.4 (%.be5) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.WithSelf.Op(%Copy.facet.loc10_10.3) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.loc9_7.1 (%T.67d)) -> out %return.param: @F.%ptr (%ptr.e8f8f9.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc9_7.1 (%T.67d) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.e8f8f9.2) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.4 (%.be5) = impl_witness_access constants.%Copy.lookup_impl_witness.1da, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %Copy.facet.loc10_10.1: %Copy.type = facet_value constants.%ptr.e8f8f9.2, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_10.1: %Copy.type = converted constants.%ptr.e8f8f9.2, %Copy.facet.loc10_10.1 [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %Copy.facet.loc10_10.2: %Copy.type = facet_value constants.%ptr.e8f8f9.2, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_10.2: %Copy.type = converted constants.%ptr.e8f8f9.2, %Copy.facet.loc10_10.2 [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr (%ptr.e8f8f9.2) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Self.99f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%.Self.frozen.073) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.patt.loc7_21.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_53.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.774fa9.1
// CHECK:STDOUT:   %.loc7_55 => constants.%.f8fe26.1
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_55.2 => constants.%Ptr.impl_witness.548f1d.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet.a0fe31.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.67d) {
// CHECK:STDOUT:   %U.patt.loc7_21.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc7_53.2 => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.774fa9.2
// CHECK:STDOUT:   %.loc7_55 => constants.%.f8fe26.2
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_55.2 => constants.%Ptr.impl_witness.548f1d.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet.a0fe31.2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc9_7.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T.67d
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.51d
// CHECK:STDOUT:   %t.patt.loc9_20.2 => constants.%t.patt
// CHECK:STDOUT:   %t.param_patt.loc9_15.2 => constants.%t.param_patt
// CHECK:STDOUT:   %.loc9_29.1 => constants.%.ed1
// CHECK:STDOUT:   %Ptr.impl_witness => constants.%Ptr.impl_witness.548f1d.2
// CHECK:STDOUT:   %Ptr.facet.loc9_29.1 => constants.%Ptr.facet.a0fe31.2
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %.loc9_29.2 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type.loc9_29 => constants.%pattern_type.4f4
// CHECK:STDOUT:   %return.param_patt.loc9_29.2 => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt.loc9_25.2 => constants.%return.patt.d67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.99f: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.b1a: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt.d47: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.073: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.073 [symbolic_self]
// CHECK:STDOUT:   %.c6b: <witness> = impl_self_witness %.Self.frozen.073, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.949: type = impl_witness_access %.c6b, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %.Self: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.51f: <witness> = impl_self_witness %.Self, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.406: type = impl_witness_access %.51f, element0 [symbolic_self]
// CHECK:STDOUT:   %Ptr_where.type.774: type = facet_type <@Ptr where %impl.elem0.406 = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %.f8f: <witness> = impl_self_witness %U.67d, @Ptr [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.548: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %Ptr.facet: %Ptr.type = facet_value %U.67d, (%Ptr.impl_witness.548) [symbolic]
// CHECK:STDOUT:   %pattern_type.f5d: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.patt.a2e: %pattern_type.f5d = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.76f: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.468: type = facet_access_type %T.76f [symbolic]
// CHECK:STDOUT:   %pattern_type.756: type = pattern_type %T.as_type.468 [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.756 = ref_binding_pattern t [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.756 = var_param_pattern %t.patt [symbolic]
// CHECK:STDOUT:   %ptr.ac4: type = ptr_type %T.as_type.468 [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.c27: type = facet_type <@Ptr where %impl.elem0.406 = %ptr.ac4> [symbolic]
// CHECK:STDOUT:   %.47c: <witness> = impl_self_witness %T.as_type.468, @Ptr [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.8bb: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.as_type.468) [symbolic]
// CHECK:STDOUT:   %.166: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.as_type.468) [symbolic]
// CHECK:STDOUT:   %.e16: Core.Form = init_form %ptr.ac4 [symbolic]
// CHECK:STDOUT:   %pattern_type.f62: type = pattern_type %ptr.ac4 [symbolic]
// CHECK:STDOUT:   %return.param_patt.b93: %pattern_type.f62 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.058: %pattern_type.f62 = return_slot_pattern %return.param_patt.b93, %ptr.ac4 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.35c: <witness> = require_complete_type %T.as_type.468 [symbolic]
// CHECK:STDOUT:   %require_complete.841: <witness> = require_complete_type %ptr.ac4 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d90: <witness> = lookup_impl_witness %ptr.ac4, @Copy [symbolic]
// CHECK:STDOUT:   %.ee9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.468) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.ac4, (%Copy.lookup_impl_witness.d90) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a39: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.229: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a39, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1f8: %.229 = impl_witness_access %Copy.lookup_impl_witness.d90, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d80: <specific function> = specific_impl_function %impl.elem0.1f8, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.patt.loc7_21.1: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_21.2 (constants.%U.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_29: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.frozen.loc7_38: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.073]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Ptr.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.frozen.loc7_38 [symbolic_self = constants.%.Self.frozen.073]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc7_44: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.b1a]
// CHECK:STDOUT:     %impl.elem0.loc7_44.1: type = impl_witness_access constants.%.c6b, element0 [symbolic_self = constants.%impl.elem0.949]
// CHECK:STDOUT:     %U.ref.loc7_52: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_53.1: type = ptr_type %U.ref.loc7_52 [symbolic = %ptr.loc7_53.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %rewrite.loc7_50.1 = requirement_rewrite %impl.elem0.loc7_44.1, %ptr.loc7_53.1 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc7_44.2: type = impl_witness_access constants.%.51f, element0 [symbolic_self = constants.%impl.elem0.406]
// CHECK:STDOUT:     %rewrite.loc7_50.2 = requirement_rewrite %impl.elem0.loc7_44.2, %ptr.loc7_53.1 [concrete]
// CHECK:STDOUT:     %.loc7_38: type = where_expr [symbolic = %Ptr_where.type (constants.%Ptr_where.type.774)] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Ptr.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc7_50.2 = requirement_rewrite %impl.elem0.loc7_44.2, %ptr.loc7_53.1 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_23.1: type = splice_block %.loc7_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc7_21: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc7_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_21.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @U.as.Ptr.impl.%U.ref.loc7_29, @Ptr [symbolic = @U.as.Ptr.impl.%.loc7_55 (constants.%.f8f)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_7.1: %pattern_type.f5d = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.a2e)]
// CHECK:STDOUT:     %t.patt.loc9_19.1: @F.%pattern_type.loc9_19 (%pattern_type.756) = ref_binding_pattern t [symbolic = %t.patt.loc9_19.2 (constants.%t.patt)]
// CHECK:STDOUT:     %t.param_patt.loc9_14.1: @F.%pattern_type.loc9_19 (%pattern_type.756) = var_param_pattern %t.patt.loc9_19.1 [symbolic = %t.param_patt.loc9_14.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %return.param_patt.loc9_28.1: @F.%pattern_type.loc9_28 (%pattern_type.f62) = out_param_pattern [symbolic = %return.param_patt.loc9_28.2 (constants.%return.param_patt.b93)]
// CHECK:STDOUT:     %return.patt.loc9_24.1: @F.%pattern_type.loc9_28 (%pattern_type.f62) = return_slot_pattern %return.param_patt.loc9_28.1, %impl.elem0.loc9 [symbolic = %return.patt.loc9_24.2 (constants.%return.patt.058)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_27: %Ptr.type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.76f)]
// CHECK:STDOUT:     %T.as_type.loc9_28: type = facet_access_type %T.ref.loc9_27 [symbolic = %T.as_type.loc9_21.1 (constants.%T.as_type.468)]
// CHECK:STDOUT:     %.loc9_28.3: type = converted %T.ref.loc9_27, %T.as_type.loc9_28 [symbolic = %T.as_type.loc9_21.1 (constants.%T.as_type.468)]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.b1a]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.8bb, element0 [symbolic = %ptr (constants.%ptr.ac4)]
// CHECK:STDOUT:     %.loc9_28.4: Core.Form = init_form %impl.elem0.loc9 [symbolic = %.loc9_28.2 (constants.%.e16)]
// CHECK:STDOUT:     %.loc9_9: type = splice_block %Ptr.ref [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_7.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.76f)]
// CHECK:STDOUT:     %t.param: ref @F.%T.as_type.loc9_21.1 (%T.as_type.468) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_21.1: type = splice_block %.loc9_21.2 [symbolic = %T.as_type.loc9_21.1 (constants.%T.as_type.468)] {
// CHECK:STDOUT:       %T.ref.loc9_21: %Ptr.type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.76f)]
// CHECK:STDOUT:       %T.as_type.loc9_21.2: type = facet_access_type %T.ref.loc9_21 [symbolic = %T.as_type.loc9_21.1 (constants.%T.as_type.468)]
// CHECK:STDOUT:       %.loc9_21.2: type = converted %T.ref.loc9_21, %T.as_type.loc9_21.2 [symbolic = %T.as_type.loc9_21.1 (constants.%T.as_type.468)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.as_type.loc9_21.1 (%T.as_type.468) = wrapper_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.ac4) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.ac4) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.99f]
// CHECK:STDOUT:   %Ptr.WithSelf.decl = interface_with_self_decl @Ptr [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete = constants.%assoc0.b1a]
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
// CHECK:STDOUT: generic final impl @U.as.Ptr.impl(%U.loc7_21.1: type) {
// CHECK:STDOUT:   %U.patt.loc7_21.2: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_21.2 (constants.%U.patt.d47)]
// CHECK:STDOUT:   %U.loc7_21.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc7_53.2: type = ptr_type %U.loc7_21.2 [symbolic = %ptr.loc7_53.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.406 = %ptr.loc7_53.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.774)]
// CHECK:STDOUT:   %.loc7_55: <witness> = impl_self_witness %U.loc7_21.2, @Ptr [symbolic = %.loc7_55 (constants.%.f8f)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_55.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_21.2) [symbolic = %Ptr.impl_witness.loc7_55.2 (constants.%Ptr.impl_witness.548)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   final impl: %U.ref.loc7_29 as %Ptr.ref {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc7_55.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc7_55.2 (constants.%Ptr.impl_witness.548)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc7_53.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Ptr.ref
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc7_55.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_7.2: %Ptr.type) {
// CHECK:STDOUT:   %T.patt.loc9_7.2: %pattern_type.f5d = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.a2e)]
// CHECK:STDOUT:   %T.loc9_7.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.76f)]
// CHECK:STDOUT:   %T.as_type.loc9_21.1: type = facet_access_type %T.loc9_7.1 [symbolic = %T.as_type.loc9_21.1 (constants.%T.as_type.468)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.as_type.loc9_21.1 [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.756)]
// CHECK:STDOUT:   %t.patt.loc9_19.2: @F.%pattern_type.loc9_19 (%pattern_type.756) = ref_binding_pattern t [symbolic = %t.patt.loc9_19.2 (constants.%t.patt)]
// CHECK:STDOUT:   %t.param_patt.loc9_14.2: @F.%pattern_type.loc9_19 (%pattern_type.756) = var_param_pattern %t.patt.loc9_19.2 [symbolic = %t.param_patt.loc9_14.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %.loc9_28.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.as_type.loc9_21.1) [symbolic = %.loc9_28.1 (constants.%.166)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_21.1 [symbolic = %ptr (constants.%ptr.ac4)]
// CHECK:STDOUT:   %.loc9_28.2: Core.Form = init_form %ptr [symbolic = %.loc9_28.2 (constants.%.e16)]
// CHECK:STDOUT:   %pattern_type.loc9_28: type = pattern_type %ptr [symbolic = %pattern_type.loc9_28 (constants.%pattern_type.f62)]
// CHECK:STDOUT:   %return.param_patt.loc9_28.2: @F.%pattern_type.loc9_28 (%pattern_type.f62) = out_param_pattern [symbolic = %return.param_patt.loc9_28.2 (constants.%return.param_patt.b93)]
// CHECK:STDOUT:   %return.patt.loc9_24.2: @F.%pattern_type.loc9_28 (%pattern_type.f62) = return_slot_pattern %return.param_patt.loc9_28.2, %ptr [symbolic = %return.patt.loc9_24.2 (constants.%return.patt.058)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %T.as_type.loc9_21.1 [symbolic = %require_complete.loc9_14 (constants.%require_complete.35c)]
// CHECK:STDOUT:   %require_complete.loc9_24: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_24 (constants.%require_complete.841)]
// CHECK:STDOUT:   %.loc10_10.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc9_21.1) [symbolic = %.loc10_10.3 (constants.%.ee9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d90)]
// CHECK:STDOUT:   %Copy.facet.loc10_10.3: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc10_10.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.a39)]
// CHECK:STDOUT:   %.loc10_10.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc10_10.3 [symbolic = %.loc10_10.4 (constants.%.229)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.4 (%.229) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.1f8)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.WithSelf.Op(%Copy.facet.loc10_10.3) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.d80)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.as_type.loc9_21.1 (%T.as_type.468)) -> out %return.param: @F.%ptr (%ptr.ac4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc9_21.1 (%T.as_type.468) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.ac4) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.4 (%.229) = impl_witness_access constants.%Copy.lookup_impl_witness.d90, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.1f8)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %Copy.facet.loc10_10.1: %Copy.type = facet_value constants.%ptr.ac4, (constants.%Copy.lookup_impl_witness.d90) [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_10.1: %Copy.type = converted constants.%ptr.ac4, %Copy.facet.loc10_10.1 [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %Copy.facet.loc10_10.2: %Copy.type = facet_value constants.%ptr.ac4, (constants.%Copy.lookup_impl_witness.d90) [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_10.2: %Copy.type = converted constants.%ptr.ac4, %Copy.facet.loc10_10.2 [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.d80)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr (%ptr.ac4) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Self.99f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%.Self.frozen.073) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.patt.loc7_21.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_53.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.774
// CHECK:STDOUT:   %.loc7_55 => constants.%.f8f
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_55.2 => constants.%Ptr.impl_witness.548
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%T.76f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.as_type.468) {
// CHECK:STDOUT:   %U.patt.loc7_21.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%T.as_type.468
// CHECK:STDOUT:   %ptr.loc7_53.2 => constants.%ptr.ac4
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.c27
// CHECK:STDOUT:   %.loc7_55 => constants.%.47c
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_55.2 => constants.%Ptr.impl_witness.8bb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.76f) {
// CHECK:STDOUT:   %T.patt.loc9_7.2 => constants.%T.patt.a2e
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T.76f
// CHECK:STDOUT:   %T.as_type.loc9_21.1 => constants.%T.as_type.468
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.756
// CHECK:STDOUT:   %t.patt.loc9_19.2 => constants.%t.patt
// CHECK:STDOUT:   %t.param_patt.loc9_14.2 => constants.%t.param_patt
// CHECK:STDOUT:   %.loc9_28.1 => constants.%.166
// CHECK:STDOUT:   %ptr => constants.%ptr.ac4
// CHECK:STDOUT:   %.loc9_28.2 => constants.%.e16
// CHECK:STDOUT:   %pattern_type.loc9_28 => constants.%pattern_type.f62
// CHECK:STDOUT:   %return.param_patt.loc9_28.2 => constants.%return.param_patt.b93
// CHECK:STDOUT:   %return.patt.loc9_24.2 => constants.%return.patt.058
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- final_impl_rewrite_of_symbolic_through_facet_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Ptr.type: type = facet_type <@Ptr> [concrete]
// CHECK:STDOUT:   %Self.99f: %Ptr.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Ptr.assoc_type: type = assoc_entity_type @Ptr [concrete]
// CHECK:STDOUT:   %assoc0.b1a: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt.d47: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.073: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.073 [symbolic_self]
// CHECK:STDOUT:   %.c6b: <witness> = impl_self_witness %.Self.frozen.073, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.949: type = impl_witness_access %.c6b, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %.Self: %Ptr.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.51f: <witness> = impl_self_witness %.Self, @Ptr [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.406: type = impl_witness_access %.51f, element0 [symbolic_self]
// CHECK:STDOUT:   %Ptr_where.type.774: type = facet_type <@Ptr where %impl.elem0.406 = %ptr.e8f8f9.1> [symbolic]
// CHECK:STDOUT:   %.f8f: <witness> = impl_self_witness %U.67d, @Ptr [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.548: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%U.67d) [symbolic]
// CHECK:STDOUT:   %Ptr.facet: %Ptr.type = facet_value %U.67d, (%Ptr.impl_witness.548) [symbolic]
// CHECK:STDOUT:   %pattern_type.f5d: type = pattern_type %Ptr.type [concrete]
// CHECK:STDOUT:   %T.patt.a2e: %pattern_type.f5d = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.76f: %Ptr.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.468: type = facet_access_type %T.76f [symbolic]
// CHECK:STDOUT:   %pattern_type.756: type = pattern_type %T.as_type.468 [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.756 = ref_binding_pattern t [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.756 = var_param_pattern %t.patt [symbolic]
// CHECK:STDOUT:   %ptr.ac4: type = ptr_type %T.as_type.468 [symbolic]
// CHECK:STDOUT:   %Ptr_where.type.c27: type = facet_type <@Ptr where %impl.elem0.406 = %ptr.ac4> [symbolic]
// CHECK:STDOUT:   %.47c: <witness> = impl_self_witness %T.as_type.468, @Ptr [symbolic]
// CHECK:STDOUT:   %Ptr.impl_witness.8bb: <witness> = impl_witness @U.as.Ptr.impl.%Ptr.impl_witness_table, @U.as.Ptr.impl(%T.as_type.468) [symbolic]
// CHECK:STDOUT:   %.166: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.as_type.468) [symbolic]
// CHECK:STDOUT:   %.e16: Core.Form = init_form %ptr.ac4 [symbolic]
// CHECK:STDOUT:   %pattern_type.f62: type = pattern_type %ptr.ac4 [symbolic]
// CHECK:STDOUT:   %return.param_patt.b93: %pattern_type.f62 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.058: %pattern_type.f62 = return_slot_pattern %return.param_patt.b93, %ptr.ac4 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.35c: <witness> = require_complete_type %T.as_type.468 [symbolic]
// CHECK:STDOUT:   %require_complete.841: <witness> = require_complete_type %ptr.ac4 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d90: <witness> = lookup_impl_witness %ptr.ac4, @Copy [symbolic]
// CHECK:STDOUT:   %.ee9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.468) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.ac4, (%Copy.lookup_impl_witness.d90) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a39: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.229: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a39, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1f8: %.229 = impl_witness_access %Copy.lookup_impl_witness.d90, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d80: <specific function> = specific_impl_function %impl.elem0.1f8, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:     %U.patt.loc7_21.1: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_21.2 (constants.%U.patt.d47)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc7_29: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %Ptr.ref: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %.Self.frozen.loc7_38: %Ptr.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.073]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Ptr.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %Ptr.type = name_ref .Self, %.Self.frozen.loc7_38 [symbolic_self = constants.%.Self.frozen.073]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc7_44: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.b1a]
// CHECK:STDOUT:     %impl.elem0.loc7_44.1: type = impl_witness_access constants.%.c6b, element0 [symbolic_self = constants.%impl.elem0.949]
// CHECK:STDOUT:     %U.ref.loc7_52: type = name_ref U, %U.loc7_21.1 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc7_53.1: type = ptr_type %U.ref.loc7_52 [symbolic = %ptr.loc7_53.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %rewrite.loc7_50.1 = requirement_rewrite %impl.elem0.loc7_44.1, %ptr.loc7_53.1 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc7_44.2: type = impl_witness_access constants.%.51f, element0 [symbolic_self = constants.%impl.elem0.406]
// CHECK:STDOUT:     %rewrite.loc7_50.2 = requirement_rewrite %impl.elem0.loc7_44.2, %ptr.loc7_53.1 [concrete]
// CHECK:STDOUT:     %.loc7_38: type = where_expr [symbolic = %Ptr_where.type (constants.%Ptr_where.type.774)] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Ptr.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc7_50.2 = requirement_rewrite %impl.elem0.loc7_44.2, %ptr.loc7_53.1 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_23.1: type = splice_block %.loc7_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc7_21: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %.loc7_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_21.1: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @U.as.Ptr.impl.%U.ref.loc7_29, @Ptr [symbolic = @U.as.Ptr.impl.%.loc7_55 (constants.%.f8f)]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_7.1: %pattern_type.f5d = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.a2e)]
// CHECK:STDOUT:     %t.patt.loc9_19.1: @F.%pattern_type.loc9_19 (%pattern_type.756) = ref_binding_pattern t [symbolic = %t.patt.loc9_19.2 (constants.%t.patt)]
// CHECK:STDOUT:     %t.param_patt.loc9_14.1: @F.%pattern_type.loc9_19 (%pattern_type.756) = var_param_pattern %t.patt.loc9_19.1 [symbolic = %t.param_patt.loc9_14.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %return.param_patt.loc9_28.1: @F.%pattern_type.loc9_28 (%pattern_type.f62) = out_param_pattern [symbolic = %return.param_patt.loc9_28.2 (constants.%return.param_patt.b93)]
// CHECK:STDOUT:     %return.patt.loc9_24.1: @F.%pattern_type.loc9_28 (%pattern_type.f62) = return_slot_pattern %return.param_patt.loc9_28.1, %impl.elem0.loc9 [symbolic = %return.patt.loc9_24.2 (constants.%return.patt.058)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_27: %Ptr.type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.76f)]
// CHECK:STDOUT:     %Ptr.ref.loc9_30: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     %Type.ref: %Ptr.assoc_type = name_ref Type, @Type.%assoc0 [concrete = constants.%assoc0.b1a]
// CHECK:STDOUT:     %impl.elem0.loc9: type = impl_witness_access constants.%Ptr.impl_witness.8bb, element0 [symbolic = %ptr (constants.%ptr.ac4)]
// CHECK:STDOUT:     %.loc9_28.3: Core.Form = init_form %impl.elem0.loc9 [symbolic = %.loc9_28.2 (constants.%.e16)]
// CHECK:STDOUT:     %.loc9_9: type = splice_block %Ptr.ref.loc9_9 [concrete = constants.%Ptr.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %Ptr.ref.loc9_9: type = name_ref Ptr, file.%Ptr.decl [concrete = constants.%Ptr.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_7.2: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.76f)]
// CHECK:STDOUT:     %t.param: ref @F.%T.as_type.loc9_21.1 (%T.as_type.468) = ref_param call_param0
// CHECK:STDOUT:     %.loc9_21.1: type = splice_block %.loc9_21.2 [symbolic = %T.as_type.loc9_21.1 (constants.%T.as_type.468)] {
// CHECK:STDOUT:       %T.ref.loc9_21: %Ptr.type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.76f)]
// CHECK:STDOUT:       %T.as_type.loc9_21.2: type = facet_access_type %T.ref.loc9_21 [symbolic = %T.as_type.loc9_21.1 (constants.%T.as_type.468)]
// CHECK:STDOUT:       %.loc9_21.2: type = converted %T.ref.loc9_21, %T.as_type.loc9_21.2 [symbolic = %T.as_type.loc9_21.1 (constants.%T.as_type.468)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.as_type.loc9_21.1 (%T.as_type.468) = wrapper_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr (%ptr.ac4) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr (%ptr.ac4) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ptr {
// CHECK:STDOUT:   %Self: %Ptr.type = symbolic_binding Self, 0 [symbolic = constants.%Self.99f]
// CHECK:STDOUT:   %Ptr.WithSelf.decl = interface_with_self_decl @Ptr [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Type: type = assoc_const_decl @Type [concrete] {
// CHECK:STDOUT:     %assoc0: %Ptr.assoc_type = assoc_entity element0, @Ptr.WithSelf.%Type [concrete = constants.%assoc0.b1a]
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
// CHECK:STDOUT: generic final impl @U.as.Ptr.impl(%U.loc7_21.1: type) {
// CHECK:STDOUT:   %U.patt.loc7_21.2: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_21.2 (constants.%U.patt.d47)]
// CHECK:STDOUT:   %U.loc7_21.2: type = symbolic_binding U, 0 [symbolic = %U.loc7_21.2 (constants.%U.67d)]
// CHECK:STDOUT:   %ptr.loc7_53.2: type = ptr_type %U.loc7_21.2 [symbolic = %ptr.loc7_53.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %Ptr_where.type: type = facet_type <@Ptr where constants.%impl.elem0.406 = %ptr.loc7_53.2> [symbolic = %Ptr_where.type (constants.%Ptr_where.type.774)]
// CHECK:STDOUT:   %.loc7_55: <witness> = impl_self_witness %U.loc7_21.2, @Ptr [symbolic = %.loc7_55 (constants.%.f8f)]
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_55.2: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(%U.loc7_21.2) [symbolic = %Ptr.impl_witness.loc7_55.2 (constants.%Ptr.impl_witness.548)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   final impl: %U.ref.loc7_29 as %Ptr.ref {
// CHECK:STDOUT:     %Ptr.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as.Ptr.impl [concrete]
// CHECK:STDOUT:     %Ptr.impl_witness.loc7_55.1: <witness> = impl_witness %Ptr.impl_witness_table, @U.as.Ptr.impl(constants.%U.67d) [symbolic = %Ptr.impl_witness.loc7_55.2 (constants.%Ptr.impl_witness.548)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ptr.e8f8f9.1 [symbolic = %ptr.loc7_53.2 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Ptr.ref
// CHECK:STDOUT:     witness = %Ptr.impl_witness.loc7_55.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_7.2: %Ptr.type) {
// CHECK:STDOUT:   %T.patt.loc9_7.2: %pattern_type.f5d = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_7.2 (constants.%T.patt.a2e)]
// CHECK:STDOUT:   %T.loc9_7.1: %Ptr.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.76f)]
// CHECK:STDOUT:   %T.as_type.loc9_21.1: type = facet_access_type %T.loc9_7.1 [symbolic = %T.as_type.loc9_21.1 (constants.%T.as_type.468)]
// CHECK:STDOUT:   %pattern_type.loc9_19: type = pattern_type %T.as_type.loc9_21.1 [symbolic = %pattern_type.loc9_19 (constants.%pattern_type.756)]
// CHECK:STDOUT:   %t.patt.loc9_19.2: @F.%pattern_type.loc9_19 (%pattern_type.756) = ref_binding_pattern t [symbolic = %t.patt.loc9_19.2 (constants.%t.patt)]
// CHECK:STDOUT:   %t.param_patt.loc9_14.2: @F.%pattern_type.loc9_19 (%pattern_type.756) = var_param_pattern %t.patt.loc9_19.2 [symbolic = %t.param_patt.loc9_14.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %.loc9_28.1: require_specific_def_type = require_specific_def @U.as.Ptr.impl(%T.as_type.loc9_21.1) [symbolic = %.loc9_28.1 (constants.%.166)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_21.1 [symbolic = %ptr (constants.%ptr.ac4)]
// CHECK:STDOUT:   %.loc9_28.2: Core.Form = init_form %ptr [symbolic = %.loc9_28.2 (constants.%.e16)]
// CHECK:STDOUT:   %pattern_type.loc9_28: type = pattern_type %ptr [symbolic = %pattern_type.loc9_28 (constants.%pattern_type.f62)]
// CHECK:STDOUT:   %return.param_patt.loc9_28.2: @F.%pattern_type.loc9_28 (%pattern_type.f62) = out_param_pattern [symbolic = %return.param_patt.loc9_28.2 (constants.%return.param_patt.b93)]
// CHECK:STDOUT:   %return.patt.loc9_24.2: @F.%pattern_type.loc9_28 (%pattern_type.f62) = return_slot_pattern %return.param_patt.loc9_28.2, %ptr [symbolic = %return.patt.loc9_24.2 (constants.%return.patt.058)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %T.as_type.loc9_21.1 [symbolic = %require_complete.loc9_14 (constants.%require_complete.35c)]
// CHECK:STDOUT:   %require_complete.loc9_24: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_24 (constants.%require_complete.841)]
// CHECK:STDOUT:   %.loc10_10.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.as_type.loc9_21.1) [symbolic = %.loc10_10.3 (constants.%.ee9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d90)]
// CHECK:STDOUT:   %Copy.facet.loc10_10.3: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc10_10.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.a39)]
// CHECK:STDOUT:   %.loc10_10.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc10_10.3 [symbolic = %.loc10_10.4 (constants.%.229)]
// CHECK:STDOUT:   %impl.elem0.loc10_10.2: @F.%.loc10_10.4 (%.229) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.1f8)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_10.2: <specific function> = specific_impl_function %impl.elem0.loc10_10.2, @Copy.WithSelf.Op(%Copy.facet.loc10_10.3) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.d80)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.as_type.loc9_21.1 (%T.as_type.468)) -> out %return.param: @F.%ptr (%ptr.ac4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc9_21.1 (%T.as_type.468) = name_ref t, %t
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.ac4) = addr_of %t.ref
// CHECK:STDOUT:     %impl.elem0.loc10_10.1: @F.%.loc10_10.4 (%.229) = impl_witness_access constants.%Copy.lookup_impl_witness.d90, element0 [symbolic = %impl.elem0.loc10_10.2 (constants.%impl.elem0.1f8)]
// CHECK:STDOUT:     %bound_method.loc10_10.1: <bound method> = bound_method %addr, %impl.elem0.loc10_10.1
// CHECK:STDOUT:     %Copy.facet.loc10_10.1: %Copy.type = facet_value constants.%ptr.ac4, (constants.%Copy.lookup_impl_witness.d90) [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_10.1: %Copy.type = converted constants.%ptr.ac4, %Copy.facet.loc10_10.1 [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %Copy.facet.loc10_10.2: %Copy.type = facet_value constants.%ptr.ac4, (constants.%Copy.lookup_impl_witness.d90) [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_10.2: %Copy.type = converted constants.%ptr.ac4, %Copy.facet.loc10_10.2 [symbolic = %Copy.facet.loc10_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_10.1: <specific function> = specific_impl_function %impl.elem0.loc10_10.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc10_10.2 (constants.%specific_impl_fn.d80)]
// CHECK:STDOUT:     %bound_method.loc10_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc10_10.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr (%ptr.ac4) = call %bound_method.loc10_10.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Self.99f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%.Self.frozen.073) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%U.67d) {
// CHECK:STDOUT:   %U.patt.loc7_21.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%U.67d
// CHECK:STDOUT:   %ptr.loc7_53.2 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.774
// CHECK:STDOUT:   %.loc7_55 => constants.%.f8f
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_55.2 => constants.%Ptr.impl_witness.548
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%Ptr.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as.Ptr.impl(constants.%T.as_type.468) {
// CHECK:STDOUT:   %U.patt.loc7_21.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc7_21.2 => constants.%T.as_type.468
// CHECK:STDOUT:   %ptr.loc7_53.2 => constants.%ptr.ac4
// CHECK:STDOUT:   %Ptr_where.type => constants.%Ptr_where.type.c27
// CHECK:STDOUT:   %.loc7_55 => constants.%.47c
// CHECK:STDOUT:   %Ptr.impl_witness.loc7_55.2 => constants.%Ptr.impl_witness.8bb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Ptr.WithSelf(constants.%T.76f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.76f) {
// CHECK:STDOUT:   %T.patt.loc9_7.2 => constants.%T.patt.a2e
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T.76f
// CHECK:STDOUT:   %T.as_type.loc9_21.1 => constants.%T.as_type.468
// CHECK:STDOUT:   %pattern_type.loc9_19 => constants.%pattern_type.756
// CHECK:STDOUT:   %t.patt.loc9_19.2 => constants.%t.patt
// CHECK:STDOUT:   %t.param_patt.loc9_14.2 => constants.%t.param_patt
// CHECK:STDOUT:   %.loc9_28.1 => constants.%.166
// CHECK:STDOUT:   %ptr => constants.%ptr.ac4
// CHECK:STDOUT:   %.loc9_28.2 => constants.%.e16
// CHECK:STDOUT:   %pattern_type.loc9_28 => constants.%pattern_type.f62
// CHECK:STDOUT:   %return.param_patt.loc9_28.2 => constants.%return.param_patt.b93
// CHECK:STDOUT:   %return.patt.loc9_24.2 => constants.%return.patt.058
// CHECK:STDOUT: }
// CHECK:STDOUT:

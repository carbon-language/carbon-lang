// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/self_in_interface_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/self_in_interface_param.carbon

// --- period_self_param.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
interface I(T:! type) {
  let I1:! type;
}

fn F(T:! I(.Self) where .I1 = ()) -> T.I1 {
  return ();
}

fn G(T:! I(.Self as type) where .I1 = ()) -> T.I1 {
  return ();
}
//@dump-sem-ir-end

// --- underscore_identifier_name.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  let I1:! type;
}

// Underscore as the identifier name produces a different parse tree for the
// binding pattern.
fn G(_:! I(.Self) where .I1 = ()) {}

// --- todo_fail_returned_period_self_has_full_facet_type.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F() -> T;
}

fn G2(U:! I(.Self)) {}

fn G(U:! I(.Self)) {
  // Caller sees the returned `.Self` type from `F()` as the full `U`.
  U.F().F();
  // Conversion to `type` retains access to all of `U`.
  (U as type).F().F();
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+7]]:4: error: cannot convert non-type value of type `U` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   (U.F() as type).F();
  // CHECK:STDERR:    ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+4]]:4: note: type `U` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   (U.F() as type).F();
  // CHECK:STDERR:    ^~~~~~~~~~~~~
  // CHECK:STDERR:
  (U.F() as type).F();

  // Using the returned `.Self` as a facet value works, not just compound member lookup.
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+7]]:3: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   G2(U.F());
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE-20]]:7: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn G2(U:! I(.Self)) {}
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  G2(U.F());
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+7]]:6: error: cannot convert non-type value of type `U` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   G2(U.F() as type);}
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+4]]:6: note: type `U` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   G2(U.F() as type);}
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR:
  G2(U.F() as type);}

// --- todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F() -> T;
}

interface J {
    let J1:! type;
}

fn G2(U:! I(.Self) where .Self impls (J where .J1 = ())) {}

fn G(U:! I(.Self) where .Self impls (J where .J1 = ())) {
  // Caller sees the returned `.Self` type from `F()` as the full `U`.
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+7]]:10: error: cannot implicitly convert non-type value of type `U` into type implementing `J` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   let x: U.F().(J.J1) = ();
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+4]]:10: note: type `U` does not implement interface `Core.ImplicitAs(J)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let x: U.F().(J.J1) = ();
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR:
  let x: U.F().(J.J1) = ();
  // Conversion to `type` retains access to all of `U`.
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+7]]:10: error: cannot implicitly convert non-type value of type `U` into type implementing `J` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   let y: (U as type).F().(J.J1) = ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+4]]:10: note: type `U` does not implement interface `Core.ImplicitAs(J)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let y: (U as type).F().(J.J1) = ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let y: (U as type).F().(J.J1) = ();
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+7]]:11: error: cannot convert non-type value of type `U` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   let z: (U.F() as type).(J.J1) = ();
  // CHECK:STDERR:           ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+4]]:11: note: type `U` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let z: (U.F() as type).(J.J1) = ();
  // CHECK:STDERR:           ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let z: (U.F() as type).(J.J1) = ();

  // Using the returned `.Self` as a facet value works, not just compound member lookup.
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+7]]:3: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   G2(U.F());
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE-34]]:7: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn G2(U:! I(.Self) where .Self impls (J where .J1 = ())) {}
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  G2(U.F());
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+7]]:6: error: cannot convert non-type value of type `U` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   G2(U.F() as type);
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+4]]:6: note: type `U` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   G2(U.F() as type);
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR:
  G2(U.F() as type);
}

// --- todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F() -> T;
}

interface J {
    let J1:! type;
}

fn G2(U:! I(.Self) & J where .J1 = ()) {}

fn G(U:! I(.Self) & J where .J1 = ()) {
  // Caller sees the returned `.Self` type from `F()` as the full `U`.
  let x: U.F().J1 = ();
  // Conversion to `type` retains access to all of `U`.
  let y: (U as type).F().J1 = ();
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+7]]:11: error: cannot convert non-type value of type `U` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   let z: (U.F() as type).J1 = ();
  // CHECK:STDERR:           ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+4]]:11: note: type `U` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let z: (U.F() as type).J1 = ();
  // CHECK:STDERR:           ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let z: (U.F() as type).J1 = ();

  // Using the returned `.Self` as a facet value works, not just compound member lookup.
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+7]]:3: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   G2(U.F());
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE-20]]:7: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn G2(U:! I(.Self) & J where .J1 = ()) {}
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  G2(U.F());
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+7]]:6: error: cannot convert non-type value of type `U` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   G2(U.F() as type);
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+4]]:6: note: type `U` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   G2(U.F() as type);
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR:
  G2(U.F() as type);
}

// --- fail_template_access.carbon
library "[[@TEST_NAME]]";

// TODO: Improved error: no member named `I1` in `T` of type `type`.
//
// CHECK:STDERR: fail_template_access.carbon:[[@LINE+4]]:36: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: interface I(template T:! type, N:! T.I1) {
// CHECK:STDERR:                                    ^~~~
// CHECK:STDERR:
interface I(template T:! type, N:! T.I1) {
    let I1:! type;
}

// CHECK:STDOUT: --- period_self_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.186b81.1: %type = bind_symbolic_name .Self(-2) [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.209: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.1e5: type = assoc_entity_type @I, @I(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.ab0: %I.assoc_type.1e5 = assoc_entity element0, @I.%I1 [symbolic]
// CHECK:STDOUT:   %.c10bb5.1: type = deferred_bind_symbolic_self %.Self.186b81.1, .Self(-2) [symbolic_self]
// CHECK:STDOUT:   %I.type.295e66.1: type = facet_type <@I, @I(%.c10bb5.1)> [symbolic_self]
// CHECK:STDOUT:   %.Self.bbd876.1: %I.type.295e66.1 = bind_symbolic_name .Self(-2) [symbolic_self]
// CHECK:STDOUT:   %Self.a63420.1: %I.type.295e66.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.fc5023.1: type = assoc_entity_type @I, @I(%.c10bb5.1) [symbolic_self]
// CHECK:STDOUT:   %assoc0.43280d.1: %I.assoc_type.fc5023.1 = assoc_entity element0, @I.%I1 [symbolic_self]
// CHECK:STDOUT:   %.0e5e66.1: type = deferred_bind_symbolic_self %.Self.bbd876.1, .Self(-2) [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.ce8298.1: <witness> = lookup_impl_witness %.Self.bbd876.1, @I, @I(%.c10bb5.1) [symbolic_self]
// CHECK:STDOUT:   %I.facet.0e5757.1: %I.type.295e66.1 = facet_value %.0e5e66.1, (%I.lookup_impl_witness.ce8298.1) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7fe8bd.1: type = impl_witness_access %I.lookup_impl_witness.ce8298.1, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.b4a2b3.1: type = facet_type <@I, @I(%.c10bb5.1) where %impl.elem0.7fe8bd.1 = %empty_tuple.type> [symbolic_self]
// CHECK:STDOUT:   %T.2e6660.1: %I_where.type.b4a2b3.1 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.015a58.1: type = pattern_type %I_where.type.b4a2b3.1 [symbolic_self]
// CHECK:STDOUT:   %T.as_type.a2ae6e.1: type = facet_access_type %T.2e6660.1 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.0abc80.1: <witness> = lookup_impl_witness %T.2e6660.1, @I, @I(%.c10bb5.1) [symbolic]
// CHECK:STDOUT:   %I.facet.d6049c.1: %I.type.295e66.1 = facet_value %T.as_type.a2ae6e.1, (%I.lookup_impl_witness.0abc80.1) [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.186b81.2: %type = bind_symbolic_name .Self(-2) [symbolic_self]
// CHECK:STDOUT:   %.c10bb5.2: type = deferred_bind_symbolic_self %.Self.186b81.2, .Self(-2) [symbolic_self]
// CHECK:STDOUT:   %I.type.295e66.2: type = facet_type <@I, @I(%.c10bb5.2)> [symbolic_self]
// CHECK:STDOUT:   %.Self.bbd876.2: %I.type.295e66.2 = bind_symbolic_name .Self(-2) [symbolic_self]
// CHECK:STDOUT:   %Self.a63420.2: %I.type.295e66.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.fc5023.2: type = assoc_entity_type @I, @I(%.c10bb5.2) [symbolic_self]
// CHECK:STDOUT:   %assoc0.43280d.2: %I.assoc_type.fc5023.2 = assoc_entity element0, @I.%I1 [symbolic_self]
// CHECK:STDOUT:   %.0e5e66.2: type = deferred_bind_symbolic_self %.Self.bbd876.2, .Self(-2) [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.ce8298.2: <witness> = lookup_impl_witness %.Self.bbd876.2, @I, @I(%.c10bb5.2) [symbolic_self]
// CHECK:STDOUT:   %I.facet.0e5757.2: %I.type.295e66.2 = facet_value %.0e5e66.2, (%I.lookup_impl_witness.ce8298.2) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7fe8bd.2: type = impl_witness_access %I.lookup_impl_witness.ce8298.2, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.b4a2b3.2: type = facet_type <@I, @I(%.c10bb5.2) where %impl.elem0.7fe8bd.2 = %empty_tuple.type> [symbolic_self]
// CHECK:STDOUT:   %T.2e6660.2: %I_where.type.b4a2b3.2 = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.015a58.2: type = pattern_type %I_where.type.b4a2b3.2 [symbolic_self]
// CHECK:STDOUT:   %T.as_type.a2ae6e.2: type = facet_access_type %T.2e6660.2 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.0abc80.2: <witness> = lookup_impl_witness %T.2e6660.2, @I, @I(%.c10bb5.2) [symbolic]
// CHECK:STDOUT:   %I.facet.d6049c.2: %I.type.295e66.2 = facet_value %T.as_type.a2ae6e.2, (%I.lookup_impl_witness.0abc80.2) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.015a58.1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I_where.type.b4a2b3.1 = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T.2e6660.1)]
// CHECK:STDOUT:     %.loc8_39.1: %I.assoc_type.fc5023.1 = specific_constant @I1.%assoc0, @I(constants.%.c10bb5.1) [symbolic_self = constants.%assoc0.43280d.1]
// CHECK:STDOUT:     %I1.ref.loc8_39: %I.assoc_type.fc5023.1 = name_ref I1, %.loc8_39.1 [symbolic_self = constants.%assoc0.43280d.1]
// CHECK:STDOUT:     %T.as_type.loc8_39.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type.a2ae6e.1)]
// CHECK:STDOUT:     %.loc8_39.2: type = converted %T.ref, %T.as_type.loc8_39.2 [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type.a2ae6e.1)]
// CHECK:STDOUT:     %impl.elem0.loc8_39: type = impl_witness_access constants.%I.lookup_impl_witness.0abc80.1, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [symbolic_self = constants.%I_where.type.b4a2b3.1] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.Self.ref.loc8_12: %type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.186b81.1]
// CHECK:STDOUT:       %.Self.as_type.loc8_17: type = facet_access_type %.Self.ref.loc8_12 [symbolic_self = constants.%.c10bb5.1]
// CHECK:STDOUT:       %.loc8_17: type = converted %.Self.ref.loc8_12, %.Self.as_type.loc8_17 [symbolic_self = constants.%.c10bb5.1]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%.c10bb5.1)> [symbolic_self = constants.%I.type.295e66.1]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc8_25: %I.type.295e66.1 = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.bbd876.1]
// CHECK:STDOUT:       %.loc8_25.1: %I.assoc_type.fc5023.1 = specific_constant @I1.%assoc0, @I(constants.%.c10bb5.1) [symbolic_self = constants.%assoc0.43280d.1]
// CHECK:STDOUT:       %I1.ref.loc8_25: %I.assoc_type.fc5023.1 = name_ref I1, %.loc8_25.1 [symbolic_self = constants.%assoc0.43280d.1]
// CHECK:STDOUT:       %.Self.as_type.loc8_25: type = facet_access_type %.Self.ref.loc8_25 [symbolic_self = constants.%.0e5e66.1]
// CHECK:STDOUT:       %.loc8_25.2: type = converted %.Self.ref.loc8_25, %.Self.as_type.loc8_25 [symbolic_self = constants.%.0e5e66.1]
// CHECK:STDOUT:       %impl.elem0.loc8_25: type = impl_witness_access constants.%I.lookup_impl_witness.ce8298.1, element0 [symbolic_self = constants.%impl.elem0.7fe8bd.1]
// CHECK:STDOUT:       %.loc8_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc8_32.2: type = converted %.loc8_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.2 [symbolic_self = constants.%I_where.type.b4a2b3.1] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type.295e66.1
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc8_25, %.loc8_32.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %I_where.type.b4a2b3.1 = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.1 (constants.%T.2e6660.1)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.015a58.2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I_where.type.b4a2b3.2 = name_ref T, %T.loc12_6.2 [symbolic = %T.loc12_6.1 (constants.%T.2e6660.2)]
// CHECK:STDOUT:     %.loc12_47.1: %I.assoc_type.fc5023.2 = specific_constant @I1.%assoc0, @I(constants.%.c10bb5.2) [symbolic_self = constants.%assoc0.43280d.2]
// CHECK:STDOUT:     %I1.ref.loc12_47: %I.assoc_type.fc5023.2 = name_ref I1, %.loc12_47.1 [symbolic_self = constants.%assoc0.43280d.2]
// CHECK:STDOUT:     %T.as_type.loc12_47.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc12_47.1 (constants.%T.as_type.a2ae6e.2)]
// CHECK:STDOUT:     %.loc12_47.2: type = converted %T.ref, %T.as_type.loc12_47.2 [symbolic = %T.as_type.loc12_47.1 (constants.%T.as_type.a2ae6e.2)]
// CHECK:STDOUT:     %impl.elem0.loc12_47: type = impl_witness_access constants.%I.lookup_impl_witness.0abc80.2, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc12_27.1: type = splice_block %.loc12_27.2 [symbolic_self = constants.%I_where.type.b4a2b3.2] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.Self.ref.loc12_12: %type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.186b81.2]
// CHECK:STDOUT:       %.Self.as_type.loc12_18: type = facet_access_type %.Self.ref.loc12_12 [symbolic_self = constants.%.c10bb5.2]
// CHECK:STDOUT:       %.loc12_18: type = converted %.Self.ref.loc12_12, %.Self.as_type.loc12_18 [symbolic_self = constants.%.c10bb5.2]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%.c10bb5.2)> [symbolic_self = constants.%I.type.295e66.2]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc12_33: %I.type.295e66.2 = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.bbd876.2]
// CHECK:STDOUT:       %.loc12_33.1: %I.assoc_type.fc5023.2 = specific_constant @I1.%assoc0, @I(constants.%.c10bb5.2) [symbolic_self = constants.%assoc0.43280d.2]
// CHECK:STDOUT:       %I1.ref.loc12_33: %I.assoc_type.fc5023.2 = name_ref I1, %.loc12_33.1 [symbolic_self = constants.%assoc0.43280d.2]
// CHECK:STDOUT:       %.Self.as_type.loc12_33: type = facet_access_type %.Self.ref.loc12_33 [symbolic_self = constants.%.0e5e66.2]
// CHECK:STDOUT:       %.loc12_33.2: type = converted %.Self.ref.loc12_33, %.Self.as_type.loc12_33 [symbolic_self = constants.%.0e5e66.2]
// CHECK:STDOUT:       %impl.elem0.loc12_33: type = impl_witness_access constants.%I.lookup_impl_witness.ce8298.2, element0 [symbolic_self = constants.%impl.elem0.7fe8bd.2]
// CHECK:STDOUT:       %.loc12_40.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc12_40.2: type = converted %.loc12_40.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc12_27.2: type = where_expr %.Self.2 [symbolic_self = constants.%I_where.type.b4a2b3.2] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type.295e66.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc12_33, %.loc12_40.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_6.2: %I_where.type.b4a2b3.2 = bind_symbolic_name T, 0 [symbolic = %T.loc12_6.1 (constants.%T.2e6660.2)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T.loc4_13.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.1e5)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.1e5) = assoc_entity element0, %I1 [symbolic = %assoc0 (constants.%assoc0.ab0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %I1: type = assoc_const_decl @I1 [concrete] {
// CHECK:STDOUT:       %assoc0: @I.%I.assoc_type (%I.assoc_type.1e5) = assoc_entity element0, @I.%I1 [symbolic = @I.%assoc0 (constants.%assoc0.ab0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .I1 = @I1.%assoc0
// CHECK:STDOUT:     witness = (%I1)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @I1(@I.%T.loc4_13.2: type, @I.%Self.1: @I.%I.type (%I.type.325)) {
// CHECK:STDOUT:   assoc_const I1:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %I_where.type.b4a2b3.1) {
// CHECK:STDOUT:   %T.loc8_6.1: %I_where.type.b4a2b3.1 = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.1 (constants.%T.2e6660.1)]
// CHECK:STDOUT:   %T.as_type.loc8_39.1: type = facet_access_type %T.loc8_6.1 [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type.a2ae6e.1)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_6.1, @I, @I(constants.%.c10bb5.1) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.0abc80.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc9_11: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_12: %empty_tuple.type = converted %.loc9_11, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc9_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc12_6.2: %I_where.type.b4a2b3.2) {
// CHECK:STDOUT:   %T.loc12_6.1: %I_where.type.b4a2b3.2 = bind_symbolic_name T, 0 [symbolic = %T.loc12_6.1 (constants.%T.2e6660.2)]
// CHECK:STDOUT:   %T.as_type.loc12_47.1: type = facet_access_type %T.loc12_6.1 [symbolic = %T.as_type.loc12_47.1 (constants.%T.as_type.a2ae6e.2)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc12_6.1, @I, @I(constants.%.c10bb5.2) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.0abc80.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc13_11: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13_12: %empty_tuple.type = converted %.loc13_11, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc13_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1(constants.%T.8b3, constants.%Self.209) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.c10bb5.1) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%.c10bb5.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.295e66.1
// CHECK:STDOUT:   %Self.2 => constants.%Self.a63420.1
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.fc5023.1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.43280d.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1(constants.%.c10bb5.1, constants.%I.facet.0e5757.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1(constants.%.c10bb5.1, constants.%I.facet.d6049c.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.2e6660.1) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T.2e6660.1
// CHECK:STDOUT:   %T.as_type.loc8_39.1 => constants.%T.as_type.a2ae6e.1
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.0abc80.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.c10bb5.2) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%.c10bb5.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.295e66.2
// CHECK:STDOUT:   %Self.2 => constants.%Self.a63420.2
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.fc5023.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.43280d.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1(constants.%.c10bb5.2, constants.%I.facet.0e5757.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1(constants.%.c10bb5.2, constants.%I.facet.d6049c.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.2e6660.2) {
// CHECK:STDOUT:   %T.loc12_6.1 => constants.%T.2e6660.2
// CHECK:STDOUT:   %T.as_type.loc12_47.1 => constants.%T.as_type.a2ae6e.2
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.0abc80.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

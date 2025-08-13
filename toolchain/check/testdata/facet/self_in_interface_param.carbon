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
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+4]]:3: error: type `.Self` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   U.F().F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  U.F().F();
  // Conversion to `type` retains access to all of `U`.
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+4]]:3: error: type `.Self` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   (U as type).F().F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (U as type).F().F();
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+7]]:4: error: cannot convert non-type value of type `.Self` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   (U.F() as type).F();
  // CHECK:STDERR:    ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+4]]:4: note: type `.Self` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   (U.F() as type).F();
  // CHECK:STDERR:    ^~~~~~~~~~~~~
  // CHECK:STDERR:
  (U.F() as type).F();

  // Using the returned `.Self` as a facet value works, not just compound member lookup.
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+7]]:3: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   G2(U.F());
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE-28]]:7: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn G2(U:! I(.Self)) {}
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  G2(U.F());
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+7]]:6: error: cannot convert non-type value of type `.Self` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   G2(U.F() as type);}
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type.carbon:[[@LINE+4]]:6: note: type `.Self` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+7]]:10: error: cannot implicitly convert non-type value of type `.Self` into type implementing `J` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   let x: U.F().(J.J1) = ();
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+4]]:10: note: type `.Self` does not implement interface `Core.ImplicitAs(J)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let x: U.F().(J.J1) = ();
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR:
  let x: U.F().(J.J1) = ();
  // Conversion to `type` retains access to all of `U`.
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+7]]:10: error: cannot implicitly convert non-type value of type `.Self` into type implementing `J` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   let y: (U as type).F().(J.J1) = ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+4]]:10: note: type `.Self` does not implement interface `Core.ImplicitAs(J)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let y: (U as type).F().(J.J1) = ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let y: (U as type).F().(J.J1) = ();
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+7]]:11: error: cannot convert non-type value of type `.Self` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   let z: (U.F() as type).(J.J1) = ();
  // CHECK:STDERR:           ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+4]]:11: note: type `.Self` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+7]]:6: error: cannot convert non-type value of type `.Self` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   G2(U.F() as type);
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_impls_interface.carbon:[[@LINE+4]]:6: note: type `.Self` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+4]]:10: error: type `.Self` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   let x: U.F().J1 = ();
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR:
  let x: U.F().J1 = ();
  // Conversion to `type` retains access to all of `U`.
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+4]]:10: error: type `.Self` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   let y: (U as type).F().J1 = ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let y: (U as type).F().J1 = ();
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+7]]:11: error: cannot convert non-type value of type `.Self` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   let z: (U.F() as type).J1 = ();
  // CHECK:STDERR:           ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+4]]:11: note: type `.Self` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let z: (U.F() as type).J1 = ();
  // CHECK:STDERR:           ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let z: (U.F() as type).J1 = ();

  // Using the returned `.Self` as a facet value works, not just compound member lookup.
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+7]]:3: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   G2(U.F());
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE-28]]:7: note: initializing generic parameter `U` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn G2(U:! I(.Self) & J where .J1 = ()) {}
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  G2(U.F());
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+7]]:6: error: cannot convert non-type value of type `.Self` to `type` with `as` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   G2(U.F() as type);
  // CHECK:STDERR:      ^~~~~~~~~~~~~
  // CHECK:STDERR: todo_fail_returned_period_self_has_full_facet_type_with_extends_interface.carbon:[[@LINE+4]]:6: note: type `.Self` does not implement interface `Core.As(type)` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T.8b3)> [symbolic]
// CHECK:STDOUT:   %Self.209: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.Self.6442b1.2: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type.1e5: type = assoc_entity_type @I, @I(%T.8b3) [symbolic]
// CHECK:STDOUT:   %assoc0.ab0: %I.assoc_type.1e5 = assoc_entity element0, @I.%I1 [symbolic]
// CHECK:STDOUT:   %I.type.962: type = facet_type <@I, @I(%.Self.6442b1.2)> [symbolic_self]
// CHECK:STDOUT:   %.Self.50d: %I.type.962 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Self.e0b: %I.type.962 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.180: type = assoc_entity_type @I, @I(%.Self.6442b1.2) [symbolic_self]
// CHECK:STDOUT:   %assoc0.357: %I.assoc_type.180 = assoc_entity element0, @I.%I1 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.50d [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.aee: <witness> = lookup_impl_witness %.Self.50d, @I, @I(%.Self.6442b1.2) [symbolic_self]
// CHECK:STDOUT:   %I.facet.5e0: %I.type.962 = facet_value %.Self.as_type, (%I.lookup_impl_witness.aee) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness.aee, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I, @I(%.Self.6442b1.2) where %impl.elem0 = %empty_tuple.type> [symbolic_self]
// CHECK:STDOUT:   %T.735: %I_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3cf: type = pattern_type %I_where.type [symbolic_self]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.735 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.262: <witness> = lookup_impl_witness %T.735, @I, @I(%.Self.6442b1.2) [symbolic]
// CHECK:STDOUT:   %I.facet.2b9: %I.type.962 = facet_value %T.as_type, (%I.lookup_impl_witness.262) [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.3cf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I_where.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T.735)]
// CHECK:STDOUT:     %.loc8_39.1: %I.assoc_type.180 = specific_constant @I1.%assoc0, @I(constants.%.Self.6442b1.2) [symbolic_self = constants.%assoc0.357]
// CHECK:STDOUT:     %I1.ref.loc8_39: %I.assoc_type.180 = name_ref I1, %.loc8_39.1 [symbolic_self = constants.%assoc0.357]
// CHECK:STDOUT:     %T.as_type.loc8_39.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc8_39.2: type = converted %T.ref, %T.as_type.loc8_39.2 [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc8_39: type = impl_witness_access constants.%I.lookup_impl_witness.262, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.Self.ref.loc8_12: type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.6442b1.2]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%.Self.6442b1.2)> [symbolic_self = constants.%I.type.962]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc8_25: %I.type.962 = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.50d]
// CHECK:STDOUT:       %.loc8_25.1: %I.assoc_type.180 = specific_constant @I1.%assoc0, @I(constants.%.Self.6442b1.2) [symbolic_self = constants.%assoc0.357]
// CHECK:STDOUT:       %I1.ref.loc8_25: %I.assoc_type.180 = name_ref I1, %.loc8_25.1 [symbolic_self = constants.%assoc0.357]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref.loc8_25 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc8_25.2: type = converted %.Self.ref.loc8_25, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc8_25: type = impl_witness_access constants.%I.lookup_impl_witness.aee, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc8_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc8_32.2: type = converted %.loc8_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.2 [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type.962
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc8_25, %.loc8_32.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.1 (constants.%T.735)]
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
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %I_where.type) {
// CHECK:STDOUT:   %T.loc8_6.1: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.1 (constants.%T.735)]
// CHECK:STDOUT:   %T.as_type.loc8_39.1: type = facet_access_type %T.loc8_6.1 [symbolic = %T.as_type.loc8_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_6.1, @I, @I(constants.%.Self.6442b1.2) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.262)]
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
// CHECK:STDOUT: specific @I(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1(constants.%T.8b3, constants.%Self.209) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.Self.6442b1.2) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%.Self.6442b1.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.962
// CHECK:STDOUT:   %Self.2 => constants.%Self.e0b
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.180
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1(constants.%.Self.6442b1.2, constants.%I.facet.5e0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I1(constants.%.Self.6442b1.2, constants.%I.facet.2b9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.735) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T.735
// CHECK:STDOUT:   %T.as_type.loc8_39.1 => constants.%T.as_type
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.262
// CHECK:STDOUT: }
// CHECK:STDOUT:

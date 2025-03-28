// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/impl_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/impl_assoc_const.carbon

// --- success.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; }

impl () as I where .T = {} {}

// --- redecl.carbon
library "[[@TEST_NAME]]";

interface I2 { let T2:! type; }

impl () as I2 where .T2 = {};
impl () as I2 where .T2 = {} {}

// --- fail_redecl_adds_rewrites.carbon
library "[[@TEST_NAME]]";

interface I3 { let T3:! type; }

// CHECK:STDERR: fail_redecl_adds_rewrites.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as I3;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I3;
impl () as I3 where .T3 = {} {}

// --- fail_mismatch.carbon
library "[[@TEST_NAME]]";

interface J { let U:! type; }

// CHECK:STDERR: fail_mismatch.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as J where .U = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as J where .U = {};
impl () as J where .U = () {}

// --- fail_mismatch_bad_value.carbon
library "[[@TEST_NAME]]";

interface I4 {
  let T4:! type;
  let T5:! type;
  let T6:! type;
}

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+8]]:27: error: name `BAD1` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+4]]:61: error: name `BAD2` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR:                                                             ^~~~
// CHECK:STDERR:
impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+12]]:46: error: name `BAD3` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:                                              ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+8]]:61: error: name `BAD4` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:                                                             ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-10]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}

// --- fail_missing_on_definition.carbon
library "[[@TEST_NAME]]";

interface K { let V:! type; }

impl () as K where .V = {};
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE+11]]:12: error: associated constant V not given a value in impl of interface K [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl () as K {}
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-6]]:19: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR: interface K { let V:! type; }
// CHECK:STDERR:                   ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-8]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as K where .V = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as K {}

// --- fail_two_different.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

// CHECK:STDERR: fail_two_different.carbon:[[@LINE+4]]:12: error: associated constant W given two different values [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl () as L where .W = {} and .W = () {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as L where .W = {} and .W = () {}

// --- fail_two_different_first_bad.carbon
library "[[@TEST_NAME]]";

interface L2 { let W2:! type; }

// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+4]]:27: error: name `BAD5` not found [NameNotFound]
// CHECK:STDERR: impl () as L2 where .W2 = BAD5 and .W2 = () {}
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
impl () as L2 where .W2 = BAD5 and .W2 = () {}

// --- fail_two_different_second_bad.carbon
library "[[@TEST_NAME]]";

interface L3 { let W3:! type; }

// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+4]]:40: error: name `BAD6` not found [NameNotFound]
// CHECK:STDERR: impl () as L3 where .W3 = {} and .W3 = BAD6 {}
// CHECK:STDERR:                                        ^~~~
// CHECK:STDERR:
impl () as L3 where .W3 = {} and .W3 = BAD6 {}

// --- fail_two_different_both_bad.carbon
library "[[@TEST_NAME]]";

interface L4 { let W4:! type; }

// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+8]]:27: error: name `BAD7` not found [NameNotFound]
// CHECK:STDERR: impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+4]]:42: error: name `BAD8` not found [NameNotFound]
// CHECK:STDERR: impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}
// CHECK:STDERR:                                          ^~~~
// CHECK:STDERR:
impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}

// --- repeated.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

impl () as M where .X = {} and .X = {} {}

// --- non-type.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y:! {.a: {}};
}

impl () as N where .Y = {.a = {}} { }

// --- fail_where_rewrite_function.carbon
library "[[@TEST_NAME]]";

interface IF { fn F(); }

class CD { }

// CHECK:STDERR: fail_where_rewrite_function.carbon:[[@LINE+4]]:12: error: rewrite specified for associated function F [RewriteForAssociatedFunction]
// CHECK:STDERR: impl CD as IF where .F = 0 {
// CHECK:STDERR:            ^~~~~~~~~~~~~~~
// CHECK:STDERR:
impl CD as IF where .F = 0 {
  fn F() {}
}

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc5_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_7.2: type = converted %.loc5_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, @T.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc5_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc5_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc5_7.2 as %.loc5_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I2.type: type = facet_type <@I2> [concrete]
// CHECK:STDOUT:   %Self: %I2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I2.assoc_type: type = assoc_entity_type %I2.type [concrete]
// CHECK:STDOUT:   %assoc0: %I2.assoc_type = assoc_entity element0, @I2.%T2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %I2.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I2.facet: %I2.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %I2_where.type: type = facet_type <@I2 where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I2 = %I2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I2.decl: type = interface_decl @I2 [concrete = constants.%I2.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc5_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_7.2: type = converted %.loc5_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I2.ref.loc5: type = name_ref I2, file.%I2.decl [concrete = constants.%I2.type]
// CHECK:STDOUT:     %.Self.1: %I2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc5: %I2.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T2.ref.loc5: %I2.assoc_type = name_ref T2, @T2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc5: type = facet_access_type %.Self.ref.loc5 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc5_21: type = converted %.Self.ref.loc5, %.Self.as_type.loc5 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc5: <witness> = facet_access_witness %.Self.ref.loc5, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc5: type = impl_witness_access %.Self.as_wit.iface0.loc5, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_28.2: type = converted %.loc5_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_15: type = where_expr %.Self.1 [concrete = constants.%I2_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc5, %.loc5_28.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I2.ref.loc6: type = name_ref I2, file.%I2.decl [concrete = constants.%I2.type]
// CHECK:STDOUT:     %.Self.2: %I2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc6: %I2.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T2.ref.loc6: %I2.assoc_type = name_ref T2, @T2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc6: type = facet_access_type %.Self.ref.loc6 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc6_21: type = converted %.Self.ref.loc6, %.Self.as_type.loc6 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc6: <witness> = facet_access_witness %.Self.ref.loc6, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc6: type = impl_witness_access %.Self.as_wit.iface0.loc6, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc6_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_28.2: type = converted %.loc6_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc6_15: type = where_expr %.Self.2 [concrete = constants.%I2_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6, %.loc6_28.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I2 {
// CHECK:STDOUT:   %Self: %I2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T2: type = assoc_const_decl @T2 [concrete] {
// CHECK:STDOUT:     %assoc0: %I2.assoc_type = assoc_entity element0, @I2.%T2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T2 = @T2.%assoc0
// CHECK:STDOUT:   witness = (%T2)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T2(@I2.%Self: %I2.type) {
// CHECK:STDOUT:   assoc_const T2:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc5_7.2 as %.loc5_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T2(constants.%I2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_adds_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3> [concrete]
// CHECK:STDOUT:   %Self: %I3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I3.assoc_type: type = assoc_entity_type %I3.type [concrete]
// CHECK:STDOUT:   %assoc0: %I3.assoc_type = assoc_entity element0, @I3.%T3 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness.1bc: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %.Self: %I3.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I3.facet: %I3.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %I3_where.type: type = facet_type <@I3 where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %impl_witness.6de: <witness> = impl_witness (%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I3 = %I3.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I3.decl: type = interface_decl @I3 [concrete = constants.%I3.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.b46 [concrete] {} {
// CHECK:STDOUT:     %.loc9_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I3.ref: type = name_ref I3, file.%I3.decl [concrete = constants.%I3.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc9: <witness> = impl_witness () [concrete = constants.%impl_witness.1bc]
// CHECK:STDOUT:   impl_decl @impl.2b6 [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I3.ref: type = name_ref I3, file.%I3.decl [concrete = constants.%I3.type]
// CHECK:STDOUT:     %.Self: %I3.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T3.ref: %I3.assoc_type = name_ref T3, @T3.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc10_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_28.2: type = converted %.loc10_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_15: type = where_expr %.Self [concrete = constants.%I3_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc10_28.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc10: <witness> = impl_witness (constants.%empty_struct_type) [concrete = constants.%impl_witness.6de]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I3 {
// CHECK:STDOUT:   %Self: %I3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T3: type = assoc_const_decl @T3 [concrete] {
// CHECK:STDOUT:     %assoc0: %I3.assoc_type = assoc_entity element0, @I3.%T3 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T3 = @T3.%assoc0
// CHECK:STDOUT:   witness = (%T3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T3(@I3.%Self: %I3.type) {
// CHECK:STDOUT:   assoc_const T3:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b46: %.loc9_7.2 as %I3.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2b6: %.loc10_7.2 as %.loc10_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T3(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T3(constants.%I3.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %J_where.type.5f6: type = facet_type <@J where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %impl_witness.6de: <witness> = impl_witness (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %J_where.type.143: type = facet_type <@J where %impl.elem0 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %impl_witness.2a6: <witness> = impl_witness (%empty_tuple.type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.8c7 [concrete] {} {
// CHECK:STDOUT:     %.loc9_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_26.2: type = converted %.loc9_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc9_14: type = where_expr %.Self [concrete = constants.%J_where.type.5f6] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc9_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc9: <witness> = impl_witness (constants.%empty_struct_type) [concrete = constants.%impl_witness.6de]
// CHECK:STDOUT:   impl_decl @impl.6a8 [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc10_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc10_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_26.2: type = converted %.loc10_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_14: type = where_expr %.Self [concrete = constants.%J_where.type.143] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc10_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc10: <witness> = impl_witness (constants.%empty_tuple.type) [concrete = constants.%impl_witness.2a6]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   witness = (%U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8c7: %.loc9_7.2 as %.loc9_14;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6a8: %.loc10_7.2 as %.loc10_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_bad_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I4.type: type = facet_type <@I4> [concrete]
// CHECK:STDOUT:   %Self: %I4.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I4.assoc_type: type = assoc_entity_type %I4.type [concrete]
// CHECK:STDOUT:   %assoc0: %I4.assoc_type = assoc_entity element0, @I4.%T4 [concrete]
// CHECK:STDOUT:   %assoc1: %I4.assoc_type = assoc_entity element1, @I4.%T5 [concrete]
// CHECK:STDOUT:   %assoc2: %I4.assoc_type = assoc_entity element2, @I4.%T6 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %I4.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %I4.facet: %I4.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %.Self.as_wit.iface0, element1 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %impl.elem2: type = impl_witness_access %.Self.as_wit.iface0, element2 [symbolic_self]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I4 = %I4.decl
// CHECK:STDOUT:     .BAD1 = <poisoned>
// CHECK:STDOUT:     .BAD2 = <poisoned>
// CHECK:STDOUT:     .BAD3 = <poisoned>
// CHECK:STDOUT:     .BAD4 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I4.decl: type = interface_decl @I4 [concrete = constants.%I4.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.de506c.1 [concrete] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I4.ref: type = name_ref I4, file.%I4.decl [concrete = constants.%I4.type]
// CHECK:STDOUT:     %.Self: %I4.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc17_21: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T4.ref: %I4.assoc_type = name_ref T4, @T4.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc17_21: type = facet_access_type %.Self.ref.loc17_21 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc17_21: type = converted %.Self.ref.loc17_21, %.Self.as_type.loc17_21 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc17_21: <witness> = facet_access_witness %.Self.ref.loc17_21, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0.loc17_21, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD1.ref: <error> = name_ref BAD1, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc17_36: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T5.ref: %I4.assoc_type = name_ref T5, @T5.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_type.loc17_36: type = facet_access_type %.Self.ref.loc17_36 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc17_36: type = converted %.Self.ref.loc17_36, %.Self.as_type.loc17_36 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc17_36: <witness> = facet_access_witness %.Self.ref.loc17_36, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem1: type = impl_witness_access %.Self.as_wit.iface0.loc17_36, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:     %.loc17_48.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_48.2: type = converted %.loc17_48.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %.Self.ref.loc17_55: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T6.ref: %I4.assoc_type = name_ref T6, @T6.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %.Self.as_type.loc17_55: type = facet_access_type %.Self.ref.loc17_55 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc17_55: type = converted %.Self.ref.loc17_55, %.Self.as_type.loc17_55 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc17_55: <witness> = facet_access_witness %.Self.ref.loc17_55, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem2: type = impl_witness_access %.Self.as_wit.iface0.loc17_55, element2 [symbolic_self = constants.%impl.elem2]
// CHECK:STDOUT:     %BAD2.ref: <error> = name_ref BAD2, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc17_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem1, %struct_type.a
// CHECK:STDOUT:       requirement_rewrite %impl.elem2, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.de506c.2 [concrete] {} {
// CHECK:STDOUT:     %.loc31_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc31_7.2: type = converted %.loc31_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I4.ref: type = name_ref I4, file.%I4.decl [concrete = constants.%I4.type]
// CHECK:STDOUT:     %.Self: %I4.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc31_21: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T4.ref: %I4.assoc_type = name_ref T4, @T4.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc31_21: type = facet_access_type %.Self.ref.loc31_21 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc31_21: type = converted %.Self.ref.loc31_21, %.Self.as_type.loc31_21 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc31_21: <witness> = facet_access_witness %.Self.ref.loc31_21, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0.loc31_21, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc31_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc31_33.2: type = converted %.loc31_33.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     %.Self.ref.loc31_40: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T5.ref: %I4.assoc_type = name_ref T5, @T5.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_type.loc31_40: type = facet_access_type %.Self.ref.loc31_40 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc31_40: type = converted %.Self.ref.loc31_40, %.Self.as_type.loc31_40 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc31_40: <witness> = facet_access_witness %.Self.ref.loc31_40, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem1: type = impl_witness_access %.Self.as_wit.iface0.loc31_40, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:     %BAD3.ref: <error> = name_ref BAD3, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc31_55: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T6.ref: %I4.assoc_type = name_ref T6, @T6.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %.Self.as_type.loc31_55: type = facet_access_type %.Self.ref.loc31_55 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc31_55: type = converted %.Self.ref.loc31_55, %.Self.as_type.loc31_55 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc31_55: <witness> = facet_access_witness %.Self.ref.loc31_55, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem2: type = impl_witness_access %.Self.as_wit.iface0.loc31_55, element2 [symbolic_self = constants.%impl.elem2]
// CHECK:STDOUT:     %BAD4.ref: <error> = name_ref BAD4, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc31_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %struct_type.b
// CHECK:STDOUT:       requirement_rewrite %impl.elem1, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem2, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I4 {
// CHECK:STDOUT:   %Self: %I4.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T4: type = assoc_const_decl @T4 [concrete] {
// CHECK:STDOUT:     %assoc0: %I4.assoc_type = assoc_entity element0, @I4.%T4 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T5: type = assoc_const_decl @T5 [concrete] {
// CHECK:STDOUT:     %assoc1: %I4.assoc_type = assoc_entity element1, @I4.%T5 [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T6: type = assoc_const_decl @T6 [concrete] {
// CHECK:STDOUT:     %assoc2: %I4.assoc_type = assoc_entity element2, @I4.%T6 [concrete = constants.%assoc2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T4 = @T4.%assoc0
// CHECK:STDOUT:   .T5 = @T5.%assoc1
// CHECK:STDOUT:   .T6 = @T6.%assoc2
// CHECK:STDOUT:   witness = (%T4, %T5, %T6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T4(@I4.%Self: %I4.type) {
// CHECK:STDOUT:   assoc_const T4:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T5(@I4.%Self: %I4.type) {
// CHECK:STDOUT:   assoc_const T5:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T6(@I4.%Self: %I4.type) {
// CHECK:STDOUT:   assoc_const T6:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.de506c.1: %.loc17_7.2 as %.loc17_15;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.de506c.2: %.loc31_7.2 as %.loc31_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T4(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T5(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T6(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T4(constants.%I4.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T5(constants.%I4.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T6(constants.%I4.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_on_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type %K.type [concrete]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %K.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %K.facet: %K.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %K_where.type: type = facet_type <@K where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %impl_witness.6de: <witness> = impl_witness (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %impl_witness.85b: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [concrete = constants.%K.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.a6a [concrete] {} {
// CHECK:STDOUT:     %.loc5_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_7.2: type = converted %.loc5_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     %.Self: %K.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %K.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: %K.assoc_type = name_ref V, @V.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc5_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_14: type = where_expr %.Self [concrete = constants.%K_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc5_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc5: <witness> = impl_witness (constants.%empty_struct_type) [concrete = constants.%impl_witness.6de]
// CHECK:STDOUT:   impl_decl @impl.4b0 [concrete] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85b]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %V: type = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (%V)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@K.%Self: %K.type) {
// CHECK:STDOUT:   assoc_const V:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.a6a: %.loc5_7.2 as %.loc5_14;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4b0: %.loc17_7.2 as %K.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%K.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %Self: %L.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L.assoc_type: type = assoc_entity_type %L.type [concrete]
// CHECK:STDOUT:   %assoc0: %L.assoc_type = assoc_entity element0, @L.%W [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %L.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %L.facet: %L.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %L_where.type: type = facet_type <@L where %impl.elem0 = %empty_tuple.type and %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L.decl: type = interface_decl @L [concrete = constants.%L.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %L.ref: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     %.Self: %L.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc9_20: %L.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.ref.loc9_20: %L.assoc_type = name_ref W, @W.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc9_20: type = facet_access_type %.Self.ref.loc9_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_20: type = converted %.Self.ref.loc9_20, %.Self.as_type.loc9_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc9_20: <witness> = facet_access_witness %.Self.ref.loc9_20, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc9_20: type = impl_witness_access %.Self.as_wit.iface0.loc9_20, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_26.2: type = converted %.loc9_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.Self.ref.loc9_32: %L.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.ref.loc9_32: %L.assoc_type = name_ref W, @W.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc9_32: type = facet_access_type %.Self.ref.loc9_32 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_32: type = converted %.Self.ref.loc9_32, %.Self.as_type.loc9_32 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc9_32: <witness> = facet_access_witness %.Self.ref.loc9_32, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc9_32: type = impl_witness_access %.Self.as_wit.iface0.loc9_32, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc9_38.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_38.2: type = converted %.loc9_38.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_14: type = where_expr %.Self [concrete = constants.%L_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_20, %.loc9_26.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_32, %.loc9_38.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %L.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %W: type = assoc_const_decl @W [concrete] {
// CHECK:STDOUT:     %assoc0: %L.assoc_type = assoc_entity element0, @L.%W [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .W = @W.%assoc0
// CHECK:STDOUT:   witness = (%W)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @W(@L.%Self: %L.type) {
// CHECK:STDOUT:   assoc_const W:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc9_7.2 as %.loc9_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W(constants.%L.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_first_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L2.type: type = facet_type <@L2> [concrete]
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L2.assoc_type: type = assoc_entity_type %L2.type [concrete]
// CHECK:STDOUT:   %assoc0: %L2.assoc_type = assoc_entity element0, @L2.%W2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %L2.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %L2.facet: %L2.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .L2 = %L2.decl
// CHECK:STDOUT:     .BAD5 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L2.decl: type = interface_decl @L2 [concrete = constants.%L2.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %L2.ref: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     %.Self: %L2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc9_21: %L2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W2.ref.loc9_21: %L2.assoc_type = name_ref W2, @W2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc9_21: type = facet_access_type %.Self.ref.loc9_21 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_21: type = converted %.Self.ref.loc9_21, %.Self.as_type.loc9_21 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc9_21: <witness> = facet_access_witness %.Self.ref.loc9_21, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc9_21: type = impl_witness_access %.Self.as_wit.iface0.loc9_21, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD5.ref: <error> = name_ref BAD5, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc9_36: %L2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W2.ref.loc9_36: %L2.assoc_type = name_ref W2, @W2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc9_36: type = facet_access_type %.Self.ref.loc9_36 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_36: type = converted %.Self.ref.loc9_36, %.Self.as_type.loc9_36 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc9_36: <witness> = facet_access_witness %.Self.ref.loc9_36, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc9_36: type = impl_witness_access %.Self.as_wit.iface0.loc9_36, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc9_43.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_43.2: type = converted %.loc9_43.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_21, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_36, %.loc9_43.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L2 {
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %W2: type = assoc_const_decl @W2 [concrete] {
// CHECK:STDOUT:     %assoc0: %L2.assoc_type = assoc_entity element0, @L2.%W2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .W2 = @W2.%assoc0
// CHECK:STDOUT:   witness = (%W2)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @W2(@L2.%Self: %L2.type) {
// CHECK:STDOUT:   assoc_const W2:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc9_7.2 as %.loc9_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W2(constants.%L2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_second_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L3.type: type = facet_type <@L3> [concrete]
// CHECK:STDOUT:   %Self: %L3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L3.assoc_type: type = assoc_entity_type %L3.type [concrete]
// CHECK:STDOUT:   %assoc0: %L3.assoc_type = assoc_entity element0, @L3.%W3 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %L3.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %L3.facet: %L3.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .L3 = %L3.decl
// CHECK:STDOUT:     .BAD6 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L3.decl: type = interface_decl @L3 [concrete = constants.%L3.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc9_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %L3.ref: type = name_ref L3, file.%L3.decl [concrete = constants.%L3.type]
// CHECK:STDOUT:     %.Self: %L3.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc9_21: %L3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W3.ref.loc9_21: %L3.assoc_type = name_ref W3, @W3.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc9_21: type = facet_access_type %.Self.ref.loc9_21 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_21: type = converted %.Self.ref.loc9_21, %.Self.as_type.loc9_21 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc9_21: <witness> = facet_access_witness %.Self.ref.loc9_21, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc9_21: type = impl_witness_access %.Self.as_wit.iface0.loc9_21, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc9_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_28.2: type = converted %.loc9_28.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.Self.ref.loc9_34: %L3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W3.ref.loc9_34: %L3.assoc_type = name_ref W3, @W3.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc9_34: type = facet_access_type %.Self.ref.loc9_34 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc9_34: type = converted %.Self.ref.loc9_34, %.Self.as_type.loc9_34 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc9_34: <witness> = facet_access_witness %.Self.ref.loc9_34, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc9_34: type = impl_witness_access %.Self.as_wit.iface0.loc9_34, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD6.ref: <error> = name_ref BAD6, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc9_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_21, %.loc9_28.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc9_34, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L3 {
// CHECK:STDOUT:   %Self: %L3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %W3: type = assoc_const_decl @W3 [concrete] {
// CHECK:STDOUT:     %assoc0: %L3.assoc_type = assoc_entity element0, @L3.%W3 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .W3 = @W3.%assoc0
// CHECK:STDOUT:   witness = (%W3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @W3(@L3.%Self: %L3.type) {
// CHECK:STDOUT:   assoc_const W3:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc9_7.2 as %.loc9_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W3(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W3(constants.%L3.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_both_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L4.type: type = facet_type <@L4> [concrete]
// CHECK:STDOUT:   %Self: %L4.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L4.assoc_type: type = assoc_entity_type %L4.type [concrete]
// CHECK:STDOUT:   %assoc0: %L4.assoc_type = assoc_entity element0, @L4.%W4 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %L4.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %L4.facet: %L4.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .L4 = %L4.decl
// CHECK:STDOUT:     .BAD7 = <poisoned>
// CHECK:STDOUT:     .BAD8 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L4.decl: type = interface_decl @L4 [concrete = constants.%L4.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc13_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_7.2: type = converted %.loc13_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %L4.ref: type = name_ref L4, file.%L4.decl [concrete = constants.%L4.type]
// CHECK:STDOUT:     %.Self: %L4.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc13_21: %L4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W4.ref.loc13_21: %L4.assoc_type = name_ref W4, @W4.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc13_21: type = facet_access_type %.Self.ref.loc13_21 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_21: type = converted %.Self.ref.loc13_21, %.Self.as_type.loc13_21 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc13_21: <witness> = facet_access_witness %.Self.ref.loc13_21, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc13_21: type = impl_witness_access %.Self.as_wit.iface0.loc13_21, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD7.ref: <error> = name_ref BAD7, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc13_36: %L4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W4.ref.loc13_36: %L4.assoc_type = name_ref W4, @W4.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc13_36: type = facet_access_type %.Self.ref.loc13_36 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_36: type = converted %.Self.ref.loc13_36, %.Self.as_type.loc13_36 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc13_36: <witness> = facet_access_witness %.Self.ref.loc13_36, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc13_36: type = impl_witness_access %.Self.as_wit.iface0.loc13_36, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD8.ref: <error> = name_ref BAD8, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc13_21, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc13_36, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L4 {
// CHECK:STDOUT:   %Self: %L4.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %W4: type = assoc_const_decl @W4 [concrete] {
// CHECK:STDOUT:     %assoc0: %L4.assoc_type = assoc_entity element0, @L4.%W4 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .W4 = @W4.%assoc0
// CHECK:STDOUT:   witness = (%W4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @W4(@L4.%Self: %L4.type) {
// CHECK:STDOUT:   assoc_const W4:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc13_7.2 as %.loc13_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W4(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W4(constants.%L4.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type %M.type [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.%X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %M_where.type: type = facet_type <@M where %impl.elem0 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc5_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_7.2: type = converted %.loc5_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc5_20: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref.loc5_20: %M.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc5_20: type = facet_access_type %.Self.ref.loc5_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc5_20: type = converted %.Self.ref.loc5_20, %.Self.as_type.loc5_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc5_20: <witness> = facet_access_witness %.Self.ref.loc5_20, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc5_20: type = impl_witness_access %.Self.as_wit.iface0.loc5_20, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.Self.ref.loc5_32: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref.loc5_32: %M.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc5_32: type = facet_access_type %.Self.ref.loc5_32 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc5_32: type = converted %.Self.ref.loc5_32, %.Self.as_type.loc5_32 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0.loc5_32: <witness> = facet_access_witness %.Self.ref.loc5_32, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0.loc5_32: type = impl_witness_access %.Self.as_wit.iface0.loc5_32, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_38.2: type = converted %.loc5_38.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_14: type = where_expr %.Self [concrete = constants.%M_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc5_20, %.loc5_26.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc5_32, %.loc5_38.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%X [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (%X)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc5_7.2 as %.loc5_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%M.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- non-type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %Self: %N.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.a.225: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N.assoc_type: type = assoc_entity_type %N.type [concrete]
// CHECK:STDOUT:   %assoc0: %N.assoc_type = assoc_entity element0, @N.%Y [concrete]
// CHECK:STDOUT:   %.Self: %N.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %N.facet: %N.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.a.225 = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.225 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %N_where.type: type = facet_type <@N where %impl.elem0 = %struct> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%struct) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: type = interface_decl @N [concrete = constants.%N.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc7_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_7.2: type = converted %.loc7_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %N.ref: type = name_ref N, file.%N.decl [concrete = constants.%N.type]
// CHECK:STDOUT:     %.Self: %N.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %N.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Y.ref: %N.assoc_type = name_ref Y, @Y.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc7_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: %struct_type.a.225 = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc7_32: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_33.1: %struct_type.a.225 = struct_literal (%.loc7_32)
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_33.2: %empty_struct_type = converted %.loc7_32, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %struct: %struct_type.a.225 = struct_value (%.loc7_33.2) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc7_33.3: %struct_type.a.225 = converted %.loc7_33.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc7_14: type = where_expr %.Self [concrete = constants.%N_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc7_33.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%struct) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @N {
// CHECK:STDOUT:   %Self: %N.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Y: %struct_type.a.225 = assoc_const_decl @Y [concrete] {
// CHECK:STDOUT:     %assoc0: %N.assoc_type = assoc_entity element0, @N.%Y [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = @Y.%assoc0
// CHECK:STDOUT:   witness = (%Y)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Y(@N.%Self: %N.type) {
// CHECK:STDOUT:   assoc_const Y:! %struct_type.a.225;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc7_7.2 as %.loc7_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%N.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_where_rewrite_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IF.type: type = facet_type <@IF> [concrete]
// CHECK:STDOUT:   %Self: %IF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.496: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.1de: %F.type.496 = struct_value () [concrete]
// CHECK:STDOUT:   %IF.assoc_type: type = assoc_entity_type %IF.type [concrete]
// CHECK:STDOUT:   %assoc0: %IF.assoc_type = assoc_entity element0, @IF.%F.decl [concrete]
// CHECK:STDOUT:   %CD: type = class_type @CD [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: %IF.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self, element0 [symbolic_self]
// CHECK:STDOUT:   %IF.facet.94f: %IF.type = facet_value %.Self.as_type, (%.Self.as_wit.iface0) [symbolic_self]
// CHECK:STDOUT:   %.bf5: type = fn_type_with_self_type %F.type.496, %IF.facet.94f [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %.bf5 = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %IF_where.type: type = facet_type <@IF where %impl.elem0 = %int_0> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.2b5: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.479: %F.type.2b5 = struct_value () [concrete]
// CHECK:STDOUT:   %IF.facet.0db: %IF.type = facet_value %CD, (%impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IF = %IF.decl
// CHECK:STDOUT:     .CD = %CD.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IF.decl: type = interface_decl @IF [concrete = constants.%IF.type] {} {}
// CHECK:STDOUT:   %CD.decl: type = class_decl @CD [concrete = constants.%CD] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %CD.ref: type = name_ref CD, file.%CD.decl [concrete = constants.%CD]
// CHECK:STDOUT:     %IF.ref: type = name_ref IF, file.%IF.decl [concrete = constants.%IF.type]
// CHECK:STDOUT:     %.Self: %IF.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %IF.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %F.ref: %IF.assoc_type = name_ref F, @IF.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc11_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit.iface0: <witness> = facet_access_witness %.Self.ref, element0 [symbolic_self = constants.%.Self.as_wit.iface0]
// CHECK:STDOUT:     %impl.elem0: %.bf5 = impl_witness_access %.Self.as_wit.iface0, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc11_15: type = where_expr %.Self [concrete = constants.%IF_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %int_0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @IF {
// CHECK:STDOUT:   %Self: %IF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.496 = fn_decl @F.1 [concrete = constants.%F.1de] {} {}
// CHECK:STDOUT:   %assoc0: %IF.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %CD.ref as %.loc11_15 {
// CHECK:STDOUT:   %F.decl: %F.type.2b5 = fn_decl @F.2 [concrete = constants.%F.479] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CD {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%CD
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@IF.%Self: %IF.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%IF.facet.0db) {}
// CHECK:STDOUT:

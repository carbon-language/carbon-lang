// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-prelude-import
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/impl_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/impl_assoc_const.carbon

// --- success.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; }

impl () as I where .T = {} {}

// --- success_associated.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; let U:! type; }

impl () as I where .T = .U and .U = {} {}

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

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+8]]:46: error: name `BAD3` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:                                              ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+4]]:61: error: name `BAD4` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:                                                             ^~~~
// CHECK:STDERR:
impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}

// --- fail_missing_on_definition.carbon
library "[[@TEST_NAME]]";

interface K { let V:! type; }

impl () as K where .V = {};
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE+11]]:1: error: associated constant V not given a value in impl of interface K [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl () as K {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
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

// CHECK:STDERR: fail_two_different.carbon:[[@LINE+4]]:12: error: associated constant `.(L.W)` given two different values `()` and `{}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl () as L where .W = {} and .W = () {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as L where .W = {} and .W = () {}

// --- fail_two_different_first_associated.carbon
library "[[@TEST_NAME]]";

interface L2 { let W2:! type; let X2:! type; }

// CHECK:STDERR: fail_two_different_first_associated.carbon:[[@LINE+11]]:12: error: associated constant `.(L2.W2)` given two different values `()` and `.(L2.X2)` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl () as L2 where .W2 = .X2 and .W2 = () {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_first_associated.carbon:[[@LINE+7]]:1: error: associated constant X2 not given a value in impl of interface L2 [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl () as L2 where .W2 = .X2 and .W2 = () {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_two_different_first_associated.carbon:[[@LINE-9]]:35: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR: interface L2 { let W2:! type; let X2:! type; }
// CHECK:STDERR:                                   ^~~~~~~~~
// CHECK:STDERR:
impl () as L2 where .W2 = .X2 and .W2 = () {}

// --- fail_two_different_second_associated.carbon
library "[[@TEST_NAME]]";

interface L2 { let W2:! type; let X2:! type; }

// CHECK:STDERR: fail_two_different_second_associated.carbon:[[@LINE+11]]:12: error: associated constant `.(L2.W2)` given two different values `()` and `.(L2.X2)` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl () as L2 where .W2 = () and .W2 = .X2 {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_second_associated.carbon:[[@LINE+7]]:1: error: associated constant X2 not given a value in impl of interface L2 [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl () as L2 where .W2 = () and .W2 = .X2 {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_two_different_second_associated.carbon:[[@LINE-9]]:35: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR: interface L2 { let W2:! type; let X2:! type; }
// CHECK:STDERR:                                   ^~~~~~~~~
// CHECK:STDERR:
impl () as L2 where .W2 = () and .W2 = .X2 {}

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

// --- fail_many_different.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

// The facet type should have a single rewrite with `<error>` in the RHS (they
// are all errors, and they get deduped).
//
//@dump-sem-ir-begin
// CHECK:STDERR: fail_many_different.carbon:[[@LINE+4]]:12: error: associated constant `.(L.W)` given two different values `((), (), ())` and `({}, (), ())` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl () as L where .W = ((), (), ()) and .W = ({}, (), ()) and .W = ({}, {}, ()) and .W = ({}, (), {}) {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as L where .W = ((), (), ()) and .W = ({}, (), ()) and .W = ({}, {}, ()) and .W = ({}, (), {}) {}
//@dump-sem-ir-end

// --- repeated.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

impl () as M where .X = {} and .X = {} {}

// --- repeated_associated.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

impl () as M where .X = .Y and .X = .Y and .Y = () {}

// --- fail_repeated_and_different.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

// CHECK:STDERR: fail_repeated_and_different.carbon:[[@LINE+4]]:12: error: associated constant `.(M.X)` given two different values `()` and `{}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl () as M where .X = {} and .X = () and .X = {} {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as M where .X = {} and .X = () and .X = {} {}

// --- fail_cycle_single.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

// This fails because it resolves to `.X = .X` which is cyclical.
//
// CHECK:STDERR: fail_cycle_single.carbon:[[@LINE+4]]:12: error: found cycle in facet type constraint for `.(M.X)` [FacetTypeConstraintCycle]
// CHECK:STDERR: impl () as M where .X = .X {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as M where .X = .X {}

// --- fail_cycle.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; let Z:! type; }

// This fails because it resolves to `.X = .X` which is cyclical.
// The value of .X and .Y becomes <error> but .Z is still valid.
//
//@dump-sem-ir-begin
// CHECK:STDERR: fail_cycle.carbon:[[@LINE+4]]:12: error: found cycle in facet type constraint for `.(M.X)` [FacetTypeConstraintCycle]
// CHECK:STDERR: impl () as M where .X = .Y and .Y = .X and .Z = () {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as M where .X = .Y and .Y = .X and .Z = () {}
//@dump-sem-ir-end

// --- fail_todo_cycle_between_interfaces.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}
interface J {
  let X3:! type;
}

// TODO: This should fail due to a cyclic definition of each value. But right
// now it's failing due to compound member access in a rewrite constraint not
// working.
//
// CHECK:STDERR: fail_todo_cycle_between_interfaces.carbon:[[@LINE+12]]:45: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR: impl () as I where .Self impls J and .X1 = .(J.X3) and .X2 = .X1 and .(J.X3) = .X2 {}
// CHECK:STDERR:                                             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_cycle_between_interfaces.carbon:[[@LINE+8]]:45: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: impl () as I where .Self impls J and .X1 = .(J.X3) and .X2 = .X1 and .(J.X3) = .X2 {}
// CHECK:STDERR:                                             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_cycle_between_interfaces.carbon:[[@LINE+4]]:71: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR: impl () as I where .Self impls J and .X1 = .(J.X3) and .X2 = .X1 and .(J.X3) = .X2 {}
// CHECK:STDERR:                                                                       ^
// CHECK:STDERR:
impl () as I where .Self impls J and .X1 = .(J.X3) and .X2 = .X1 and .(J.X3) = .X2 {}

// --- non-type.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y:! {.a: {}};
}

impl () as N where .Y = {.a = {}} { }

// --- non-type_repeated.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y:! {.a: {}};
}

impl () as N where .Y = {.a = {}} and .Y = {.a = {}} { }

// --- fail_non-type_different.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y:! {.a: type};
}

// CHECK:STDERR: fail_non-type_different.carbon:[[@LINE+4]]:12: error: associated constant `.(N.Y)` given two different values `{.a = {}}` and `{.a = ()}` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl () as N where .Y = {.a = {}} and .Y = {.a = ()} {}
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as N where .Y = {.a = {}} and .Y = {.a = ()} {}

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

// CHECK:STDOUT: --- fail_many_different.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %L.assoc_type: type = assoc_entity_type @L [concrete]
// CHECK:STDOUT:   %assoc0: %L.assoc_type = assoc_entity element0, @L.%W [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %L.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %L.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @L [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %L.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %tuple.type.2d5: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.e5a: type = tuple_type (%empty_struct_type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.type.d7e: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.type.bd8: type = tuple_type (%empty_struct_type, %empty_tuple.type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %L_where.type: type = facet_type <@L where %impl.elem0 = <error>> [concrete]
// CHECK:STDOUT:   %L.impl_witness: <witness> = impl_witness file.%L.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc13_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_7.2: type = converted %.loc13_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %L.ref: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref.loc13_20: %L.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.ref.loc13_20: %L.assoc_type = name_ref W, @W.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc13_20: type = facet_access_type %.Self.ref.loc13_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_20: type = converted %.Self.ref.loc13_20, %.Self.as_type.loc13_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0.loc13_20: type = impl_witness_access constants.%L.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc13_27: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_31: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_36.1: %tuple.type.2d5 = tuple_literal (%.loc13_27, %.loc13_31, %.loc13_35)
// CHECK:STDOUT:     %.loc13_36.2: type = converted %.loc13_27, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13_36.3: type = converted %.loc13_31, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13_36.4: type = converted %.loc13_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13_36.5: type = converted %.loc13_36.1, constants.%tuple.type.2d5 [concrete = constants.%tuple.type.2d5]
// CHECK:STDOUT:     %.Self.ref.loc13_42: %L.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.ref.loc13_42: %L.assoc_type = name_ref W, @W.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc13_42: type = facet_access_type %.Self.ref.loc13_42 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_42: type = converted %.Self.ref.loc13_42, %.Self.as_type.loc13_42 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0.loc13_42: type = impl_witness_access constants.%L.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc13_49: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_53: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_57: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_58.1: %tuple.type.e5a = tuple_literal (%.loc13_49, %.loc13_53, %.loc13_57)
// CHECK:STDOUT:     %.loc13_58.2: type = converted %.loc13_49, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc13_58.3: type = converted %.loc13_53, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13_58.4: type = converted %.loc13_57, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13_58.5: type = converted %.loc13_58.1, constants.%tuple.type.e5a [concrete = constants.%tuple.type.e5a]
// CHECK:STDOUT:     %.Self.ref.loc13_64: %L.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.ref.loc13_64: %L.assoc_type = name_ref W, @W.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc13_64: type = facet_access_type %.Self.ref.loc13_64 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_64: type = converted %.Self.ref.loc13_64, %.Self.as_type.loc13_64 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0.loc13_64: type = impl_witness_access constants.%L.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc13_71: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_75: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_79: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_80.1: %tuple.type.d7e = tuple_literal (%.loc13_71, %.loc13_75, %.loc13_79)
// CHECK:STDOUT:     %.loc13_80.2: type = converted %.loc13_71, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc13_80.3: type = converted %.loc13_75, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc13_80.4: type = converted %.loc13_79, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13_80.5: type = converted %.loc13_80.1, constants.%tuple.type.d7e [concrete = constants.%tuple.type.d7e]
// CHECK:STDOUT:     %.Self.ref.loc13_86: %L.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.ref.loc13_86: %L.assoc_type = name_ref W, @W.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc13_86: type = facet_access_type %.Self.ref.loc13_86 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_86: type = converted %.Self.ref.loc13_86, %.Self.as_type.loc13_86 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0.loc13_86: type = impl_witness_access constants.%L.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc13_93: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_97: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_101: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_102.1: %tuple.type.bd8 = tuple_literal (%.loc13_93, %.loc13_97, %.loc13_101)
// CHECK:STDOUT:     %.loc13_102.2: type = converted %.loc13_93, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc13_102.3: type = converted %.loc13_97, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13_102.4: type = converted %.loc13_101, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc13_102.5: type = converted %.loc13_102.1, constants.%tuple.type.bd8 [concrete = constants.%tuple.type.bd8]
// CHECK:STDOUT:     %.loc13_14: type = where_expr %.Self [concrete = constants.%L_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc13_20, %.loc13_36.5
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc13_42, %.loc13_58.5
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc13_64, %.loc13_80.5
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc13_86, %.loc13_102.5
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L.impl_witness_table = impl_witness_table (<error>), @impl [concrete]
// CHECK:STDOUT:   %L.impl_witness: <witness> = impl_witness %L.impl_witness_table [concrete = constants.%L.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc13_7.2 as %.loc13_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%L.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_cycle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type @M [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.%X [concrete]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%Y [concrete]
// CHECK:STDOUT:   %assoc2: %M.assoc_type = assoc_entity element2, @M.%Z [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %M.lookup_impl_witness: <witness> = lookup_impl_witness %.Self, @M [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %M.lookup_impl_witness, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %M.lookup_impl_witness, element1 [symbolic_self]
// CHECK:STDOUT:   %impl.elem2: type = impl_witness_access %M.lookup_impl_witness, element2 [symbolic_self]
// CHECK:STDOUT:   %M_where.type: type = facet_type <@M where %impl.elem0 = <error> and %impl.elem1 = <error> and %impl.elem2 = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness file.%M.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc13_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_7.2: type = converted %.loc13_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.Self.ref.loc13_20: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref.loc13_20: %M.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc13_20: type = facet_access_type %.Self.ref.loc13_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_20: type = converted %.Self.ref.loc13_20, %.Self.as_type.loc13_20 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0.loc13_20: type = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.Self.ref.loc13_25: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Y.ref.loc13_25: %M.assoc_type = name_ref Y, @Y.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_type.loc13_25: type = facet_access_type %.Self.ref.loc13_25 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_25: type = converted %.Self.ref.loc13_25, %.Self.as_type.loc13_25 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem1.loc13_25: type = impl_witness_access constants.%M.lookup_impl_witness, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:     %.Self.ref.loc13_32: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Y.ref.loc13_32: %M.assoc_type = name_ref Y, @Y.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_type.loc13_32: type = facet_access_type %.Self.ref.loc13_32 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_32: type = converted %.Self.ref.loc13_32, %.Self.as_type.loc13_32 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem1.loc13_32: type = impl_witness_access constants.%M.lookup_impl_witness, element1 [symbolic_self = constants.%impl.elem1]
// CHECK:STDOUT:     %.Self.ref.loc13_37: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref.loc13_37: %M.assoc_type = name_ref X, @X.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type.loc13_37: type = facet_access_type %.Self.ref.loc13_37 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_37: type = converted %.Self.ref.loc13_37, %.Self.as_type.loc13_37 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem0.loc13_37: type = impl_witness_access constants.%M.lookup_impl_witness, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.Self.ref.loc13_44: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %.Self.as_type.loc13_44: type = facet_access_type %.Self.ref.loc13_44 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc13_44: type = converted %.Self.ref.loc13_44, %.Self.as_type.loc13_44 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %impl.elem2: type = impl_witness_access constants.%M.lookup_impl_witness, element2 [symbolic_self = constants.%impl.elem2]
// CHECK:STDOUT:     %.loc13_50.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_50.2: type = converted %.loc13_50.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13_14: type = where_expr %.Self [concrete = constants.%M_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc13_20, %impl.elem1.loc13_25
// CHECK:STDOUT:       requirement_rewrite %impl.elem1.loc13_32, %impl.elem0.loc13_37
// CHECK:STDOUT:       requirement_rewrite %impl.elem2, %.loc13_50.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.impl_witness_table = impl_witness_table (<error>, <error>, %impl_witness_assoc_constant), @impl [concrete]
// CHECK:STDOUT:   %M.impl_witness: <witness> = impl_witness %M.impl_witness_table [concrete = constants.%M.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc13_7.2 as %.loc13_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%M.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:

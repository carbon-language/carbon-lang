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

// --- repeated.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

impl () as M where .X = {} and .X = {} {}

// --- repeated_associated.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

impl () as M where .X = .Y and .X = .Y and .Y = () {}

// --- todo_fail_cycle_single.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

// TODO: This should fail, as it resolves to `.X = .X` which is cyclical.
impl () as M where .X = .X {}

// --- todo_fail_cycle.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; let Y:! type; }

// TODO: This should fail, as it resolves to `.X = .X` which is cyclical.
impl () as M where .X = .Y and .Y = .X {}

// --- fail_todo_cycle_between_interfaces.carbon
library "[[@TEST_NAME]]";

interface I {
  let X1:! type;
  let X2:! type;
}
interface J {
  let X3:! type;
}

// TODO: This should fail, due to a cyclic definition of each value. But right
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


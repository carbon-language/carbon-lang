// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/nested_facet_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/nested_facet_types.carbon

// --- nested_facet_types.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  let U:! type;
}

fn F(FF:! (Z where .T = ()) where .U = .T) -> FF.U {
  // CHECK:STDERR: nested_facet_types.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `()` to `.(Z.T)` [ConversionFailure]
  // CHECK:STDERR:   return ();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: nested_facet_types.carbon:[[@LINE+4]]:3: note: type `()` does not implement interface `Core.ImplicitAs(.(Z.T))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return ();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  return ();
}

// --- nested_facet_types_same.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
}

fn F(FF:! ((Z where .T = ()) where .T = ()) where .T = ()) -> FF.T {
  return ();
}

// --- nested_facet_types_same_associated.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  let U:! type;
}

fn F(FF:! ((Z where .T = .U) where .T = .U) where .T = .U) {}

// --- nested_facet_types_same_two_associated.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  let U:! type;
  let V:! type;
}

fn F(FF:! ((Z where .T = .U and .U = .V) where .T = .U and .U = .V) where .T = .U and .U = .V) {}

// --- nested_facet_types_same_associated_in_generic_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  let U:! type;
}
class C(T:! type) {}

fn F(FF:! ((Z where .T = C(.U)) where .T = C(.U)) where .T = C(.U)) {}

// --- fail_nested_facet_types_different.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
}

fn F(FF:! (Z where .T = ()) where .T = {}) {}

// --- fail_nested_facet_types_different_with_associated.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  let U:! type;
}

fn F(FF:! (Z where .T = .U) where .T = {}) {}

// --- fail_nested_facet_types_different_with_associated_in_generic_parameter.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  let U:! type;
}
class C(T:! type) {}

fn F(FF:! (Z where .T = C(.U)) where .T = {}) {}

// --- nested_facet_types_same_with_bitand.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  let U:! type;
}

fn F(FF:! ((Z where .T = .U) where .T = .U) & (Z where .T = .U)) {}

// --- fail_nested_facet_types_different_with_bitand.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  let U:! type;
}

fn F(FF:! ((Z where .T = .U) where .T = .U) & (Z where .T = {})) {}

// --- fail_nested_facet_type_cycle.carbon
library "[[@TEST_NAME]]";

interface Z {
  let T:! type;
  let U:! type;
}

fn F(FF:! (Z where .T = .U) where .U = .T) {}

// --- fail_nested_facet_types_different_separated_order_one.carbon
library "[[@TEST_NAME]]";

interface Z {
  let S:! type;
  let T:! type;
  let U:! type;
  let V:! type;
}

// The ordering here is meant to find a way to insert a LHS between the two `.S`
// ids, by producing different orderings to get an id between them.

fn F(FF:! (((Z where .S = .U) where .V = {}) where .T = .U) & (Z where .S = {})) {}

// --- fail_nested_facet_types_different_separated_order_two.carbon
library "[[@TEST_NAME]]";

interface Z {
  let S:! type;
  let T:! type;
  let U:! type;
  let V:! type;
}

// The ordering here is meant to find a way to insert a LHS between the two `.S`
// ids, by producing different orderings to get an id between them. As of today,
// this is the case where `.T` is sorted between the two `.S` rewrite rules.

fn F(FF:! (((Z where .V = {}) where .S = .U) where .T = .U) & (Z where .S = {})) {}

// --- fail_nested_facet_types_different_separated_order_three.carbon
library "[[@TEST_NAME]]";

interface Z {
  let S:! type;
  let T:! type;
  let U:! type;
  let V:! type;
}

// The ordering here is meant to find a way to insert a LHS between the two `.S`
// ids, by producing different orderings to get an id between them.

fn F(FF:! (((Z where .V = {}) where .T = .U) where .S = .U) & (Z where .S = {})) {}

// --- repeated_with_value_available.carbon
library "[[@TEST_NAME]]";

interface Z { let X:! type; let Y:! type; }

fn F(T:! ((Z where .Y = ()) where .X = .Y) where .X = .Y) -> T.X {
  // CHECK:STDERR: repeated_with_value_available.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `()` to `.(Z.Y)` [ConversionFailure]
  // CHECK:STDERR:   return ();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: repeated_with_value_available.carbon:[[@LINE+4]]:3: note: type `()` does not implement interface `Core.ImplicitAs(.(Z.Y))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return ();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  return ();
}

class C(T:! type) { adapt (); }

fn F1(T:! (Z where .Y = C(()) and .X = C(.Y)) where .X = C(C(()))) -> T.X {
  return () as C(C(()));
}

fn F2(T:! ((Z where .Y = C(())) where .X = C(.Y)) where .X = C(C(()))) -> T.X {
  // CHECK:STDERR: repeated_with_value_available.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `C(C(()))` to `C(.(Z.Y))` [ConversionFailure]
  // CHECK:STDERR:   return () as C(C(()));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: repeated_with_value_available.carbon:[[@LINE+4]]:3: note: type `C(C(()))` does not implement interface `Core.ImplicitAs(C(.(Z.Y)))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return () as C(C(()));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return () as C(C(()));
}

fn F3(T:! Z where .Y = C(()) and .X = C(.Y) and .X = C(C(()))) -> T.X {
  return () as C(C(()));
}

// --- conflicting_syntax_same_value.carbon
library "[[@TEST_NAME]]";

interface Z { let X:! type; let Y:! type; }

fn F1(T:! ((Z where .Y = ()) where .X = .Y) where .X = ()) -> T.X {
  // CHECK:STDERR: conflicting_syntax_same_value.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `()` to `.(Z.Y)` [ConversionFailure]
  // CHECK:STDERR:   return ();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: conflicting_syntax_same_value.carbon:[[@LINE+4]]:3: note: type `()` does not implement interface `Core.ImplicitAs(.(Z.Y))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return ();
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  return ();
}

fn F2(T:! ((Z where .Y = ()) where .X = ()) where .X = .Y) -> T.X {
  return ();
}

// --- nested_impls_repeated.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface Z {
  let T:! type;
  let U:! type;
}
class C(T:! type) {}

fn F(FF:! ((Y where .Self impls (Z where .T = C(.U)))
                where .Self impls (Z where .T = C(.U)))
                  where .Self impls (Z where .T = C(.U))) {}

// --- nested_impl_unique.carbon
library "[[@TEST_NAME]]";

interface Y {
  let Y1:! type;
}
interface Z {
  let Z1:! type;
  let Z2:! type;
}
class C(T:! type) { adapt (); }

// CHECK:STDERR: nested_impl_unique.carbon:[[@LINE+4]]:90: error: cannot convert type `FF` that implements `Y where .(Y.Y1) = () and...` into type implementing `Z` [ConversionFailureFacetToFacet]
// CHECK:STDERR: fn F(FF:! ((Y where .Y1 = ()) where .Self impls (Z where .Z1 = C(.Z2) and .Z2 = ()))) -> FF.(Z.Z1) {
// CHECK:STDERR:                                                                                          ^~~~~~~~~
// CHECK:STDERR:
fn F(FF:! ((Y where .Y1 = ()) where .Self impls (Z where .Z1 = C(.Z2) and .Z2 = ()))) -> FF.(Z.Z1) {
  return () as C(());
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/partially_identified.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/partially_identified.carbon

// --- fail_self_doesnt_meet_constraint.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface NeedsY(T:! Y) {}
constraint W {
  // CHECK:STDERR: fail_self_doesnt_meet_constraint.carbon:[[@LINE+7]]:24: error: cannot convert type `Self` that implements `W` into type implementing `Y` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   extend require impls NeedsY(Self);
  // CHECK:STDERR:                        ^~~~~~~~~~~~
  // CHECK:STDERR: fail_self_doesnt_meet_constraint.carbon:[[@LINE-5]]:18: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface NeedsY(T:! Y) {}
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  extend require impls NeedsY(Self);
}

// --- self_meets_constraint.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface NeedsY(T:! Y) {}
constraint W {
  require impls Y;
  extend require impls NeedsY(Self);
}

fn F(w:! W) {
  w as Y;
  w as NeedsY(w);
}

// --- self_meets_constraint_through_indirection.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface NeedsY(T:! Y) {}
constraint GivesY {
  require impls Y;
}
constraint W {
  require impls GivesY;
  extend require impls NeedsY(Self);
}

fn F(w:! W) {
  w as Y;
  w as NeedsY(w);
}

// --- self_in_constraint_specific_interface.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface NeedsY(T:! Y) {}

// All types impl Y. This requires an impl lookup to find a witness, and it will
// be a symbolic witness for a symbolic self type.
impl forall [T:! type] T as Y {}

constraint W {
  // Constructs a FacetValue of `Self as Y`, containing the impl lookup
  // instruction as its witness, which will be part of the generic definition of
  // this require.
  require impls NeedsY(Self);
}

class C {}
impl C as NeedsY(C) {}

fn G(w:! W) {
  // The witness of NeedsY(Self) monomorphized by `w` must be the same as the
  // witness of `NeedsY(w)` constructed directly here in order for this to find
  // `NeedsY(w)` in the facet type `W`.
  w as NeedsY(w);
}

fn F() {
  // Further monomorphize `NeedsY(w)` to `NeedsY(C)`.
  G(C);
}

// --- fail_todo_facet_access_type_of_self_meets_constraint.carbon
library "[[@TEST_NAME]]";

class C(T:! type) {}

interface Y {}
interface NeedsY(T:! Y) {}
constraint W {
  require C(Self) impls Y;
  // CHECK:STDERR: fail_todo_facet_access_type_of_self_meets_constraint.carbon:[[@LINE+7]]:17: error: cannot convert type `C(Self)` into type implementing `Y` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   require impls NeedsY(C(Self));
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_facet_access_type_of_self_meets_constraint.carbon:[[@LINE-6]]:18: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface NeedsY(T:! Y) {}
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  require impls NeedsY(C(Self));
}

fn F(w:! W) {
  // TODO: Lookups for `C(w) as <something>` need to look into the facet types
  // of any facet value in the type `C(w)`, so that it can find a witness from
  // the constraint `W`.
  // CHECK:STDERR: fail_todo_facet_access_type_of_self_meets_constraint.carbon:[[@LINE+4]]:3: error: cannot convert type `C(w)` into type implementing `Y` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C(w) as Y;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  C(w) as Y;
  // CHECK:STDERR: fail_todo_facet_access_type_of_self_meets_constraint.carbon:[[@LINE+7]]:11: error: cannot convert type `C(w)` into type implementing `Y` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C(w) as NeedsY(C(w));
  // CHECK:STDERR:           ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_facet_access_type_of_self_meets_constraint.carbon:[[@LINE-25]]:18: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface NeedsY(T:! Y) {}
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  C(w) as NeedsY(C(w));
}

// --- multiple_use_of_self.carbon
library "[[@TEST_NAME]]";

interface Y {}
interface NeedsY(T:! Y) {}
interface NeedsNeedsY(T:! Y, U:! NeedsY(T)) {}
constraint W {
  require impls Y;
  // The identified facet type of Self is {Y}.
  extend require impls NeedsY(Self);
  // The identified facet type of Self is {Y, NeedsY}. Ensure we don't keep
  // seeing the previous identified facet type of Self.
  extend require impls NeedsNeedsY(Self, Self);
}

fn F(w:! W) {
  w as Y;
  w as NeedsY(w);
}

// --- access_through_previous_require.carbon
library "[[@TEST_NAME]]";

interface Z { let Z0:! type; }
interface Y {}
interface X {
  let X0:! type;
  fn XF() -> X0;
}

class P(T:! type) { adapt (); }

constraint M {
  require impls Z where .Z0 = {};
  require impls Y;
}
constraint N {
  require impls M;
  // The facts from M travel back to here.
  require impls X where .X0 = P(Self.(Z.Z0));
}

fn F(T:! N) -> T.(X.X0) { return T.(X.XF)(); }

fn G() -> P({}) {
  class C {}
  impl C as Z where .Z0 = {} {}
  impl C as Y {}
  impl C as X where .X0 = P({}) {
    fn XF() -> P({}) { return () as P({}); }
  }
  return F(C);
}

// --- fail_impl_lookup_target_partially_identified.carbon
library "[[@TEST_NAME]]";

interface Z {}

interface X { let XZ:! Z; }
constraint W {
  require impls Z;
  // Target of impl lookup can not be partially identified.
  // CHECK:STDERR: fail_impl_lookup_target_partially_identified.carbon:[[@LINE+7]]:32: error: facet type `Z & W` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   require impls X where .XZ = (Self as (W & Z));
  // CHECK:STDERR:                                ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_lookup_target_partially_identified.carbon:[[@LINE-6]]:1: note: constraint is currently being defined [NamedConstraintIncompleteWithinDefinition]
  // CHECK:STDERR: constraint W {
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  require impls X where .XZ = (Self as (W & Z));
}

// --- convert_from_forward_declaration.carbon
library "[[@TEST_NAME]]";

interface Z {}
// The facet type `W` is partially identified as soon as it is declared.
// Converting from a facet requires its facet type to be at least partially
// identified, not fully identified.
constraint W;

fn F(T:! W & Z) {
  T as Z;
}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/access.carbon

// --- impl_witness_access_in_impl_type_structure.carbon
library "[[@TEST_NAME]]";

interface X(T:! type) {}
interface Y {
  let Y1:! type;
}
interface Z {
  let Z1:! type;
}

// There's 2 ImplWitnessAccess instructions in the type, but they together
// resolve to a symbolic type value, so the type structure is: `? as X(?)`
impl forall [T:! Z where .Z1 impls Y] T as X(T.Z1.(Y.Y1)) {}

class C {
  impl as Y where .Y1 = {} {}
}

fn F(V:! Z where .Z1 = C) {
  // The type stucture is `? as X({})` which will match the impl's less specific
  // `? as X(?)`, then the impl will deduce the parameter of `X` to be `{}` from
  // the type of `V`. This would fail if ImplWitnessAccess instructions were
  // treated as Concrete in the type structure, since the impl would have a
  // different concrete value (an ImplWitnessAccess) than the query (a
  // StructValue).
  V as X({});
}

fn G(V:! Z where .Z1 impls Y) {
  // The type structure is `? as X(?)`, also built from ImplWitnessAccess insts,
  // which will match the impl's `? as X(?)`.
  V as X(V.(Z.Z1).(Y.Y1));
}

fn H(U:! type, V:! Z where .Z1 impls (Y where .Y1 = U)) {
  // The type structure is `? as X(?)`, without using an ImplWitnessAccess,
  // which will match the impl's `? as X(?)`. This would fail if
  // ImplWitnessAccess instructions were treated as Concrete in the type
  // structure, since the impl's type structure would be more specific than the
  // query's.
  V as X(U);
}

// --- fail_concrete_impl_witness_access_in_type_structure.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let Z1:! type;
}
interface Y {}

class C;

// Converting to this facet type requires a lookup for `C.(Z(C).Z1) as Y`. That
// has a type structure that contains a fully concrete ImplWitnessAccess. But if
// the lookup fails because there is no impl for `C as Z(C)`, then the access
// remains in the type structure as is. This test ensures the type structure can
// handle this edge case.
fn F(unused T:! Z(.Self) where C impls (Z(C) where .Z1 impls Y)) {}

fn G(T:! Z(.Self)) {
  // CHECK:STDERR: fail_concrete_impl_witness_access_in_type_structure.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `Z(.Self)` into type implementing `Z(.Self) where C impls Z(C) and C.(Z(C).Z1) impls Y` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   F(T);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_concrete_impl_witness_access_in_type_structure.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F(unused T:! Z(.Self) where C impls (Z(C) where .Z1 impls Y)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(T);
}

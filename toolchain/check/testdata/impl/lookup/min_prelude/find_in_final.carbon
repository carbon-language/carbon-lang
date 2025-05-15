// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/find_in_final.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/find_in_final.carbon

// --- final_impl_precedence_over_facet.carbon
library "[[@TEST_NAME]]";
interface I {
  let T:! type;
}

final impl forall [U:! type] U as I where .T = () {}

fn F(V:! I) -> V.T {
  // Even though we have a witness that `V impls I` from the constraint on `I`,
  // we should do an impl lookup to see if any effectively final impl applies
  // when we find an unknown value in that witness. In this case, that lookup
  // would find an impl with more specific values for associated constants that
  // we should merge.
  return ();
}

// --- final_impl_precedence_over_facet_with_where.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
  let Y:! type;
}

final impl forall [T:! type] T as Z where .X = () and .Y = () {}

fn F(ZZ:! Z where .X = ()) {
  // Z.Y is unspecified on `ZZ` so it's found on the final impl where it's known
  // to be the concrete type (), which can then be used in this generic
  // function.
  let a: ZZ.Y = ();
}

// --- final_impl_precedence_over_facet_access_type_with_where.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
  let Y:! type;
}

final impl forall [T:! type] T as Z where .X = () and .Y = () {}

fn F[T:! Z where .X = ()](z: T) {
  // z.Y is unspecified on `ZZ` so it's found on the final impl where it's known
  // to be the concrete type (), which can then be used in this generic
  // function.
  let a: z.Y = ();
}

// --- final_impl_makes_compatible_facet_values.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }
interface J {}

final impl forall [T:! J] T as I where .X = () {}

class C(T:! I) {
  var b: T.X;
}

class D(T:! J) {
  var c: C(T)*;
}

fn F(T:! I & J) -> () {
  // The witness for `I` found here directly, and inside `D` from a FacetValue
  // come from the same facet, the `T` binding in the params of F, so they
  // should be compatible.
  var x: C(T);
  var y: D(T);
  y.c = &x;
  return (*y.c).b;
}

// --- non_final_impl_makes_compatible_facet_values.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }
interface J {}

impl forall [T:! J] T as I where .X = () {}

class C(T:! I) {
  var b: T.X;
}

class D(T:! J) {
  var c: C(T)*;
}

fn F(T:! I & J) -> T.(I.X) {
  var x: C(T);
  var y: D(T);
  // The witness for `I` found here directly, and inside `D` from a FacetValue
  // come from the same facet, the `T` binding in the params of F, so they
  // should be compatible.
  y.c = &x;
  return (*y.c).b;
}

// --- todo_fail_facet_value_rewrite_incompatible_with_final_impl.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

final impl forall [T:! type] T as Z where .X = () {}

// TODO: This should be diagnosed as there is a final impl defining `.X = ()`,
// which makes the LHS of this rewrite constraint concrete. And since the RHS is
// not the same (or convertible from), the rewrite is impossible.
fn F(ZZ:! Z where .X = {.r: ()}) {}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --no-dump-sem-ir --custom-core
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

// --- fail_todo_final_impl_makes_compatible_facet_values.carbon
library "[[@TEST_NAME]]";

interface I {}
interface J {}

final impl forall [T:! J] T as I {}

class C(T:! I) {}

class D(T:! J) {
  // Finds the final impl decl; The facet value of `T` in `C(T)` holds an
  // `ImplWitness`.
  var c: C(T)*;
}

fn F(T:! I & J) {
  // `I` found in the facet type; The facet value of `T` in `C(T)` holds a `FacetAccessWitness`.
  // TODO: It should use the `final impl` and thus hold the same `ImplWitness`
  var x: C(T);
  // `D.c` used the final impl decl to make a facet value with an `ImplWitness`
  // for its `T` in `C(T)`.
  var y: D(T);
  // This converts a pointer to a facet value (should be `ImplWitness` but
  // isn't) to a pointer to a facet value (with `ImplWitness`).
  //
  // TODO: It should type check when they are both the same generic type
  // `C(T:! I)` defined by the `final impl` of `I`.
  //
  // CHECK:STDERR: fail_todo_final_impl_makes_compatible_facet_values.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `C(T as I)*` to `C(T as I)*` [ConversionFailure]
  // CHECK:STDERR:   y.c = &x;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_final_impl_makes_compatible_facet_values.carbon:[[@LINE+4]]:3: note: type `C(T as I)*` does not implement interface `Core.ImplicitAs(C(T as I)*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   y.c = &x;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  y.c = &x;
}

// --- fail_todo_convert_facet_value_of_facet_access_witness_to_facet_value_of_lookup_impl_witness.carbon
library "[[@TEST_NAME]]";

interface I { let X:! type; }
interface J {}

impl forall [T:! J] T as I where .X = () {}

class C(T:! I) {
  var b: T.X;
}

class D(T:! J) {
  // Finds the impl decl; The facet value of `T` in `C(T)` holds a
  // `LookupImplWitness`.
  var c: C(T)*;
}

fn F(T:! I & J) -> T.(I.X) {
  // `I` found in the facet type; The facet value of `T` in `C(T)` holds a
  // `FacetAccessWitness`.
  var x: C(T);
  // `D.c` used the final impl decl to make a facet value with a
  // `LookupImplWitness` for its `T` in `C(T)`.
  var y: D(T);
  // This converts a pointer to a facet value (with `FacetAccessWitness`) to a
  // pointer to a facet value (with `LookupImplWitness`).
  //
  // TODO: It should type check as a `LookupImplWitness` will find the same
  // witness that `FacetAccessWitness` is accessing from the caller.
  //
  // CHECK:STDERR: fail_todo_convert_facet_value_of_facet_access_witness_to_facet_value_of_lookup_impl_witness.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `C(T as I)*` to `C(T as I)*` [ConversionFailure]
  // CHECK:STDERR:   y.c = &x;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_convert_facet_value_of_facet_access_witness_to_facet_value_of_lookup_impl_witness.carbon:[[@LINE+4]]:3: note: type `C(T as I)*` does not implement interface `Core.ImplicitAs(C(T as I)*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   y.c = &x;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  y.c = &x;
  // CHECK:STDERR: fail_todo_convert_facet_value_of_facet_access_witness_to_facet_value_of_lookup_impl_witness.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `(T as J as I).(I.X)` to `T.(TODO: element 0 in <witness for T, interface 0>)` [ConversionFailure]
  // CHECK:STDERR:   return (*y.c).b;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_convert_facet_value_of_facet_access_witness_to_facet_value_of_lookup_impl_witness.carbon:[[@LINE+4]]:3: note: type `(T as J as I).(I.X)` does not implement interface `Core.ImplicitAs(T.(TODO: element 0 in <witness for T, interface 0>))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return (*y.c).b;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return (*y.c).b;
}

class E1 {}
impl E1 as J {}

fn G1() {
  // Doesn't see the `final impl` below. `E1.(I.X)` is `()`.
  let a1: () = F(E1);
}

final impl forall [T:! J] T as I where .X = {} {}

class E2 {}
impl E2 as J {}

fn G2() {
  // TODO: Does see the `final impl` above. `E2.(I.X)` is `{}`.
  // CHECK:STDERR: fail_todo_convert_facet_value_of_facet_access_witness_to_facet_value_of_lookup_impl_witness.carbon:[[@LINE+7]]:16: error: cannot implicitly convert expression of type `()` to `{}` [ConversionFailure]
  // CHECK:STDERR:   let a2: {} = F(E2);
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR: fail_todo_convert_facet_value_of_facet_access_witness_to_facet_value_of_lookup_impl_witness.carbon:[[@LINE+4]]:16: note: type `()` does not implement interface `Core.ImplicitAs({})` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a2: {} = F(E2);
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let a2: {} = F(E2);
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

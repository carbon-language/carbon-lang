// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/observe.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/observe.carbon

// --- transitive.carbon
library "[[@TEST_NAME]]";

interface P { fn InP(self); }
interface Q { fn InQ(self); }
interface R { fn InR(self); }

//@dump-sem-ir-begin
interface Transitive {
  let A: P & Core.Destroy;
  let B: (Q where .Self == A) & Core.Destroy;
  let C: (R where .Self == B) & Core.Destroy;

  fn GetA(self) -> A;
  fn TakesC(self, c: C);

  observe A == B == C;
}
//@dump-sem-ir-end

// --- fail_todo_equivalent.carbon
library "[[@TEST_NAME]]";

import library "transitive";

fn F[T: Transitive](t: T) {
  var a: T.A = t.GetA();

  // CHECK:STDERR: fail_todo_equivalent.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `T.(Transitive.A)` to `T.(Transitive.B)` [ConversionFailure]
  // CHECK:STDERR:   var b: T.B = a;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_equivalent.carbon:[[@LINE+4]]:3: note: type `T.(Transitive.A)` does not implement interface `Core.ImplicitAs(T.(Transitive.B))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var b: T.B = a;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  var b: T.B = a;
  b.InQ();

  // CHECK:STDERR: fail_todo_equivalent.carbon:[[@LINE+8]]:3: error: cannot convert type `T.(Transitive.A)` that implements `Core.Destroy & P` into type implementing `R` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   a.(T.C.InR)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_equivalent.carbon:[[@LINE-18]]:1: in import [InImport]
  // CHECK:STDERR: transitive.carbon:5:15: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: interface R { fn InR(self); }
  // CHECK:STDERR:               ^~~~~~~~~~~~~
  // CHECK:STDERR:
  a.(T.C.InR)();
  // CHECK:STDERR: fail_todo_equivalent.carbon:[[@LINE+7]]:4: error: cannot convert expression of type `T.(Transitive.A)` to `T.(Transitive.C)` with `as` [ConversionFailure]
  // CHECK:STDERR:   (a as T.C).InR();
  // CHECK:STDERR:    ^~~~~~~~
  // CHECK:STDERR: fail_todo_equivalent.carbon:[[@LINE+4]]:4: note: type `T.(Transitive.A)` does not implement interface `Core.As(T.(Transitive.C))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   (a as T.C).InR();
  // CHECK:STDERR:    ^~~~~~~~
  // CHECK:STDERR:
  (a as T.C).InR();
}

// --- fail_todo_impls.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B { require impls A; }
interface C { require impls B; }

fn RequiresA[T: A](unused x: T) {}
fn F[T: C](x: T) {
  observe T impls B;

  // CHECK:STDERR: fail_todo_impls.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `C` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_impls.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T: A](unused x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(x);
}

// --- fail_todo_equivalent_impls.carbon
library "[[@TEST_NAME]]";

import library "transitive";

fn TakesPQR[U: P & Q & R](unused u: U) {}
fn G[T: Transitive](t: T) {
  var a: T.A = t.GetA();

  observe T.A == T.B impls Q;
  // CHECK:STDERR: fail_todo_equivalent_impls.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Q` in type `T.(Transitive.A)` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a.(Q.InQ)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  a.(Q.InQ)();

  observe T.A == T.C impls R;
  // CHECK:STDERR: fail_todo_equivalent_impls.carbon:[[@LINE+7]]:3: error: cannot convert type `T.(Transitive.A)` that implements `P & Core.Destroy` into type implementing `P & Q & R` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   TakesPQR(a);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_equivalent_impls.carbon:[[@LINE-15]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TakesPQR[U: P & Q & R](unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakesPQR(a);
}

// --- fail_equivalent_impls_without_observe.carbon
library "[[@TEST_NAME]]";

import library "transitive";

fn TakesPQR[U: P & Q & R](unused u: U) {}
fn G[T: Transitive](t: T) {
  var a: T.A = t.GetA();

  // CHECK:STDERR: fail_equivalent_impls_without_observe.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Q` in type `T.(Transitive.A)` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a.(Q.InQ)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  a.(Q.InQ)();

  // CHECK:STDERR: fail_equivalent_impls_without_observe.carbon:[[@LINE+7]]:3: error: cannot convert type `T.(Transitive.A)` that implements `P & Core.Destroy` into type implementing `P & Q & R` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   TakesPQR(a);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_equivalent_impls_without_observe.carbon:[[@LINE-13]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TakesPQR[U: P & Q & R](unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakesPQR(a);
}

// --- fail_todo_same_observe_in_both_scopes.carbon
library "[[@TEST_NAME]]";

import library "transitive";

fn G[T: Transitive](t: T) {
  observe T.A == T.B == T.C;

  var a: T.A = t.GetA();
  // CHECK:STDERR: fail_todo_same_observe_in_both_scopes.carbon:[[@LINE+8]]:3: error: cannot convert type `T.(Transitive.A)` that implements `Core.Destroy & P` into type implementing `R` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   a.(T.C.InR)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_same_observe_in_both_scopes.carbon:[[@LINE-9]]:1: in import [InImport]
  // CHECK:STDERR: transitive.carbon:5:15: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: interface R { fn InR(self); }
  // CHECK:STDERR:               ^~~~~~~~~~~~~
  // CHECK:STDERR:
  a.(T.C.InR)();
}

// --- blanket_impls.carbon
library "[[@TEST_NAME]]";

interface A { }
interface B { }
interface C { }
interface D { }

impl forall [T: A] T as B { }
impl forall [T: B] T as C { }
impl forall [T: C] T as D { }

fn RequiresB[T: B](unused x: T) {}
fn RequiresD[T: D](unused x: T) {}

//@dump-sem-ir-begin
fn RequiresA[T: A](x: T) {
  RequiresB(x);

  observe T impls B;
  observe T impls C;
  RequiresD(x);
}
//@dump-sem-ir-end

// --- fail_todo_nested_functions_in_outer.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

//@dump-sem-ir-begin
fn RequiresA[T: A](unused x: T) {}
fn F[T: A, U: B where .Self == T]() {
  observe T == U;

  fn InnerF(x: U) {
    // CHECK:STDERR: fail_todo_nested_functions_in_outer.carbon:[[@LINE+7]]:5: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     RequiresA(x);
    // CHECK:STDERR:     ^~~~~~~~~~~~
    // CHECK:STDERR: fail_todo_nested_functions_in_outer.carbon:[[@LINE-8]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR: fn RequiresA[T: A](unused x: T) {}
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    RequiresA(x);
  }
}
//@dump-sem-ir-end

// --- fail_todo_nested_functions_in_inner.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T: A](unused x: T) {}
fn F[T: A, U: B where .Self == T]() {

  fn InnerF(x: U) {
    observe T == U;

    // CHECK:STDERR: fail_todo_nested_functions_in_inner.carbon:[[@LINE+7]]:5: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     RequiresA(x);
    // CHECK:STDERR:     ^~~~~~~~~~~~
    // CHECK:STDERR: fail_todo_nested_functions_in_inner.carbon:[[@LINE-9]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR: fn RequiresA[T: A](unused x: T) {}
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    RequiresA(x);
  }
}

// --- fail_nested_functions_lexical_order.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T: A](unused x: T) {}
fn F[T: A, U: B where .Self == T]() {

  fn InnerF(x: U) {
    // CHECK:STDERR: fail_nested_functions_lexical_order.carbon:[[@LINE+7]]:5: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     RequiresA(x);
    // CHECK:STDERR:     ^~~~~~~~~~~~
    // CHECK:STDERR: fail_nested_functions_lexical_order.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR: fn RequiresA[T: A](unused x: T) {}
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    RequiresA(x);
  }

  observe T == U;
}

// --- fail_nested_functions_leak_upward.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T: A](unused x: T) {}
fn F[T: A, U: B where .Self == T](x: U) {

  fn InnerF(unused x: U) {
    observe T == U;
  }

  // CHECK:STDERR: fail_nested_functions_leak_upward.carbon:[[@LINE+7]]:3: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_nested_functions_leak_upward.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T: A](unused x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(x);
}

// --- fail_not_observed.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T: A](unused x: T) {}
fn F[T: A, U: B where .Self == T, V: B](x: U) {
  observe T == V;

  // CHECK:STDERR: fail_not_observed.carbon:[[@LINE+7]]:3: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_not_observed.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T: A](unused x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(x);
}

// --- fail_invalid_equivalent.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn TakesA[T: A](unused x: T) {}
fn F[T: A, U: B](x: U) {
  observe T == U;
  // CHECK:STDERR: fail_invalid_equivalent.carbon:[[@LINE+7]]:3: error: cannot convert type `U` that implements `B` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   TakesA(x);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_invalid_equivalent.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TakesA[T: A](unused x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakesA(x);
}

// --- fail_wrong_scope_class.carbon
library "[[@TEST_NAME]]";

class C(T: type, U: type) {
  // CHECK:STDERR: fail_wrong_scope_class.carbon:[[@LINE+4]]:3: error: `observe` can only be used in an `interface` or `function` [ObserveInWrongScope]
  // CHECK:STDERR:   observe T == U;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  observe T == U;
}

// --- fail_wrong_scope_named_constraint.carbon
library "[[@TEST_NAME]]";

constraint C(T: type, U: type) {
  // CHECK:STDERR: fail_wrong_scope_named_constraint.carbon:[[@LINE+4]]:3: error: `observe` can only be used in an `interface` or `function` [ObserveInWrongScope]
  // CHECK:STDERR:   observe T == U;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  observe T == U;
}

// --- fail_non_facet_type_in_impls.carbon
library "[[@TEST_NAME]]";

class C {}
interface A {}

fn F[T: A]() {
  // CHECK:STDERR: fail_non_facet_type_in_impls.carbon:[[@LINE+4]]:19: error: right argument of `impls` requirement must be a facet type [ImplsOnNonFacetType]
  // CHECK:STDERR:   observe T impls C;
  // CHECK:STDERR:                   ^
  // CHECK:STDERR:
  observe T impls C;
}

// --- generic_observe.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}
//@dump-sem-ir-begin
interface I(T: A & Core.Destroy) {
  let U: (B where .Self == T) & Core.Destroy;

  fn GetU(self) -> U;
  fn TakesT(self, t: T);

  observe T == U;
}
//@dump-sem-ir-end

// --- fail_todo_generic_import.carbon
library "[[@TEST_NAME]]";

import library "generic_observe";

class C {}
impl C as A {}

fn F[T: I(C)](t: T) {
  var u: T.U = t.GetU();

  // CHECK:STDERR: fail_todo_generic_import.carbon:[[@LINE+8]]:3: error: cannot convert type `T.(I(C as A & Core.Destroy).U)` that implements `Core.Destroy & B where...` into type implementing `A & Core.Destroy` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   t.TakesT(u);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_import.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: generic_observe.carbon:10:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:   fn TakesT(self, t: T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  t.TakesT(u);
}

// CHECK:STDOUT: --- transitive.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.77c: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %facet_type.b1b: type = facet_type <@P & @Destroy> [concrete]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.1be: %Transitive.assoc_type = assoc_entity element0, @Transitive.WithSelf.%A [concrete]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self.77c, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.130: type = facet_type <@Q & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.WithSelf.%B [concrete]
// CHECK:STDOUT:   %impl.elem1: %facet_type.130 = impl_witness_access %Transitive.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %facet_type.4c8: type = facet_type <@R & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.WithSelf.%C [concrete]
// CHECK:STDOUT:   %Self.as_type.7f2: type = facet_access_type %Self.77c [symbolic]
// CHECK:STDOUT:   %pattern_type.b5c: type = pattern_type %Self.as_type.7f2 [symbolic]
// CHECK:STDOUT:   %self.param_patt.ae9: %pattern_type.b5c = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.12b: %pattern_type.b5c = at_binding_pattern self, %self.param_patt.ae9 [symbolic]
// CHECK:STDOUT:   %as_type.d30: type = facet_access_type %impl.elem0 [symbolic]
// CHECK:STDOUT:   %.117: Core.Form = init_form %as_type.d30 [symbolic]
// CHECK:STDOUT:   %pattern_type.51a: type = pattern_type %as_type.d30 [symbolic]
// CHECK:STDOUT:   %return.param_patt.931: %pattern_type.51a = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.680: %pattern_type.51a = return_slot_pattern %return.param_patt.931, %as_type.d30 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%Self.77c) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA: %Transitive.WithSelf.GetA.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, @Transitive.WithSelf.%Transitive.WithSelf.GetA.decl [concrete]
// CHECK:STDOUT:   %impl.elem2: %facet_type.4c8 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %as_type.472: type = facet_access_type %impl.elem2 [symbolic]
// CHECK:STDOUT:   %pattern_type.007: type = pattern_type %as_type.472 [symbolic]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.007 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %c.patt: %pattern_type.007 = at_binding_pattern c, %c.param_patt [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%Self.77c) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC: %Transitive.WithSelf.TakesC.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, @Transitive.WithSelf.%Transitive.WithSelf.TakesC.decl [concrete]
// CHECK:STDOUT:   %as_type.aed: type = facet_access_type %impl.elem1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Transitive.decl: type = interface_decl @Transitive [concrete = constants.%Transitive.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = constants.%Self.77c]
// CHECK:STDOUT:   %Transitive.WithSelf.decl = interface_with_self_decl @Transitive [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A: %facet_type.b1b = assoc_const_decl @A [concrete] {
// CHECK:STDOUT:     %assoc0: %Transitive.assoc_type = assoc_entity element0, @Transitive.WithSelf.%A [concrete = constants.%assoc0.1be]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: %facet_type.130 = assoc_const_decl @B [concrete] {
// CHECK:STDOUT:     %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.WithSelf.%B [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: %facet_type.4c8 = assoc_const_decl @C [concrete] {
// CHECK:STDOUT:     %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.WithSelf.%C [concrete = constants.%assoc2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.decl: @Transitive.WithSelf.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type) = fn_decl @Transitive.WithSelf.GetA [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA)] {
// CHECK:STDOUT:     %self.param_patt.loc13_11.1: @Transitive.WithSelf.GetA.%pattern_type.loc13_11 (%pattern_type.b5c) = value_param_pattern [symbolic = %self.param_patt.loc13_11.2 (constants.%self.param_patt.ae9)]
// CHECK:STDOUT:     %self.patt.loc13_11.1: @Transitive.WithSelf.GetA.%pattern_type.loc13_11 (%pattern_type.b5c) = at_binding_pattern self, %self.param_patt.loc13_11.1 [symbolic = %self.patt.loc13_11.2 (constants.%self.patt.12b)]
// CHECK:STDOUT:     %return.param_patt.loc13_20.1: @Transitive.WithSelf.GetA.%pattern_type.loc13_20 (%pattern_type.51a) = out_param_pattern [symbolic = %return.param_patt.loc13_20.2 (constants.%return.param_patt.931)]
// CHECK:STDOUT:     %return.patt.loc13_17.1: @Transitive.WithSelf.GetA.%pattern_type.loc13_20 (%pattern_type.51a) = return_slot_pattern %return.param_patt.loc13_20.1, %.loc13_20.2 [symbolic = %return.patt.loc13_17.2 (constants.%return.patt.680)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc13_20.2: %facet_type.b1b = impl_witness_access constants.%Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_20.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %A.ref: %facet_type.b1b = name_ref A, %impl.elem0.loc13_20.2 [symbolic = %impl.elem0.loc13_20.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %A.as_type: type = facet_access_type %A.ref [symbolic = %as_type (constants.%as_type.d30)]
// CHECK:STDOUT:     %.loc13_20.2: type = converted %A.ref, %A.as_type [symbolic = %as_type (constants.%as_type.d30)]
// CHECK:STDOUT:     %.loc13_20.3: Core.Form = init_form %.loc13_20.2 [symbolic = %.loc13_20.1 (constants.%.117)]
// CHECK:STDOUT:     %self.param: @Transitive.WithSelf.GetA.%Self.as_type.loc13_11.1 (%Self.as_type.7f2) = value_param call_param0
// CHECK:STDOUT:     %.loc13_11.1: type = splice_block %.loc13_11.2 [symbolic = %Self.as_type.loc13_11.1 (constants.%Self.as_type.7f2)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self [symbolic = %Self (constants.%Self.77c)]
// CHECK:STDOUT:       %Self.as_type.loc13_11.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc13_11.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:       %.loc13_11.2: type = converted %Self.ref, %Self.as_type.loc13_11.2 [symbolic = %Self.as_type.loc13_11.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.WithSelf.GetA.%Self.as_type.loc13_11.1 (%Self.as_type.7f2) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Transitive.WithSelf.GetA.%as_type (%as_type.d30) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Transitive.WithSelf.GetA.%as_type (%as_type.d30) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, %Transitive.WithSelf.GetA.decl [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.decl: @Transitive.WithSelf.%Transitive.WithSelf.TakesC.type (%Transitive.WithSelf.TakesC.type) = fn_decl @Transitive.WithSelf.TakesC [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.TakesC (constants.%Transitive.WithSelf.TakesC)] {
// CHECK:STDOUT:     %self.param_patt.loc14_13.1: @Transitive.WithSelf.TakesC.%pattern_type.loc14_13 (%pattern_type.b5c) = value_param_pattern [symbolic = %self.param_patt.loc14_13.2 (constants.%self.param_patt.ae9)]
// CHECK:STDOUT:     %self.patt.loc14_13.1: @Transitive.WithSelf.TakesC.%pattern_type.loc14_13 (%pattern_type.b5c) = at_binding_pattern self, %self.param_patt.loc14_13.1 [symbolic = %self.patt.loc14_13.2 (constants.%self.patt.12b)]
// CHECK:STDOUT:     %c.param_patt.loc14_20.1: @Transitive.WithSelf.TakesC.%pattern_type.loc14_20 (%pattern_type.007) = value_param_pattern [symbolic = %c.param_patt.loc14_20.2 (constants.%c.param_patt)]
// CHECK:STDOUT:     %c.patt.loc14_20.1: @Transitive.WithSelf.TakesC.%pattern_type.loc14_20 (%pattern_type.007) = at_binding_pattern c, %c.param_patt.loc14_20.1 [symbolic = %c.patt.loc14_20.2 (constants.%c.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Transitive.WithSelf.TakesC.%Self.as_type.loc14_13.1 (%Self.as_type.7f2) = value_param call_param0
// CHECK:STDOUT:     %.loc14_13.1: type = splice_block %.loc14_13.2 [symbolic = %Self.as_type.loc14_13.1 (constants.%Self.as_type.7f2)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self [symbolic = %Self (constants.%Self.77c)]
// CHECK:STDOUT:       %Self.as_type.loc14_13.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_13.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:       %.loc14_13.2: type = converted %Self.ref, %Self.as_type.loc14_13.2 [symbolic = %Self.as_type.loc14_13.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.WithSelf.TakesC.%Self.as_type.loc14_13.1 (%Self.as_type.7f2) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %c.param: @Transitive.WithSelf.TakesC.%as_type (%as_type.472) = value_param call_param1
// CHECK:STDOUT:     %.loc14_22.1: type = splice_block %.loc14_22.2 [symbolic = %as_type (constants.%as_type.472)] {
// CHECK:STDOUT:       %impl.elem2.loc14_22.2: %facet_type.4c8 = impl_witness_access constants.%Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc14_22.1 (constants.%impl.elem2)]
// CHECK:STDOUT:       %C.ref: %facet_type.4c8 = name_ref C, %impl.elem2.loc14_22.2 [symbolic = %impl.elem2.loc14_22.1 (constants.%impl.elem2)]
// CHECK:STDOUT:       %C.as_type: type = facet_access_type %C.ref [symbolic = %as_type (constants.%as_type.472)]
// CHECK:STDOUT:       %.loc14_22.2: type = converted %C.ref, %C.as_type [symbolic = %as_type (constants.%as_type.472)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @Transitive.WithSelf.TakesC.%as_type (%as_type.472) = wrapper_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, %Transitive.WithSelf.TakesC.decl [concrete = constants.%assoc4]
// CHECK:STDOUT:   %impl.elem0.loc16: %facet_type.b1b = impl_witness_access constants.%Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10 (constants.%impl.elem0)]
// CHECK:STDOUT:   %A.ref: %facet_type.b1b = name_ref A, %impl.elem0.loc16 [symbolic = %impl.elem0.loc10 (constants.%impl.elem0)]
// CHECK:STDOUT:   %impl.elem1.loc16: %facet_type.130 = impl_witness_access constants.%Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc11 (constants.%impl.elem1)]
// CHECK:STDOUT:   %B.ref: %facet_type.130 = name_ref B, %impl.elem1.loc16 [symbolic = %impl.elem1.loc11 (constants.%impl.elem1)]
// CHECK:STDOUT:   %B.as_type.loc16_16.1: type = facet_access_type %B.ref [symbolic = %as_type.loc16_16 (constants.%as_type.aed)]
// CHECK:STDOUT:   %.loc16_16.1: type = converted %B.ref, %B.as_type.loc16_16.1 [symbolic = %as_type.loc16_16 (constants.%as_type.aed)]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A.ref [symbolic = %as_type.loc16_11 (constants.%as_type.d30)]
// CHECK:STDOUT:   %.loc16_11: type = converted %A.ref, %A.as_type [symbolic = %as_type.loc16_11 (constants.%as_type.d30)]
// CHECK:STDOUT:   %impl.elem2.loc16_21.1: %facet_type.4c8 = impl_witness_access constants.%Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc16_21.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %C.ref: %facet_type.4c8 = name_ref C, %impl.elem2.loc16_21.1 [symbolic = %impl.elem2.loc16_21.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %C.as_type: type = facet_access_type %C.ref [symbolic = %as_type.loc16_21 (constants.%as_type.472)]
// CHECK:STDOUT:   %.loc16_21: type = converted %C.ref, %C.as_type [symbolic = %as_type.loc16_21 (constants.%as_type.472)]
// CHECK:STDOUT:   %B.as_type.loc16_16.2: type = facet_access_type %B.ref [symbolic = %as_type.loc16_16 (constants.%as_type.aed)]
// CHECK:STDOUT:   %.loc16_16.2: type = converted %B.ref, %B.as_type.loc16_16.2 [symbolic = %as_type.loc16_16 (constants.%as_type.aed)]
// CHECK:STDOUT:   %Transitive.observe.decl = observe_decl @Transitive.observe [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .P = <poisoned>
// CHECK:STDOUT:   .Q = <poisoned>
// CHECK:STDOUT:   .R = <poisoned>
// CHECK:STDOUT:   .P = <poisoned>
// CHECK:STDOUT:   .A = @A.%assoc0
// CHECK:STDOUT:   .Q = <poisoned>
// CHECK:STDOUT:   .B = @B.%assoc1
// CHECK:STDOUT:   .R = <poisoned>
// CHECK:STDOUT:   .C = @C.%assoc2
// CHECK:STDOUT:   .GetA = @Transitive.WithSelf.%assoc3
// CHECK:STDOUT:   .TakesC = @Transitive.WithSelf.%assoc4
// CHECK:STDOUT:   witness = (@Transitive.WithSelf.%A, @Transitive.WithSelf.%B, @Transitive.WithSelf.%C, @Transitive.WithSelf.%Transitive.WithSelf.GetA.decl, @Transitive.WithSelf.%Transitive.WithSelf.TakesC.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT:   @Transitive.observe {
// CHECK:STDOUT:     observe constants.%impl.elem0 == constants.%impl.elem1 == constants.%impl.elem2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.77c)]
// CHECK:STDOUT:   %Self.as_type.loc13_11.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc13_11.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:   %pattern_type.loc13_11: type = pattern_type %Self.as_type.loc13_11.1 [symbolic = %pattern_type.loc13_11 (constants.%pattern_type.b5c)]
// CHECK:STDOUT:   %self.param_patt.loc13_11.2: @Transitive.WithSelf.GetA.%pattern_type.loc13_11 (%pattern_type.b5c) = value_param_pattern [symbolic = %self.param_patt.loc13_11.2 (constants.%self.param_patt.ae9)]
// CHECK:STDOUT:   %self.patt.loc13_11.2: @Transitive.WithSelf.GetA.%pattern_type.loc13_11 (%pattern_type.b5c) = at_binding_pattern self, %self.param_patt.loc13_11.2 [symbolic = %self.patt.loc13_11.2 (constants.%self.patt.12b)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc13_20.1: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_20.1 (constants.%impl.elem0)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc13_20.1 [symbolic = %as_type (constants.%as_type.d30)]
// CHECK:STDOUT:   %.loc13_20.1: Core.Form = init_form %as_type [symbolic = %.loc13_20.1 (constants.%.117)]
// CHECK:STDOUT:   %pattern_type.loc13_20: type = pattern_type %as_type [symbolic = %pattern_type.loc13_20 (constants.%pattern_type.51a)]
// CHECK:STDOUT:   %return.param_patt.loc13_20.2: @Transitive.WithSelf.GetA.%pattern_type.loc13_20 (%pattern_type.51a) = out_param_pattern [symbolic = %return.param_patt.loc13_20.2 (constants.%return.param_patt.931)]
// CHECK:STDOUT:   %return.patt.loc13_17.2: @Transitive.WithSelf.GetA.%pattern_type.loc13_20 (%pattern_type.51a) = return_slot_pattern %return.param_patt.loc13_20.2, %as_type [symbolic = %return.patt.loc13_17.2 (constants.%return.patt.680)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.WithSelf.GetA.%Self.as_type.loc13_11.1 (%Self.as_type.7f2)) -> out %return.param: @Transitive.WithSelf.GetA.%as_type (%as_type.d30);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.77c)]
// CHECK:STDOUT:   %Self.as_type.loc14_13.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_13.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:   %pattern_type.loc14_13: type = pattern_type %Self.as_type.loc14_13.1 [symbolic = %pattern_type.loc14_13 (constants.%pattern_type.b5c)]
// CHECK:STDOUT:   %self.param_patt.loc14_13.2: @Transitive.WithSelf.TakesC.%pattern_type.loc14_13 (%pattern_type.b5c) = value_param_pattern [symbolic = %self.param_patt.loc14_13.2 (constants.%self.param_patt.ae9)]
// CHECK:STDOUT:   %self.patt.loc14_13.2: @Transitive.WithSelf.TakesC.%pattern_type.loc14_13 (%pattern_type.b5c) = at_binding_pattern self, %self.param_patt.loc14_13.2 [symbolic = %self.patt.loc14_13.2 (constants.%self.patt.12b)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem2.loc14_22.1: %facet_type.4c8 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc14_22.1 (constants.%impl.elem2)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2.loc14_22.1 [symbolic = %as_type (constants.%as_type.472)]
// CHECK:STDOUT:   %pattern_type.loc14_20: type = pattern_type %as_type [symbolic = %pattern_type.loc14_20 (constants.%pattern_type.007)]
// CHECK:STDOUT:   %c.param_patt.loc14_20.2: @Transitive.WithSelf.TakesC.%pattern_type.loc14_20 (%pattern_type.007) = value_param_pattern [symbolic = %c.param_patt.loc14_20.2 (constants.%c.param_patt)]
// CHECK:STDOUT:   %c.patt.loc14_20.2: @Transitive.WithSelf.TakesC.%pattern_type.loc14_20 (%pattern_type.007) = at_binding_pattern c, %c.param_patt.loc14_20.2 [symbolic = %c.patt.loc14_20.2 (constants.%c.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.WithSelf.TakesC.%Self.as_type.loc14_13.1 (%Self.as_type.7f2), %c.param: @Transitive.WithSelf.TakesC.%as_type (%as_type.472));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%Self.77c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.77c) {
// CHECK:STDOUT:   %Self => constants.%Self.77c
// CHECK:STDOUT:   %Self.as_type.loc13_11.1 => constants.%Self.as_type.7f2
// CHECK:STDOUT:   %pattern_type.loc13_11 => constants.%pattern_type.b5c
// CHECK:STDOUT:   %self.param_patt.loc13_11.2 => constants.%self.param_patt.ae9
// CHECK:STDOUT:   %self.patt.loc13_11.2 => constants.%self.patt.12b
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem0.loc13_20.1 => constants.%impl.elem0
// CHECK:STDOUT:   %as_type => constants.%as_type.d30
// CHECK:STDOUT:   %.loc13_20.1 => constants.%.117
// CHECK:STDOUT:   %pattern_type.loc13_20 => constants.%pattern_type.51a
// CHECK:STDOUT:   %return.param_patt.loc13_20.2 => constants.%return.param_patt.931
// CHECK:STDOUT:   %return.patt.loc13_17.2 => constants.%return.patt.680
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.77c) {
// CHECK:STDOUT:   %Self => constants.%Self.77c
// CHECK:STDOUT:   %Self.as_type.loc14_13.1 => constants.%Self.as_type.7f2
// CHECK:STDOUT:   %pattern_type.loc14_13 => constants.%pattern_type.b5c
// CHECK:STDOUT:   %self.param_patt.loc14_13.2 => constants.%self.param_patt.ae9
// CHECK:STDOUT:   %self.patt.loc14_13.2 => constants.%self.patt.12b
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem2.loc14_22.1 => constants.%impl.elem2
// CHECK:STDOUT:   %as_type => constants.%as_type.472
// CHECK:STDOUT:   %pattern_type.loc14_20 => constants.%pattern_type.007
// CHECK:STDOUT:   %c.param_patt.loc14_20.2 => constants.%c.param_patt
// CHECK:STDOUT:   %c.patt.loc14_20.2 => constants.%c.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- blanket_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %D.type: type = facet_type <@D> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt.a4b: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.626: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.3b8: type = facet_access_type %T.626 [symbolic]
// CHECK:STDOUT:   %RequiresB.type: type = fn_type @RequiresB [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %RequiresB: %RequiresB.type = struct_value () [concrete]
// CHECK:STDOUT:   %RequiresD.type: type = fn_type @RequiresD [concrete]
// CHECK:STDOUT:   %RequiresD: %RequiresD.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %T.as_type.3b8 [symbolic]
// CHECK:STDOUT:   %x.param_patt.241: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.19b9f1.1: %pattern_type.a0e = at_binding_pattern x, %x.param_patt.241 [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.94f: <witness> = require_complete_type %T.as_type.3b8 [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.626, @B [symbolic]
// CHECK:STDOUT:   %.714: require_specific_def_type = require_specific_def @T.as_type.as.B.impl(%T.626) [symbolic]
// CHECK:STDOUT:   %B.facet.c85: %B.type = facet_value %T.as_type.3b8, (%B.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %RequiresB.specific_fn: <specific function> = specific_function %RequiresB, @RequiresB(%B.facet.c85) [symbolic]
// CHECK:STDOUT:   %impls.be1 = observe_impls %T.626, %B.type [symbolic]
// CHECK:STDOUT:   %impls.86f = observe_impls %T.626, %C.type [symbolic]
// CHECK:STDOUT:   %D.lookup_impl_witness: <witness> = lookup_impl_witness %T.626, @D [symbolic]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %T.626, @C [symbolic]
// CHECK:STDOUT:   %C.facet.124: %C.type = facet_value %T.as_type.3b8, (%C.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.f88: require_specific_def_type = require_specific_def @T.as_type.as.D.impl(%C.facet.124) [symbolic]
// CHECK:STDOUT:   %D.facet.5cf: %D.type = facet_value %T.as_type.3b8, (%D.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %RequiresD.specific_fn: <specific function> = specific_function %RequiresD, @RequiresD(%D.facet.5cf) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt.loc16_15.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_15.2 (constants.%T.patt.a4b)]
// CHECK:STDOUT:     %x.param_patt.loc16_21.1: @RequiresA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc16_21.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:     %x.patt.loc16_21.1: @RequiresA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc16_21.1 [symbolic = %x.patt.loc16_21.2 (constants.%x.patt.19b9f1.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_17: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_15.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc16_15.1 (constants.%T.626)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.as_type.loc16_23.1 (%T.as_type.3b8) = value_param call_param0
// CHECK:STDOUT:     %.loc16_23.1: type = splice_block %.loc16_23.2 [symbolic = %T.as_type.loc16_23.1 (constants.%T.as_type.3b8)] {
// CHECK:STDOUT:       %T.ref.loc16: %A.type = name_ref T, %T.loc16_15.2 [symbolic = %T.loc16_15.1 (constants.%T.626)]
// CHECK:STDOUT:       %T.as_type.loc16_23.2: type = facet_access_type %T.ref.loc16 [symbolic = %T.as_type.loc16_23.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:       %.loc16_23.2: type = converted %T.ref.loc16, %T.as_type.loc16_23.2 [symbolic = %T.as_type.loc16_23.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.as_type.loc16_23.1 (%T.as_type.3b8) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc16_15.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc16_15.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_15.2 (constants.%T.patt.a4b)]
// CHECK:STDOUT:   %T.loc16_15.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc16_15.1 (constants.%T.626)]
// CHECK:STDOUT:   %T.as_type.loc16_23.1: type = facet_access_type %T.loc16_15.1 [symbolic = %T.as_type.loc16_23.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc16_23.1 [symbolic = %pattern_type (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %x.param_patt.loc16_21.2: @RequiresA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc16_21.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:   %x.patt.loc16_21.2: @RequiresA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc16_21.2 [symbolic = %x.patt.loc16_21.2 (constants.%x.patt.19b9f1.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc16_23.1 [symbolic = %require_complete (constants.%require_complete.94f)]
// CHECK:STDOUT:   %.loc17_14.3: require_specific_def_type = require_specific_def @T.as_type.as.B.impl(%T.loc16_15.1) [symbolic = %.loc17_14.3 (constants.%.714)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc16_15.1, @B [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness)]
// CHECK:STDOUT:   %B.facet.loc17_14.3: %B.type = facet_value %T.as_type.loc16_23.1, (%B.lookup_impl_witness) [symbolic = %B.facet.loc17_14.3 (constants.%B.facet.c85)]
// CHECK:STDOUT:   %RequiresB.specific_fn.loc17_3.2: <specific function> = specific_function constants.%RequiresB, @RequiresB(%B.facet.loc17_14.3) [symbolic = %RequiresB.specific_fn.loc17_3.2 (constants.%RequiresB.specific_fn)]
// CHECK:STDOUT:   %impls.loc19 = observe_impls %T.loc16_15.1, constants.%B.type [symbolic = %impls.loc19 (constants.%impls.be1)]
// CHECK:STDOUT:   %impls.loc20 = observe_impls %T.loc16_15.1, constants.%C.type [symbolic = %impls.loc20 (constants.%impls.86f)]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc16_15.1, @C [symbolic = %C.lookup_impl_witness (constants.%C.lookup_impl_witness)]
// CHECK:STDOUT:   %C.facet: %C.type = facet_value %T.as_type.loc16_23.1, (%C.lookup_impl_witness) [symbolic = %C.facet (constants.%C.facet.124)]
// CHECK:STDOUT:   %.loc21_14.3: require_specific_def_type = require_specific_def @T.as_type.as.D.impl(%C.facet) [symbolic = %.loc21_14.3 (constants.%.f88)]
// CHECK:STDOUT:   %D.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc16_15.1, @D [symbolic = %D.lookup_impl_witness (constants.%D.lookup_impl_witness)]
// CHECK:STDOUT:   %D.facet.loc21_14.3: %D.type = facet_value %T.as_type.loc16_23.1, (%D.lookup_impl_witness) [symbolic = %D.facet.loc21_14.3 (constants.%D.facet.5cf)]
// CHECK:STDOUT:   %RequiresD.specific_fn.loc21_3.2: <specific function> = specific_function constants.%RequiresD, @RequiresD(%D.facet.loc21_14.3) [symbolic = %RequiresD.specific_fn.loc21_3.2 (constants.%RequiresD.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.as_type.loc16_23.1 (%T.as_type.3b8)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RequiresB.ref: %RequiresB.type = name_ref RequiresB, file.%RequiresB.decl [concrete = constants.%RequiresB]
// CHECK:STDOUT:     %x.ref.loc17: @RequiresA.%T.as_type.loc16_23.1 (%T.as_type.3b8) = name_ref x, %x
// CHECK:STDOUT:     %B.facet.loc17_14.1: %B.type = facet_value constants.%T.as_type.3b8, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc17_14.3 (constants.%B.facet.c85)]
// CHECK:STDOUT:     %.loc17_14.1: %B.type = converted constants.%T.as_type.3b8, %B.facet.loc17_14.1 [symbolic = %B.facet.loc17_14.3 (constants.%B.facet.c85)]
// CHECK:STDOUT:     %B.facet.loc17_14.2: %B.type = facet_value constants.%T.as_type.3b8, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc17_14.3 (constants.%B.facet.c85)]
// CHECK:STDOUT:     %.loc17_14.2: %B.type = converted constants.%T.as_type.3b8, %B.facet.loc17_14.2 [symbolic = %B.facet.loc17_14.3 (constants.%B.facet.c85)]
// CHECK:STDOUT:     %RequiresB.specific_fn.loc17_3.1: <specific function> = specific_function %RequiresB.ref, @RequiresB(constants.%B.facet.c85) [symbolic = %RequiresB.specific_fn.loc17_3.2 (constants.%RequiresB.specific_fn)]
// CHECK:STDOUT:     %RequiresB.call: init %empty_tuple.type = call %RequiresB.specific_fn.loc17_3.1(%x.ref.loc17)
// CHECK:STDOUT:     %T.ref.loc19: %A.type = name_ref T, %T.loc16_15.2 [symbolic = %T.loc16_15.1 (constants.%T.626)]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %T.as_type.loc19: type = facet_access_type %T.ref.loc19 [symbolic = %T.as_type.loc16_23.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     %.loc19: type = converted %T.ref.loc19, %T.as_type.loc19 [symbolic = %T.as_type.loc16_23.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     %RequiresA.observe.decl.loc19 = observe_decl @RequiresA.observe.loc19 [concrete]
// CHECK:STDOUT:     %T.ref.loc20: %A.type = name_ref T, %T.loc16_15.2 [symbolic = %T.loc16_15.1 (constants.%T.626)]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     %T.as_type.loc20: type = facet_access_type %T.ref.loc20 [symbolic = %T.as_type.loc16_23.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     %.loc20: type = converted %T.ref.loc20, %T.as_type.loc20 [symbolic = %T.as_type.loc16_23.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     %RequiresA.observe.decl.loc20 = observe_decl @RequiresA.observe.loc20 [concrete]
// CHECK:STDOUT:     %RequiresD.ref: %RequiresD.type = name_ref RequiresD, file.%RequiresD.decl [concrete = constants.%RequiresD]
// CHECK:STDOUT:     %x.ref.loc21: @RequiresA.%T.as_type.loc16_23.1 (%T.as_type.3b8) = name_ref x, %x
// CHECK:STDOUT:     %D.facet.loc21_14.1: %D.type = facet_value constants.%T.as_type.3b8, (constants.%D.lookup_impl_witness) [symbolic = %D.facet.loc21_14.3 (constants.%D.facet.5cf)]
// CHECK:STDOUT:     %.loc21_14.1: %D.type = converted constants.%T.as_type.3b8, %D.facet.loc21_14.1 [symbolic = %D.facet.loc21_14.3 (constants.%D.facet.5cf)]
// CHECK:STDOUT:     %D.facet.loc21_14.2: %D.type = facet_value constants.%T.as_type.3b8, (constants.%D.lookup_impl_witness) [symbolic = %D.facet.loc21_14.3 (constants.%D.facet.5cf)]
// CHECK:STDOUT:     %.loc21_14.2: %D.type = converted constants.%T.as_type.3b8, %D.facet.loc21_14.2 [symbolic = %D.facet.loc21_14.3 (constants.%D.facet.5cf)]
// CHECK:STDOUT:     %RequiresD.specific_fn.loc21_3.1: <specific function> = specific_function %RequiresD.ref, @RequiresD(constants.%D.facet.5cf) [symbolic = %RequiresD.specific_fn.loc21_3.2 (constants.%RequiresD.specific_fn)]
// CHECK:STDOUT:     %RequiresD.call: init %empty_tuple.type = call %RequiresD.specific_fn.loc21_3.1(%x.ref.loc21)
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @RequiresA.observe.loc19 {
// CHECK:STDOUT:       observe constants.%T.626 impls constants.%B.type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     @RequiresA.observe.loc20 {
// CHECK:STDOUT:       observe constants.%T.626 impls constants.%C.type
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T.626) {
// CHECK:STDOUT:   %T.patt.loc16_15.2 => constants.%T.patt.a4b
// CHECK:STDOUT:   %T.loc16_15.1 => constants.%T.626
// CHECK:STDOUT:   %T.as_type.loc16_23.1 => constants.%T.as_type.3b8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc16_21.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc16_21.2 => constants.%x.patt.19b9f1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_nested_functions_in_outer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.241: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.19b: %pattern_type.a0e = at_binding_pattern x, %x.param_patt.241 [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.94f: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.Self.frozen.f7e: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %eq = observe_equivalent %T, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.d51: type = pattern_type %U.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.fbe: %pattern_type.d51 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.cdb: %pattern_type.d51 = at_binding_pattern x, %x.param_patt.fbe [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.963: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt.loc7_15.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc7_28.1: @RequiresA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_28.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:     %x.patt.loc7_28.1: @RequiresA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_28.1 [symbolic = %x.patt.loc7_28.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_17: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.as_type.loc7_30.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_30.1: type = splice_block %.loc7_30.2 [symbolic = %T.as_type.loc7_30.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_15.2 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7_30.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_30.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_30.2: type = converted %T.ref, %T.as_type.loc7_30.2 [symbolic = %T.as_type.loc7_30.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.as_type.loc7_30.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc8_13.1: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_13.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_9: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_17.1: type = splice_block %.loc8_17.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_13: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_17: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_23.1: %B.type = name_ref .Self, %.Self.frozen.loc8_17 [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %T.ref.loc8: %A.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %equiv.loc8_29.1 = requirement_equivalent %.Self.ref.loc8_23.1, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       %.Self: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.Self.ref.loc8_23.2: %B.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %equiv.loc8_29.2 = requirement_equivalent %.Self.ref.loc8_23.2, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       %.loc8_17.2: type = where_expr [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_29.2 = requirement_equivalent %.Self.ref.loc8_23.2, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc7_15.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_15.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc7_30.1: type = facet_access_type %T.loc7_15.1 [symbolic = %T.as_type.loc7_30.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_30.1 [symbolic = %pattern_type (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %x.param_patt.loc7_28.2: @RequiresA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_28.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:   %x.patt.loc7_28.2: @RequiresA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_28.2 [symbolic = %x.patt.loc7_28.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_30.1 [symbolic = %require_complete (constants.%require_complete.94f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.as_type.loc7_30.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %A.type, %U.loc8_13.2: %B_where.type) {
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_7.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc8_13.2: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_13.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc8_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.as_type.loc9_16.2: type = facet_access_type %U.loc8_13.1 [symbolic = %U.as_type.loc9_16.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %T.as_type.loc9_11.2: type = facet_access_type %T.loc8_7.1 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %eq = observe_equivalent %T.loc8_7.1, %U.loc8_13.1 [symbolic = %eq (constants.%eq)]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF, @F(%T.loc8_7.1, %U.loc8_13.1) [symbolic = %InnerF.type (constants.%InnerF.type)]
// CHECK:STDOUT:   %InnerF: @F.%InnerF.type (%InnerF.type) = struct_value () [symbolic = %InnerF (constants.%InnerF)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc9: %A.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref: %B_where.type = name_ref U, %U.loc8_13.2 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc9_16.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc9_16.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc9_16: type = converted %U.ref, %U.as_type.loc9_16.1 [symbolic = %U.as_type.loc9_16.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %T.as_type.loc9_11.1: type = facet_access_type %T.ref.loc9 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_11: type = converted %T.ref.loc9, %T.as_type.loc9_11.1 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %F.observe.decl = observe_decl @F.observe [concrete]
// CHECK:STDOUT:     %InnerF.decl: @F.%InnerF.type (%InnerF.type) = fn_decl @InnerF [symbolic = @F.%InnerF (constants.%InnerF)] {
// CHECK:STDOUT:       %x.param_patt.loc11_14.1: @InnerF.%pattern_type (%pattern_type.d51) = value_param_pattern [symbolic = %x.param_patt.loc11_14.2 (constants.%x.param_patt.fbe)]
// CHECK:STDOUT:       %x.patt.loc11_14.1: @InnerF.%pattern_type (%pattern_type.d51) = at_binding_pattern x, %x.param_patt.loc11_14.1 [symbolic = %x.patt.loc11_14.2 (constants.%x.patt.cdb)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%U.as_type.loc11_16.1 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc11_16.1: type = splice_block %.loc11_16.2 [symbolic = %U.as_type.loc11_16.1 (constants.%U.as_type)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc8_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type.loc11_16.2: type = facet_access_type %U.ref [symbolic = %U.as_type.loc11_16.1 (constants.%U.as_type)]
// CHECK:STDOUT:         %.loc11_16.2: type = converted %U.ref, %U.as_type.loc11_16.2 [symbolic = %U.as_type.loc11_16.1 (constants.%U.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%U.as_type.loc11_16.1 (%U.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @F.observe {
// CHECK:STDOUT:       observe constants.%T == constants.%U
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc8_7.2: %A.type, @F.%U.loc8_13.2: %B_where.type) {
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.as_type.loc11_16.1: type = facet_access_type %U [symbolic = %U.as_type.loc11_16.1 (constants.%U.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc11_16.1 [symbolic = %pattern_type (constants.%pattern_type.d51)]
// CHECK:STDOUT:   %x.param_patt.loc11_14.2: @InnerF.%pattern_type (%pattern_type.d51) = value_param_pattern [symbolic = %x.param_patt.loc11_14.2 (constants.%x.param_patt.fbe)]
// CHECK:STDOUT:   %x.patt.loc11_14.2: @InnerF.%pattern_type (%pattern_type.d51) = at_binding_pattern x, %x.param_patt.loc11_14.2 [symbolic = %x.patt.loc11_14.2 (constants.%x.patt.cdb)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc11_16.1 [symbolic = %require_complete (constants.%require_complete.963)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%U.as_type.loc11_16.1 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @InnerF.%U.as_type.loc11_16.1 (%U.as_type) = name_ref x, %x
// CHECK:STDOUT:     %B.type: type = facet_type <@B> [concrete = constants.%B.type]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_15.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_30.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc7_28.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc7_28.2 => constants.%x.patt.19b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc8_13.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc8_13.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.as_type.loc11_16.1 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d51
// CHECK:STDOUT:   %x.param_patt.loc11_14.2 => constants.%x.param_patt.fbe
// CHECK:STDOUT:   %x.patt.loc11_14.2 => constants.%x.patt.cdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_observe.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.802: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.802 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.76f: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.eef: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.76f, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.36e: <bound method> = bound_method %A.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.8cb: type = facet_type <@A & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.827: type = pattern_type %facet_type.8cb [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.827 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %facet_type.8cb = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.715: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.dca: %I.type.715 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %facet_type.208: type = facet_type <@B & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.93a: %I.assoc_type = assoc_entity element0, @I.WithSelf.%U [symbolic]
// CHECK:STDOUT:   %Self.as_type.37b: type = facet_access_type %Self.dca [symbolic]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %Self.as_type.37b [symbolic]
// CHECK:STDOUT:   %self.param_patt.ee4: %pattern_type.f76 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.2e1: %pattern_type.f76 = at_binding_pattern self, %self.param_patt.ee4 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self.dca, @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %impl.elem0: %facet_type.208 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic]
// CHECK:STDOUT:   %.344: Core.Form = init_form %as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.079: type = pattern_type %as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.82c: %pattern_type.079 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.4f7: %pattern_type.079 = return_slot_pattern %return.param_patt.82c, %as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.GetU.type: type = fn_type @I.WithSelf.GetU, @I.WithSelf(%T, %Self.dca) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.GetU: %I.WithSelf.GetU.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.WithSelf.%I.WithSelf.GetU.decl [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.bc3: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.bc3 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.bc3 = at_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %I.WithSelf.TakesT.type: type = fn_type @I.WithSelf.TakesT, @I.WithSelf(%T, %Self.dca) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.TakesT: %I.WithSelf.TakesT.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc2: %I.assoc_type = assoc_entity element2, @I.WithSelf.%I.WithSelf.TakesT.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc6_14.1: %pattern_type.827 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.3 [concrete = constants.%facet_type.8cb] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0: %.eef = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %A.ref, %impl.elem0 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.36e]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%A.ref, %Destroy.ref) [concrete = constants.%facet_type.8cb]
// CHECK:STDOUT:       %.loc6_18.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type.8cb]
// CHECK:STDOUT:       %.loc6_18.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc6_18.2 [concrete = constants.%facet_type.8cb]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_14.2: %facet_type.8cb = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc6_14.2: %facet_type.8cb) {
// CHECK:STDOUT:   %T.patt.loc6_14.2: %pattern_type.827 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_14.1: %facet_type.8cb = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc6_14.1)> [symbolic = %I.type (constants.%I.type.715)]
// CHECK:STDOUT:   %Self.loc6_34.2: @I.%I.type (%I.type.715) = symbolic_binding Self, 1 [symbolic = %Self.loc6_34.2 (constants.%Self.dca)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc6_34.1: @I.%I.type (%I.type.715) = symbolic_binding Self, 1 [symbolic = %Self.loc6_34.2 (constants.%Self.dca)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %U: %facet_type.208 = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:       %assoc0: @I.WithSelf.%I.assoc_type (%I.assoc_type) = assoc_entity element0, @I.WithSelf.%U [symbolic = @I.WithSelf.%assoc0 (constants.%assoc0.93a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.WithSelf.GetU.decl: @I.WithSelf.%I.WithSelf.GetU.type (%I.WithSelf.GetU.type) = fn_decl @I.WithSelf.GetU [symbolic = @I.WithSelf.%I.WithSelf.GetU (constants.%I.WithSelf.GetU)] {
// CHECK:STDOUT:       %self.param_patt.loc9_11.1: @I.WithSelf.GetU.%pattern_type.loc9_11 (%pattern_type.f76) = value_param_pattern [symbolic = %self.param_patt.loc9_11.2 (constants.%self.param_patt.ee4)]
// CHECK:STDOUT:       %self.patt.loc9_11.1: @I.WithSelf.GetU.%pattern_type.loc9_11 (%pattern_type.f76) = at_binding_pattern self, %self.param_patt.loc9_11.1 [symbolic = %self.patt.loc9_11.2 (constants.%self.patt.2e1)]
// CHECK:STDOUT:       %return.param_patt.loc9_20.1: @I.WithSelf.GetU.%pattern_type.loc9_20 (%pattern_type.079) = out_param_pattern [symbolic = %return.param_patt.loc9_20.2 (constants.%return.param_patt.82c)]
// CHECK:STDOUT:       %return.patt.loc9_17.1: @I.WithSelf.GetU.%pattern_type.loc9_20 (%pattern_type.079) = return_slot_pattern %return.param_patt.loc9_20.1, %.loc9_20.2 [symbolic = %return.patt.loc9_17.2 (constants.%return.patt.4f7)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %impl.elem0.loc9_20.2: %facet_type.208 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_20.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.ref: %facet_type.208 = name_ref U, %impl.elem0.loc9_20.2 [symbolic = %impl.elem0.loc9_20.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:       %.loc9_20.2: type = converted %U.ref, %U.as_type [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:       %.loc9_20.3: Core.Form = init_form %.loc9_20.2 [symbolic = %.loc9_20.1 (constants.%.344)]
// CHECK:STDOUT:       %self.param: @I.WithSelf.GetU.%Self.as_type.loc9_11.1 (%Self.as_type.37b) = value_param call_param0
// CHECK:STDOUT:       %.loc9_11.1: type = splice_block %.loc9_11.3 [symbolic = %Self.as_type.loc9_11.1 (constants.%Self.as_type.37b)] {
// CHECK:STDOUT:         %.loc9_11.2: @I.WithSelf.GetU.%I.type (%I.type.715) = specific_constant @I.%Self.loc6_34.1, @I(constants.%T) [symbolic = %Self (constants.%Self.dca)]
// CHECK:STDOUT:         %Self.ref: @I.WithSelf.GetU.%I.type (%I.type.715) = name_ref Self, %.loc9_11.2 [symbolic = %Self (constants.%Self.dca)]
// CHECK:STDOUT:         %Self.as_type.loc9_11.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_11.1 (constants.%Self.as_type.37b)]
// CHECK:STDOUT:         %.loc9_11.3: type = converted %Self.ref, %Self.as_type.loc9_11.2 [symbolic = %Self.as_type.loc9_11.1 (constants.%Self.as_type.37b)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @I.WithSelf.GetU.%Self.as_type.loc9_11.1 (%Self.as_type.37b) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @I.WithSelf.GetU.%as_type (%as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @I.WithSelf.GetU.%as_type (%as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc1.loc9_21.1: @I.WithSelf.%I.assoc_type (%I.assoc_type) = assoc_entity element1, %I.WithSelf.GetU.decl [symbolic = %assoc1.loc9_21.2 (constants.%assoc1)]
// CHECK:STDOUT:     %I.WithSelf.TakesT.decl: @I.WithSelf.%I.WithSelf.TakesT.type (%I.WithSelf.TakesT.type) = fn_decl @I.WithSelf.TakesT [symbolic = @I.WithSelf.%I.WithSelf.TakesT (constants.%I.WithSelf.TakesT)] {
// CHECK:STDOUT:       %self.param_patt.loc10_13.1: @I.WithSelf.TakesT.%pattern_type.loc10_13 (%pattern_type.f76) = value_param_pattern [symbolic = %self.param_patt.loc10_13.2 (constants.%self.param_patt.ee4)]
// CHECK:STDOUT:       %self.patt.loc10_13.1: @I.WithSelf.TakesT.%pattern_type.loc10_13 (%pattern_type.f76) = at_binding_pattern self, %self.param_patt.loc10_13.1 [symbolic = %self.patt.loc10_13.2 (constants.%self.patt.2e1)]
// CHECK:STDOUT:       %t.param_patt.loc10_20.1: @I.WithSelf.TakesT.%pattern_type.loc10_20 (%pattern_type.bc3) = value_param_pattern [symbolic = %t.param_patt.loc10_20.2 (constants.%t.param_patt)]
// CHECK:STDOUT:       %t.patt.loc10_20.1: @I.WithSelf.TakesT.%pattern_type.loc10_20 (%pattern_type.bc3) = at_binding_pattern t, %t.param_patt.loc10_20.1 [symbolic = %t.patt.loc10_20.2 (constants.%t.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @I.WithSelf.TakesT.%Self.as_type.loc10_13.1 (%Self.as_type.37b) = value_param call_param0
// CHECK:STDOUT:       %.loc10_13.1: type = splice_block %.loc10_13.3 [symbolic = %Self.as_type.loc10_13.1 (constants.%Self.as_type.37b)] {
// CHECK:STDOUT:         %.loc10_13.2: @I.WithSelf.TakesT.%I.type (%I.type.715) = specific_constant @I.%Self.loc6_34.1, @I(constants.%T) [symbolic = %Self (constants.%Self.dca)]
// CHECK:STDOUT:         %Self.ref: @I.WithSelf.TakesT.%I.type (%I.type.715) = name_ref Self, %.loc10_13.2 [symbolic = %Self (constants.%Self.dca)]
// CHECK:STDOUT:         %Self.as_type.loc10_13.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_13.1 (constants.%Self.as_type.37b)]
// CHECK:STDOUT:         %.loc10_13.3: type = converted %Self.ref, %Self.as_type.loc10_13.2 [symbolic = %Self.as_type.loc10_13.1 (constants.%Self.as_type.37b)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @I.WithSelf.TakesT.%Self.as_type.loc10_13.1 (%Self.as_type.37b) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %t.param: @I.WithSelf.TakesT.%T.as_type.loc10_22.1 (%T.as_type) = value_param call_param1
// CHECK:STDOUT:       %.loc10_22.1: type = splice_block %.loc10_22.2 [symbolic = %T.as_type.loc10_22.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref: %facet_type.8cb = name_ref T, @I.%T.loc6_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:         %T.as_type.loc10_22.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_22.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc10_22.2: type = converted %T.ref, %T.as_type.loc10_22.2 [symbolic = %T.as_type.loc10_22.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: @I.WithSelf.TakesT.%T.as_type.loc10_22.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc2.loc10_24.1: @I.WithSelf.%I.assoc_type (%I.assoc_type) = assoc_entity element2, %I.WithSelf.TakesT.decl [symbolic = %assoc2.loc10_24.2 (constants.%assoc2)]
// CHECK:STDOUT:     %T.ref: %facet_type.8cb = name_ref T, @I.%T.loc6_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %impl.elem0.loc12_16.1: %facet_type.208 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_16.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.ref: %facet_type.208 = name_ref U, %impl.elem0.loc12_16.1 [symbolic = %impl.elem0.loc12_16.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.as_type: type = facet_access_type %U.ref [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:     %.loc12_16: type = converted %U.ref, %U.as_type [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:     %T.as_type.loc12_11.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc12_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc12_11: type = converted %T.ref, %T.as_type.loc12_11.1 [symbolic = %T.as_type.loc12_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %I.observe.decl = observe_decl @I.observe [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc6_34.1
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .U = @U.%assoc0
// CHECK:STDOUT:     .GetU = @I.WithSelf.%assoc1.loc9_21.1
// CHECK:STDOUT:     .TakesT = @I.WithSelf.%assoc2.loc10_24.1
// CHECK:STDOUT:     witness = (@I.WithSelf.%U, @I.WithSelf.%I.WithSelf.GetU.decl, @I.WithSelf.%I.WithSelf.TakesT.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @I.observe {
// CHECK:STDOUT:       observe constants.%T == constants.%impl.elem0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.GetU(@I.%T.loc6_14.2: %facet_type.8cb, @I.%Self.loc6_34.1: @I.%I.type (%I.type.715)) {
// CHECK:STDOUT:   %T: %facet_type.8cb = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.715)]
// CHECK:STDOUT:   %Self: @I.WithSelf.GetU.%I.type (%I.type.715) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.dca)]
// CHECK:STDOUT:   %Self.as_type.loc9_11.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_11.1 (constants.%Self.as_type.37b)]
// CHECK:STDOUT:   %pattern_type.loc9_11: type = pattern_type %Self.as_type.loc9_11.1 [symbolic = %pattern_type.loc9_11 (constants.%pattern_type.f76)]
// CHECK:STDOUT:   %self.param_patt.loc9_11.2: @I.WithSelf.GetU.%pattern_type.loc9_11 (%pattern_type.f76) = value_param_pattern [symbolic = %self.param_patt.loc9_11.2 (constants.%self.param_patt.ee4)]
// CHECK:STDOUT:   %self.patt.loc9_11.2: @I.WithSelf.GetU.%pattern_type.loc9_11 (%pattern_type.f76) = at_binding_pattern self, %self.param_patt.loc9_11.2 [symbolic = %self.patt.loc9_11.2 (constants.%self.patt.2e1)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I, @I(%T) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc9_20.1: %facet_type.208 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_20.1 (constants.%impl.elem0)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc9_20.1 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:   %.loc9_20.1: Core.Form = init_form %as_type [symbolic = %.loc9_20.1 (constants.%.344)]
// CHECK:STDOUT:   %pattern_type.loc9_20: type = pattern_type %as_type [symbolic = %pattern_type.loc9_20 (constants.%pattern_type.079)]
// CHECK:STDOUT:   %return.param_patt.loc9_20.2: @I.WithSelf.GetU.%pattern_type.loc9_20 (%pattern_type.079) = out_param_pattern [symbolic = %return.param_patt.loc9_20.2 (constants.%return.param_patt.82c)]
// CHECK:STDOUT:   %return.patt.loc9_17.2: @I.WithSelf.GetU.%pattern_type.loc9_20 (%pattern_type.079) = return_slot_pattern %return.param_patt.loc9_20.2, %as_type [symbolic = %return.patt.loc9_17.2 (constants.%return.patt.4f7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.GetU.%Self.as_type.loc9_11.1 (%Self.as_type.37b)) -> out %return.param: @I.WithSelf.GetU.%as_type (%as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.TakesT(@I.%T.loc6_14.2: %facet_type.8cb, @I.%Self.loc6_34.1: @I.%I.type (%I.type.715)) {
// CHECK:STDOUT:   %T: %facet_type.8cb = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.715)]
// CHECK:STDOUT:   %Self: @I.WithSelf.TakesT.%I.type (%I.type.715) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.dca)]
// CHECK:STDOUT:   %Self.as_type.loc10_13.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_13.1 (constants.%Self.as_type.37b)]
// CHECK:STDOUT:   %pattern_type.loc10_13: type = pattern_type %Self.as_type.loc10_13.1 [symbolic = %pattern_type.loc10_13 (constants.%pattern_type.f76)]
// CHECK:STDOUT:   %self.param_patt.loc10_13.2: @I.WithSelf.TakesT.%pattern_type.loc10_13 (%pattern_type.f76) = value_param_pattern [symbolic = %self.param_patt.loc10_13.2 (constants.%self.param_patt.ee4)]
// CHECK:STDOUT:   %self.patt.loc10_13.2: @I.WithSelf.TakesT.%pattern_type.loc10_13 (%pattern_type.f76) = at_binding_pattern self, %self.param_patt.loc10_13.2 [symbolic = %self.patt.loc10_13.2 (constants.%self.patt.2e1)]
// CHECK:STDOUT:   %T.as_type.loc10_22.1: type = facet_access_type %T [symbolic = %T.as_type.loc10_22.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc10_20: type = pattern_type %T.as_type.loc10_22.1 [symbolic = %pattern_type.loc10_20 (constants.%pattern_type.bc3)]
// CHECK:STDOUT:   %t.param_patt.loc10_20.2: @I.WithSelf.TakesT.%pattern_type.loc10_20 (%pattern_type.bc3) = value_param_pattern [symbolic = %t.param_patt.loc10_20.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc10_20.2: @I.WithSelf.TakesT.%pattern_type.loc10_20 (%pattern_type.bc3) = at_binding_pattern t, %t.param_patt.loc10_20.2 [symbolic = %t.patt.loc10_20.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.TakesT.%Self.as_type.loc10_13.1 (%Self.as_type.37b), %t.param: @I.WithSelf.TakesT.%T.as_type.loc10_22.1 (%T.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.dca) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.GetU(constants.%T, constants.%Self.dca) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.715
// CHECK:STDOUT:   %Self => constants.%Self.dca
// CHECK:STDOUT:   %Self.as_type.loc9_11.1 => constants.%Self.as_type.37b
// CHECK:STDOUT:   %pattern_type.loc9_11 => constants.%pattern_type.f76
// CHECK:STDOUT:   %self.param_patt.loc9_11.2 => constants.%self.param_patt.ee4
// CHECK:STDOUT:   %self.patt.loc9_11.2 => constants.%self.patt.2e1
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem0.loc9_20.1 => constants.%impl.elem0
// CHECK:STDOUT:   %as_type => constants.%as_type
// CHECK:STDOUT:   %.loc9_20.1 => constants.%.344
// CHECK:STDOUT:   %pattern_type.loc9_20 => constants.%pattern_type.079
// CHECK:STDOUT:   %return.param_patt.loc9_20.2 => constants.%return.param_patt.82c
// CHECK:STDOUT:   %return.patt.loc9_17.2 => constants.%return.patt.4f7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.TakesT(constants.%T, constants.%Self.dca) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.715
// CHECK:STDOUT:   %Self => constants.%Self.dca
// CHECK:STDOUT:   %Self.as_type.loc10_13.1 => constants.%Self.as_type.37b
// CHECK:STDOUT:   %pattern_type.loc10_13 => constants.%pattern_type.f76
// CHECK:STDOUT:   %self.param_patt.loc10_13.2 => constants.%self.param_patt.ee4
// CHECK:STDOUT:   %self.patt.loc10_13.2 => constants.%self.patt.2e1
// CHECK:STDOUT:   %T.as_type.loc10_22.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc10_20 => constants.%pattern_type.bc3
// CHECK:STDOUT:   %t.param_patt.loc10_20.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc10_20.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:

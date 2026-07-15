// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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

interface Transitive {
  let A:! P & Core.Destroy;
  let B:! (Q where .Self == A) & Core.Destroy;
  let C:! (R where .Self == B) & Core.Destroy;

  fn GetA(self) -> A;
  fn TakesC(self, c: C);

  observe A == B == C;
}

// --- equivalent.carbon
library "[[@TEST_NAME]]";

import library "transitive";

fn F[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  var b: T.B = a;
  b.InQ();

  t.TakesC(a);

  (a as T.C).InR();
}

// --- fail_todo_impls.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B { require impls A; }
interface C { require impls B; }

fn RequiresA[T:! A](unused x: T) {}
fn F[T:! C](x: T) {
  observe T impls B;

  // CHECK:STDERR: fail_todo_impls.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `C` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_impls.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T:! A](unused x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(x);
}

// --- fail_todo_equivalent_impls.carbon
library "[[@TEST_NAME]]";

import library "transitive";

fn TakesPQR[U:! P & Q & R](unused u: U) {}
fn G[T:! Transitive](t: T) {
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
  // CHECK:STDERR: fn TakesPQR[U:! P & Q & R](unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakesPQR(a);
}

// --- fail_equivalent_impls_without_observe.carbon
library "[[@TEST_NAME]]";

import library "transitive";

fn TakesPQR[U:! P & Q & R](unused u: U) {}
fn G[T:! Transitive](t: T) {
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
  // CHECK:STDERR: fn TakesPQR[U:! P & Q & R](unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakesPQR(a);
}

// --- same_observe_in_both_scopes.carbon
library "[[@TEST_NAME]]";

import library "transitive";

fn G[T:! Transitive](t: T) {
  observe T.A == T.B == T.C;

  var a: T.A = t.GetA();
  t.TakesC(a);
}

// --- blanket_impls.carbon
library "[[@TEST_NAME]]";

interface A { }
interface B { }
interface C { }
interface D { }

impl forall [T:! A] T as B { }
impl forall [T:! B] T as C { }
impl forall [T:! C] T as D { }

fn RequiresB[T:! B](unused x: T) {}
fn RequiresD[T:! D](unused x: T) {}

fn RequiresA[T:! A](x: T) {
  RequiresB(x);

  observe T impls B;
  observe T impls C;
  RequiresD(x);
}

// --- nested_functions_in_outer.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}
interface C {}

fn TakesA[T:! A](unused x: T) {}
fn F[T:! A, U:! B where .Self == T, V:! C where .Self == U]() {
  observe T == U == V;

  fn InnerF(x: V) {
    TakesA(x);
  }
}

// --- nested_functions_in_inner.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}
interface C {}

fn TakesA[T:! A](unused x: T) {}
fn F[T:! A, U:! B where .Self == T, V:! C where .Self == U]() {

  fn InnerF(x: V) {
      observe T == U == V;

    TakesA(x);
  }
}

// --- fail_nested_functions_lexical_order.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}
interface C {}

fn TakesA[T:! A](unused x: T) {}
fn F[T:! A, U:! B where .Self == T, V:! C where .Self == U]() {

  fn InnerF(x: V) {
    // CHECK:STDERR: fail_nested_functions_lexical_order.carbon:[[@LINE+7]]:5: error: cannot convert type `V` that implements `C where...` into type implementing `A` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     TakesA(x);
    // CHECK:STDERR:     ^~~~~~~~~
    // CHECK:STDERR: fail_nested_functions_lexical_order.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR: fn TakesA[T:! A](unused x: T) {}
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    TakesA(x);
  }

  observe T == U == V;
}

// --- fail_nested_functions_leak_upward.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}
interface C {}

fn TakesA[T:! A](unused x: T) {}
fn F[T:! A, U:! B where .Self == T, V:! C where .Self == U](x: V) {

  fn InnerF() {
    observe T == U == V;
  }

  // CHECK:STDERR: fail_nested_functions_leak_upward.carbon:[[@LINE+7]]:3: error: cannot convert type `V` that implements `C where...` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   TakesA(x);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_nested_functions_leak_upward.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TakesA[T:! A](unused x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakesA(x);
}

// --- fail_not_observed.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}
interface C {}

fn TakesA[T:! A](unused x: T) {}
fn F[T:! A, U:! B where .Self == T, V:! C where .Self == U](x: V) {
  observe T == U;

  // CHECK:STDERR: fail_not_observed.carbon:[[@LINE+7]]:3: error: cannot convert type `V` that implements `C where...` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   TakesA(x);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_not_observed.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TakesA[T:! A](unused x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TakesA(x);
}

// --- todo_fail_invalid_equivalent.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn F[T:! A, U:! B]() {
  // TODO: This should be diagnosed, since T and U are not equal by a
  // single same-type where equality constraint.
  observe T == U;
}

// --- fail_wrong_scope_class.carbon
library "[[@TEST_NAME]]";

class C(T:! type, U:! type) {
  // CHECK:STDERR: fail_wrong_scope_class.carbon:[[@LINE+4]]:3: error: `observe` can only be used in an `interface` or `function` [ObserveInWrongScope]
  // CHECK:STDERR:   observe T == U;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  observe T == U;
}

// --- fail_wrong_scope_named_constraint.carbon
library "[[@TEST_NAME]]";

constraint C(T:! type, U:! type) {
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

fn F[T:! A]() {
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
interface C {}
interface I(T:! A) {
  let U:! (B where .Self == T) & Core.Destroy;
  let V:! (C where .Self == U) & Core.Destroy;

  fn GetV(self) -> V;
  fn TakesT(self, t: T);

  observe T == U == V;
}

// --- generic_import.carbon
library "[[@TEST_NAME]]";

import library "generic_observe";

fn F[W:! A, X:! I(W)](x: X) {
  var v: X.V = x.GetV();

  x.TakesT(v);
}

// CHECK:STDOUT: --- transitive.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.801: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b49: type = facet_access_type %Self.801 [symbolic]
// CHECK:STDOUT:   %pattern_type.fc0: type = pattern_type %Self.as_type.b49 [symbolic]
// CHECK:STDOUT:   %self.param_patt.97c: %pattern_type.fc0 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.090: %pattern_type.fc0 = at_binding_pattern self, %self.param_patt.97c [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P.WithSelf(%Self.801) [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %P.assoc_type: type = assoc_entity_type @P [concrete]
// CHECK:STDOUT:   %assoc0.1ba: %P.assoc_type = assoc_entity element0, @P.WithSelf.%P.WithSelf.InP.decl [concrete]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.ca9: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7c9: type = facet_access_type %Self.ca9 [symbolic]
// CHECK:STDOUT:   %pattern_type.684: type = pattern_type %Self.as_type.7c9 [symbolic]
// CHECK:STDOUT:   %self.param_patt.9cf: %pattern_type.684 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.f63: %pattern_type.684 = at_binding_pattern self, %self.param_patt.9cf [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Self.ca9) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ: %Q.WithSelf.InQ.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.26a: %Q.assoc_type = assoc_entity element0, @Q.WithSelf.%Q.WithSelf.InQ.decl [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.9c1: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.85f: type = facet_access_type %Self.9c1 [symbolic]
// CHECK:STDOUT:   %pattern_type.ebd: type = pattern_type %Self.as_type.85f [symbolic]
// CHECK:STDOUT:   %self.param_patt.659: %pattern_type.ebd = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.da0: %pattern_type.ebd = at_binding_pattern self, %self.param_patt.659 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R.WithSelf(%Self.9c1) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR: %R.WithSelf.InR.type = struct_value () [symbolic]
// CHECK:STDOUT:   %R.assoc_type: type = assoc_entity_type @R [concrete]
// CHECK:STDOUT:   %assoc0.782: %R.assoc_type = assoc_entity element0, @R.WithSelf.%R.WithSelf.InR.decl [concrete]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.77c: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .P = %P.decl
// CHECK:STDOUT:     .Q = %Q.decl
// CHECK:STDOUT:     .R = %R.decl
// CHECK:STDOUT:     .Transitive = %Transitive.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %P.decl: type = interface_decl @P [concrete = constants.%P.type] {} {}
// CHECK:STDOUT:   %Q.decl: type = interface_decl @Q [concrete = constants.%Q.type] {} {}
// CHECK:STDOUT:   %R.decl: type = interface_decl @R [concrete = constants.%R.type] {} {}
// CHECK:STDOUT:   %Transitive.decl: type = interface_decl @Transitive [concrete = constants.%Transitive.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @P {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = constants.%Self.801]
// CHECK:STDOUT:   %P.WithSelf.decl = interface_with_self_decl @P [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %P.WithSelf.InP.decl: @P.WithSelf.%P.WithSelf.InP.type (%P.WithSelf.InP.type) = fn_decl @P.WithSelf.InP [symbolic = @P.WithSelf.%P.WithSelf.InP (constants.%P.WithSelf.InP)] {
// CHECK:STDOUT:     %self.param_patt.loc3_22.1: @P.WithSelf.InP.%pattern_type (%pattern_type.fc0) = value_param_pattern [symbolic = %self.param_patt.loc3_22.2 (constants.%self.param_patt.97c)]
// CHECK:STDOUT:     %self.patt.loc3_22.1: @P.WithSelf.InP.%pattern_type (%pattern_type.fc0) = at_binding_pattern self, %self.param_patt.loc3_22.1 [symbolic = %self.patt.loc3_22.2 (constants.%self.patt.090)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @P.WithSelf.InP.%Self.as_type.loc3_22.1 (%Self.as_type.b49) = value_param call_param0
// CHECK:STDOUT:     %.loc3_22.1: type = splice_block %.loc3_22.2 [symbolic = %Self.as_type.loc3_22.1 (constants.%Self.as_type.b49)] {
// CHECK:STDOUT:       %Self.ref: %P.type = name_ref Self, @P.%Self [symbolic = %Self (constants.%Self.801)]
// CHECK:STDOUT:       %Self.as_type.loc3_22.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc3_22.1 (constants.%Self.as_type.b49)]
// CHECK:STDOUT:       %.loc3_22.2: type = converted %Self.ref, %Self.as_type.loc3_22.2 [symbolic = %Self.as_type.loc3_22.1 (constants.%Self.as_type.b49)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @P.WithSelf.InP.%Self.as_type.loc3_22.1 (%Self.as_type.b49) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %P.assoc_type = assoc_entity element0, %P.WithSelf.InP.decl [concrete = constants.%assoc0.1ba]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InP = @P.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@P.WithSelf.%P.WithSelf.InP.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ca9]
// CHECK:STDOUT:   %Q.WithSelf.decl = interface_with_self_decl @Q [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Q.WithSelf.InQ.decl: @Q.WithSelf.%Q.WithSelf.InQ.type (%Q.WithSelf.InQ.type) = fn_decl @Q.WithSelf.InQ [symbolic = @Q.WithSelf.%Q.WithSelf.InQ (constants.%Q.WithSelf.InQ)] {
// CHECK:STDOUT:     %self.param_patt.loc4_22.1: @Q.WithSelf.InQ.%pattern_type (%pattern_type.684) = value_param_pattern [symbolic = %self.param_patt.loc4_22.2 (constants.%self.param_patt.9cf)]
// CHECK:STDOUT:     %self.patt.loc4_22.1: @Q.WithSelf.InQ.%pattern_type (%pattern_type.684) = at_binding_pattern self, %self.param_patt.loc4_22.1 [symbolic = %self.patt.loc4_22.2 (constants.%self.patt.f63)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Q.WithSelf.InQ.%Self.as_type.loc4_22.1 (%Self.as_type.7c9) = value_param call_param0
// CHECK:STDOUT:     %.loc4_22.1: type = splice_block %.loc4_22.2 [symbolic = %Self.as_type.loc4_22.1 (constants.%Self.as_type.7c9)] {
// CHECK:STDOUT:       %Self.ref: %Q.type = name_ref Self, @Q.%Self [symbolic = %Self (constants.%Self.ca9)]
// CHECK:STDOUT:       %Self.as_type.loc4_22.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_22.1 (constants.%Self.as_type.7c9)]
// CHECK:STDOUT:       %.loc4_22.2: type = converted %Self.ref, %Self.as_type.loc4_22.2 [symbolic = %Self.as_type.loc4_22.1 (constants.%Self.as_type.7c9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Q.WithSelf.InQ.%Self.as_type.loc4_22.1 (%Self.as_type.7c9) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Q.assoc_type = assoc_entity element0, %Q.WithSelf.InQ.decl [concrete = constants.%assoc0.26a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InQ = @Q.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Q.WithSelf.%Q.WithSelf.InQ.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = constants.%Self.9c1]
// CHECK:STDOUT:   %R.WithSelf.decl = interface_with_self_decl @R [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %R.WithSelf.InR.decl: @R.WithSelf.%R.WithSelf.InR.type (%R.WithSelf.InR.type) = fn_decl @R.WithSelf.InR [symbolic = @R.WithSelf.%R.WithSelf.InR (constants.%R.WithSelf.InR)] {
// CHECK:STDOUT:     %self.param_patt.loc5_22.1: @R.WithSelf.InR.%pattern_type (%pattern_type.ebd) = value_param_pattern [symbolic = %self.param_patt.loc5_22.2 (constants.%self.param_patt.659)]
// CHECK:STDOUT:     %self.patt.loc5_22.1: @R.WithSelf.InR.%pattern_type (%pattern_type.ebd) = at_binding_pattern self, %self.param_patt.loc5_22.1 [symbolic = %self.patt.loc5_22.2 (constants.%self.patt.da0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @R.WithSelf.InR.%Self.as_type.loc5_22.1 (%Self.as_type.85f) = value_param call_param0
// CHECK:STDOUT:     %.loc5_22.1: type = splice_block %.loc5_22.2 [symbolic = %Self.as_type.loc5_22.1 (constants.%Self.as_type.85f)] {
// CHECK:STDOUT:       %Self.ref: %R.type = name_ref Self, @R.%Self [symbolic = %Self (constants.%Self.9c1)]
// CHECK:STDOUT:       %Self.as_type.loc5_22.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_22.1 (constants.%Self.as_type.85f)]
// CHECK:STDOUT:       %.loc5_22.2: type = converted %Self.ref, %Self.as_type.loc5_22.2 [symbolic = %Self.as_type.loc5_22.1 (constants.%Self.as_type.85f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @R.WithSelf.InR.%Self.as_type.loc5_22.1 (%Self.as_type.85f) = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %R.assoc_type = assoc_entity element0, %R.WithSelf.InR.decl [concrete = constants.%assoc0.782]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InR = @R.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@R.WithSelf.%R.WithSelf.InR.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
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
// CHECK:STDOUT:     %self.param_patt.loc12_11.1: @Transitive.WithSelf.GetA.%pattern_type.loc12_11 (%pattern_type.b5c) = value_param_pattern [symbolic = %self.param_patt.loc12_11.2 (constants.%self.param_patt.ae9)]
// CHECK:STDOUT:     %self.patt.loc12_11.1: @Transitive.WithSelf.GetA.%pattern_type.loc12_11 (%pattern_type.b5c) = at_binding_pattern self, %self.param_patt.loc12_11.1 [symbolic = %self.patt.loc12_11.2 (constants.%self.patt.12b)]
// CHECK:STDOUT:     %return.param_patt.loc12_20.1: @Transitive.WithSelf.GetA.%pattern_type.loc12_20 (%pattern_type.51a) = out_param_pattern [symbolic = %return.param_patt.loc12_20.2 (constants.%return.param_patt.931)]
// CHECK:STDOUT:     %return.patt.loc12_17.1: @Transitive.WithSelf.GetA.%pattern_type.loc12_20 (%pattern_type.51a) = return_slot_pattern %return.param_patt.loc12_20.1, %.loc12_20.2 [symbolic = %return.patt.loc12_17.2 (constants.%return.patt.680)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc12_20.2: %facet_type.b1b = impl_witness_access constants.%Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_20.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %A.ref: %facet_type.b1b = name_ref A, %impl.elem0.loc12_20.2 [symbolic = %impl.elem0.loc12_20.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %A.as_type: type = facet_access_type %A.ref [symbolic = %as_type (constants.%as_type.d30)]
// CHECK:STDOUT:     %.loc12_20.2: type = converted %A.ref, %A.as_type [symbolic = %as_type (constants.%as_type.d30)]
// CHECK:STDOUT:     %.loc12_20.3: Core.Form = init_form %.loc12_20.2 [symbolic = %.loc12_20.1 (constants.%.117)]
// CHECK:STDOUT:     %self.param: @Transitive.WithSelf.GetA.%Self.as_type.loc12_11.1 (%Self.as_type.7f2) = value_param call_param0
// CHECK:STDOUT:     %.loc12_11.1: type = splice_block %.loc12_11.2 [symbolic = %Self.as_type.loc12_11.1 (constants.%Self.as_type.7f2)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self [symbolic = %Self (constants.%Self.77c)]
// CHECK:STDOUT:       %Self.as_type.loc12_11.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_11.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:       %.loc12_11.2: type = converted %Self.ref, %Self.as_type.loc12_11.2 [symbolic = %Self.as_type.loc12_11.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.WithSelf.GetA.%Self.as_type.loc12_11.1 (%Self.as_type.7f2) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Transitive.WithSelf.GetA.%as_type (%as_type.d30) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Transitive.WithSelf.GetA.%as_type (%as_type.d30) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, %Transitive.WithSelf.GetA.decl [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.decl: @Transitive.WithSelf.%Transitive.WithSelf.TakesC.type (%Transitive.WithSelf.TakesC.type) = fn_decl @Transitive.WithSelf.TakesC [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.TakesC (constants.%Transitive.WithSelf.TakesC)] {
// CHECK:STDOUT:     %self.param_patt.loc13_13.1: @Transitive.WithSelf.TakesC.%pattern_type.loc13_13 (%pattern_type.b5c) = value_param_pattern [symbolic = %self.param_patt.loc13_13.2 (constants.%self.param_patt.ae9)]
// CHECK:STDOUT:     %self.patt.loc13_13.1: @Transitive.WithSelf.TakesC.%pattern_type.loc13_13 (%pattern_type.b5c) = at_binding_pattern self, %self.param_patt.loc13_13.1 [symbolic = %self.patt.loc13_13.2 (constants.%self.patt.12b)]
// CHECK:STDOUT:     %c.param_patt.loc13_20.1: @Transitive.WithSelf.TakesC.%pattern_type.loc13_20 (%pattern_type.007) = value_param_pattern [symbolic = %c.param_patt.loc13_20.2 (constants.%c.param_patt)]
// CHECK:STDOUT:     %c.patt.loc13_20.1: @Transitive.WithSelf.TakesC.%pattern_type.loc13_20 (%pattern_type.007) = at_binding_pattern c, %c.param_patt.loc13_20.1 [symbolic = %c.patt.loc13_20.2 (constants.%c.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Transitive.WithSelf.TakesC.%Self.as_type.loc13_13.1 (%Self.as_type.7f2) = value_param call_param0
// CHECK:STDOUT:     %.loc13_13.1: type = splice_block %.loc13_13.2 [symbolic = %Self.as_type.loc13_13.1 (constants.%Self.as_type.7f2)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self [symbolic = %Self (constants.%Self.77c)]
// CHECK:STDOUT:       %Self.as_type.loc13_13.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc13_13.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:       %.loc13_13.2: type = converted %Self.ref, %Self.as_type.loc13_13.2 [symbolic = %Self.as_type.loc13_13.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.WithSelf.TakesC.%Self.as_type.loc13_13.1 (%Self.as_type.7f2) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %c.param: @Transitive.WithSelf.TakesC.%as_type (%as_type.472) = value_param call_param1
// CHECK:STDOUT:     %.loc13_22.1: type = splice_block %.loc13_22.2 [symbolic = %as_type (constants.%as_type.472)] {
// CHECK:STDOUT:       %impl.elem2.loc13_22.2: %facet_type.4c8 = impl_witness_access constants.%Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc13_22.1 (constants.%impl.elem2)]
// CHECK:STDOUT:       %C.ref: %facet_type.4c8 = name_ref C, %impl.elem2.loc13_22.2 [symbolic = %impl.elem2.loc13_22.1 (constants.%impl.elem2)]
// CHECK:STDOUT:       %C.as_type: type = facet_access_type %C.ref [symbolic = %as_type (constants.%as_type.472)]
// CHECK:STDOUT:       %.loc13_22.2: type = converted %C.ref, %C.as_type [symbolic = %as_type (constants.%as_type.472)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @Transitive.WithSelf.TakesC.%as_type (%as_type.472) = wrapper_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, %Transitive.WithSelf.TakesC.decl [concrete = constants.%assoc4]
// CHECK:STDOUT:   %impl.elem0.loc15: %facet_type.b1b = impl_witness_access constants.%Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9 (constants.%impl.elem0)]
// CHECK:STDOUT:   %A.ref: %facet_type.b1b = name_ref A, %impl.elem0.loc15 [symbolic = %impl.elem0.loc9 (constants.%impl.elem0)]
// CHECK:STDOUT:   %impl.elem1.loc15: %facet_type.130 = impl_witness_access constants.%Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc10 (constants.%impl.elem1)]
// CHECK:STDOUT:   %B.ref: %facet_type.130 = name_ref B, %impl.elem1.loc15 [symbolic = %impl.elem1.loc10 (constants.%impl.elem1)]
// CHECK:STDOUT:   %B.as_type.loc15_16.1: type = facet_access_type %B.ref [symbolic = %as_type.loc15_16 (constants.%as_type.aed)]
// CHECK:STDOUT:   %.loc15_16.1: type = converted %B.ref, %B.as_type.loc15_16.1 [symbolic = %as_type.loc15_16 (constants.%as_type.aed)]
// CHECK:STDOUT:   %A.as_type: type = facet_access_type %A.ref [symbolic = %as_type.loc15_11 (constants.%as_type.d30)]
// CHECK:STDOUT:   %.loc15_11: type = converted %A.ref, %A.as_type [symbolic = %as_type.loc15_11 (constants.%as_type.d30)]
// CHECK:STDOUT:   %impl.elem2.loc15_21.1: %facet_type.4c8 = impl_witness_access constants.%Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc15_21.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %C.ref: %facet_type.4c8 = name_ref C, %impl.elem2.loc15_21.1 [symbolic = %impl.elem2.loc15_21.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %C.as_type: type = facet_access_type %C.ref [symbolic = %as_type.loc15_21 (constants.%as_type.472)]
// CHECK:STDOUT:   %.loc15_21: type = converted %C.ref, %C.as_type [symbolic = %as_type.loc15_21 (constants.%as_type.472)]
// CHECK:STDOUT:   %B.as_type.loc15_16.2: type = facet_access_type %B.ref [symbolic = %as_type.loc15_16 (constants.%as_type.aed)]
// CHECK:STDOUT:   %.loc15_16.2: type = converted %B.ref, %B.as_type.loc15_16.2 [symbolic = %as_type.loc15_16 (constants.%as_type.aed)]
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
// CHECK:STDOUT: generic fn @P.WithSelf.InP(@P.%Self: %P.type) {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.801)]
// CHECK:STDOUT:   %Self.as_type.loc3_22.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc3_22.1 (constants.%Self.as_type.b49)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc3_22.1 [symbolic = %pattern_type (constants.%pattern_type.fc0)]
// CHECK:STDOUT:   %self.param_patt.loc3_22.2: @P.WithSelf.InP.%pattern_type (%pattern_type.fc0) = value_param_pattern [symbolic = %self.param_patt.loc3_22.2 (constants.%self.param_patt.97c)]
// CHECK:STDOUT:   %self.patt.loc3_22.2: @P.WithSelf.InP.%pattern_type (%pattern_type.fc0) = at_binding_pattern self, %self.param_patt.loc3_22.2 [symbolic = %self.patt.loc3_22.2 (constants.%self.patt.090)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @P.WithSelf.InP.%Self.as_type.loc3_22.1 (%Self.as_type.b49));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(@Q.%Self: %Q.type) {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.ca9)]
// CHECK:STDOUT:   %Self.as_type.loc4_22.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_22.1 (constants.%Self.as_type.7c9)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc4_22.1 [symbolic = %pattern_type (constants.%pattern_type.684)]
// CHECK:STDOUT:   %self.param_patt.loc4_22.2: @Q.WithSelf.InQ.%pattern_type (%pattern_type.684) = value_param_pattern [symbolic = %self.param_patt.loc4_22.2 (constants.%self.param_patt.9cf)]
// CHECK:STDOUT:   %self.patt.loc4_22.2: @Q.WithSelf.InQ.%pattern_type (%pattern_type.684) = at_binding_pattern self, %self.param_patt.loc4_22.2 [symbolic = %self.patt.loc4_22.2 (constants.%self.patt.f63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Q.WithSelf.InQ.%Self.as_type.loc4_22.1 (%Self.as_type.7c9));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(@R.%Self: %R.type) {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9c1)]
// CHECK:STDOUT:   %Self.as_type.loc5_22.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_22.1 (constants.%Self.as_type.85f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_22.1 [symbolic = %pattern_type (constants.%pattern_type.ebd)]
// CHECK:STDOUT:   %self.param_patt.loc5_22.2: @R.WithSelf.InR.%pattern_type (%pattern_type.ebd) = value_param_pattern [symbolic = %self.param_patt.loc5_22.2 (constants.%self.param_patt.659)]
// CHECK:STDOUT:   %self.patt.loc5_22.2: @R.WithSelf.InR.%pattern_type (%pattern_type.ebd) = at_binding_pattern self, %self.param_patt.loc5_22.2 [symbolic = %self.patt.loc5_22.2 (constants.%self.patt.da0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @R.WithSelf.InR.%Self.as_type.loc5_22.1 (%Self.as_type.85f));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.77c)]
// CHECK:STDOUT:   %Self.as_type.loc12_11.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_11.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:   %pattern_type.loc12_11: type = pattern_type %Self.as_type.loc12_11.1 [symbolic = %pattern_type.loc12_11 (constants.%pattern_type.b5c)]
// CHECK:STDOUT:   %self.param_patt.loc12_11.2: @Transitive.WithSelf.GetA.%pattern_type.loc12_11 (%pattern_type.b5c) = value_param_pattern [symbolic = %self.param_patt.loc12_11.2 (constants.%self.param_patt.ae9)]
// CHECK:STDOUT:   %self.patt.loc12_11.2: @Transitive.WithSelf.GetA.%pattern_type.loc12_11 (%pattern_type.b5c) = at_binding_pattern self, %self.param_patt.loc12_11.2 [symbolic = %self.patt.loc12_11.2 (constants.%self.patt.12b)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc12_20.1: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_20.1 (constants.%impl.elem0)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc12_20.1 [symbolic = %as_type (constants.%as_type.d30)]
// CHECK:STDOUT:   %.loc12_20.1: Core.Form = init_form %as_type [symbolic = %.loc12_20.1 (constants.%.117)]
// CHECK:STDOUT:   %pattern_type.loc12_20: type = pattern_type %as_type [symbolic = %pattern_type.loc12_20 (constants.%pattern_type.51a)]
// CHECK:STDOUT:   %return.param_patt.loc12_20.2: @Transitive.WithSelf.GetA.%pattern_type.loc12_20 (%pattern_type.51a) = out_param_pattern [symbolic = %return.param_patt.loc12_20.2 (constants.%return.param_patt.931)]
// CHECK:STDOUT:   %return.patt.loc12_17.2: @Transitive.WithSelf.GetA.%pattern_type.loc12_20 (%pattern_type.51a) = return_slot_pattern %return.param_patt.loc12_20.2, %as_type [symbolic = %return.patt.loc12_17.2 (constants.%return.patt.680)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.WithSelf.GetA.%Self.as_type.loc12_11.1 (%Self.as_type.7f2)) -> out %return.param: @Transitive.WithSelf.GetA.%as_type (%as_type.d30);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.77c)]
// CHECK:STDOUT:   %Self.as_type.loc13_13.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc13_13.1 (constants.%Self.as_type.7f2)]
// CHECK:STDOUT:   %pattern_type.loc13_13: type = pattern_type %Self.as_type.loc13_13.1 [symbolic = %pattern_type.loc13_13 (constants.%pattern_type.b5c)]
// CHECK:STDOUT:   %self.param_patt.loc13_13.2: @Transitive.WithSelf.TakesC.%pattern_type.loc13_13 (%pattern_type.b5c) = value_param_pattern [symbolic = %self.param_patt.loc13_13.2 (constants.%self.param_patt.ae9)]
// CHECK:STDOUT:   %self.patt.loc13_13.2: @Transitive.WithSelf.TakesC.%pattern_type.loc13_13 (%pattern_type.b5c) = at_binding_pattern self, %self.param_patt.loc13_13.2 [symbolic = %self.patt.loc13_13.2 (constants.%self.patt.12b)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem2.loc13_22.1: %facet_type.4c8 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc13_22.1 (constants.%impl.elem2)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2.loc13_22.1 [symbolic = %as_type (constants.%as_type.472)]
// CHECK:STDOUT:   %pattern_type.loc13_20: type = pattern_type %as_type [symbolic = %pattern_type.loc13_20 (constants.%pattern_type.007)]
// CHECK:STDOUT:   %c.param_patt.loc13_20.2: @Transitive.WithSelf.TakesC.%pattern_type.loc13_20 (%pattern_type.007) = value_param_pattern [symbolic = %c.param_patt.loc13_20.2 (constants.%c.param_patt)]
// CHECK:STDOUT:   %c.patt.loc13_20.2: @Transitive.WithSelf.TakesC.%pattern_type.loc13_20 (%pattern_type.007) = at_binding_pattern c, %c.param_patt.loc13_20.2 [symbolic = %c.patt.loc13_20.2 (constants.%c.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.WithSelf.TakesC.%Self.as_type.loc13_13.1 (%Self.as_type.7f2), %c.param: @Transitive.WithSelf.TakesC.%as_type (%as_type.472));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf(constants.%Self.801) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.801
// CHECK:STDOUT:   %P.WithSelf.InP.type => constants.%P.WithSelf.InP.type
// CHECK:STDOUT:   %P.WithSelf.InP => constants.%P.WithSelf.InP
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf.InP(constants.%Self.801) {
// CHECK:STDOUT:   %Self => constants.%Self.801
// CHECK:STDOUT:   %Self.as_type.loc3_22.1 => constants.%Self.as_type.b49
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fc0
// CHECK:STDOUT:   %self.param_patt.loc3_22.2 => constants.%self.param_patt.97c
// CHECK:STDOUT:   %self.patt.loc3_22.2 => constants.%self.patt.090
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Self.ca9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.ca9
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.ca9) {
// CHECK:STDOUT:   %Self => constants.%Self.ca9
// CHECK:STDOUT:   %Self.as_type.loc4_22.1 => constants.%Self.as_type.7c9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.684
// CHECK:STDOUT:   %self.param_patt.loc4_22.2 => constants.%self.param_patt.9cf
// CHECK:STDOUT:   %self.patt.loc4_22.2 => constants.%self.patt.f63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%Self.9c1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.9c1
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%Self.9c1) {
// CHECK:STDOUT:   %Self => constants.%Self.9c1
// CHECK:STDOUT:   %Self.as_type.loc5_22.1 => constants.%Self.as_type.85f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ebd
// CHECK:STDOUT:   %self.param_patt.loc5_22.2 => constants.%self.param_patt.659
// CHECK:STDOUT:   %self.patt.loc5_22.2 => constants.%self.patt.da0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%Self.77c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.77c) {
// CHECK:STDOUT:   %Self => constants.%Self.77c
// CHECK:STDOUT:   %Self.as_type.loc12_11.1 => constants.%Self.as_type.7f2
// CHECK:STDOUT:   %pattern_type.loc12_11 => constants.%pattern_type.b5c
// CHECK:STDOUT:   %self.param_patt.loc12_11.2 => constants.%self.param_patt.ae9
// CHECK:STDOUT:   %self.patt.loc12_11.2 => constants.%self.patt.12b
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem0.loc12_20.1 => constants.%impl.elem0
// CHECK:STDOUT:   %as_type => constants.%as_type.d30
// CHECK:STDOUT:   %.loc12_20.1 => constants.%.117
// CHECK:STDOUT:   %pattern_type.loc12_20 => constants.%pattern_type.51a
// CHECK:STDOUT:   %return.param_patt.loc12_20.2 => constants.%return.param_patt.931
// CHECK:STDOUT:   %return.patt.loc12_17.2 => constants.%return.patt.680
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.77c) {
// CHECK:STDOUT:   %Self => constants.%Self.77c
// CHECK:STDOUT:   %Self.as_type.loc13_13.1 => constants.%Self.as_type.7f2
// CHECK:STDOUT:   %pattern_type.loc13_13 => constants.%pattern_type.b5c
// CHECK:STDOUT:   %self.param_patt.loc13_13.2 => constants.%self.param_patt.ae9
// CHECK:STDOUT:   %self.patt.loc13_13.2 => constants.%self.patt.12b
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem2.loc13_22.1 => constants.%impl.elem2
// CHECK:STDOUT:   %as_type => constants.%as_type.472
// CHECK:STDOUT:   %pattern_type.loc13_20 => constants.%pattern_type.007
// CHECK:STDOUT:   %c.param_patt.loc13_20.2 => constants.%c.param_patt
// CHECK:STDOUT:   %c.patt.loc13_20.2 => constants.%c.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- equivalent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.744: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.ab0b8d.1: <witness> = lookup_impl_witness %Self.744, @Transitive [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.be6: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type.4bf: type = fn_type @R.WithSelf.InR, @R.WithSelf(%Self.be6) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.345: %R.WithSelf.InR.type.4bf = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.b02: type = facet_access_type %Self.be6 [symbolic]
// CHECK:STDOUT:   %pattern_type.60e: type = pattern_type %Self.as_type.b02 [symbolic]
// CHECK:STDOUT:   %self.param_patt.640: %pattern_type.60e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.894: %pattern_type.60e = at_binding_pattern self, %self.param_patt.640 [symbolic]
// CHECK:STDOUT:   %facet_type.0e9: type = facet_type <@Destroy & @R where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem2.279493.1: %facet_type.0e9 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element2 [symbolic]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.afd: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.fb9: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Self.afd) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.daf: %Q.WithSelf.InQ.type.fb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.430: type = facet_access_type %Self.afd [symbolic]
// CHECK:STDOUT:   %pattern_type.ade: type = pattern_type %Self.as_type.430 [symbolic]
// CHECK:STDOUT:   %self.param_patt.271: %pattern_type.ade = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b5e: %pattern_type.ade = at_binding_pattern self, %self.param_patt.271 [symbolic]
// CHECK:STDOUT:   %facet_type.1b9: type = facet_type <@Destroy & @Q where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem1.1413d6.1: %facet_type.1b9 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element1 [symbolic]
// CHECK:STDOUT:   %eq.be77bc.1 = observe_equivalent %impl.elem1.1413d6.1, %impl.elem2.279493.1 [symbolic]
// CHECK:STDOUT:   %as_type.b2aa43.1: type = facet_access_type %impl.elem2.279493.1 [symbolic]
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.104: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P.WithSelf(%Self.104) [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.cc3: type = facet_access_type %Self.104 [symbolic]
// CHECK:STDOUT:   %pattern_type.3ef: type = pattern_type %Self.as_type.cc3 [symbolic]
// CHECK:STDOUT:   %self.param_patt.88d: %pattern_type.3ef = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.6de: %pattern_type.3ef = at_binding_pattern self, %self.param_patt.88d [symbolic]
// CHECK:STDOUT:   %facet_type.1ff: type = facet_type <@Destroy & @P> [concrete]
// CHECK:STDOUT:   %impl.elem0.8aa205.1: %facet_type.1ff = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element0 [symbolic]
// CHECK:STDOUT:   %eq.20ba2e.1 = observe_equivalent %impl.elem0.8aa205.1, %impl.elem1.1413d6.1 [symbolic]
// CHECK:STDOUT:   %as_type.a0914b.1: type = facet_access_type %impl.elem0.8aa205.1 [symbolic]
// CHECK:STDOUT:   %as_type.f86e2d.1: type = facet_access_type %impl.elem1.1413d6.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.d86d9f.1: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%Self.744) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.f276cd.1: %Transitive.WithSelf.TakesC.type.d86d9f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.56c: type = facet_access_type %Self.744 [symbolic]
// CHECK:STDOUT:   %pattern_type.68e728.1: type = pattern_type %Self.as_type.56c [symbolic]
// CHECK:STDOUT:   %self.param_patt.e90fff.1: %pattern_type.68e728.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d2dab3.1: %pattern_type.68e728.1 = at_binding_pattern self, %self.param_patt.e90fff.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.d96bbe.1: type = pattern_type %as_type.b2aa43.1 [symbolic]
// CHECK:STDOUT:   %c.param_patt.b7bdaf.1: %pattern_type.d96bbe.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %c.patt.464614.1: %pattern_type.d96bbe.1 = at_binding_pattern c, %c.param_patt.b7bdaf.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.0b3c58.1: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%Self.744) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.e3e09c.1: %Transitive.WithSelf.GetA.type.0b3c58.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.77f685.1: type = pattern_type %as_type.a0914b.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.11f0b9.1: %pattern_type.77f685.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.87a9a6.1: %pattern_type.77f685.1 = return_slot_pattern %return.param_patt.11f0b9.1, invalid [symbolic]
// CHECK:STDOUT:   %.3ef823.1: Core.Form = init_form %as_type.a0914b.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.bd9: type = pattern_type %Transitive.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Transitive.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.68e728.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.68e728.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.68e728.2 = at_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fbd: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.ab0b8d.2: <witness> = lookup_impl_witness %T, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.8aa205.2: %facet_type.1ff = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.1413d6.2: %facet_type.1b9 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.0b3c58.2: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.e3e09c.2: %Transitive.WithSelf.GetA.type.0b3c58.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.d86d9f.2: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.f276cd.2: %Transitive.WithSelf.TakesC.type.d86d9f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %as_type.f86e2d.2: type = facet_access_type %impl.elem1.1413d6.2 [symbolic]
// CHECK:STDOUT:   %as_type.a0914b.2: type = facet_access_type %impl.elem0.8aa205.2 [symbolic]
// CHECK:STDOUT:   %eq.20ba2e.2 = observe_equivalent %impl.elem0.8aa205.2, %impl.elem1.1413d6.2 [symbolic]
// CHECK:STDOUT:   %impl.elem2.279493.2: %facet_type.0e9 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element2 [symbolic]
// CHECK:STDOUT:   %as_type.b2aa43.2: type = facet_access_type %impl.elem2.279493.2 [symbolic]
// CHECK:STDOUT:   %eq.be77bc.2 = observe_equivalent %impl.elem1.1413d6.2, %impl.elem2.279493.2 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.e77: %Transitive.assoc_type = assoc_entity element0, imports.%Main.import_ref.0a7 [concrete]
// CHECK:STDOUT:   %require_complete.6cf: <witness> = require_complete_type %as_type.a0914b.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.77f685.2: type = pattern_type %as_type.a0914b.2 [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.77f685.2 = ref_binding_pattern a [symbolic]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.77f685.2 = var_pattern %a.patt [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, imports.%Main.import_ref.06c [concrete]
// CHECK:STDOUT:   %.657: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type.0b3c58.2, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.657 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element3 [symbolic]
// CHECK:STDOUT:   %self.param_patt.e90fff.2: %pattern_type.68e728.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d2dab3.2: %pattern_type.68e728.2 = at_binding_pattern self, %self.param_patt.e90fff.2 [symbolic]
// CHECK:STDOUT:   %.3ef823.2: Core.Form = init_form %as_type.a0914b.2 [symbolic]
// CHECK:STDOUT:   %return.param_patt.11f0b9.2: %pattern_type.77f685.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.87a9a6.2: %pattern_type.77f685.2 = return_slot_pattern %return.param_patt.11f0b9.2, invalid [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.72f: <specific function> = specific_impl_function %impl.elem3, @Transitive.WithSelf.GetA(%T) [symbolic]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, imports.%Main.import_ref.dae [concrete]
// CHECK:STDOUT:   %require_complete.0ea: <witness> = require_complete_type %as_type.f86e2d.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a: type = pattern_type %as_type.f86e2d.2 [symbolic]
// CHECK:STDOUT:   %b.patt: %pattern_type.86a = ref_binding_pattern b [symbolic]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.86a = var_pattern %b.patt [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.c89: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%impl.elem1.1413d6.2) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.aef: %Q.WithSelf.InQ.type.c89 = struct_value () [symbolic]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.6a1: %Q.assoc_type = assoc_entity element0, imports.%Main.import_ref.636 [concrete]
// CHECK:STDOUT:   %facet_type.aa5: type = facet_type <@Destroy & @Q> [concrete]
// CHECK:STDOUT:   %Q.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.1413d6.2, @Q [symbolic]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.f86e2d.2, (%Q.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.0c5: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Q.facet) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.9e5: %Q.WithSelf.InQ.type.0c5 = struct_value () [symbolic]
// CHECK:STDOUT:   %.786: type = fn_type_with_self_type %Q.WithSelf.InQ.type.0c5, %Q.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.8e3: %.786 = impl_witness_access %Q.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.321: %pattern_type.86a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.7e6: %pattern_type.86a = at_binding_pattern self, %self.param_patt.321 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.957: <specific function> = specific_impl_function %impl.elem0.8e3, @Q.WithSelf.InQ(%Q.facet) [symbolic]
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, imports.%Main.import_ref.034 [concrete]
// CHECK:STDOUT:   %.4ff: type = fn_type_with_self_type %Transitive.WithSelf.TakesC.type.d86d9f.2, %T [symbolic]
// CHECK:STDOUT:   %impl.elem4: %.4ff = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element4 [symbolic]
// CHECK:STDOUT:   %as_type.84f: %facet_type.0e9 = facet_access_type %impl.elem0.8aa205.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.d96bbe.2: type = pattern_type %as_type.b2aa43.2 [symbolic]
// CHECK:STDOUT:   %c.param_patt.b7bdaf.2: %pattern_type.d96bbe.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %c.patt.464614.2: %pattern_type.d96bbe.2 = at_binding_pattern c, %c.param_patt.b7bdaf.2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.183: <specific function> = specific_impl_function %impl.elem4, @Transitive.WithSelf.TakesC(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.6e3: <witness> = require_complete_type %as_type.b2aa43.2 [symbolic]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, imports.%Main.import_ref.56a [concrete]
// CHECK:STDOUT:   %R.WithSelf.InR.type.a87: type = fn_type @R.WithSelf.InR, @R.WithSelf(%impl.elem2.279493.2) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.704: %R.WithSelf.InR.type.a87 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.assoc_type: type = assoc_entity_type @R [concrete]
// CHECK:STDOUT:   %assoc0.e94: %R.assoc_type = assoc_entity element0, imports.%Main.import_ref.8ad [concrete]
// CHECK:STDOUT:   %facet_type.fcc: type = facet_type <@Destroy & @R> [concrete]
// CHECK:STDOUT:   %R.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem2.279493.2, @R [symbolic]
// CHECK:STDOUT:   %R.facet: %R.type = facet_value %as_type.b2aa43.2, (%R.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type.56d: type = fn_type @R.WithSelf.InR, @R.WithSelf(%R.facet) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.9a8: %R.WithSelf.InR.type.56d = struct_value () [symbolic]
// CHECK:STDOUT:   %.f43: type = fn_type_with_self_type %R.WithSelf.InR.type.56d, %R.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.8cc: %.f43 = impl_witness_access %R.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.3a1: %pattern_type.d96bbe.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.9df: %pattern_type.d96bbe.2 = at_binding_pattern self, %self.param_patt.3a1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.543: <specific function> = specific_impl_function %impl.elem0.8cc, @R.WithSelf.InR(%R.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.32f: <witness> = lookup_impl_witness %impl.elem1.1413d6.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.853: %Destroy.type = facet_value %as_type.f86e2d.2, (%Destroy.lookup_impl_witness.32f) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.bb2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.853) [symbolic]
// CHECK:STDOUT:   %.ee0: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.bb2, %Destroy.facet.853 [symbolic]
// CHECK:STDOUT:   %impl.elem0.087: %.ee0 = impl_witness_access %Destroy.lookup_impl_witness.32f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ef3: <specific function> = specific_impl_function %impl.elem0.087, @Destroy.WithSelf.Op(%Destroy.facet.853) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.1c9: <witness> = lookup_impl_witness %impl.elem0.8aa205.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.27e: %Destroy.type = facet_value %as_type.a0914b.2, (%Destroy.lookup_impl_witness.1c9) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.74e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.27e) [symbolic]
// CHECK:STDOUT:   %.569: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.74e, %Destroy.facet.27e [symbolic]
// CHECK:STDOUT:   %impl.elem0.95f: %.569 = impl_witness_access %Destroy.lookup_impl_witness.1c9, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.584: <specific function> = specific_impl_function %impl.elem0.95f, @Destroy.WithSelf.Op(%Destroy.facet.27e) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.P = import_ref Main//transitive, P, unloaded
// CHECK:STDOUT:   %Main.Q = import_ref Main//transitive, Q, unloaded
// CHECK:STDOUT:   %Main.R = import_ref Main//transitive, R, unloaded
// CHECK:STDOUT:   %Main.Transitive: type = import_ref Main//transitive, Transitive, loaded [concrete = constants.%Transitive.type]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.72c: %Transitive.assoc_type = import_ref Main//transitive, loc8_8, loaded [concrete = constants.%assoc0.e77]
// CHECK:STDOUT:   %Main.import_ref.e5e: %Transitive.assoc_type = import_ref Main//transitive, loc9_8, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Main.import_ref.c9c: %Transitive.assoc_type = import_ref Main//transitive, loc10_8, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Main.import_ref.5c5: %Transitive.assoc_type = import_ref Main//transitive, loc12_21, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Main.import_ref.8a0: %Transitive.assoc_type = import_ref Main//transitive, loc13_24, loaded [concrete = constants.%assoc4]
// CHECK:STDOUT:   %Main.A = import_ref Main//transitive, A, unloaded
// CHECK:STDOUT:   %Main.B = import_ref Main//transitive, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//transitive, C, unloaded
// CHECK:STDOUT:   %Main.GetA = import_ref Main//transitive, GetA, unloaded
// CHECK:STDOUT:   %Main.TakesC = import_ref Main//transitive, TakesC, unloaded
// CHECK:STDOUT:   %Main.import_ref.682: %R.assoc_type = import_ref Main//transitive, loc5_27, loaded [concrete = constants.%assoc0.e94]
// CHECK:STDOUT:   %Main.InR = import_ref Main//transitive, InR, unloaded
// CHECK:STDOUT:   %Main.import_ref.f6968b.2: %R.type = import_ref Main//transitive, loc5_13, loaded [symbolic = constants.%Self.be6]
// CHECK:STDOUT:   %Main.import_ref.659 = import_ref Main//transitive, loc5_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.e2a: %Q.assoc_type = import_ref Main//transitive, loc4_27, loaded [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:   %Main.InQ = import_ref Main//transitive, InQ, unloaded
// CHECK:STDOUT:   %Main.import_ref.c0a1f5.2: %Q.type = import_ref Main//transitive, loc4_13, loaded [symbolic = constants.%Self.afd]
// CHECK:STDOUT:   %Main.import_ref.4f3 = import_ref Main//transitive, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.fec = import_ref Main//transitive, loc3_27, unloaded
// CHECK:STDOUT:   %Main.InP = import_ref Main//transitive, InP, unloaded
// CHECK:STDOUT:   %Main.import_ref.aef2fa.2: %P.type = import_ref Main//transitive, loc3_13, loaded [symbolic = constants.%Self.104]
// CHECK:STDOUT:   %Main.import_ref.96f = import_ref Main//transitive, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.df6636.2: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.744]
// CHECK:STDOUT:   %Main.import_ref.df6636.3: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.744]
// CHECK:STDOUT:   %Main.import_ref.0d5 = import_ref Main//transitive, loc7_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.0a7: %facet_type.1ff = import_ref Main//transitive, loc8_8, loaded [concrete = %A]
// CHECK:STDOUT:   %A: %facet_type.1ff = assoc_const_decl @A [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.06c: @Transitive.WithSelf.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type.0b3c58.1) = import_ref Main//transitive, loc12_21, loaded [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA.e3e09c.1)]
// CHECK:STDOUT:   %Main.import_ref.dae: %facet_type.1b9 = import_ref Main//transitive, loc9_8, loaded [concrete = %B]
// CHECK:STDOUT:   %B: %facet_type.1b9 = assoc_const_decl @B [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.636: @Q.WithSelf.%Q.WithSelf.InQ.type (%Q.WithSelf.InQ.type.fb9) = import_ref Main//transitive, loc4_27, loaded [symbolic = @Q.WithSelf.%Q.WithSelf.InQ (constants.%Q.WithSelf.InQ.daf)]
// CHECK:STDOUT:   %Main.import_ref.034: @Transitive.WithSelf.%Transitive.WithSelf.TakesC.type (%Transitive.WithSelf.TakesC.type.d86d9f.1) = import_ref Main//transitive, loc13_24, loaded [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.TakesC (constants.%Transitive.WithSelf.TakesC.f276cd.1)]
// CHECK:STDOUT:   %Main.import_ref.56a: %facet_type.0e9 = import_ref Main//transitive, loc10_8, loaded [concrete = %C]
// CHECK:STDOUT:   %C: %facet_type.0e9 = assoc_const_decl @C [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.8ad: @R.WithSelf.%R.WithSelf.InR.type (%R.WithSelf.InR.type.4bf) = import_ref Main//transitive, loc5_27, loaded [symbolic = @R.WithSelf.%R.WithSelf.InR (constants.%R.WithSelf.InR.345)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .P = imports.%Main.P
// CHECK:STDOUT:     .Q = imports.%Main.Q
// CHECK:STDOUT:     .R = imports.%Main.R
// CHECK:STDOUT:     .Transitive = imports.%Main.Transitive
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_7.1: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc5_23.1: @F.%pattern_type.loc5 (%pattern_type.68e728.2) = value_param_pattern [symbolic = %t.param_patt.loc5_23.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc5_23.1: @F.%pattern_type.loc5 (%pattern_type.68e728.2) = at_binding_pattern t, %t.param_patt.loc5_23.1 [symbolic = %t.patt.loc5_23.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_10: type = splice_block %Transitive.ref [concrete = constants.%Transitive.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Transitive.ref: type = name_ref Transitive, imports.%Main.Transitive [concrete = constants.%Transitive.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_7.2: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.as_type.loc5_25.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_25.1: type = splice_block %.loc5_25.2 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc5: %Transitive.type = name_ref T, %T.loc5_7.2 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc5_25.2: type = facet_access_type %T.ref.loc5 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc5_25.2: type = converted %T.ref.loc5, %T.as_type.loc5_25.2 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.as_type.loc5_25.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.0d5
// CHECK:STDOUT:   .A = imports.%Main.import_ref.72c
// CHECK:STDOUT:   .B = imports.%Main.import_ref.e5e
// CHECK:STDOUT:   .C = imports.%Main.import_ref.c9c
// CHECK:STDOUT:   .GetA = imports.%Main.import_ref.5c5
// CHECK:STDOUT:   .TakesC = imports.%Main.import_ref.8a0
// CHECK:STDOUT:   witness = (imports.%Main.A, imports.%Main.B, imports.%Main.C, imports.%Main.GetA, imports.%Main.TakesC)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT:   @Transitive.observe {
// CHECK:STDOUT:     observe constants.%impl.elem0.8aa205.1 == constants.%impl.elem1.1413d6.1 == constants.%impl.elem2.279493.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.659
// CHECK:STDOUT:   .InR = imports.%Main.import_ref.682
// CHECK:STDOUT:   witness = (imports.%Main.InR)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.4f3
// CHECK:STDOUT:   .InQ = imports.%Main.import_ref.e2a
// CHECK:STDOUT:   witness = (imports.%Main.InQ)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @P [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.96f
// CHECK:STDOUT:   .InP = imports.%Main.import_ref.fec
// CHECK:STDOUT:   witness = (imports.%Main.InP)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(imports.%Main.import_ref.f6968b.2: %R.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.be6)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.b02)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.60e)]
// CHECK:STDOUT:   %self.param_patt: @R.WithSelf.InR.%pattern_type (%pattern_type.60e) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.640)]
// CHECK:STDOUT:   %self.patt: @R.WithSelf.InR.%pattern_type (%pattern_type.60e) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.894)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(imports.%Main.import_ref.c0a1f5.2: %Q.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.afd)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.430)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.ade)]
// CHECK:STDOUT:   %self.param_patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.ade) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.271)]
// CHECK:STDOUT:   %self.patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.ade) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.b5e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.WithSelf.InP(imports.%Main.import_ref.aef2fa.2: %P.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.104)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.cc3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.3ef)]
// CHECK:STDOUT:   %self.param_patt: @P.WithSelf.InP.%pattern_type (%pattern_type.3ef) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.88d)]
// CHECK:STDOUT:   %self.patt: @P.WithSelf.InP.%pattern_type (%pattern_type.3ef) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(imports.%Main.import_ref.df6636.2: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.744)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.68e728.1)]
// CHECK:STDOUT:   %self.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.1 (%pattern_type.68e728.1) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.e90fff.1)]
// CHECK:STDOUT:   %self.patt: @Transitive.WithSelf.TakesC.%pattern_type.1 (%pattern_type.68e728.1) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.d2dab3.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.1)]
// CHECK:STDOUT:   %impl.elem2: %facet_type.0e9 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2.279493.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2 [symbolic = %as_type (constants.%as_type.b2aa43.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.d96bbe.1)]
// CHECK:STDOUT:   %c.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.2 (%pattern_type.d96bbe.1) = value_param_pattern [symbolic = %c.param_patt (constants.%c.param_patt.b7bdaf.1)]
// CHECK:STDOUT:   %c.patt: @Transitive.WithSelf.TakesC.%pattern_type.2 (%pattern_type.d96bbe.1) = at_binding_pattern c, %c.param_patt [symbolic = %c.patt (constants.%c.patt.464614.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(imports.%Main.import_ref.df6636.3: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.744)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.68e728.1)]
// CHECK:STDOUT:   %self.param_patt: @Transitive.WithSelf.GetA.%pattern_type.1 (%pattern_type.68e728.1) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.e90fff.1)]
// CHECK:STDOUT:   %self.patt: @Transitive.WithSelf.GetA.%pattern_type.1 (%pattern_type.68e728.1) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.d2dab3.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.1)]
// CHECK:STDOUT:   %impl.elem0: %facet_type.1ff = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.8aa205.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic = %as_type (constants.%as_type.a0914b.1)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %as_type [symbolic = %.1 (constants.%.3ef823.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.77f685.1)]
// CHECK:STDOUT:   %return.param_patt: @Transitive.WithSelf.GetA.%pattern_type.2 (%pattern_type.77f685.1) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.11f0b9.1)]
// CHECK:STDOUT:   %return.patt: @Transitive.WithSelf.GetA.%pattern_type.2 (%pattern_type.77f685.1) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.87a9a6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_7.2: %Transitive.type) {
// CHECK:STDOUT:   %T.patt.loc5_7.2: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_7.1: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc5_25.1: type = facet_access_type %T.loc5_7.1 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5: type = pattern_type %T.as_type.loc5_25.1 [symbolic = %pattern_type.loc5 (constants.%pattern_type.68e728.2)]
// CHECK:STDOUT:   %t.param_patt.loc5_23.2: @F.%pattern_type.loc5 (%pattern_type.68e728.2) = value_param_pattern [symbolic = %t.param_patt.loc5_23.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc5_23.2: @F.%pattern_type.loc5 (%pattern_type.68e728.2) = at_binding_pattern t, %t.param_patt.loc5_23.2 [symbolic = %t.patt.loc5_23.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5: <witness> = require_complete_type %T.as_type.loc5_25.1 [symbolic = %require_complete.loc5 (constants.%require_complete.fbd)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc5_7.1, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.2)]
// CHECK:STDOUT:   %impl.elem0.loc6_11.2: %facet_type.1ff = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_11.2 (constants.%impl.elem0.8aa205.2)]
// CHECK:STDOUT:   %as_type.loc6_11.2: type = facet_access_type %impl.elem0.loc6_11.2 [symbolic = %as_type.loc6_11.2 (constants.%as_type.a0914b.2)]
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %as_type.loc6_11.2 [symbolic = %require_complete.loc6 (constants.%require_complete.6cf)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %as_type.loc6_11.2 [symbolic = %pattern_type.loc6 (constants.%pattern_type.77f685.2)]
// CHECK:STDOUT:   %a.patt.loc6_8.2: @F.%pattern_type.loc6 (%pattern_type.77f685.2) = ref_binding_pattern a [symbolic = %a.patt.loc6_8.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.var_patt.loc6_3.2: @F.%pattern_type.loc6 (%pattern_type.77f685.2) = var_pattern %a.patt.loc6_8.2 [symbolic = %a.var_patt.loc6_3.2 (constants.%a.var_patt)]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T.loc5_7.1) [symbolic = %Transitive.WithSelf.GetA.type (constants.%Transitive.WithSelf.GetA.type.0b3c58.2)]
// CHECK:STDOUT:   %.loc6_17: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type, %T.loc5_7.1 [symbolic = %.loc6_17 (constants.%.657)]
// CHECK:STDOUT:   %impl.elem3.loc6_17.2: @F.%.loc6_17 (%.657) = impl_witness_access %Transitive.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc6_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_17.2: <specific function> = specific_impl_function %impl.elem3.loc6_17.2, @Transitive.WithSelf.GetA(%T.loc5_7.1) [symbolic = %specific_impl_fn.loc6_17.2 (constants.%specific_impl_fn.72f)]
// CHECK:STDOUT:   %impl.elem1.loc8_11.2: %facet_type.1b9 = impl_witness_access %Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc8_11.2 (constants.%impl.elem1.1413d6.2)]
// CHECK:STDOUT:   %as_type.loc8_11.2: type = facet_access_type %impl.elem1.loc8_11.2 [symbolic = %as_type.loc8_11.2 (constants.%as_type.f86e2d.2)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %as_type.loc8_11.2 [symbolic = %require_complete.loc8 (constants.%require_complete.0ea)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %as_type.loc8_11.2 [symbolic = %pattern_type.loc8 (constants.%pattern_type.86a)]
// CHECK:STDOUT:   %b.patt.loc8_8.2: @F.%pattern_type.loc8 (%pattern_type.86a) = ref_binding_pattern b [symbolic = %b.patt.loc8_8.2 (constants.%b.patt)]
// CHECK:STDOUT:   %b.var_patt.loc8_3.2: @F.%pattern_type.loc8 (%pattern_type.86a) = var_pattern %b.patt.loc8_8.2 [symbolic = %b.var_patt.loc8_3.2 (constants.%b.var_patt)]
// CHECK:STDOUT:   %Q.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.loc8_11.2, @Q [symbolic = %Q.lookup_impl_witness (constants.%Q.lookup_impl_witness)]
// CHECK:STDOUT:   %Q.facet.loc9_4: %Q.type = facet_value %as_type.loc8_11.2, (%Q.lookup_impl_witness) [symbolic = %Q.facet.loc9_4 (constants.%Q.facet)]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Q.facet.loc9_4) [symbolic = %Q.WithSelf.InQ.type (constants.%Q.WithSelf.InQ.type.0c5)]
// CHECK:STDOUT:   %.loc9_4: type = fn_type_with_self_type %Q.WithSelf.InQ.type, %Q.facet.loc9_4 [symbolic = %.loc9_4 (constants.%.786)]
// CHECK:STDOUT:   %impl.elem0.loc9_4.2: @F.%.loc9_4 (%.786) = impl_witness_access %Q.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0.8e3)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_4.2: <specific function> = specific_impl_function %impl.elem0.loc9_4.2, @Q.WithSelf.InQ(%Q.facet.loc9_4) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn.957)]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%T.loc5_7.1) [symbolic = %Transitive.WithSelf.TakesC.type (constants.%Transitive.WithSelf.TakesC.type.d86d9f.2)]
// CHECK:STDOUT:   %.loc11_4: type = fn_type_with_self_type %Transitive.WithSelf.TakesC.type, %T.loc5_7.1 [symbolic = %.loc11_4 (constants.%.4ff)]
// CHECK:STDOUT:   %impl.elem4.loc11_4.2: @F.%.loc11_4 (%.4ff) = impl_witness_access %Transitive.lookup_impl_witness, element4 [symbolic = %impl.elem4.loc11_4.2 (constants.%impl.elem4)]
// CHECK:STDOUT:   %as_type.loc11_13: %facet_type.0e9 = facet_access_type %impl.elem0.loc6_11.2 [symbolic = %as_type.loc11_13 (constants.%as_type.84f)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_4.2: <specific function> = specific_impl_function %impl.elem4.loc11_4.2, @Transitive.WithSelf.TakesC(%T.loc5_7.1) [symbolic = %specific_impl_fn.loc11_4.2 (constants.%specific_impl_fn.183)]
// CHECK:STDOUT:   %impl.elem2.loc11: %facet_type.0e9 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc11 (constants.%impl.elem2.279493.2)]
// CHECK:STDOUT:   %as_type.loc11_12: type = facet_access_type %impl.elem2.loc11 [symbolic = %as_type.loc11_12 (constants.%as_type.b2aa43.2)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %as_type.loc11_12 [symbolic = %require_complete.loc11 (constants.%require_complete.6e3)]
// CHECK:STDOUT:   %R.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem2.loc11, @R [symbolic = %R.lookup_impl_witness (constants.%R.lookup_impl_witness)]
// CHECK:STDOUT:   %R.facet.loc13_13: %R.type = facet_value %as_type.loc11_12, (%R.lookup_impl_witness) [symbolic = %R.facet.loc13_13 (constants.%R.facet)]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R.WithSelf(%R.facet.loc13_13) [symbolic = %R.WithSelf.InR.type (constants.%R.WithSelf.InR.type.56d)]
// CHECK:STDOUT:   %.loc13_13: type = fn_type_with_self_type %R.WithSelf.InR.type, %R.facet.loc13_13 [symbolic = %.loc13_13 (constants.%.f43)]
// CHECK:STDOUT:   %impl.elem0.loc13_13.2: @F.%.loc13_13 (%.f43) = impl_witness_access %R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_13.2 (constants.%impl.elem0.8cc)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_13.2: <specific function> = specific_impl_function %impl.elem0.loc13_13.2, @R.WithSelf.InR(%R.facet.loc13_13) [symbolic = %specific_impl_fn.loc13_13.2 (constants.%specific_impl_fn.543)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %impl.elem1.loc8_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc8 (constants.%Destroy.lookup_impl_witness.32f)]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.3: %Destroy.type = facet_value %as_type.loc8_11.2, (%Destroy.lookup_impl_witness.loc8) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.853)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc8_3.3) [symbolic = %Destroy.WithSelf.Op.type.loc8 (constants.%Destroy.WithSelf.Op.type.bb2)]
// CHECK:STDOUT:   %.loc8_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc8, %Destroy.facet.loc8_3.3 [symbolic = %.loc8_3.4 (constants.%.ee0)]
// CHECK:STDOUT:   %impl.elem0.loc8_3.2: @F.%.loc8_3.4 (%.ee0) = impl_witness_access %Destroy.lookup_impl_witness.loc8, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.087)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_3.2: <specific function> = specific_impl_function %impl.elem0.loc8_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc8_3.3) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.ef3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %impl.elem0.loc6_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc6 (constants.%Destroy.lookup_impl_witness.1c9)]
// CHECK:STDOUT:   %Destroy.facet.loc6_3.3: %Destroy.type = facet_value %as_type.loc6_11.2, (%Destroy.lookup_impl_witness.loc6) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.27e)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc6_3.3) [symbolic = %Destroy.WithSelf.Op.type.loc6 (constants.%Destroy.WithSelf.Op.type.74e)]
// CHECK:STDOUT:   %.loc6_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc6, %Destroy.facet.loc6_3.3 [symbolic = %.loc6_3.4 (constants.%.569)]
// CHECK:STDOUT:   %impl.elem0.loc6_3.2: @F.%.loc6_3.4 (%.569) = impl_witness_access %Destroy.lookup_impl_witness.loc6, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.95f)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_3.2: <specific function> = specific_impl_function %impl.elem0.loc6_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc6_3.3) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.584)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.as_type.loc5_25.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.var: ref @F.%as_type.loc6_11.2 (%as_type.a0914b.2) = var_storage %a.var_patt.loc6_3.1
// CHECK:STDOUT:     %t.ref.loc6: @F.%T.as_type.loc5_25.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %GetA.ref: %Transitive.assoc_type = name_ref GetA, imports.%Main.import_ref.5c5 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc6_17.1: @F.%.loc6_17 (%.657) = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element3 [symbolic = %impl.elem3.loc6_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc6_17: <bound method> = bound_method %t.ref.loc6, %impl.elem3.loc6_17.1
// CHECK:STDOUT:     %.loc6_23.1: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_23.2: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_17.1: <specific function> = specific_impl_function %impl.elem3.loc6_17.1, @Transitive.WithSelf.GetA(constants.%T) [symbolic = %specific_impl_fn.loc6_17.2 (constants.%specific_impl_fn.72f)]
// CHECK:STDOUT:     %bound_method.loc6_23: <bound method> = bound_method %t.ref.loc6, %specific_impl_fn.loc6_17.1
// CHECK:STDOUT:     %.loc6_3.1: ref @F.%as_type.loc6_11.2 (%as_type.a0914b.2) = splice_block %a.var {}
// CHECK:STDOUT:     %Transitive.WithSelf.GetA.call: init @F.%as_type.loc6_11.2 (%as_type.a0914b.2) to %.loc6_3.1 = call %bound_method.loc6_23(%t.ref.loc6)
// CHECK:STDOUT:     assign %a.var, %Transitive.WithSelf.GetA.call
// CHECK:STDOUT:     %.loc6_11.1: type = splice_block %.loc6_11.3 [symbolic = %as_type.loc6_11.2 (constants.%as_type.a0914b.2)] {
// CHECK:STDOUT:       %T.ref.loc6: %Transitive.type = name_ref T, %T.loc5_7.2 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6: type = facet_access_type %T.ref.loc6 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_11.2: type = converted %T.ref.loc6, %T.as_type.loc6 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %A.ref: %Transitive.assoc_type = name_ref A, imports.%Main.import_ref.72c [concrete = constants.%assoc0.e77]
// CHECK:STDOUT:       %impl.elem0.loc6_11.1: %facet_type.1ff = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic = %impl.elem0.loc6_11.2 (constants.%impl.elem0.8aa205.2)]
// CHECK:STDOUT:       %as_type.loc6_11.1: type = facet_access_type %impl.elem0.loc6_11.1 [symbolic = %as_type.loc6_11.2 (constants.%as_type.a0914b.2)]
// CHECK:STDOUT:       %.loc6_11.3: type = converted %impl.elem0.loc6_11.1, %as_type.loc6_11.1 [symbolic = %as_type.loc6_11.2 (constants.%as_type.a0914b.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: ref @F.%as_type.loc6_11.2 (%as_type.a0914b.2) = wrapper_binding a, %a.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc6_8.1: @F.%pattern_type.loc6 (%pattern_type.77f685.2) = ref_binding_pattern a [symbolic = %a.patt.loc6_8.2 (constants.%a.patt)]
// CHECK:STDOUT:       %a.var_patt.loc6_3.1: @F.%pattern_type.loc6 (%pattern_type.77f685.2) = var_pattern %a.patt.loc6_8.1 [symbolic = %a.var_patt.loc6_3.2 (constants.%a.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.var: ref @F.%as_type.loc8_11.2 (%as_type.f86e2d.2) = var_storage %b.var_patt.loc8_3.1
// CHECK:STDOUT:     %a.ref.loc8: ref @F.%as_type.loc6_11.2 (%as_type.a0914b.2) = name_ref a, %a
// CHECK:STDOUT:     %.loc8_3.1: ref @F.%as_type.loc8_11.2 (%as_type.f86e2d.2) = as_compatible %a.ref.loc8
// CHECK:STDOUT:     assign %b.var, %.loc8_3.1
// CHECK:STDOUT:     %.loc8_11.1: type = splice_block %.loc8_11.3 [symbolic = %as_type.loc8_11.2 (constants.%as_type.f86e2d.2)] {
// CHECK:STDOUT:       %T.ref.loc8: %Transitive.type = name_ref T, %T.loc5_7.2 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8: type = facet_access_type %T.ref.loc8 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_11.2: type = converted %T.ref.loc8, %T.as_type.loc8 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %B.ref: %Transitive.assoc_type = name_ref B, imports.%Main.import_ref.e5e [concrete = constants.%assoc1]
// CHECK:STDOUT:       %impl.elem1.loc8_11.1: %facet_type.1b9 = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element1 [symbolic = %impl.elem1.loc8_11.2 (constants.%impl.elem1.1413d6.2)]
// CHECK:STDOUT:       %as_type.loc8_11.1: type = facet_access_type %impl.elem1.loc8_11.1 [symbolic = %as_type.loc8_11.2 (constants.%as_type.f86e2d.2)]
// CHECK:STDOUT:       %.loc8_11.3: type = converted %impl.elem1.loc8_11.1, %as_type.loc8_11.1 [symbolic = %as_type.loc8_11.2 (constants.%as_type.f86e2d.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: ref @F.%as_type.loc8_11.2 (%as_type.f86e2d.2) = wrapper_binding b, %b.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %b.patt.loc8_8.1: @F.%pattern_type.loc8 (%pattern_type.86a) = ref_binding_pattern b [symbolic = %b.patt.loc8_8.2 (constants.%b.patt)]
// CHECK:STDOUT:       %b.var_patt.loc8_3.1: @F.%pattern_type.loc8 (%pattern_type.86a) = var_pattern %b.patt.loc8_8.1 [symbolic = %b.var_patt.loc8_3.2 (constants.%b.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.ref: ref @F.%as_type.loc8_11.2 (%as_type.f86e2d.2) = name_ref b, %b
// CHECK:STDOUT:     %InQ.ref: %Q.assoc_type = name_ref InQ, imports.%Main.import_ref.e2a [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:     %facet_type.loc9: type = facet_type <@Destroy & @Q> [concrete = constants.%facet_type.aa5]
// CHECK:STDOUT:     %impl.elem0.loc9_4.1: @F.%.loc9_4 (%.786) = impl_witness_access constants.%Q.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0.8e3)]
// CHECK:STDOUT:     %bound_method.loc9_4: <bound method> = bound_method %b.ref, %impl.elem0.loc9_4.1
// CHECK:STDOUT:     %Q.facet.loc9_9.1: %Q.type = facet_value constants.%as_type.f86e2d.2, (constants.%Q.lookup_impl_witness) [symbolic = %Q.facet.loc9_4 (constants.%Q.facet)]
// CHECK:STDOUT:     %.loc9_9.1: %Q.type = converted constants.%as_type.f86e2d.2, %Q.facet.loc9_9.1 [symbolic = %Q.facet.loc9_4 (constants.%Q.facet)]
// CHECK:STDOUT:     %Q.facet.loc9_9.2: %Q.type = facet_value constants.%as_type.f86e2d.2, (constants.%Q.lookup_impl_witness) [symbolic = %Q.facet.loc9_4 (constants.%Q.facet)]
// CHECK:STDOUT:     %.loc9_9.2: %Q.type = converted constants.%as_type.f86e2d.2, %Q.facet.loc9_9.2 [symbolic = %Q.facet.loc9_4 (constants.%Q.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_4.1: <specific function> = specific_impl_function %impl.elem0.loc9_4.1, @Q.WithSelf.InQ(constants.%Q.facet) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn.957)]
// CHECK:STDOUT:     %bound_method.loc9_9: <bound method> = bound_method %b.ref, %specific_impl_fn.loc9_4.1
// CHECK:STDOUT:     %.loc9_3: @F.%as_type.loc8_11.2 (%as_type.f86e2d.2) = acquire_value %b.ref
// CHECK:STDOUT:     %Q.WithSelf.InQ.call: init %empty_tuple.type = call %bound_method.loc9_9(%.loc9_3)
// CHECK:STDOUT:     %t.ref.loc11: @F.%T.as_type.loc5_25.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %TakesC.ref: %Transitive.assoc_type = name_ref TakesC, imports.%Main.import_ref.8a0 [concrete = constants.%assoc4]
// CHECK:STDOUT:     %impl.elem4.loc11_4.1: @F.%.loc11_4 (%.4ff) = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element4 [symbolic = %impl.elem4.loc11_4.2 (constants.%impl.elem4)]
// CHECK:STDOUT:     %bound_method.loc11_4: <bound method> = bound_method %t.ref.loc11, %impl.elem4.loc11_4.1
// CHECK:STDOUT:     %a.ref.loc11: ref @F.%as_type.loc6_11.2 (%as_type.a0914b.2) = name_ref a, %a
// CHECK:STDOUT:     %.loc11_13.1: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc11_13.2: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc11_13.3: %facet_type.0e9 = as_compatible constants.%as_type.a0914b.2 [symbolic = %as_type.loc11_13 (constants.%as_type.84f)]
// CHECK:STDOUT:     %.loc11_13.4: %facet_type.0e9 = as_compatible constants.%as_type.a0914b.2 [symbolic = %as_type.loc11_13 (constants.%as_type.84f)]
// CHECK:STDOUT:     %specific_impl_fn.loc11_4.1: <specific function> = specific_impl_function %impl.elem4.loc11_4.1, @Transitive.WithSelf.TakesC(constants.%T) [symbolic = %specific_impl_fn.loc11_4.2 (constants.%specific_impl_fn.183)]
// CHECK:STDOUT:     %bound_method.loc11_13: <bound method> = bound_method %t.ref.loc11, %specific_impl_fn.loc11_4.1
// CHECK:STDOUT:     %.loc11_12: ref @F.%as_type.loc11_12 (%as_type.b2aa43.2) = as_compatible %a.ref.loc11
// CHECK:STDOUT:     %Transitive.WithSelf.TakesC.call: init %empty_tuple.type = call %bound_method.loc11_13(%t.ref.loc11, %.loc11_12)
// CHECK:STDOUT:     %a.ref.loc13: ref @F.%as_type.loc6_11.2 (%as_type.a0914b.2) = name_ref a, %a
// CHECK:STDOUT:     %T.ref.loc13: %Transitive.type = name_ref T, %T.loc5_7.2 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc13: type = facet_access_type %T.ref.loc13 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc13_10.1: type = converted %T.ref.loc13, %T.as_type.loc13 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %C.ref: %Transitive.assoc_type = name_ref C, imports.%Main.import_ref.c9c [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc13: %facet_type.0e9 = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element2 [symbolic = %impl.elem2.loc11 (constants.%impl.elem2.279493.2)]
// CHECK:STDOUT:     %as_type.loc13: type = facet_access_type %impl.elem2.loc13 [symbolic = %as_type.loc11_12 (constants.%as_type.b2aa43.2)]
// CHECK:STDOUT:     %.loc13_10.2: type = converted %impl.elem2.loc13, %as_type.loc13 [symbolic = %as_type.loc11_12 (constants.%as_type.b2aa43.2)]
// CHECK:STDOUT:     %.loc13_6.1: ref @F.%as_type.loc11_12 (%as_type.b2aa43.2) = as_compatible %a.ref.loc13
// CHECK:STDOUT:     %InR.ref: %R.assoc_type = name_ref InR, imports.%Main.import_ref.682 [concrete = constants.%assoc0.e94]
// CHECK:STDOUT:     %facet_type.loc13: type = facet_type <@Destroy & @R> [concrete = constants.%facet_type.fcc]
// CHECK:STDOUT:     %impl.elem0.loc13_13.1: @F.%.loc13_13 (%.f43) = impl_witness_access constants.%R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_13.2 (constants.%impl.elem0.8cc)]
// CHECK:STDOUT:     %bound_method.loc13_13: <bound method> = bound_method %.loc13_6.1, %impl.elem0.loc13_13.1
// CHECK:STDOUT:     %R.facet.loc13_18.1: %R.type = facet_value constants.%as_type.b2aa43.2, (constants.%R.lookup_impl_witness) [symbolic = %R.facet.loc13_13 (constants.%R.facet)]
// CHECK:STDOUT:     %.loc13_18.1: %R.type = converted constants.%as_type.b2aa43.2, %R.facet.loc13_18.1 [symbolic = %R.facet.loc13_13 (constants.%R.facet)]
// CHECK:STDOUT:     %R.facet.loc13_18.2: %R.type = facet_value constants.%as_type.b2aa43.2, (constants.%R.lookup_impl_witness) [symbolic = %R.facet.loc13_13 (constants.%R.facet)]
// CHECK:STDOUT:     %.loc13_18.2: %R.type = converted constants.%as_type.b2aa43.2, %R.facet.loc13_18.2 [symbolic = %R.facet.loc13_13 (constants.%R.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc13_13.1: <specific function> = specific_impl_function %impl.elem0.loc13_13.1, @R.WithSelf.InR(constants.%R.facet) [symbolic = %specific_impl_fn.loc13_13.2 (constants.%specific_impl_fn.543)]
// CHECK:STDOUT:     %bound_method.loc13_18: <bound method> = bound_method %.loc13_6.1, %specific_impl_fn.loc13_13.1
// CHECK:STDOUT:     %.loc13_6.2: @F.%as_type.loc11_12 (%as_type.b2aa43.2) = acquire_value %.loc13_6.1
// CHECK:STDOUT:     %R.WithSelf.InR.call: init %empty_tuple.type = call %bound_method.loc13_18(%.loc13_6.2)
// CHECK:STDOUT:     %impl.elem0.loc8_3.1: @F.%.loc8_3.4 (%.ee0) = impl_witness_access constants.%Destroy.lookup_impl_witness.32f, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.087)]
// CHECK:STDOUT:     %bound_method.loc8_3.1: <bound method> = bound_method %b.var, %impl.elem0.loc8_3.1
// CHECK:STDOUT:     %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%as_type.f86e2d.2, (constants.%Destroy.lookup_impl_witness.32f) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.853)]
// CHECK:STDOUT:     %.loc8_3.2: %Destroy.type = converted constants.%as_type.f86e2d.2, %Destroy.facet.loc8_3.1 [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.853)]
// CHECK:STDOUT:     %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%as_type.f86e2d.2, (constants.%Destroy.lookup_impl_witness.32f) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.853)]
// CHECK:STDOUT:     %.loc8_3.3: %Destroy.type = converted constants.%as_type.f86e2d.2, %Destroy.facet.loc8_3.2 [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet.853)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_3.1: <specific function> = specific_impl_function %impl.elem0.loc8_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.853) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.ef3)]
// CHECK:STDOUT:     %bound_method.loc8_3.2: <bound method> = bound_method %b.var, %specific_impl_fn.loc8_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8_3.2(%b.var)
// CHECK:STDOUT:     %impl.elem0.loc6_3.1: @F.%.loc6_3.4 (%.569) = impl_witness_access constants.%Destroy.lookup_impl_witness.1c9, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.95f)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc6_3.1
// CHECK:STDOUT:     %Destroy.facet.loc6_3.1: %Destroy.type = facet_value constants.%as_type.a0914b.2, (constants.%Destroy.lookup_impl_witness.1c9) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.27e)]
// CHECK:STDOUT:     %.loc6_3.2: %Destroy.type = converted constants.%as_type.a0914b.2, %Destroy.facet.loc6_3.1 [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.27e)]
// CHECK:STDOUT:     %Destroy.facet.loc6_3.2: %Destroy.type = facet_value constants.%as_type.a0914b.2, (constants.%Destroy.lookup_impl_witness.1c9) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.27e)]
// CHECK:STDOUT:     %.loc6_3.3: %Destroy.type = converted constants.%as_type.a0914b.2, %Destroy.facet.loc6_3.2 [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet.27e)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_3.1: <specific function> = specific_impl_function %impl.elem0.loc6_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.27e) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.584)]
// CHECK:STDOUT:     %bound_method.loc6_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc6_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc6: init %empty_tuple.type = call %bound_method.loc6_3.2(%a.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%Self.be6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.be6
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type.4bf
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR.345
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%Self.be6) {
// CHECK:STDOUT:   %Self => constants.%Self.be6
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.b02
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.60e
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.640
// CHECK:STDOUT:   %self.patt => constants.%self.patt.894
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Self.afd) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.afd
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.fb9
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.daf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.afd) {
// CHECK:STDOUT:   %Self => constants.%Self.afd
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.430
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ade
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.271
// CHECK:STDOUT:   %self.patt => constants.%self.patt.b5e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf(constants.%Self.104) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.104
// CHECK:STDOUT:   %P.WithSelf.InP.type => constants.%P.WithSelf.InP.type
// CHECK:STDOUT:   %P.WithSelf.InP => constants.%P.WithSelf.InP
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf.InP(constants.%Self.104) {
// CHECK:STDOUT:   %Self => constants.%Self.104
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.cc3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3ef
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.88d
// CHECK:STDOUT:   %self.patt => constants.%self.patt.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%Self.744) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8aa205.1
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.1413d6.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.0b3c58.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.e3e09c.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.d86d9f.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.f276cd.1
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.f86e2d.1
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.a0914b.1
// CHECK:STDOUT:   %eq.1 => constants.%eq.20ba2e.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.279493.1
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.b2aa43.1
// CHECK:STDOUT:   %eq.2 => constants.%eq.be77bc.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.744) {
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.1
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.279493.1
// CHECK:STDOUT:   %as_type => constants.%as_type.b2aa43.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.d96bbe.1
// CHECK:STDOUT:   %c.param_patt => constants.%c.param_patt.b7bdaf.1
// CHECK:STDOUT:   %c.patt => constants.%c.patt.464614.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.744) {
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.1
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8aa205.1
// CHECK:STDOUT:   %as_type => constants.%as_type.a0914b.1
// CHECK:STDOUT:   %.1 => constants.%.3ef823.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.77f685.1
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.11f0b9.1
// CHECK:STDOUT:   %return.patt => constants.%return.patt.87a9a6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc5_25.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc5 => constants.%pattern_type.68e728.2
// CHECK:STDOUT:   %t.param_patt.loc5_23.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc5_23.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8aa205.2
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.1413d6.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.0b3c58.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.e3e09c.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.d86d9f.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.f276cd.2
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.f86e2d.2
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.a0914b.2
// CHECK:STDOUT:   %eq.1 => constants.%eq.20ba2e.2
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.279493.2
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.b2aa43.2
// CHECK:STDOUT:   %eq.2 => constants.%eq.be77bc.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.2
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.2
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.2
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8aa205.2
// CHECK:STDOUT:   %as_type => constants.%as_type.a0914b.2
// CHECK:STDOUT:   %.1 => constants.%.3ef823.2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.77f685.2
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.11f0b9.2
// CHECK:STDOUT:   %return.patt => constants.%return.patt.87a9a6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%impl.elem1.1413d6.2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%impl.elem1.1413d6.2
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.c89
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.aef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Q.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Q.facet
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.0c5
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.9e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Q.facet) {
// CHECK:STDOUT:   %Self => constants.%Q.facet
// CHECK:STDOUT:   %Self.as_type => constants.%as_type.f86e2d.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.86a
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.321
// CHECK:STDOUT:   %self.patt => constants.%self.patt.7e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.2
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.2
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.2
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.2
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.279493.2
// CHECK:STDOUT:   %as_type => constants.%as_type.b2aa43.2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.d96bbe.2
// CHECK:STDOUT:   %c.param_patt => constants.%c.param_patt.b7bdaf.2
// CHECK:STDOUT:   %c.patt => constants.%c.patt.464614.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%impl.elem2.279493.2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%impl.elem2.279493.2
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type.a87
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR.704
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%R.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%R.facet
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type.56d
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR.9a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%R.facet) {
// CHECK:STDOUT:   %Self => constants.%R.facet
// CHECK:STDOUT:   %Self.as_type => constants.%as_type.b2aa43.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d96bbe.2
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.3a1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.9df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.d0c: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.b45: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.e8c: type = facet_access_type %Self.b45 [symbolic]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.62a: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.187: type = facet_access_type %Self.62a [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt.a4b: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.626: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.3b8: type = facet_access_type %T.626 [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %T.as_type.3b8 [symbolic]
// CHECK:STDOUT:   %x.param_patt.241: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.19b: %pattern_type.a0e = at_binding_pattern x, %x.param_patt.241 [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.94f: <witness> = require_complete_type %T.as_type.3b8 [symbolic]
// CHECK:STDOUT:   %pattern_type.647: type = pattern_type %C.type [concrete]
// CHECK:STDOUT:   %T.patt.ece: %pattern_type.647 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.659: %C.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.c69: type = facet_access_type %T.659 [symbolic]
// CHECK:STDOUT:   %pattern_type.df0: type = pattern_type %T.as_type.c69 [symbolic]
// CHECK:STDOUT:   %x.param_patt.948: %pattern_type.df0 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.f9a: %pattern_type.df0 = at_binding_pattern x, %x.param_patt.948 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.8a2: <witness> = require_complete_type %T.as_type.c69 [symbolic]
// CHECK:STDOUT:   %impls = observe_impls %T.659, %B.type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .RequiresA = %RequiresA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt.loc7_15.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt.a4b)]
// CHECK:STDOUT:     %x.param_patt.loc7_29.1: @RequiresA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_29.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:     %x.patt.loc7_29.1: @RequiresA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_29.1 [symbolic = %x.patt.loc7_29.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T.626)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.as_type.loc7_31.1 (%T.as_type.3b8) = value_param call_param0
// CHECK:STDOUT:     %.loc7_31.1: type = splice_block %.loc7_31.2 [symbolic = %T.as_type.loc7_31.1 (constants.%T.as_type.3b8)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_15.2 [symbolic = %T.loc7_15.1 (constants.%T.626)]
// CHECK:STDOUT:       %T.as_type.loc7_31.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_31.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:       %.loc7_31.2: type = converted %T.ref, %T.as_type.loc7_31.2 [symbolic = %T.as_type.loc7_31.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.as_type.loc7_31.1 (%T.as_type.3b8) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.647 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt.ece)]
// CHECK:STDOUT:     %x.param_patt.loc8_14.1: @F.%pattern_type (%pattern_type.df0) = value_param_pattern [symbolic = %x.param_patt.loc8_14.2 (constants.%x.param_patt.948)]
// CHECK:STDOUT:     %x.patt.loc8_14.1: @F.%pattern_type (%pattern_type.df0) = at_binding_pattern x, %x.param_patt.loc8_14.1 [symbolic = %x.patt.loc8_14.2 (constants.%x.patt.f9a)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %C.ref [concrete = constants.%C.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %C.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T.659)]
// CHECK:STDOUT:     %x.param: @F.%T.as_type.loc8_16.1 (%T.as_type.c69) = value_param call_param0
// CHECK:STDOUT:     %.loc8_16.1: type = splice_block %.loc8_16.2 [symbolic = %T.as_type.loc8_16.1 (constants.%T.as_type.c69)] {
// CHECK:STDOUT:       %T.ref.loc8: %C.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T.659)]
// CHECK:STDOUT:       %T.as_type.loc8_16.2: type = facet_access_type %T.ref.loc8 [symbolic = %T.as_type.loc8_16.1 (constants.%T.as_type.c69)]
// CHECK:STDOUT:       %.loc8_16.2: type = converted %T.ref.loc8, %T.as_type.loc8_16.2 [symbolic = %T.as_type.loc8_16.1 (constants.%T.as_type.c69)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.as_type.loc8_16.1 (%T.as_type.c69) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0c]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b45]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %B.WithSelf.Self.as_type.impls.A.type.require.decl = require_decl @B.WithSelf.Self.as_type.impls.A.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc4_23.1 impls %A.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc4_23.1: type = facet_access_type @B.%Self [symbolic = %Self.as_type.loc4_23.2 (constants.%Self.as_type.e8c)]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @B.WithSelf.Self.as_type.impls.A.type.require {
// CHECK:STDOUT:     require @B.WithSelf.Self.as_type.impls.A.type.require.%Self.as_type.loc4_23.1 impls @B.WithSelf.Self.as_type.impls.A.type.require.%A.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.62a]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %C.WithSelf.Self.as_type.impls.B.type.require.decl = require_decl @C.WithSelf.Self.as_type.impls.B.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type.loc5_23.1 impls %B.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc5_23.1: type = facet_access_type @C.%Self [symbolic = %Self.as_type.loc5_23.2 (constants.%Self.as_type.187)]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @C.WithSelf.Self.as_type.impls.B.type.require {
// CHECK:STDOUT:     require @C.WithSelf.Self.as_type.impls.B.type.require.%Self.as_type.loc5_23.1 impls @C.WithSelf.Self.as_type.impls.B.type.require.%B.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @B.WithSelf.Self.as_type.impls.A.type.require(@B.%Self: %B.type) {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.b45)]
// CHECK:STDOUT:   %Self.as_type.loc4_23.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_23.2 (constants.%Self.as_type.e8c)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @C.WithSelf.Self.as_type.impls.B.type.require(@C.%Self: %C.type) {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.62a)]
// CHECK:STDOUT:   %Self.as_type.loc5_23.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_23.2 (constants.%Self.as_type.187)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc7_15.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt.a4b)]
// CHECK:STDOUT:   %T.loc7_15.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T.626)]
// CHECK:STDOUT:   %T.as_type.loc7_31.1: type = facet_access_type %T.loc7_15.1 [symbolic = %T.as_type.loc7_31.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_31.1 [symbolic = %pattern_type (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %x.param_patt.loc7_29.2: @RequiresA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_29.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:   %x.patt.loc7_29.2: @RequiresA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_29.2 [symbolic = %x.patt.loc7_29.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_31.1 [symbolic = %require_complete (constants.%require_complete.94f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.as_type.loc7_31.1 (%T.as_type.3b8)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %C.type) {
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.647 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt.ece)]
// CHECK:STDOUT:   %T.loc8_7.1: %C.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T.659)]
// CHECK:STDOUT:   %T.as_type.loc8_16.1: type = facet_access_type %T.loc8_7.1 [symbolic = %T.as_type.loc8_16.1 (constants.%T.as_type.c69)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc8_16.1 [symbolic = %pattern_type (constants.%pattern_type.df0)]
// CHECK:STDOUT:   %x.param_patt.loc8_14.2: @F.%pattern_type (%pattern_type.df0) = value_param_pattern [symbolic = %x.param_patt.loc8_14.2 (constants.%x.param_patt.948)]
// CHECK:STDOUT:   %x.patt.loc8_14.2: @F.%pattern_type (%pattern_type.df0) = at_binding_pattern x, %x.param_patt.loc8_14.2 [symbolic = %x.patt.loc8_14.2 (constants.%x.patt.f9a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc8_16.1 [symbolic = %require_complete (constants.%require_complete.8a2)]
// CHECK:STDOUT:   %impls = observe_impls %T.loc8_7.1, constants.%B.type [symbolic = %impls (constants.%impls)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.as_type.loc8_16.1 (%T.as_type.c69)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc9: %C.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T.659)]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type %T.ref.loc9 [symbolic = %T.as_type.loc8_16.1 (constants.%T.as_type.c69)]
// CHECK:STDOUT:     %.loc9: type = converted %T.ref.loc9, %T.as_type.loc9 [symbolic = %T.as_type.loc8_16.1 (constants.%T.as_type.c69)]
// CHECK:STDOUT:     %F.observe.decl = observe_decl @F.observe [concrete]
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @F.%T.as_type.loc8_16.1 (%T.as_type.c69) = name_ref x, %x
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @F.observe {
// CHECK:STDOUT:       observe constants.%T.659 impls constants.%B.type
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.d0c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.b45) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf.Self.as_type.impls.A.type.require(constants.%Self.b45) {
// CHECK:STDOUT:   %Self => constants.%Self.b45
// CHECK:STDOUT:   %Self.as_type.loc4_23.2 => constants.%Self.as_type.e8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.62a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf.Self.as_type.impls.B.type.require(constants.%Self.62a) {
// CHECK:STDOUT:   %Self => constants.%Self.62a
// CHECK:STDOUT:   %Self.as_type.loc5_23.2 => constants.%Self.as_type.187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T.626) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt.a4b
// CHECK:STDOUT:   %T.loc7_15.1 => constants.%T.626
// CHECK:STDOUT:   %T.as_type.loc7_31.1 => constants.%T.as_type.3b8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc7_29.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc7_29.2 => constants.%x.patt.19b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.659) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt.ece
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T.659
// CHECK:STDOUT:   %T.as_type.loc8_16.1 => constants.%T.as_type.c69
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.df0
// CHECK:STDOUT:   %x.param_patt.loc8_14.2 => constants.%x.param_patt.948
// CHECK:STDOUT:   %x.patt.loc8_14.2 => constants.%x.patt.f9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_equivalent_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.104: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P.WithSelf(%Self.104) [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.cc3: type = facet_access_type %Self.104 [symbolic]
// CHECK:STDOUT:   %pattern_type.3ef: type = pattern_type %Self.as_type.cc3 [symbolic]
// CHECK:STDOUT:   %self.param_patt.88d: %pattern_type.3ef = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.6de: %pattern_type.3ef = at_binding_pattern self, %self.param_patt.88d [symbolic]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.afd: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Self.afd) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ: %Q.WithSelf.InQ.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.430: type = facet_access_type %Self.afd [symbolic]
// CHECK:STDOUT:   %pattern_type.ade: type = pattern_type %Self.as_type.430 [symbolic]
// CHECK:STDOUT:   %self.param_patt.271: %pattern_type.ade = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b5e: %pattern_type.ade = at_binding_pattern self, %self.param_patt.271 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.802: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.802 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.38f: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d56: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.38f, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.dd7: <bound method> = bound_method %P.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.bc0: type = facet_type <@P & @Q> [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.be6: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R.WithSelf(%Self.be6) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR: %R.WithSelf.InR.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.b02: type = facet_access_type %Self.be6 [symbolic]
// CHECK:STDOUT:   %pattern_type.60e: type = pattern_type %Self.as_type.b02 [symbolic]
// CHECK:STDOUT:   %self.param_patt.640: %pattern_type.60e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.894: %pattern_type.60e = at_binding_pattern self, %self.param_patt.640 [symbolic]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.588: <bound method> = bound_method %facet_type.bc0, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.0b6: type = facet_type <@P & @Q & @R> [concrete]
// CHECK:STDOUT:   %pattern_type.075: type = pattern_type %facet_type.0b6 [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.075 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: %facet_type.0b6 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %pattern_type.54d: type = pattern_type %U.as_type [symbolic]
// CHECK:STDOUT:   %u.param_patt: %pattern_type.54d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt: %pattern_type.54d = at_binding_pattern u, %u.param_patt [symbolic]
// CHECK:STDOUT:   %TakesPQR.type: type = fn_type @TakesPQR [concrete]
// CHECK:STDOUT:   %TakesPQR: %TakesPQR.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.192: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.744: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.ab0b8d.1: <witness> = lookup_impl_witness %Self.744, @Transitive [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type.4c8: type = facet_type <@R & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem2.731f97.1: %facet_type.4c8 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element2 [symbolic]
// CHECK:STDOUT:   %facet_type.130: type = facet_type <@Q & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem1.6be46c.1: %facet_type.130 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element1 [symbolic]
// CHECK:STDOUT:   %eq.033b19.1 = observe_equivalent %impl.elem1.6be46c.1, %impl.elem2.731f97.1 [symbolic]
// CHECK:STDOUT:   %as_type.2fadd6.1: type = facet_access_type %impl.elem2.731f97.1 [symbolic]
// CHECK:STDOUT:   %facet_type.b1b: type = facet_type <@P & @Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem0.0439e7.1: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element0 [symbolic]
// CHECK:STDOUT:   %eq.4be6e4.1 = observe_equivalent %impl.elem0.0439e7.1, %impl.elem1.6be46c.1 [symbolic]
// CHECK:STDOUT:   %as_type.6a024f.1: type = facet_access_type %impl.elem0.0439e7.1 [symbolic]
// CHECK:STDOUT:   %as_type.dca436.1: type = facet_access_type %impl.elem1.6be46c.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.d86d9f.1: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%Self.744) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.f276cd.1: %Transitive.WithSelf.TakesC.type.d86d9f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.56c: type = facet_access_type %Self.744 [symbolic]
// CHECK:STDOUT:   %pattern_type.68e728.1: type = pattern_type %Self.as_type.56c [symbolic]
// CHECK:STDOUT:   %self.param_patt.e90fff.1: %pattern_type.68e728.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d2dab3.1: %pattern_type.68e728.1 = at_binding_pattern self, %self.param_patt.e90fff.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.ac0: type = pattern_type %as_type.2fadd6.1 [symbolic]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.ac0 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %c.patt: %pattern_type.ac0 = at_binding_pattern c, %c.param_patt [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.0b3c58.1: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%Self.744) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.e3e09c.1: %Transitive.WithSelf.GetA.type.0b3c58.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.c73ac0.1: type = pattern_type %as_type.6a024f.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.057e89.1: %pattern_type.c73ac0.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.4e0711.1: %pattern_type.c73ac0.1 = return_slot_pattern %return.param_patt.057e89.1, invalid [symbolic]
// CHECK:STDOUT:   %.2b1a45.1: Core.Form = init_form %as_type.6a024f.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.bd9: type = pattern_type %Transitive.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Transitive.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.68e728.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.68e728.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.68e728.2 = at_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fbd: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.ab0b8d.2: <witness> = lookup_impl_witness %T, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.0439e7.2: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.6be46c.2: %facet_type.130 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.0b3c58.2: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.e3e09c.2: %Transitive.WithSelf.GetA.type.0b3c58.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.d86d9f.2: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.f276cd.2: %Transitive.WithSelf.TakesC.type.d86d9f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %as_type.dca436.2: type = facet_access_type %impl.elem1.6be46c.2 [symbolic]
// CHECK:STDOUT:   %as_type.6a024f.2: type = facet_access_type %impl.elem0.0439e7.2 [symbolic]
// CHECK:STDOUT:   %eq.4be6e4.2 = observe_equivalent %impl.elem0.0439e7.2, %impl.elem1.6be46c.2 [symbolic]
// CHECK:STDOUT:   %impl.elem2.731f97.2: %facet_type.4c8 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element2 [symbolic]
// CHECK:STDOUT:   %as_type.2fadd6.2: type = facet_access_type %impl.elem2.731f97.2 [symbolic]
// CHECK:STDOUT:   %eq.033b19.2 = observe_equivalent %impl.elem1.6be46c.2, %impl.elem2.731f97.2 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.7cd: %Transitive.assoc_type = assoc_entity element0, imports.%Main.import_ref.d38 [concrete]
// CHECK:STDOUT:   %require_complete.c2d: <witness> = require_complete_type %as_type.6a024f.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.c73ac0.2: type = pattern_type %as_type.6a024f.2 [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.c73ac0.2 = ref_binding_pattern a [symbolic]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.c73ac0.2 = var_pattern %a.patt [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, imports.%Main.import_ref.06c [concrete]
// CHECK:STDOUT:   %.657: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type.0b3c58.2, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.657 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element3 [symbolic]
// CHECK:STDOUT:   %self.param_patt.e90fff.2: %pattern_type.68e728.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d2dab3.2: %pattern_type.68e728.2 = at_binding_pattern self, %self.param_patt.e90fff.2 [symbolic]
// CHECK:STDOUT:   %.2b1a45.2: Core.Form = init_form %as_type.6a024f.2 [symbolic]
// CHECK:STDOUT:   %return.param_patt.057e89.2: %pattern_type.c73ac0.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.4e0711.2: %pattern_type.c73ac0.2 = return_slot_pattern %return.param_patt.057e89.2, invalid [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.72f: <specific function> = specific_impl_function %impl.elem3, @Transitive.WithSelf.GetA(%T) [symbolic]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, imports.%Main.import_ref.9f3 [concrete]
// CHECK:STDOUT:   %impls.885 = observe_impls %impl.elem1.6be46c.2, %Q.type [symbolic]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.6a1: %Q.assoc_type = assoc_entity element0, imports.%Main.import_ref.636 [concrete]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, imports.%Main.import_ref.a5f [concrete]
// CHECK:STDOUT:   %eq.fcd = observe_equivalent %impl.elem0.0439e7.2, %impl.elem2.731f97.2 [symbolic]
// CHECK:STDOUT:   %impls.adc = observe_impls %impl.elem2.731f97.2, %R.type [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.0439e7.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.6a024f.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5e1: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.e75: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5e1, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.678: %.e75 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e8e: <specific function> = specific_impl_function %impl.elem0.678, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.P: type = import_ref Main//transitive, P, loaded [concrete = constants.%P.type]
// CHECK:STDOUT:   %Main.Q: type = import_ref Main//transitive, Q, loaded [concrete = constants.%Q.type]
// CHECK:STDOUT:   %Main.R: type = import_ref Main//transitive, R, loaded [concrete = constants.%R.type]
// CHECK:STDOUT:   %Main.Transitive: type = import_ref Main//transitive, Transitive, loaded [concrete = constants.%Transitive.type]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.fec = import_ref Main//transitive, loc3_27, unloaded
// CHECK:STDOUT:   %Main.InP = import_ref Main//transitive, InP, unloaded
// CHECK:STDOUT:   %Main.import_ref.aef2fa.2: %P.type = import_ref Main//transitive, loc3_13, loaded [symbolic = constants.%Self.104]
// CHECK:STDOUT:   %Main.import_ref.96f = import_ref Main//transitive, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.e2a: %Q.assoc_type = import_ref Main//transitive, loc4_27, loaded [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:   %Main.InQ = import_ref Main//transitive, InQ, unloaded
// CHECK:STDOUT:   %Main.import_ref.c0a1f5.2: %Q.type = import_ref Main//transitive, loc4_13, loaded [symbolic = constants.%Self.afd]
// CHECK:STDOUT:   %Main.import_ref.4f3 = import_ref Main//transitive, loc4_13, unloaded
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.0ce = import_ref Main//transitive, loc5_27, unloaded
// CHECK:STDOUT:   %Main.InR = import_ref Main//transitive, InR, unloaded
// CHECK:STDOUT:   %Main.import_ref.f6968b.2: %R.type = import_ref Main//transitive, loc5_13, loaded [symbolic = constants.%Self.be6]
// CHECK:STDOUT:   %Main.import_ref.659 = import_ref Main//transitive, loc5_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.72c: %Transitive.assoc_type = import_ref Main//transitive, loc8_8, loaded [concrete = constants.%assoc0.7cd]
// CHECK:STDOUT:   %Main.import_ref.e5e: %Transitive.assoc_type = import_ref Main//transitive, loc9_8, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Main.import_ref.c9c: %Transitive.assoc_type = import_ref Main//transitive, loc10_8, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Main.import_ref.5c5: %Transitive.assoc_type = import_ref Main//transitive, loc12_21, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Main.import_ref.ae2 = import_ref Main//transitive, loc13_24, unloaded
// CHECK:STDOUT:   %Main.A = import_ref Main//transitive, A, unloaded
// CHECK:STDOUT:   %Main.B = import_ref Main//transitive, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//transitive, C, unloaded
// CHECK:STDOUT:   %Main.GetA = import_ref Main//transitive, GetA, unloaded
// CHECK:STDOUT:   %Main.TakesC = import_ref Main//transitive, TakesC, unloaded
// CHECK:STDOUT:   %Main.import_ref.df6636.2: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.744]
// CHECK:STDOUT:   %Main.import_ref.df6636.3: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.744]
// CHECK:STDOUT:   %Main.import_ref.0d5 = import_ref Main//transitive, loc7_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.d38: %facet_type.b1b = import_ref Main//transitive, loc8_8, loaded [concrete = %A]
// CHECK:STDOUT:   %A: %facet_type.b1b = assoc_const_decl @A [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.06c: @Transitive.WithSelf.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type.0b3c58.1) = import_ref Main//transitive, loc12_21, loaded [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA.e3e09c.1)]
// CHECK:STDOUT:   %Main.import_ref.9f3: %facet_type.130 = import_ref Main//transitive, loc9_8, loaded [concrete = %B]
// CHECK:STDOUT:   %B: %facet_type.130 = assoc_const_decl @B [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.636: @Q.WithSelf.%Q.WithSelf.InQ.type (%Q.WithSelf.InQ.type) = import_ref Main//transitive, loc4_27, loaded [symbolic = @Q.WithSelf.%Q.WithSelf.InQ (constants.%Q.WithSelf.InQ)]
// CHECK:STDOUT:   %Main.import_ref.a5f: %facet_type.4c8 = import_ref Main//transitive, loc10_8, loaded [concrete = %C]
// CHECK:STDOUT:   %C: %facet_type.4c8 = assoc_const_decl @C [concrete] {}
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .P = imports.%Main.P
// CHECK:STDOUT:     .Q = imports.%Main.Q
// CHECK:STDOUT:     .R = imports.%Main.R
// CHECK:STDOUT:     .Transitive = imports.%Main.Transitive
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .TakesPQR = %TakesPQR.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %TakesPQR.decl: %TakesPQR.type = fn_decl @TakesPQR [concrete = constants.%TakesPQR] {
// CHECK:STDOUT:     %U.patt.loc5_14.1: %pattern_type.075 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc5_14.2 (constants.%U.patt)]
// CHECK:STDOUT:     %u.param_patt.loc5_36.1: @TakesPQR.%pattern_type (%pattern_type.54d) = value_param_pattern [symbolic = %u.param_patt.loc5_36.2 (constants.%u.param_patt)]
// CHECK:STDOUT:     %u.patt.loc5_36.1: @TakesPQR.%pattern_type (%pattern_type.54d) = at_binding_pattern u, %u.param_patt.loc5_36.1 [symbolic = %u.patt.loc5_36.2 (constants.%u.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_23.1: type = splice_block %.loc5_23.3 [concrete = constants.%facet_type.0b6] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %P.ref: type = name_ref P, imports.%Main.P [concrete = constants.%P.type]
// CHECK:STDOUT:       %Q.ref: type = name_ref Q, imports.%Main.Q [concrete = constants.%Q.type]
// CHECK:STDOUT:       %impl.elem0.loc5_19: %.d56 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc5_19: <bound method> = bound_method %P.ref, %impl.elem0.loc5_19 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.dd7]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc5_19: init type = call %bound_method.loc5_19(%P.ref, %Q.ref) [concrete = constants.%facet_type.bc0]
// CHECK:STDOUT:       %R.ref: type = name_ref R, imports.%Main.R [concrete = constants.%R.type]
// CHECK:STDOUT:       %impl.elem0.loc5_23: %.d56 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc5_23: <bound method> = bound_method %type.as.BitAndWith.impl.Op.call.loc5_19, %impl.elem0.loc5_23 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.588]
// CHECK:STDOUT:       %.loc5_19.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc5_19 [concrete = constants.%facet_type.bc0]
// CHECK:STDOUT:       %.loc5_19.2: type = converted %type.as.BitAndWith.impl.Op.call.loc5_19, %.loc5_19.1 [concrete = constants.%facet_type.bc0]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc5_23: init type = call %bound_method.loc5_23(%.loc5_19.2, %R.ref) [concrete = constants.%facet_type.0b6]
// CHECK:STDOUT:       %.loc5_23.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc5_23 [concrete = constants.%facet_type.0b6]
// CHECK:STDOUT:       %.loc5_23.3: type = converted %type.as.BitAndWith.impl.Op.call.loc5_23, %.loc5_23.2 [concrete = constants.%facet_type.0b6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc5_14.2: %facet_type.0b6 = symbolic_binding U, 0 [symbolic = %U.loc5_14.1 (constants.%U)]
// CHECK:STDOUT:     %u.param: @TakesPQR.%U.as_type.loc5_38.1 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_38.1: type = splice_block %.loc5_38.2 [symbolic = %U.as_type.loc5_38.1 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref: %facet_type.0b6 = name_ref U, %U.loc5_14.2 [symbolic = %U.loc5_14.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc5_38.2: type = facet_access_type %U.ref [symbolic = %U.as_type.loc5_38.1 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc5_38.2: type = converted %U.ref, %U.as_type.loc5_38.2 [symbolic = %U.as_type.loc5_38.1 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @TakesPQR.%U.as_type.loc5_38.1 (%U.as_type) = wrapper_binding u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc6_23.1: @G.%pattern_type.loc6 (%pattern_type.68e728.2) = value_param_pattern [symbolic = %t.param_patt.loc6_23.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc6_23.1: @G.%pattern_type.loc6 (%pattern_type.68e728.2) = at_binding_pattern t, %t.param_patt.loc6_23.1 [symbolic = %t.patt.loc6_23.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_10: type = splice_block %Transitive.ref [concrete = constants.%Transitive.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Transitive.ref: type = name_ref Transitive, imports.%Main.Transitive [concrete = constants.%Transitive.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @G.%T.as_type.loc6_25.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_25.1: type = splice_block %.loc6_25.2 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc6: %Transitive.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6_25.2: type = facet_access_type %T.ref.loc6 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_25.2: type = converted %T.ref.loc6, %T.as_type.loc6_25.2 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.as_type.loc6_25.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @P [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.96f
// CHECK:STDOUT:   .InP = imports.%Main.import_ref.fec
// CHECK:STDOUT:   witness = (imports.%Main.InP)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.4f3
// CHECK:STDOUT:   .InQ = imports.%Main.import_ref.e2a
// CHECK:STDOUT:   witness = (imports.%Main.InQ)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.659
// CHECK:STDOUT:   .InR = imports.%Main.import_ref.0ce
// CHECK:STDOUT:   witness = (imports.%Main.InR)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.0d5
// CHECK:STDOUT:   .A = imports.%Main.import_ref.72c
// CHECK:STDOUT:   .B = imports.%Main.import_ref.e5e
// CHECK:STDOUT:   .C = imports.%Main.import_ref.c9c
// CHECK:STDOUT:   .GetA = imports.%Main.import_ref.5c5
// CHECK:STDOUT:   .TakesC = imports.%Main.import_ref.ae2
// CHECK:STDOUT:   witness = (imports.%Main.A, imports.%Main.B, imports.%Main.C, imports.%Main.GetA, imports.%Main.TakesC)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT:   @Transitive.observe {
// CHECK:STDOUT:     observe constants.%impl.elem0.0439e7.1 == constants.%impl.elem1.6be46c.1 == constants.%impl.elem2.731f97.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.WithSelf.InP(imports.%Main.import_ref.aef2fa.2: %P.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.104)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.cc3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.3ef)]
// CHECK:STDOUT:   %self.param_patt: @P.WithSelf.InP.%pattern_type (%pattern_type.3ef) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.88d)]
// CHECK:STDOUT:   %self.patt: @P.WithSelf.InP.%pattern_type (%pattern_type.3ef) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(imports.%Main.import_ref.c0a1f5.2: %Q.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.afd)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.430)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.ade)]
// CHECK:STDOUT:   %self.param_patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.ade) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.271)]
// CHECK:STDOUT:   %self.patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.ade) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.b5e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(imports.%Main.import_ref.f6968b.2: %R.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.be6)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.b02)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.60e)]
// CHECK:STDOUT:   %self.param_patt: @R.WithSelf.InR.%pattern_type (%pattern_type.60e) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.640)]
// CHECK:STDOUT:   %self.patt: @R.WithSelf.InR.%pattern_type (%pattern_type.60e) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.894)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesPQR(%U.loc5_14.2: %facet_type.0b6) {
// CHECK:STDOUT:   %U.patt.loc5_14.2: %pattern_type.075 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc5_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc5_14.1: %facet_type.0b6 = symbolic_binding U, 0 [symbolic = %U.loc5_14.1 (constants.%U)]
// CHECK:STDOUT:   %U.as_type.loc5_38.1: type = facet_access_type %U.loc5_14.1 [symbolic = %U.as_type.loc5_38.1 (constants.%U.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc5_38.1 [symbolic = %pattern_type (constants.%pattern_type.54d)]
// CHECK:STDOUT:   %u.param_patt.loc5_36.2: @TakesPQR.%pattern_type (%pattern_type.54d) = value_param_pattern [symbolic = %u.param_patt.loc5_36.2 (constants.%u.param_patt)]
// CHECK:STDOUT:   %u.patt.loc5_36.2: @TakesPQR.%pattern_type (%pattern_type.54d) = at_binding_pattern u, %u.param_patt.loc5_36.2 [symbolic = %u.patt.loc5_36.2 (constants.%u.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc5_38.1 [symbolic = %require_complete (constants.%require_complete.192)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @TakesPQR.%U.as_type.loc5_38.1 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(imports.%Main.import_ref.df6636.2: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.744)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.68e728.1)]
// CHECK:STDOUT:   %self.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.1 (%pattern_type.68e728.1) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.e90fff.1)]
// CHECK:STDOUT:   %self.patt: @Transitive.WithSelf.TakesC.%pattern_type.1 (%pattern_type.68e728.1) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.d2dab3.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.1)]
// CHECK:STDOUT:   %impl.elem2: %facet_type.4c8 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2.731f97.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2 [symbolic = %as_type (constants.%as_type.2fadd6.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.ac0)]
// CHECK:STDOUT:   %c.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.2 (%pattern_type.ac0) = value_param_pattern [symbolic = %c.param_patt (constants.%c.param_patt)]
// CHECK:STDOUT:   %c.patt: @Transitive.WithSelf.TakesC.%pattern_type.2 (%pattern_type.ac0) = at_binding_pattern c, %c.param_patt [symbolic = %c.patt (constants.%c.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(imports.%Main.import_ref.df6636.3: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.744)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.68e728.1)]
// CHECK:STDOUT:   %self.param_patt: @Transitive.WithSelf.GetA.%pattern_type.1 (%pattern_type.68e728.1) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.e90fff.1)]
// CHECK:STDOUT:   %self.patt: @Transitive.WithSelf.GetA.%pattern_type.1 (%pattern_type.68e728.1) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.d2dab3.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.1)]
// CHECK:STDOUT:   %impl.elem0: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.0439e7.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic = %as_type (constants.%as_type.6a024f.1)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %as_type [symbolic = %.1 (constants.%.2b1a45.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.c73ac0.1)]
// CHECK:STDOUT:   %return.param_patt: @Transitive.WithSelf.GetA.%pattern_type.2 (%pattern_type.c73ac0.1) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.057e89.1)]
// CHECK:STDOUT:   %return.patt: @Transitive.WithSelf.GetA.%pattern_type.2 (%pattern_type.c73ac0.1) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.4e0711.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc6_7.2: %Transitive.type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_7.1: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc6_25.1: type = facet_access_type %T.loc6_7.1 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %T.as_type.loc6_25.1 [symbolic = %pattern_type.loc6 (constants.%pattern_type.68e728.2)]
// CHECK:STDOUT:   %t.param_patt.loc6_23.2: @G.%pattern_type.loc6 (%pattern_type.68e728.2) = value_param_pattern [symbolic = %t.param_patt.loc6_23.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc6_23.2: @G.%pattern_type.loc6 (%pattern_type.68e728.2) = at_binding_pattern t, %t.param_patt.loc6_23.2 [symbolic = %t.patt.loc6_23.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %T.as_type.loc6_25.1 [symbolic = %require_complete.loc6 (constants.%require_complete.fbd)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc6_7.1, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.2)]
// CHECK:STDOUT:   %impl.elem0.loc7_11.2: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_11.2 (constants.%impl.elem0.0439e7.2)]
// CHECK:STDOUT:   %as_type.loc7_11.2: type = facet_access_type %impl.elem0.loc7_11.2 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %as_type.loc7_11.2 [symbolic = %require_complete.loc7 (constants.%require_complete.c2d)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %as_type.loc7_11.2 [symbolic = %pattern_type.loc7 (constants.%pattern_type.c73ac0.2)]
// CHECK:STDOUT:   %a.patt.loc7_8.2: @G.%pattern_type.loc7 (%pattern_type.c73ac0.2) = ref_binding_pattern a [symbolic = %a.patt.loc7_8.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.var_patt.loc7_3.2: @G.%pattern_type.loc7 (%pattern_type.c73ac0.2) = var_pattern %a.patt.loc7_8.2 [symbolic = %a.var_patt.loc7_3.2 (constants.%a.var_patt)]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T.loc6_7.1) [symbolic = %Transitive.WithSelf.GetA.type (constants.%Transitive.WithSelf.GetA.type.0b3c58.2)]
// CHECK:STDOUT:   %.loc7_17: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type, %T.loc6_7.1 [symbolic = %.loc7_17 (constants.%.657)]
// CHECK:STDOUT:   %impl.elem3.loc7_17.2: @G.%.loc7_17 (%.657) = impl_witness_access %Transitive.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc7_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_17.2: <specific function> = specific_impl_function %impl.elem3.loc7_17.2, @Transitive.WithSelf.GetA(%T.loc6_7.1) [symbolic = %specific_impl_fn.loc7_17.2 (constants.%specific_impl_fn.72f)]
// CHECK:STDOUT:   %impl.elem1.loc9_19.2: %facet_type.130 = impl_witness_access %Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc9_19.2 (constants.%impl.elem1.6be46c.2)]
// CHECK:STDOUT:   %as_type.loc9_19.3: type = facet_access_type %impl.elem1.loc9_19.2 [symbolic = %as_type.loc9_19.3 (constants.%as_type.dca436.2)]
// CHECK:STDOUT:   %eq.loc9 = observe_equivalent %impl.elem0.loc7_11.2, %impl.elem1.loc9_19.2 [symbolic = %eq.loc9 (constants.%eq.4be6e4.2)]
// CHECK:STDOUT:   %impls.loc9 = observe_impls %impl.elem1.loc9_19.2, constants.%Q.type [symbolic = %impls.loc9 (constants.%impls.885)]
// CHECK:STDOUT:   %impl.elem2.loc16_19.2: %facet_type.4c8 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc16_19.2 (constants.%impl.elem2.731f97.2)]
// CHECK:STDOUT:   %as_type.loc16_19.3: type = facet_access_type %impl.elem2.loc16_19.2 [symbolic = %as_type.loc16_19.3 (constants.%as_type.2fadd6.2)]
// CHECK:STDOUT:   %eq.loc16 = observe_equivalent %impl.elem0.loc7_11.2, %impl.elem2.loc16_19.2 [symbolic = %eq.loc16 (constants.%eq.fcd)]
// CHECK:STDOUT:   %impls.loc16 = observe_impls %impl.elem2.loc16_19.2, constants.%R.type [symbolic = %impls.loc16 (constants.%impls.adc)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc7_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value %as_type.loc7_11.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc7_3.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.5e1)]
// CHECK:STDOUT:   %.loc7_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc7_3.3 [symbolic = %.loc7_3.4 (constants.%.e75)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @G.%.loc7_3.4 (%.e75) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.678)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc7_3.3) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.e8e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @G.%T.as_type.loc6_25.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.var: ref @G.%as_type.loc7_11.2 (%as_type.6a024f.2) = var_storage %a.var_patt.loc7_3.1
// CHECK:STDOUT:     %t.ref: @G.%T.as_type.loc6_25.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %GetA.ref: %Transitive.assoc_type = name_ref GetA, imports.%Main.import_ref.5c5 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc7_17.1: @G.%.loc7_17 (%.657) = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element3 [symbolic = %impl.elem3.loc7_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc7_17: <bound method> = bound_method %t.ref, %impl.elem3.loc7_17.1
// CHECK:STDOUT:     %.loc7_23.1: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_23.2: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_17.1: <specific function> = specific_impl_function %impl.elem3.loc7_17.1, @Transitive.WithSelf.GetA(constants.%T) [symbolic = %specific_impl_fn.loc7_17.2 (constants.%specific_impl_fn.72f)]
// CHECK:STDOUT:     %bound_method.loc7_23: <bound method> = bound_method %t.ref, %specific_impl_fn.loc7_17.1
// CHECK:STDOUT:     %.loc7_3.1: ref @G.%as_type.loc7_11.2 (%as_type.6a024f.2) = splice_block %a.var {}
// CHECK:STDOUT:     %Transitive.WithSelf.GetA.call: init @G.%as_type.loc7_11.2 (%as_type.6a024f.2) to %.loc7_3.1 = call %bound_method.loc7_23(%t.ref)
// CHECK:STDOUT:     assign %a.var, %Transitive.WithSelf.GetA.call
// CHECK:STDOUT:     %.loc7_11.1: type = splice_block %.loc7_11.3 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)] {
// CHECK:STDOUT:       %T.ref.loc7: %Transitive.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7: type = facet_access_type %T.ref.loc7 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_11.2: type = converted %T.ref.loc7, %T.as_type.loc7 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %A.ref.loc7: %Transitive.assoc_type = name_ref A, imports.%Main.import_ref.72c [concrete = constants.%assoc0.7cd]
// CHECK:STDOUT:       %impl.elem0.loc7_11.1: %facet_type.b1b = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic = %impl.elem0.loc7_11.2 (constants.%impl.elem0.0439e7.2)]
// CHECK:STDOUT:       %as_type.loc7_11.1: type = facet_access_type %impl.elem0.loc7_11.1 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)]
// CHECK:STDOUT:       %.loc7_11.3: type = converted %impl.elem0.loc7_11.1, %as_type.loc7_11.1 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: ref @G.%as_type.loc7_11.2 (%as_type.6a024f.2) = wrapper_binding a, %a.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc7_8.1: @G.%pattern_type.loc7 (%pattern_type.c73ac0.2) = ref_binding_pattern a [symbolic = %a.patt.loc7_8.2 (constants.%a.patt)]
// CHECK:STDOUT:       %a.var_patt.loc7_3.1: @G.%pattern_type.loc7 (%pattern_type.c73ac0.2) = var_pattern %a.patt.loc7_8.1 [symbolic = %a.var_patt.loc7_3.2 (constants.%a.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref.loc9_11: %Transitive.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc9_12: type = facet_access_type %T.ref.loc9_11 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_12.1: type = converted %T.ref.loc9_11, %T.as_type.loc9_12 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %A.ref.loc9: %Transitive.assoc_type = name_ref A, imports.%Main.import_ref.72c [concrete = constants.%assoc0.7cd]
// CHECK:STDOUT:     %impl.elem0.loc9: %facet_type.b1b = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic = %impl.elem0.loc7_11.2 (constants.%impl.elem0.0439e7.2)]
// CHECK:STDOUT:     %T.ref.loc9_18: %Transitive.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc9_19: type = facet_access_type %T.ref.loc9_18 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_19.1: type = converted %T.ref.loc9_18, %T.as_type.loc9_19 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %B.ref: %Transitive.assoc_type = name_ref B, imports.%Main.import_ref.e5e [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc9_19.1: %facet_type.130 = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element1 [symbolic = %impl.elem1.loc9_19.2 (constants.%impl.elem1.6be46c.2)]
// CHECK:STDOUT:     %as_type.loc9_19.1: type = facet_access_type %impl.elem1.loc9_19.1 [symbolic = %as_type.loc9_19.3 (constants.%as_type.dca436.2)]
// CHECK:STDOUT:     %.loc9_19.2: type = converted %impl.elem1.loc9_19.1, %as_type.loc9_19.1 [symbolic = %as_type.loc9_19.3 (constants.%as_type.dca436.2)]
// CHECK:STDOUT:     %as_type.loc9_12: type = facet_access_type %impl.elem0.loc9 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)]
// CHECK:STDOUT:     %.loc9_12.2: type = converted %impl.elem0.loc9, %as_type.loc9_12 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)]
// CHECK:STDOUT:     %Q.ref.loc9: type = name_ref Q, imports.%Main.Q [concrete = constants.%Q.type]
// CHECK:STDOUT:     %as_type.loc9_19.2: type = facet_access_type %impl.elem1.loc9_19.1 [symbolic = %as_type.loc9_19.3 (constants.%as_type.dca436.2)]
// CHECK:STDOUT:     %.loc9_19.3: type = converted %impl.elem1.loc9_19.1, %as_type.loc9_19.2 [symbolic = %as_type.loc9_19.3 (constants.%as_type.dca436.2)]
// CHECK:STDOUT:     %G.observe.decl.loc9 = observe_decl @G.observe.loc9 [concrete]
// CHECK:STDOUT:     %a.ref.loc14: ref @G.%as_type.loc7_11.2 (%as_type.6a024f.2) = name_ref a, %a
// CHECK:STDOUT:     %Q.ref.loc14: type = name_ref Q, imports.%Main.Q [concrete = constants.%Q.type]
// CHECK:STDOUT:     %InQ.ref: %Q.assoc_type = name_ref InQ, imports.%Main.import_ref.e2a [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:     %T.ref.loc16_11: %Transitive.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc16_12: type = facet_access_type %T.ref.loc16_11 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc16_12.1: type = converted %T.ref.loc16_11, %T.as_type.loc16_12 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %A.ref.loc16: %Transitive.assoc_type = name_ref A, imports.%Main.import_ref.72c [concrete = constants.%assoc0.7cd]
// CHECK:STDOUT:     %impl.elem0.loc16: %facet_type.b1b = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic = %impl.elem0.loc7_11.2 (constants.%impl.elem0.0439e7.2)]
// CHECK:STDOUT:     %T.ref.loc16_18: %Transitive.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc16_19: type = facet_access_type %T.ref.loc16_18 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc16_19.1: type = converted %T.ref.loc16_18, %T.as_type.loc16_19 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %C.ref: %Transitive.assoc_type = name_ref C, imports.%Main.import_ref.c9c [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc16_19.1: %facet_type.4c8 = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element2 [symbolic = %impl.elem2.loc16_19.2 (constants.%impl.elem2.731f97.2)]
// CHECK:STDOUT:     %as_type.loc16_19.1: type = facet_access_type %impl.elem2.loc16_19.1 [symbolic = %as_type.loc16_19.3 (constants.%as_type.2fadd6.2)]
// CHECK:STDOUT:     %.loc16_19.2: type = converted %impl.elem2.loc16_19.1, %as_type.loc16_19.1 [symbolic = %as_type.loc16_19.3 (constants.%as_type.2fadd6.2)]
// CHECK:STDOUT:     %as_type.loc16_12: type = facet_access_type %impl.elem0.loc16 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)]
// CHECK:STDOUT:     %.loc16_12.2: type = converted %impl.elem0.loc16, %as_type.loc16_12 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)]
// CHECK:STDOUT:     %R.ref: type = name_ref R, imports.%Main.R [concrete = constants.%R.type]
// CHECK:STDOUT:     %as_type.loc16_19.2: type = facet_access_type %impl.elem2.loc16_19.1 [symbolic = %as_type.loc16_19.3 (constants.%as_type.2fadd6.2)]
// CHECK:STDOUT:     %.loc16_19.3: type = converted %impl.elem2.loc16_19.1, %as_type.loc16_19.2 [symbolic = %as_type.loc16_19.3 (constants.%as_type.2fadd6.2)]
// CHECK:STDOUT:     %G.observe.decl.loc16 = observe_decl @G.observe.loc16 [concrete]
// CHECK:STDOUT:     %TakesPQR.ref: %TakesPQR.type = name_ref TakesPQR, file.%TakesPQR.decl [concrete = constants.%TakesPQR]
// CHECK:STDOUT:     %a.ref.loc24: ref @G.%as_type.loc7_11.2 (%as_type.6a024f.2) = name_ref a, %a
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @G.%.loc7_3.4 (%.e75) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.678)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%as_type.6a024f.2, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc7_3.2: %Destroy.type = converted constants.%as_type.6a024f.2, %Destroy.facet.loc7_3.1 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%as_type.6a024f.2, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc7_3.3: %Destroy.type = converted constants.%as_type.6a024f.2, %Destroy.facet.loc7_3.2 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.e8e)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%a.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @G.observe.loc9 {
// CHECK:STDOUT:       observe constants.%impl.elem0.0439e7.2 == constants.%impl.elem1.6be46c.2 impls constants.%Q.type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     @G.observe.loc16 {
// CHECK:STDOUT:       observe constants.%impl.elem0.0439e7.2 == constants.%impl.elem2.731f97.2 impls constants.%R.type
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf(constants.%Self.104) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.104
// CHECK:STDOUT:   %P.WithSelf.InP.type => constants.%P.WithSelf.InP.type
// CHECK:STDOUT:   %P.WithSelf.InP => constants.%P.WithSelf.InP
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf.InP(constants.%Self.104) {
// CHECK:STDOUT:   %Self => constants.%Self.104
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.cc3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3ef
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.88d
// CHECK:STDOUT:   %self.patt => constants.%self.patt.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Self.afd) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.afd
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.afd) {
// CHECK:STDOUT:   %Self => constants.%Self.afd
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.430
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ade
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.271
// CHECK:STDOUT:   %self.patt => constants.%self.patt.b5e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%Self.be6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.be6
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%Self.be6) {
// CHECK:STDOUT:   %Self => constants.%Self.be6
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.b02
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.60e
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.640
// CHECK:STDOUT:   %self.patt => constants.%self.patt.894
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesPQR(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc5_14.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc5_14.1 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc5_38.1 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.54d
// CHECK:STDOUT:   %u.param_patt.loc5_36.2 => constants.%u.param_patt
// CHECK:STDOUT:   %u.patt.loc5_36.2 => constants.%u.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%Self.744) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.0439e7.1
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.6be46c.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.0b3c58.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.e3e09c.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.d86d9f.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.f276cd.1
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.dca436.1
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.6a024f.1
// CHECK:STDOUT:   %eq.1 => constants.%eq.4be6e4.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.731f97.1
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.2fadd6.1
// CHECK:STDOUT:   %eq.2 => constants.%eq.033b19.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.744) {
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.1
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.731f97.1
// CHECK:STDOUT:   %as_type => constants.%as_type.2fadd6.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.ac0
// CHECK:STDOUT:   %c.param_patt => constants.%c.param_patt
// CHECK:STDOUT:   %c.patt => constants.%c.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.744) {
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.1
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.0439e7.1
// CHECK:STDOUT:   %as_type => constants.%as_type.6a024f.1
// CHECK:STDOUT:   %.1 => constants.%.2b1a45.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.c73ac0.1
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.057e89.1
// CHECK:STDOUT:   %return.patt => constants.%return.patt.4e0711.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc6_25.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc6 => constants.%pattern_type.68e728.2
// CHECK:STDOUT:   %t.param_patt.loc6_23.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc6_23.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.0439e7.2
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.6be46c.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.0b3c58.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.e3e09c.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.d86d9f.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.f276cd.2
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.dca436.2
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.6a024f.2
// CHECK:STDOUT:   %eq.1 => constants.%eq.4be6e4.2
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.731f97.2
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.2fadd6.2
// CHECK:STDOUT:   %eq.2 => constants.%eq.033b19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.2
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.2
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.2
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.0439e7.2
// CHECK:STDOUT:   %as_type => constants.%as_type.6a024f.2
// CHECK:STDOUT:   %.1 => constants.%.2b1a45.2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.c73ac0.2
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.057e89.2
// CHECK:STDOUT:   %return.patt => constants.%return.patt.4e0711.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_equivalent_impls_without_observe.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.104: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P.WithSelf(%Self.104) [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.cc3: type = facet_access_type %Self.104 [symbolic]
// CHECK:STDOUT:   %pattern_type.3ef: type = pattern_type %Self.as_type.cc3 [symbolic]
// CHECK:STDOUT:   %self.param_patt.88d: %pattern_type.3ef = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.6de: %pattern_type.3ef = at_binding_pattern self, %self.param_patt.88d [symbolic]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.afd: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Self.afd) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ: %Q.WithSelf.InQ.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.430: type = facet_access_type %Self.afd [symbolic]
// CHECK:STDOUT:   %pattern_type.ade: type = pattern_type %Self.as_type.430 [symbolic]
// CHECK:STDOUT:   %self.param_patt.271: %pattern_type.ade = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b5e: %pattern_type.ade = at_binding_pattern self, %self.param_patt.271 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.802: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.802 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.38f: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d56: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.38f, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.dd7: <bound method> = bound_method %P.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.bc0: type = facet_type <@P & @Q> [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.be6: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R.WithSelf(%Self.be6) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR: %R.WithSelf.InR.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.b02: type = facet_access_type %Self.be6 [symbolic]
// CHECK:STDOUT:   %pattern_type.60e: type = pattern_type %Self.as_type.b02 [symbolic]
// CHECK:STDOUT:   %self.param_patt.640: %pattern_type.60e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.894: %pattern_type.60e = at_binding_pattern self, %self.param_patt.640 [symbolic]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.588: <bound method> = bound_method %facet_type.bc0, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.0b6: type = facet_type <@P & @Q & @R> [concrete]
// CHECK:STDOUT:   %pattern_type.075: type = pattern_type %facet_type.0b6 [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.075 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: %facet_type.0b6 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %pattern_type.54d: type = pattern_type %U.as_type [symbolic]
// CHECK:STDOUT:   %u.param_patt: %pattern_type.54d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt: %pattern_type.54d = at_binding_pattern u, %u.param_patt [symbolic]
// CHECK:STDOUT:   %TakesPQR.type: type = fn_type @TakesPQR [concrete]
// CHECK:STDOUT:   %TakesPQR: %TakesPQR.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.192: <witness> = require_complete_type %U.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.744: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.ab0b8d.1: <witness> = lookup_impl_witness %Self.744, @Transitive [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type.4c8: type = facet_type <@R & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem2.731f97.1: %facet_type.4c8 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element2 [symbolic]
// CHECK:STDOUT:   %facet_type.130: type = facet_type <@Q & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem1.6be46c.1: %facet_type.130 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element1 [symbolic]
// CHECK:STDOUT:   %eq.033b19.1 = observe_equivalent %impl.elem1.6be46c.1, %impl.elem2.731f97.1 [symbolic]
// CHECK:STDOUT:   %as_type.2fadd6.1: type = facet_access_type %impl.elem2.731f97.1 [symbolic]
// CHECK:STDOUT:   %facet_type.b1b: type = facet_type <@P & @Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem0.0439e7.1: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element0 [symbolic]
// CHECK:STDOUT:   %eq.4be6e4.1 = observe_equivalent %impl.elem0.0439e7.1, %impl.elem1.6be46c.1 [symbolic]
// CHECK:STDOUT:   %as_type.6a024f.1: type = facet_access_type %impl.elem0.0439e7.1 [symbolic]
// CHECK:STDOUT:   %as_type.dca436.1: type = facet_access_type %impl.elem1.6be46c.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.d86d9f.1: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%Self.744) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.f276cd.1: %Transitive.WithSelf.TakesC.type.d86d9f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.56c: type = facet_access_type %Self.744 [symbolic]
// CHECK:STDOUT:   %pattern_type.68e728.1: type = pattern_type %Self.as_type.56c [symbolic]
// CHECK:STDOUT:   %self.param_patt.e90fff.1: %pattern_type.68e728.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d2dab3.1: %pattern_type.68e728.1 = at_binding_pattern self, %self.param_patt.e90fff.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.ac0: type = pattern_type %as_type.2fadd6.1 [symbolic]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.ac0 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %c.patt: %pattern_type.ac0 = at_binding_pattern c, %c.param_patt [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.0b3c58.1: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%Self.744) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.e3e09c.1: %Transitive.WithSelf.GetA.type.0b3c58.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.c73ac0.1: type = pattern_type %as_type.6a024f.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.057e89.1: %pattern_type.c73ac0.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.4e0711.1: %pattern_type.c73ac0.1 = return_slot_pattern %return.param_patt.057e89.1, invalid [symbolic]
// CHECK:STDOUT:   %.2b1a45.1: Core.Form = init_form %as_type.6a024f.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.bd9: type = pattern_type %Transitive.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Transitive.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.68e728.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.68e728.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.68e728.2 = at_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fbd: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.ab0b8d.2: <witness> = lookup_impl_witness %T, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.0439e7.2: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.6be46c.2: %facet_type.130 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.0b3c58.2: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.e3e09c.2: %Transitive.WithSelf.GetA.type.0b3c58.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.d86d9f.2: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.f276cd.2: %Transitive.WithSelf.TakesC.type.d86d9f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %as_type.dca436.2: type = facet_access_type %impl.elem1.6be46c.2 [symbolic]
// CHECK:STDOUT:   %as_type.6a024f.2: type = facet_access_type %impl.elem0.0439e7.2 [symbolic]
// CHECK:STDOUT:   %eq.4be6e4.2 = observe_equivalent %impl.elem0.0439e7.2, %impl.elem1.6be46c.2 [symbolic]
// CHECK:STDOUT:   %impl.elem2.731f97.2: %facet_type.4c8 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element2 [symbolic]
// CHECK:STDOUT:   %as_type.2fadd6.2: type = facet_access_type %impl.elem2.731f97.2 [symbolic]
// CHECK:STDOUT:   %eq.033b19.2 = observe_equivalent %impl.elem1.6be46c.2, %impl.elem2.731f97.2 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.7cd: %Transitive.assoc_type = assoc_entity element0, imports.%Main.import_ref.d38 [concrete]
// CHECK:STDOUT:   %require_complete.c2d: <witness> = require_complete_type %as_type.6a024f.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.c73ac0.2: type = pattern_type %as_type.6a024f.2 [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.c73ac0.2 = ref_binding_pattern a [symbolic]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.c73ac0.2 = var_pattern %a.patt [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, imports.%Main.import_ref.06c [concrete]
// CHECK:STDOUT:   %.657: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type.0b3c58.2, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.657 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element3 [symbolic]
// CHECK:STDOUT:   %self.param_patt.e90fff.2: %pattern_type.68e728.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d2dab3.2: %pattern_type.68e728.2 = at_binding_pattern self, %self.param_patt.e90fff.2 [symbolic]
// CHECK:STDOUT:   %.2b1a45.2: Core.Form = init_form %as_type.6a024f.2 [symbolic]
// CHECK:STDOUT:   %return.param_patt.057e89.2: %pattern_type.c73ac0.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.4e0711.2: %pattern_type.c73ac0.2 = return_slot_pattern %return.param_patt.057e89.2, invalid [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.72f: <specific function> = specific_impl_function %impl.elem3, @Transitive.WithSelf.GetA(%T) [symbolic]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.6a1: %Q.assoc_type = assoc_entity element0, imports.%Main.import_ref.636 [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.0439e7.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.6a024f.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5e1: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.e75: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5e1, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.678: %.e75 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e8e: <specific function> = specific_impl_function %impl.elem0.678, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.P: type = import_ref Main//transitive, P, loaded [concrete = constants.%P.type]
// CHECK:STDOUT:   %Main.Q: type = import_ref Main//transitive, Q, loaded [concrete = constants.%Q.type]
// CHECK:STDOUT:   %Main.R: type = import_ref Main//transitive, R, loaded [concrete = constants.%R.type]
// CHECK:STDOUT:   %Main.Transitive: type = import_ref Main//transitive, Transitive, loaded [concrete = constants.%Transitive.type]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.fec = import_ref Main//transitive, loc3_27, unloaded
// CHECK:STDOUT:   %Main.InP = import_ref Main//transitive, InP, unloaded
// CHECK:STDOUT:   %Main.import_ref.aef2fa.2: %P.type = import_ref Main//transitive, loc3_13, loaded [symbolic = constants.%Self.104]
// CHECK:STDOUT:   %Main.import_ref.96f = import_ref Main//transitive, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.e2a: %Q.assoc_type = import_ref Main//transitive, loc4_27, loaded [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:   %Main.InQ = import_ref Main//transitive, InQ, unloaded
// CHECK:STDOUT:   %Main.import_ref.c0a1f5.2: %Q.type = import_ref Main//transitive, loc4_13, loaded [symbolic = constants.%Self.afd]
// CHECK:STDOUT:   %Main.import_ref.4f3 = import_ref Main//transitive, loc4_13, unloaded
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.0ce = import_ref Main//transitive, loc5_27, unloaded
// CHECK:STDOUT:   %Main.InR = import_ref Main//transitive, InR, unloaded
// CHECK:STDOUT:   %Main.import_ref.f6968b.2: %R.type = import_ref Main//transitive, loc5_13, loaded [symbolic = constants.%Self.be6]
// CHECK:STDOUT:   %Main.import_ref.659 = import_ref Main//transitive, loc5_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.72c: %Transitive.assoc_type = import_ref Main//transitive, loc8_8, loaded [concrete = constants.%assoc0.7cd]
// CHECK:STDOUT:   %Main.import_ref.f7f = import_ref Main//transitive, loc9_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.71d = import_ref Main//transitive, loc10_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.5c5: %Transitive.assoc_type = import_ref Main//transitive, loc12_21, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Main.import_ref.ae2 = import_ref Main//transitive, loc13_24, unloaded
// CHECK:STDOUT:   %Main.A = import_ref Main//transitive, A, unloaded
// CHECK:STDOUT:   %Main.B = import_ref Main//transitive, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//transitive, C, unloaded
// CHECK:STDOUT:   %Main.GetA = import_ref Main//transitive, GetA, unloaded
// CHECK:STDOUT:   %Main.TakesC = import_ref Main//transitive, TakesC, unloaded
// CHECK:STDOUT:   %Main.import_ref.df6636.2: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.744]
// CHECK:STDOUT:   %Main.import_ref.df6636.3: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.744]
// CHECK:STDOUT:   %Main.import_ref.0d5 = import_ref Main//transitive, loc7_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.d38: %facet_type.b1b = import_ref Main//transitive, loc8_8, loaded [concrete = %A]
// CHECK:STDOUT:   %A: %facet_type.b1b = assoc_const_decl @A [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.06c: @Transitive.WithSelf.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type.0b3c58.1) = import_ref Main//transitive, loc12_21, loaded [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA.e3e09c.1)]
// CHECK:STDOUT:   %Main.import_ref.636: @Q.WithSelf.%Q.WithSelf.InQ.type (%Q.WithSelf.InQ.type) = import_ref Main//transitive, loc4_27, loaded [symbolic = @Q.WithSelf.%Q.WithSelf.InQ (constants.%Q.WithSelf.InQ)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .P = imports.%Main.P
// CHECK:STDOUT:     .Q = imports.%Main.Q
// CHECK:STDOUT:     .R = imports.%Main.R
// CHECK:STDOUT:     .Transitive = imports.%Main.Transitive
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .TakesPQR = %TakesPQR.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %TakesPQR.decl: %TakesPQR.type = fn_decl @TakesPQR [concrete = constants.%TakesPQR] {
// CHECK:STDOUT:     %U.patt.loc5_14.1: %pattern_type.075 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc5_14.2 (constants.%U.patt)]
// CHECK:STDOUT:     %u.param_patt.loc5_36.1: @TakesPQR.%pattern_type (%pattern_type.54d) = value_param_pattern [symbolic = %u.param_patt.loc5_36.2 (constants.%u.param_patt)]
// CHECK:STDOUT:     %u.patt.loc5_36.1: @TakesPQR.%pattern_type (%pattern_type.54d) = at_binding_pattern u, %u.param_patt.loc5_36.1 [symbolic = %u.patt.loc5_36.2 (constants.%u.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_23.1: type = splice_block %.loc5_23.3 [concrete = constants.%facet_type.0b6] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %P.ref: type = name_ref P, imports.%Main.P [concrete = constants.%P.type]
// CHECK:STDOUT:       %Q.ref: type = name_ref Q, imports.%Main.Q [concrete = constants.%Q.type]
// CHECK:STDOUT:       %impl.elem0.loc5_19: %.d56 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc5_19: <bound method> = bound_method %P.ref, %impl.elem0.loc5_19 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.dd7]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc5_19: init type = call %bound_method.loc5_19(%P.ref, %Q.ref) [concrete = constants.%facet_type.bc0]
// CHECK:STDOUT:       %R.ref: type = name_ref R, imports.%Main.R [concrete = constants.%R.type]
// CHECK:STDOUT:       %impl.elem0.loc5_23: %.d56 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc5_23: <bound method> = bound_method %type.as.BitAndWith.impl.Op.call.loc5_19, %impl.elem0.loc5_23 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.588]
// CHECK:STDOUT:       %.loc5_19.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc5_19 [concrete = constants.%facet_type.bc0]
// CHECK:STDOUT:       %.loc5_19.2: type = converted %type.as.BitAndWith.impl.Op.call.loc5_19, %.loc5_19.1 [concrete = constants.%facet_type.bc0]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc5_23: init type = call %bound_method.loc5_23(%.loc5_19.2, %R.ref) [concrete = constants.%facet_type.0b6]
// CHECK:STDOUT:       %.loc5_23.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc5_23 [concrete = constants.%facet_type.0b6]
// CHECK:STDOUT:       %.loc5_23.3: type = converted %type.as.BitAndWith.impl.Op.call.loc5_23, %.loc5_23.2 [concrete = constants.%facet_type.0b6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc5_14.2: %facet_type.0b6 = symbolic_binding U, 0 [symbolic = %U.loc5_14.1 (constants.%U)]
// CHECK:STDOUT:     %u.param: @TakesPQR.%U.as_type.loc5_38.1 (%U.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_38.1: type = splice_block %.loc5_38.2 [symbolic = %U.as_type.loc5_38.1 (constants.%U.as_type)] {
// CHECK:STDOUT:       %U.ref: %facet_type.0b6 = name_ref U, %U.loc5_14.2 [symbolic = %U.loc5_14.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type.loc5_38.2: type = facet_access_type %U.ref [symbolic = %U.as_type.loc5_38.1 (constants.%U.as_type)]
// CHECK:STDOUT:       %.loc5_38.2: type = converted %U.ref, %U.as_type.loc5_38.2 [symbolic = %U.as_type.loc5_38.1 (constants.%U.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @TakesPQR.%U.as_type.loc5_38.1 (%U.as_type) = wrapper_binding u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc6_23.1: @G.%pattern_type.loc6 (%pattern_type.68e728.2) = value_param_pattern [symbolic = %t.param_patt.loc6_23.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc6_23.1: @G.%pattern_type.loc6 (%pattern_type.68e728.2) = at_binding_pattern t, %t.param_patt.loc6_23.1 [symbolic = %t.patt.loc6_23.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_10: type = splice_block %Transitive.ref [concrete = constants.%Transitive.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Transitive.ref: type = name_ref Transitive, imports.%Main.Transitive [concrete = constants.%Transitive.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @G.%T.as_type.loc6_25.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_25.1: type = splice_block %.loc6_25.2 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc6: %Transitive.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6_25.2: type = facet_access_type %T.ref.loc6 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_25.2: type = converted %T.ref.loc6, %T.as_type.loc6_25.2 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.as_type.loc6_25.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @P [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.96f
// CHECK:STDOUT:   .InP = imports.%Main.import_ref.fec
// CHECK:STDOUT:   witness = (imports.%Main.InP)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.4f3
// CHECK:STDOUT:   .InQ = imports.%Main.import_ref.e2a
// CHECK:STDOUT:   witness = (imports.%Main.InQ)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.659
// CHECK:STDOUT:   .InR = imports.%Main.import_ref.0ce
// CHECK:STDOUT:   witness = (imports.%Main.InR)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.0d5
// CHECK:STDOUT:   .A = imports.%Main.import_ref.72c
// CHECK:STDOUT:   .B = imports.%Main.import_ref.f7f
// CHECK:STDOUT:   .C = imports.%Main.import_ref.71d
// CHECK:STDOUT:   .GetA = imports.%Main.import_ref.5c5
// CHECK:STDOUT:   .TakesC = imports.%Main.import_ref.ae2
// CHECK:STDOUT:   witness = (imports.%Main.A, imports.%Main.B, imports.%Main.C, imports.%Main.GetA, imports.%Main.TakesC)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT:   @Transitive.observe {
// CHECK:STDOUT:     observe constants.%impl.elem0.0439e7.1 == constants.%impl.elem1.6be46c.1 == constants.%impl.elem2.731f97.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.WithSelf.InP(imports.%Main.import_ref.aef2fa.2: %P.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.104)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.cc3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.3ef)]
// CHECK:STDOUT:   %self.param_patt: @P.WithSelf.InP.%pattern_type (%pattern_type.3ef) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.88d)]
// CHECK:STDOUT:   %self.patt: @P.WithSelf.InP.%pattern_type (%pattern_type.3ef) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(imports.%Main.import_ref.c0a1f5.2: %Q.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.afd)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.430)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.ade)]
// CHECK:STDOUT:   %self.param_patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.ade) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.271)]
// CHECK:STDOUT:   %self.patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.ade) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.b5e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(imports.%Main.import_ref.f6968b.2: %R.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.be6)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.b02)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.60e)]
// CHECK:STDOUT:   %self.param_patt: @R.WithSelf.InR.%pattern_type (%pattern_type.60e) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.640)]
// CHECK:STDOUT:   %self.patt: @R.WithSelf.InR.%pattern_type (%pattern_type.60e) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.894)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesPQR(%U.loc5_14.2: %facet_type.0b6) {
// CHECK:STDOUT:   %U.patt.loc5_14.2: %pattern_type.075 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc5_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc5_14.1: %facet_type.0b6 = symbolic_binding U, 0 [symbolic = %U.loc5_14.1 (constants.%U)]
// CHECK:STDOUT:   %U.as_type.loc5_38.1: type = facet_access_type %U.loc5_14.1 [symbolic = %U.as_type.loc5_38.1 (constants.%U.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc5_38.1 [symbolic = %pattern_type (constants.%pattern_type.54d)]
// CHECK:STDOUT:   %u.param_patt.loc5_36.2: @TakesPQR.%pattern_type (%pattern_type.54d) = value_param_pattern [symbolic = %u.param_patt.loc5_36.2 (constants.%u.param_patt)]
// CHECK:STDOUT:   %u.patt.loc5_36.2: @TakesPQR.%pattern_type (%pattern_type.54d) = at_binding_pattern u, %u.param_patt.loc5_36.2 [symbolic = %u.patt.loc5_36.2 (constants.%u.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc5_38.1 [symbolic = %require_complete (constants.%require_complete.192)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @TakesPQR.%U.as_type.loc5_38.1 (%U.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(imports.%Main.import_ref.df6636.2: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.744)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.68e728.1)]
// CHECK:STDOUT:   %self.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.1 (%pattern_type.68e728.1) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.e90fff.1)]
// CHECK:STDOUT:   %self.patt: @Transitive.WithSelf.TakesC.%pattern_type.1 (%pattern_type.68e728.1) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.d2dab3.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.1)]
// CHECK:STDOUT:   %impl.elem2: %facet_type.4c8 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2.731f97.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2 [symbolic = %as_type (constants.%as_type.2fadd6.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.ac0)]
// CHECK:STDOUT:   %c.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.2 (%pattern_type.ac0) = value_param_pattern [symbolic = %c.param_patt (constants.%c.param_patt)]
// CHECK:STDOUT:   %c.patt: @Transitive.WithSelf.TakesC.%pattern_type.2 (%pattern_type.ac0) = at_binding_pattern c, %c.param_patt [symbolic = %c.patt (constants.%c.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(imports.%Main.import_ref.df6636.3: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.744)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.68e728.1)]
// CHECK:STDOUT:   %self.param_patt: @Transitive.WithSelf.GetA.%pattern_type.1 (%pattern_type.68e728.1) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.e90fff.1)]
// CHECK:STDOUT:   %self.patt: @Transitive.WithSelf.GetA.%pattern_type.1 (%pattern_type.68e728.1) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.d2dab3.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.1)]
// CHECK:STDOUT:   %impl.elem0: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.0439e7.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic = %as_type (constants.%as_type.6a024f.1)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %as_type [symbolic = %.1 (constants.%.2b1a45.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.c73ac0.1)]
// CHECK:STDOUT:   %return.param_patt: @Transitive.WithSelf.GetA.%pattern_type.2 (%pattern_type.c73ac0.1) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.057e89.1)]
// CHECK:STDOUT:   %return.patt: @Transitive.WithSelf.GetA.%pattern_type.2 (%pattern_type.c73ac0.1) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.4e0711.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc6_7.2: %Transitive.type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_7.1: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc6_25.1: type = facet_access_type %T.loc6_7.1 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %T.as_type.loc6_25.1 [symbolic = %pattern_type.loc6 (constants.%pattern_type.68e728.2)]
// CHECK:STDOUT:   %t.param_patt.loc6_23.2: @G.%pattern_type.loc6 (%pattern_type.68e728.2) = value_param_pattern [symbolic = %t.param_patt.loc6_23.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc6_23.2: @G.%pattern_type.loc6 (%pattern_type.68e728.2) = at_binding_pattern t, %t.param_patt.loc6_23.2 [symbolic = %t.patt.loc6_23.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %T.as_type.loc6_25.1 [symbolic = %require_complete.loc6 (constants.%require_complete.fbd)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc6_7.1, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.2)]
// CHECK:STDOUT:   %impl.elem0.loc7_11.2: %facet_type.b1b = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_11.2 (constants.%impl.elem0.0439e7.2)]
// CHECK:STDOUT:   %as_type.loc7_11.2: type = facet_access_type %impl.elem0.loc7_11.2 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %as_type.loc7_11.2 [symbolic = %require_complete.loc7 (constants.%require_complete.c2d)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %as_type.loc7_11.2 [symbolic = %pattern_type.loc7 (constants.%pattern_type.c73ac0.2)]
// CHECK:STDOUT:   %a.patt.loc7_8.2: @G.%pattern_type.loc7 (%pattern_type.c73ac0.2) = ref_binding_pattern a [symbolic = %a.patt.loc7_8.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.var_patt.loc7_3.2: @G.%pattern_type.loc7 (%pattern_type.c73ac0.2) = var_pattern %a.patt.loc7_8.2 [symbolic = %a.var_patt.loc7_3.2 (constants.%a.var_patt)]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T.loc6_7.1) [symbolic = %Transitive.WithSelf.GetA.type (constants.%Transitive.WithSelf.GetA.type.0b3c58.2)]
// CHECK:STDOUT:   %.loc7_17: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type, %T.loc6_7.1 [symbolic = %.loc7_17 (constants.%.657)]
// CHECK:STDOUT:   %impl.elem3.loc7_17.2: @G.%.loc7_17 (%.657) = impl_witness_access %Transitive.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc7_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_17.2: <specific function> = specific_impl_function %impl.elem3.loc7_17.2, @Transitive.WithSelf.GetA(%T.loc6_7.1) [symbolic = %specific_impl_fn.loc7_17.2 (constants.%specific_impl_fn.72f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc7_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value %as_type.loc7_11.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc7_3.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.5e1)]
// CHECK:STDOUT:   %.loc7_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc7_3.3 [symbolic = %.loc7_3.4 (constants.%.e75)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @G.%.loc7_3.4 (%.e75) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.678)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc7_3.3) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.e8e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @G.%T.as_type.loc6_25.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.var: ref @G.%as_type.loc7_11.2 (%as_type.6a024f.2) = var_storage %a.var_patt.loc7_3.1
// CHECK:STDOUT:     %t.ref: @G.%T.as_type.loc6_25.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %GetA.ref: %Transitive.assoc_type = name_ref GetA, imports.%Main.import_ref.5c5 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc7_17.1: @G.%.loc7_17 (%.657) = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element3 [symbolic = %impl.elem3.loc7_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc7_17: <bound method> = bound_method %t.ref, %impl.elem3.loc7_17.1
// CHECK:STDOUT:     %.loc7_23.1: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_23.2: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_17.1: <specific function> = specific_impl_function %impl.elem3.loc7_17.1, @Transitive.WithSelf.GetA(constants.%T) [symbolic = %specific_impl_fn.loc7_17.2 (constants.%specific_impl_fn.72f)]
// CHECK:STDOUT:     %bound_method.loc7_23: <bound method> = bound_method %t.ref, %specific_impl_fn.loc7_17.1
// CHECK:STDOUT:     %.loc7_3.1: ref @G.%as_type.loc7_11.2 (%as_type.6a024f.2) = splice_block %a.var {}
// CHECK:STDOUT:     %Transitive.WithSelf.GetA.call: init @G.%as_type.loc7_11.2 (%as_type.6a024f.2) to %.loc7_3.1 = call %bound_method.loc7_23(%t.ref)
// CHECK:STDOUT:     assign %a.var, %Transitive.WithSelf.GetA.call
// CHECK:STDOUT:     %.loc7_11.1: type = splice_block %.loc7_11.3 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)] {
// CHECK:STDOUT:       %T.ref.loc7: %Transitive.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7: type = facet_access_type %T.ref.loc7 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_11.2: type = converted %T.ref.loc7, %T.as_type.loc7 [symbolic = %T.as_type.loc6_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %A.ref: %Transitive.assoc_type = name_ref A, imports.%Main.import_ref.72c [concrete = constants.%assoc0.7cd]
// CHECK:STDOUT:       %impl.elem0.loc7_11.1: %facet_type.b1b = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic = %impl.elem0.loc7_11.2 (constants.%impl.elem0.0439e7.2)]
// CHECK:STDOUT:       %as_type.loc7_11.1: type = facet_access_type %impl.elem0.loc7_11.1 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)]
// CHECK:STDOUT:       %.loc7_11.3: type = converted %impl.elem0.loc7_11.1, %as_type.loc7_11.1 [symbolic = %as_type.loc7_11.2 (constants.%as_type.6a024f.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: ref @G.%as_type.loc7_11.2 (%as_type.6a024f.2) = wrapper_binding a, %a.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc7_8.1: @G.%pattern_type.loc7 (%pattern_type.c73ac0.2) = ref_binding_pattern a [symbolic = %a.patt.loc7_8.2 (constants.%a.patt)]
// CHECK:STDOUT:       %a.var_patt.loc7_3.1: @G.%pattern_type.loc7 (%pattern_type.c73ac0.2) = var_pattern %a.patt.loc7_8.1 [symbolic = %a.var_patt.loc7_3.2 (constants.%a.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.ref.loc13: ref @G.%as_type.loc7_11.2 (%as_type.6a024f.2) = name_ref a, %a
// CHECK:STDOUT:     %Q.ref: type = name_ref Q, imports.%Main.Q [concrete = constants.%Q.type]
// CHECK:STDOUT:     %InQ.ref: %Q.assoc_type = name_ref InQ, imports.%Main.import_ref.e2a [concrete = constants.%assoc0.6a1]
// CHECK:STDOUT:     %TakesPQR.ref: %TakesPQR.type = name_ref TakesPQR, file.%TakesPQR.decl [concrete = constants.%TakesPQR]
// CHECK:STDOUT:     %a.ref.loc22: ref @G.%as_type.loc7_11.2 (%as_type.6a024f.2) = name_ref a, %a
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @G.%.loc7_3.4 (%.e75) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.678)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%as_type.6a024f.2, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc7_3.2: %Destroy.type = converted constants.%as_type.6a024f.2, %Destroy.facet.loc7_3.1 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%as_type.6a024f.2, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc7_3.3: %Destroy.type = converted constants.%as_type.6a024f.2, %Destroy.facet.loc7_3.2 [symbolic = %Destroy.facet.loc7_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.e8e)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%a.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf(constants.%Self.104) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.104
// CHECK:STDOUT:   %P.WithSelf.InP.type => constants.%P.WithSelf.InP.type
// CHECK:STDOUT:   %P.WithSelf.InP => constants.%P.WithSelf.InP
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf.InP(constants.%Self.104) {
// CHECK:STDOUT:   %Self => constants.%Self.104
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.cc3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3ef
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.88d
// CHECK:STDOUT:   %self.patt => constants.%self.patt.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Self.afd) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.afd
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.afd) {
// CHECK:STDOUT:   %Self => constants.%Self.afd
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.430
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ade
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.271
// CHECK:STDOUT:   %self.patt => constants.%self.patt.b5e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%Self.be6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.be6
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%Self.be6) {
// CHECK:STDOUT:   %Self => constants.%Self.be6
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.b02
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.60e
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.640
// CHECK:STDOUT:   %self.patt => constants.%self.patt.894
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesPQR(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc5_14.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc5_14.1 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc5_38.1 => constants.%U.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.54d
// CHECK:STDOUT:   %u.param_patt.loc5_36.2 => constants.%u.param_patt
// CHECK:STDOUT:   %u.patt.loc5_36.2 => constants.%u.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%Self.744) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.0439e7.1
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.6be46c.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.0b3c58.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.e3e09c.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.d86d9f.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.f276cd.1
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.dca436.1
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.6a024f.1
// CHECK:STDOUT:   %eq.1 => constants.%eq.4be6e4.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.731f97.1
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.2fadd6.1
// CHECK:STDOUT:   %eq.2 => constants.%eq.033b19.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.744) {
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.1
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.731f97.1
// CHECK:STDOUT:   %as_type => constants.%as_type.2fadd6.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.ac0
// CHECK:STDOUT:   %c.param_patt => constants.%c.param_patt
// CHECK:STDOUT:   %c.patt => constants.%c.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.744) {
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.1
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.0439e7.1
// CHECK:STDOUT:   %as_type => constants.%as_type.6a024f.1
// CHECK:STDOUT:   %.1 => constants.%.2b1a45.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.c73ac0.1
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.057e89.1
// CHECK:STDOUT:   %return.patt => constants.%return.patt.4e0711.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc6_25.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc6 => constants.%pattern_type.68e728.2
// CHECK:STDOUT:   %t.param_patt.loc6_23.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc6_23.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.0439e7.2
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.6be46c.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.0b3c58.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.e3e09c.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.d86d9f.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.f276cd.2
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.dca436.2
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.6a024f.2
// CHECK:STDOUT:   %eq.1 => constants.%eq.4be6e4.2
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.731f97.2
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.2fadd6.2
// CHECK:STDOUT:   %eq.2 => constants.%eq.033b19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.2
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.2
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.2
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.0439e7.2
// CHECK:STDOUT:   %as_type => constants.%as_type.6a024f.2
// CHECK:STDOUT:   %.1 => constants.%.2b1a45.2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.c73ac0.2
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.057e89.2
// CHECK:STDOUT:   %return.patt => constants.%return.patt.4e0711.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_observe_in_both_scopes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.744: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.ab0b8d.1: <witness> = lookup_impl_witness %Self.744, @Transitive [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.be6: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R.WithSelf(%Self.be6) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR: %R.WithSelf.InR.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.b02: type = facet_access_type %Self.be6 [symbolic]
// CHECK:STDOUT:   %pattern_type.60e: type = pattern_type %Self.as_type.b02 [symbolic]
// CHECK:STDOUT:   %self.param_patt.640: %pattern_type.60e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.894: %pattern_type.60e = at_binding_pattern self, %self.param_patt.640 [symbolic]
// CHECK:STDOUT:   %facet_type.0e9: type = facet_type <@Destroy & @R where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem2.279493.1: %facet_type.0e9 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element2 [symbolic]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.afd: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Self.afd) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ: %Q.WithSelf.InQ.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.430: type = facet_access_type %Self.afd [symbolic]
// CHECK:STDOUT:   %pattern_type.ade: type = pattern_type %Self.as_type.430 [symbolic]
// CHECK:STDOUT:   %self.param_patt.271: %pattern_type.ade = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b5e: %pattern_type.ade = at_binding_pattern self, %self.param_patt.271 [symbolic]
// CHECK:STDOUT:   %facet_type.1b9: type = facet_type <@Destroy & @Q where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem1.1413d6.1: %facet_type.1b9 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element1 [symbolic]
// CHECK:STDOUT:   %eq.be77bc.1 = observe_equivalent %impl.elem1.1413d6.1, %impl.elem2.279493.1 [symbolic]
// CHECK:STDOUT:   %as_type.b2aa43.1: type = facet_access_type %impl.elem2.279493.1 [symbolic]
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.104: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P.WithSelf(%Self.104) [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.cc3: type = facet_access_type %Self.104 [symbolic]
// CHECK:STDOUT:   %pattern_type.3ef: type = pattern_type %Self.as_type.cc3 [symbolic]
// CHECK:STDOUT:   %self.param_patt.88d: %pattern_type.3ef = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.6de: %pattern_type.3ef = at_binding_pattern self, %self.param_patt.88d [symbolic]
// CHECK:STDOUT:   %facet_type.1ff: type = facet_type <@Destroy & @P> [concrete]
// CHECK:STDOUT:   %impl.elem0.8aa205.1: %facet_type.1ff = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.1, element0 [symbolic]
// CHECK:STDOUT:   %eq.20ba2e.1 = observe_equivalent %impl.elem0.8aa205.1, %impl.elem1.1413d6.1 [symbolic]
// CHECK:STDOUT:   %as_type.a0914b.1: type = facet_access_type %impl.elem0.8aa205.1 [symbolic]
// CHECK:STDOUT:   %as_type.f86e2d.1: type = facet_access_type %impl.elem1.1413d6.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.d86d9f.1: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%Self.744) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.f276cd.1: %Transitive.WithSelf.TakesC.type.d86d9f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.56c: type = facet_access_type %Self.744 [symbolic]
// CHECK:STDOUT:   %pattern_type.68e728.1: type = pattern_type %Self.as_type.56c [symbolic]
// CHECK:STDOUT:   %self.param_patt.e90fff.1: %pattern_type.68e728.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d2dab3.1: %pattern_type.68e728.1 = at_binding_pattern self, %self.param_patt.e90fff.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.d96bbe.1: type = pattern_type %as_type.b2aa43.1 [symbolic]
// CHECK:STDOUT:   %c.param_patt.b7bdaf.1: %pattern_type.d96bbe.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %c.patt.464614.1: %pattern_type.d96bbe.1 = at_binding_pattern c, %c.param_patt.b7bdaf.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.0b3c58.1: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%Self.744) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.e3e09c.1: %Transitive.WithSelf.GetA.type.0b3c58.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.77f685.1: type = pattern_type %as_type.a0914b.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.11f0b9.1: %pattern_type.77f685.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.87a9a6.1: %pattern_type.77f685.1 = return_slot_pattern %return.param_patt.11f0b9.1, invalid [symbolic]
// CHECK:STDOUT:   %.3ef823.1: Core.Form = init_form %as_type.a0914b.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.bd9: type = pattern_type %Transitive.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Transitive.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.68e728.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.68e728.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.68e728.2 = at_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fbd: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.ab0b8d.2: <witness> = lookup_impl_witness %T, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.8aa205.2: %facet_type.1ff = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.1413d6.2: %facet_type.1b9 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.0b3c58.2: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.e3e09c.2: %Transitive.WithSelf.GetA.type.0b3c58.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.d86d9f.2: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.f276cd.2: %Transitive.WithSelf.TakesC.type.d86d9f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %as_type.f86e2d.2: type = facet_access_type %impl.elem1.1413d6.2 [symbolic]
// CHECK:STDOUT:   %as_type.a0914b.2: type = facet_access_type %impl.elem0.8aa205.2 [symbolic]
// CHECK:STDOUT:   %eq.20ba2e.2 = observe_equivalent %impl.elem0.8aa205.2, %impl.elem1.1413d6.2 [symbolic]
// CHECK:STDOUT:   %impl.elem2.279493.2: %facet_type.0e9 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element2 [symbolic]
// CHECK:STDOUT:   %as_type.b2aa43.2: type = facet_access_type %impl.elem2.279493.2 [symbolic]
// CHECK:STDOUT:   %eq.be77bc.2 = observe_equivalent %impl.elem1.1413d6.2, %impl.elem2.279493.2 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.e77: %Transitive.assoc_type = assoc_entity element0, imports.%Main.import_ref.0a7 [concrete]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, imports.%Main.import_ref.dae [concrete]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, imports.%Main.import_ref.56a [concrete]
// CHECK:STDOUT:   %require_complete.6cf: <witness> = require_complete_type %as_type.a0914b.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.77f685.2: type = pattern_type %as_type.a0914b.2 [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.77f685.2 = ref_binding_pattern a [symbolic]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.77f685.2 = var_pattern %a.patt [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, imports.%Main.import_ref.06c [concrete]
// CHECK:STDOUT:   %.657: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type.0b3c58.2, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.657 = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element3 [symbolic]
// CHECK:STDOUT:   %self.param_patt.e90fff.2: %pattern_type.68e728.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d2dab3.2: %pattern_type.68e728.2 = at_binding_pattern self, %self.param_patt.e90fff.2 [symbolic]
// CHECK:STDOUT:   %.3ef823.2: Core.Form = init_form %as_type.a0914b.2 [symbolic]
// CHECK:STDOUT:   %return.param_patt.11f0b9.2: %pattern_type.77f685.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.87a9a6.2: %pattern_type.77f685.2 = return_slot_pattern %return.param_patt.11f0b9.2, invalid [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.72f: <specific function> = specific_impl_function %impl.elem3, @Transitive.WithSelf.GetA(%T) [symbolic]
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, imports.%Main.import_ref.034 [concrete]
// CHECK:STDOUT:   %.4ff: type = fn_type_with_self_type %Transitive.WithSelf.TakesC.type.d86d9f.2, %T [symbolic]
// CHECK:STDOUT:   %impl.elem4: %.4ff = impl_witness_access %Transitive.lookup_impl_witness.ab0b8d.2, element4 [symbolic]
// CHECK:STDOUT:   %as_type.84f: %facet_type.0e9 = facet_access_type %impl.elem0.8aa205.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.d96bbe.2: type = pattern_type %as_type.b2aa43.2 [symbolic]
// CHECK:STDOUT:   %c.param_patt.b7bdaf.2: %pattern_type.d96bbe.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %c.patt.464614.2: %pattern_type.d96bbe.2 = at_binding_pattern c, %c.param_patt.b7bdaf.2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.183: <specific function> = specific_impl_function %impl.elem4, @Transitive.WithSelf.TakesC(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.6e3: <witness> = require_complete_type %as_type.b2aa43.2 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.8aa205.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.a0914b.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.74e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.569: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.74e, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.95f: %.569 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.584: <specific function> = specific_impl_function %impl.elem0.95f, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.P = import_ref Main//transitive, P, unloaded
// CHECK:STDOUT:   %Main.Q = import_ref Main//transitive, Q, unloaded
// CHECK:STDOUT:   %Main.R = import_ref Main//transitive, R, unloaded
// CHECK:STDOUT:   %Main.Transitive: type = import_ref Main//transitive, Transitive, loaded [concrete = constants.%Transitive.type]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.72c: %Transitive.assoc_type = import_ref Main//transitive, loc8_8, loaded [concrete = constants.%assoc0.e77]
// CHECK:STDOUT:   %Main.import_ref.e5e: %Transitive.assoc_type = import_ref Main//transitive, loc9_8, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Main.import_ref.c9c: %Transitive.assoc_type = import_ref Main//transitive, loc10_8, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Main.import_ref.5c5: %Transitive.assoc_type = import_ref Main//transitive, loc12_21, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Main.import_ref.8a0: %Transitive.assoc_type = import_ref Main//transitive, loc13_24, loaded [concrete = constants.%assoc4]
// CHECK:STDOUT:   %Main.A = import_ref Main//transitive, A, unloaded
// CHECK:STDOUT:   %Main.B = import_ref Main//transitive, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//transitive, C, unloaded
// CHECK:STDOUT:   %Main.GetA = import_ref Main//transitive, GetA, unloaded
// CHECK:STDOUT:   %Main.TakesC = import_ref Main//transitive, TakesC, unloaded
// CHECK:STDOUT:   %Main.import_ref.0ce = import_ref Main//transitive, loc5_27, unloaded
// CHECK:STDOUT:   %Main.InR = import_ref Main//transitive, InR, unloaded
// CHECK:STDOUT:   %Main.import_ref.f6968b.2: %R.type = import_ref Main//transitive, loc5_13, loaded [symbolic = constants.%Self.be6]
// CHECK:STDOUT:   %Main.import_ref.659 = import_ref Main//transitive, loc5_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.849 = import_ref Main//transitive, loc4_27, unloaded
// CHECK:STDOUT:   %Main.InQ = import_ref Main//transitive, InQ, unloaded
// CHECK:STDOUT:   %Main.import_ref.c0a1f5.2: %Q.type = import_ref Main//transitive, loc4_13, loaded [symbolic = constants.%Self.afd]
// CHECK:STDOUT:   %Main.import_ref.4f3 = import_ref Main//transitive, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.fec = import_ref Main//transitive, loc3_27, unloaded
// CHECK:STDOUT:   %Main.InP = import_ref Main//transitive, InP, unloaded
// CHECK:STDOUT:   %Main.import_ref.aef2fa.2: %P.type = import_ref Main//transitive, loc3_13, loaded [symbolic = constants.%Self.104]
// CHECK:STDOUT:   %Main.import_ref.96f = import_ref Main//transitive, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.df6636.2: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.744]
// CHECK:STDOUT:   %Main.import_ref.df6636.3: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.744]
// CHECK:STDOUT:   %Main.import_ref.0d5 = import_ref Main//transitive, loc7_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.0a7: %facet_type.1ff = import_ref Main//transitive, loc8_8, loaded [concrete = %A]
// CHECK:STDOUT:   %A: %facet_type.1ff = assoc_const_decl @A [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.dae: %facet_type.1b9 = import_ref Main//transitive, loc9_8, loaded [concrete = %B]
// CHECK:STDOUT:   %B: %facet_type.1b9 = assoc_const_decl @B [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.56a: %facet_type.0e9 = import_ref Main//transitive, loc10_8, loaded [concrete = %C]
// CHECK:STDOUT:   %C: %facet_type.0e9 = assoc_const_decl @C [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.06c: @Transitive.WithSelf.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type.0b3c58.1) = import_ref Main//transitive, loc12_21, loaded [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA.e3e09c.1)]
// CHECK:STDOUT:   %Main.import_ref.034: @Transitive.WithSelf.%Transitive.WithSelf.TakesC.type (%Transitive.WithSelf.TakesC.type.d86d9f.1) = import_ref Main//transitive, loc13_24, loaded [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.TakesC (constants.%Transitive.WithSelf.TakesC.f276cd.1)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .P = imports.%Main.P
// CHECK:STDOUT:     .Q = imports.%Main.Q
// CHECK:STDOUT:     .R = imports.%Main.R
// CHECK:STDOUT:     .Transitive = imports.%Main.Transitive
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc5_7.1: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc5_23.1: @G.%pattern_type.loc5 (%pattern_type.68e728.2) = value_param_pattern [symbolic = %t.param_patt.loc5_23.2 (constants.%t.param_patt)]
// CHECK:STDOUT:     %t.patt.loc5_23.1: @G.%pattern_type.loc5 (%pattern_type.68e728.2) = at_binding_pattern t, %t.param_patt.loc5_23.1 [symbolic = %t.patt.loc5_23.2 (constants.%t.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_10: type = splice_block %Transitive.ref [concrete = constants.%Transitive.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Transitive.ref: type = name_ref Transitive, imports.%Main.Transitive [concrete = constants.%Transitive.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_7.2: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @G.%T.as_type.loc5_25.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_25.1: type = splice_block %.loc5_25.2 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc5: %Transitive.type = name_ref T, %T.loc5_7.2 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc5_25.2: type = facet_access_type %T.ref.loc5 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc5_25.2: type = converted %T.ref.loc5, %T.as_type.loc5_25.2 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.as_type.loc5_25.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.0d5
// CHECK:STDOUT:   .A = imports.%Main.import_ref.72c
// CHECK:STDOUT:   .B = imports.%Main.import_ref.e5e
// CHECK:STDOUT:   .C = imports.%Main.import_ref.c9c
// CHECK:STDOUT:   .GetA = imports.%Main.import_ref.5c5
// CHECK:STDOUT:   .TakesC = imports.%Main.import_ref.8a0
// CHECK:STDOUT:   witness = (imports.%Main.A, imports.%Main.B, imports.%Main.C, imports.%Main.GetA, imports.%Main.TakesC)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT:   @Transitive.observe {
// CHECK:STDOUT:     observe constants.%impl.elem0.8aa205.1 == constants.%impl.elem1.1413d6.1 == constants.%impl.elem2.279493.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.659
// CHECK:STDOUT:   .InR = imports.%Main.import_ref.0ce
// CHECK:STDOUT:   witness = (imports.%Main.InR)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.4f3
// CHECK:STDOUT:   .InQ = imports.%Main.import_ref.849
// CHECK:STDOUT:   witness = (imports.%Main.InQ)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @P [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.96f
// CHECK:STDOUT:   .InP = imports.%Main.import_ref.fec
// CHECK:STDOUT:   witness = (imports.%Main.InP)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(imports.%Main.import_ref.f6968b.2: %R.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.be6)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.b02)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.60e)]
// CHECK:STDOUT:   %self.param_patt: @R.WithSelf.InR.%pattern_type (%pattern_type.60e) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.640)]
// CHECK:STDOUT:   %self.patt: @R.WithSelf.InR.%pattern_type (%pattern_type.60e) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.894)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(imports.%Main.import_ref.c0a1f5.2: %Q.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.afd)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.430)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.ade)]
// CHECK:STDOUT:   %self.param_patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.ade) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.271)]
// CHECK:STDOUT:   %self.patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.ade) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.b5e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.WithSelf.InP(imports.%Main.import_ref.aef2fa.2: %P.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.104)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.cc3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.3ef)]
// CHECK:STDOUT:   %self.param_patt: @P.WithSelf.InP.%pattern_type (%pattern_type.3ef) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.88d)]
// CHECK:STDOUT:   %self.patt: @P.WithSelf.InP.%pattern_type (%pattern_type.3ef) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(imports.%Main.import_ref.df6636.2: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.744)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.68e728.1)]
// CHECK:STDOUT:   %self.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.1 (%pattern_type.68e728.1) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.e90fff.1)]
// CHECK:STDOUT:   %self.patt: @Transitive.WithSelf.TakesC.%pattern_type.1 (%pattern_type.68e728.1) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.d2dab3.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.1)]
// CHECK:STDOUT:   %impl.elem2: %facet_type.0e9 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2.279493.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2 [symbolic = %as_type (constants.%as_type.b2aa43.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.d96bbe.1)]
// CHECK:STDOUT:   %c.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.2 (%pattern_type.d96bbe.1) = value_param_pattern [symbolic = %c.param_patt (constants.%c.param_patt.b7bdaf.1)]
// CHECK:STDOUT:   %c.patt: @Transitive.WithSelf.TakesC.%pattern_type.2 (%pattern_type.d96bbe.1) = at_binding_pattern c, %c.param_patt [symbolic = %c.patt (constants.%c.patt.464614.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(imports.%Main.import_ref.df6636.3: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.744)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.56c)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.68e728.1)]
// CHECK:STDOUT:   %self.param_patt: @Transitive.WithSelf.GetA.%pattern_type.1 (%pattern_type.68e728.1) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.e90fff.1)]
// CHECK:STDOUT:   %self.patt: @Transitive.WithSelf.GetA.%pattern_type.1 (%pattern_type.68e728.1) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.d2dab3.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.1)]
// CHECK:STDOUT:   %impl.elem0: %facet_type.1ff = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.8aa205.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic = %as_type (constants.%as_type.a0914b.1)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %as_type [symbolic = %.1 (constants.%.3ef823.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.77f685.1)]
// CHECK:STDOUT:   %return.param_patt: @Transitive.WithSelf.GetA.%pattern_type.2 (%pattern_type.77f685.1) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.11f0b9.1)]
// CHECK:STDOUT:   %return.patt: @Transitive.WithSelf.GetA.%pattern_type.2 (%pattern_type.77f685.1) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.87a9a6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc5_7.2: %Transitive.type) {
// CHECK:STDOUT:   %T.patt.loc5_7.2: %pattern_type.bd9 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_7.1: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc5_25.1: type = facet_access_type %T.loc5_7.1 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5: type = pattern_type %T.as_type.loc5_25.1 [symbolic = %pattern_type.loc5 (constants.%pattern_type.68e728.2)]
// CHECK:STDOUT:   %t.param_patt.loc5_23.2: @G.%pattern_type.loc5 (%pattern_type.68e728.2) = value_param_pattern [symbolic = %t.param_patt.loc5_23.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc5_23.2: @G.%pattern_type.loc5 (%pattern_type.68e728.2) = at_binding_pattern t, %t.param_patt.loc5_23.2 [symbolic = %t.patt.loc5_23.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5: <witness> = require_complete_type %T.as_type.loc5_25.1 [symbolic = %require_complete.loc5 (constants.%require_complete.fbd)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc5_7.1, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.ab0b8d.2)]
// CHECK:STDOUT:   %impl.elem0.loc6_12.2: %facet_type.1ff = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_12.2 (constants.%impl.elem0.8aa205.2)]
// CHECK:STDOUT:   %impl.elem1.loc6_19.2: %facet_type.1b9 = impl_witness_access %Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc6_19.2 (constants.%impl.elem1.1413d6.2)]
// CHECK:STDOUT:   %as_type.loc6_19.3: type = facet_access_type %impl.elem1.loc6_19.2 [symbolic = %as_type.loc6_19.3 (constants.%as_type.f86e2d.2)]
// CHECK:STDOUT:   %as_type.loc6_12.2: type = facet_access_type %impl.elem0.loc6_12.2 [symbolic = %as_type.loc6_12.2 (constants.%as_type.a0914b.2)]
// CHECK:STDOUT:   %eq.loc6_15 = observe_equivalent %impl.elem0.loc6_12.2, %impl.elem1.loc6_19.2 [symbolic = %eq.loc6_15 (constants.%eq.20ba2e.2)]
// CHECK:STDOUT:   %impl.elem2.loc6_26.2: %facet_type.0e9 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc6_26.2 (constants.%impl.elem2.279493.2)]
// CHECK:STDOUT:   %as_type.loc6_26.2: type = facet_access_type %impl.elem2.loc6_26.2 [symbolic = %as_type.loc6_26.2 (constants.%as_type.b2aa43.2)]
// CHECK:STDOUT:   %eq.loc6_22 = observe_equivalent %impl.elem1.loc6_19.2, %impl.elem2.loc6_26.2 [symbolic = %eq.loc6_22 (constants.%eq.be77bc.2)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %as_type.loc6_12.2 [symbolic = %require_complete.loc8 (constants.%require_complete.6cf)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %as_type.loc6_12.2 [symbolic = %pattern_type.loc8 (constants.%pattern_type.77f685.2)]
// CHECK:STDOUT:   %a.patt.loc8_8.2: @G.%pattern_type.loc8 (%pattern_type.77f685.2) = ref_binding_pattern a [symbolic = %a.patt.loc8_8.2 (constants.%a.patt)]
// CHECK:STDOUT:   %a.var_patt.loc8_3.2: @G.%pattern_type.loc8 (%pattern_type.77f685.2) = var_pattern %a.patt.loc8_8.2 [symbolic = %a.var_patt.loc8_3.2 (constants.%a.var_patt)]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T.loc5_7.1) [symbolic = %Transitive.WithSelf.GetA.type (constants.%Transitive.WithSelf.GetA.type.0b3c58.2)]
// CHECK:STDOUT:   %.loc8_17: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type, %T.loc5_7.1 [symbolic = %.loc8_17 (constants.%.657)]
// CHECK:STDOUT:   %impl.elem3.loc8_17.2: @G.%.loc8_17 (%.657) = impl_witness_access %Transitive.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc8_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_17.2: <specific function> = specific_impl_function %impl.elem3.loc8_17.2, @Transitive.WithSelf.GetA(%T.loc5_7.1) [symbolic = %specific_impl_fn.loc8_17.2 (constants.%specific_impl_fn.72f)]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%T.loc5_7.1) [symbolic = %Transitive.WithSelf.TakesC.type (constants.%Transitive.WithSelf.TakesC.type.d86d9f.2)]
// CHECK:STDOUT:   %.loc9_4: type = fn_type_with_self_type %Transitive.WithSelf.TakesC.type, %T.loc5_7.1 [symbolic = %.loc9_4 (constants.%.4ff)]
// CHECK:STDOUT:   %impl.elem4.loc9_4.2: @G.%.loc9_4 (%.4ff) = impl_witness_access %Transitive.lookup_impl_witness, element4 [symbolic = %impl.elem4.loc9_4.2 (constants.%impl.elem4)]
// CHECK:STDOUT:   %as_type.loc9: %facet_type.0e9 = facet_access_type %impl.elem0.loc6_12.2 [symbolic = %as_type.loc9 (constants.%as_type.84f)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_4.2: <specific function> = specific_impl_function %impl.elem4.loc9_4.2, @Transitive.WithSelf.TakesC(%T.loc5_7.1) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn.183)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %as_type.loc6_26.2 [symbolic = %require_complete.loc9 (constants.%require_complete.6e3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc6_12.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.3: %Destroy.type = facet_value %as_type.loc6_12.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc8_3.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.74e)]
// CHECK:STDOUT:   %.loc8_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc8_3.3 [symbolic = %.loc8_3.4 (constants.%.569)]
// CHECK:STDOUT:   %impl.elem0.loc8_3.2: @G.%.loc8_3.4 (%.569) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.95f)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_3.2: <specific function> = specific_impl_function %impl.elem0.loc8_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc8_3.3) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.584)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @G.%T.as_type.loc5_25.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc6_11: %Transitive.type = name_ref T, %T.loc5_7.2 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc6_12: type = facet_access_type %T.ref.loc6_11 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc6_12.1: type = converted %T.ref.loc6_11, %T.as_type.loc6_12 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %A.ref.loc6: %Transitive.assoc_type = name_ref A, imports.%Main.import_ref.72c [concrete = constants.%assoc0.e77]
// CHECK:STDOUT:     %impl.elem0.loc6_12.1: %facet_type.1ff = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic = %impl.elem0.loc6_12.2 (constants.%impl.elem0.8aa205.2)]
// CHECK:STDOUT:     %T.ref.loc6_18: %Transitive.type = name_ref T, %T.loc5_7.2 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc6_19: type = facet_access_type %T.ref.loc6_18 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc6_19.1: type = converted %T.ref.loc6_18, %T.as_type.loc6_19 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %B.ref: %Transitive.assoc_type = name_ref B, imports.%Main.import_ref.e5e [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc6_19.1: %facet_type.1b9 = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element1 [symbolic = %impl.elem1.loc6_19.2 (constants.%impl.elem1.1413d6.2)]
// CHECK:STDOUT:     %as_type.loc6_19.1: type = facet_access_type %impl.elem1.loc6_19.1 [symbolic = %as_type.loc6_19.3 (constants.%as_type.f86e2d.2)]
// CHECK:STDOUT:     %.loc6_19.2: type = converted %impl.elem1.loc6_19.1, %as_type.loc6_19.1 [symbolic = %as_type.loc6_19.3 (constants.%as_type.f86e2d.2)]
// CHECK:STDOUT:     %as_type.loc6_12.1: type = facet_access_type %impl.elem0.loc6_12.1 [symbolic = %as_type.loc6_12.2 (constants.%as_type.a0914b.2)]
// CHECK:STDOUT:     %.loc6_12.2: type = converted %impl.elem0.loc6_12.1, %as_type.loc6_12.1 [symbolic = %as_type.loc6_12.2 (constants.%as_type.a0914b.2)]
// CHECK:STDOUT:     %T.ref.loc6_25: %Transitive.type = name_ref T, %T.loc5_7.2 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc6_26: type = facet_access_type %T.ref.loc6_25 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc6_26.1: type = converted %T.ref.loc6_25, %T.as_type.loc6_26 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %C.ref: %Transitive.assoc_type = name_ref C, imports.%Main.import_ref.c9c [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc6_26.1: %facet_type.0e9 = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element2 [symbolic = %impl.elem2.loc6_26.2 (constants.%impl.elem2.279493.2)]
// CHECK:STDOUT:     %as_type.loc6_26.1: type = facet_access_type %impl.elem2.loc6_26.1 [symbolic = %as_type.loc6_26.2 (constants.%as_type.b2aa43.2)]
// CHECK:STDOUT:     %.loc6_26.2: type = converted %impl.elem2.loc6_26.1, %as_type.loc6_26.1 [symbolic = %as_type.loc6_26.2 (constants.%as_type.b2aa43.2)]
// CHECK:STDOUT:     %as_type.loc6_19.2: type = facet_access_type %impl.elem1.loc6_19.1 [symbolic = %as_type.loc6_19.3 (constants.%as_type.f86e2d.2)]
// CHECK:STDOUT:     %.loc6_19.3: type = converted %impl.elem1.loc6_19.1, %as_type.loc6_19.2 [symbolic = %as_type.loc6_19.3 (constants.%as_type.f86e2d.2)]
// CHECK:STDOUT:     %G.observe.decl = observe_decl @G.observe [concrete]
// CHECK:STDOUT:     %a.var: ref @G.%as_type.loc6_12.2 (%as_type.a0914b.2) = var_storage %a.var_patt.loc8_3.1
// CHECK:STDOUT:     %t.ref.loc8: @G.%T.as_type.loc5_25.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %GetA.ref: %Transitive.assoc_type = name_ref GetA, imports.%Main.import_ref.5c5 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc8_17.1: @G.%.loc8_17 (%.657) = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element3 [symbolic = %impl.elem3.loc8_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc8_17: <bound method> = bound_method %t.ref.loc8, %impl.elem3.loc8_17.1
// CHECK:STDOUT:     %.loc8_23.1: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_23.2: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_17.1: <specific function> = specific_impl_function %impl.elem3.loc8_17.1, @Transitive.WithSelf.GetA(constants.%T) [symbolic = %specific_impl_fn.loc8_17.2 (constants.%specific_impl_fn.72f)]
// CHECK:STDOUT:     %bound_method.loc8_23: <bound method> = bound_method %t.ref.loc8, %specific_impl_fn.loc8_17.1
// CHECK:STDOUT:     %.loc8_3.1: ref @G.%as_type.loc6_12.2 (%as_type.a0914b.2) = splice_block %a.var {}
// CHECK:STDOUT:     %Transitive.WithSelf.GetA.call: init @G.%as_type.loc6_12.2 (%as_type.a0914b.2) to %.loc8_3.1 = call %bound_method.loc8_23(%t.ref.loc8)
// CHECK:STDOUT:     assign %a.var, %Transitive.WithSelf.GetA.call
// CHECK:STDOUT:     %.loc8_11.1: type = splice_block %.loc8_11.3 [symbolic = %as_type.loc6_12.2 (constants.%as_type.a0914b.2)] {
// CHECK:STDOUT:       %T.ref.loc8: %Transitive.type = name_ref T, %T.loc5_7.2 [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8: type = facet_access_type %T.ref.loc8 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_11.2: type = converted %T.ref.loc8, %T.as_type.loc8 [symbolic = %T.as_type.loc5_25.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %A.ref.loc8: %Transitive.assoc_type = name_ref A, imports.%Main.import_ref.72c [concrete = constants.%assoc0.e77]
// CHECK:STDOUT:       %impl.elem0.loc8_11: %facet_type.1ff = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element0 [symbolic = %impl.elem0.loc6_12.2 (constants.%impl.elem0.8aa205.2)]
// CHECK:STDOUT:       %as_type.loc8: type = facet_access_type %impl.elem0.loc8_11 [symbolic = %as_type.loc6_12.2 (constants.%as_type.a0914b.2)]
// CHECK:STDOUT:       %.loc8_11.3: type = converted %impl.elem0.loc8_11, %as_type.loc8 [symbolic = %as_type.loc6_12.2 (constants.%as_type.a0914b.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: ref @G.%as_type.loc6_12.2 (%as_type.a0914b.2) = wrapper_binding a, %a.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc8_8.1: @G.%pattern_type.loc8 (%pattern_type.77f685.2) = ref_binding_pattern a [symbolic = %a.patt.loc8_8.2 (constants.%a.patt)]
// CHECK:STDOUT:       %a.var_patt.loc8_3.1: @G.%pattern_type.loc8 (%pattern_type.77f685.2) = var_pattern %a.patt.loc8_8.1 [symbolic = %a.var_patt.loc8_3.2 (constants.%a.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t.ref.loc9: @G.%T.as_type.loc5_25.1 (%T.as_type) = name_ref t, %t
// CHECK:STDOUT:     %TakesC.ref: %Transitive.assoc_type = name_ref TakesC, imports.%Main.import_ref.8a0 [concrete = constants.%assoc4]
// CHECK:STDOUT:     %impl.elem4.loc9_4.1: @G.%.loc9_4 (%.4ff) = impl_witness_access constants.%Transitive.lookup_impl_witness.ab0b8d.2, element4 [symbolic = %impl.elem4.loc9_4.2 (constants.%impl.elem4)]
// CHECK:STDOUT:     %bound_method.loc9_4: <bound method> = bound_method %t.ref.loc9, %impl.elem4.loc9_4.1
// CHECK:STDOUT:     %a.ref: ref @G.%as_type.loc6_12.2 (%as_type.a0914b.2) = name_ref a, %a
// CHECK:STDOUT:     %.loc9_13.1: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc9_13.2: %Transitive.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc5_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc9_13.3: %facet_type.0e9 = as_compatible constants.%as_type.a0914b.2 [symbolic = %as_type.loc9 (constants.%as_type.84f)]
// CHECK:STDOUT:     %.loc9_13.4: %facet_type.0e9 = as_compatible constants.%as_type.a0914b.2 [symbolic = %as_type.loc9 (constants.%as_type.84f)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_4.1: <specific function> = specific_impl_function %impl.elem4.loc9_4.1, @Transitive.WithSelf.TakesC(constants.%T) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn.183)]
// CHECK:STDOUT:     %bound_method.loc9_13: <bound method> = bound_method %t.ref.loc9, %specific_impl_fn.loc9_4.1
// CHECK:STDOUT:     %.loc9_12: ref @G.%as_type.loc6_26.2 (%as_type.b2aa43.2) = as_compatible %a.ref
// CHECK:STDOUT:     %Transitive.WithSelf.TakesC.call: init %empty_tuple.type = call %bound_method.loc9_13(%t.ref.loc9, %.loc9_12)
// CHECK:STDOUT:     %impl.elem0.loc8_3.1: @G.%.loc8_3.4 (%.569) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.95f)]
// CHECK:STDOUT:     %bound_method.loc8_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc8_3.1
// CHECK:STDOUT:     %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%as_type.a0914b.2, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc8_3.2: %Destroy.type = converted constants.%as_type.a0914b.2, %Destroy.facet.loc8_3.1 [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%as_type.a0914b.2, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc8_3.3: %Destroy.type = converted constants.%as_type.a0914b.2, %Destroy.facet.loc8_3.2 [symbolic = %Destroy.facet.loc8_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_3.1: <specific function> = specific_impl_function %impl.elem0.loc8_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.584)]
// CHECK:STDOUT:     %bound_method.loc8_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc8_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc8_3.2(%a.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @G.observe {
// CHECK:STDOUT:       observe constants.%impl.elem0.8aa205.2 == constants.%impl.elem1.1413d6.2 == constants.%impl.elem2.279493.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%Self.be6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.be6
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%Self.be6) {
// CHECK:STDOUT:   %Self => constants.%Self.be6
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.b02
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.60e
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.640
// CHECK:STDOUT:   %self.patt => constants.%self.patt.894
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Self.afd) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.afd
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.afd) {
// CHECK:STDOUT:   %Self => constants.%Self.afd
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.430
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ade
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.271
// CHECK:STDOUT:   %self.patt => constants.%self.patt.b5e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf(constants.%Self.104) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.104
// CHECK:STDOUT:   %P.WithSelf.InP.type => constants.%P.WithSelf.InP.type
// CHECK:STDOUT:   %P.WithSelf.InP => constants.%P.WithSelf.InP
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf.InP(constants.%Self.104) {
// CHECK:STDOUT:   %Self => constants.%Self.104
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.cc3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3ef
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.88d
// CHECK:STDOUT:   %self.patt => constants.%self.patt.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%Self.744) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8aa205.1
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.1413d6.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.0b3c58.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.e3e09c.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.d86d9f.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.f276cd.1
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.f86e2d.1
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.a0914b.1
// CHECK:STDOUT:   %eq.1 => constants.%eq.20ba2e.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.279493.1
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.b2aa43.1
// CHECK:STDOUT:   %eq.2 => constants.%eq.be77bc.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.744) {
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.1
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.279493.1
// CHECK:STDOUT:   %as_type => constants.%as_type.b2aa43.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.d96bbe.1
// CHECK:STDOUT:   %c.param_patt => constants.%c.param_patt.b7bdaf.1
// CHECK:STDOUT:   %c.patt => constants.%c.patt.464614.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.744) {
// CHECK:STDOUT:   %Self => constants.%Self.744
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.56c
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.1
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8aa205.1
// CHECK:STDOUT:   %as_type => constants.%as_type.a0914b.1
// CHECK:STDOUT:   %.1 => constants.%.3ef823.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.77f685.1
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.11f0b9.1
// CHECK:STDOUT:   %return.patt => constants.%return.patt.87a9a6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc5_25.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc5 => constants.%pattern_type.68e728.2
// CHECK:STDOUT:   %t.param_patt.loc5_23.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc5_23.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8aa205.2
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.1413d6.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.0b3c58.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.e3e09c.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.d86d9f.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.f276cd.2
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.f86e2d.2
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.a0914b.2
// CHECK:STDOUT:   %eq.1 => constants.%eq.20ba2e.2
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.279493.2
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.b2aa43.2
// CHECK:STDOUT:   %eq.2 => constants.%eq.be77bc.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.2
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.2
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.2
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8aa205.2
// CHECK:STDOUT:   %as_type => constants.%as_type.a0914b.2
// CHECK:STDOUT:   %.1 => constants.%.3ef823.2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.77f685.2
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.11f0b9.2
// CHECK:STDOUT:   %return.patt => constants.%return.patt.87a9a6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.as_type => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.68e728.2
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.e90fff.2
// CHECK:STDOUT:   %self.patt => constants.%self.patt.d2dab3.2
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.ab0b8d.2
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.279493.2
// CHECK:STDOUT:   %as_type => constants.%as_type.b2aa43.2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.d96bbe.2
// CHECK:STDOUT:   %c.param_patt => constants.%c.param_patt.b7bdaf.2
// CHECK:STDOUT:   %c.patt => constants.%c.patt.464614.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- blanket_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.d0c: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.b45: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.62a: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = facet_type <@D> [concrete]
// CHECK:STDOUT:   %Self.e2d: %D.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt.a4b: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.626: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.3b8: type = facet_access_type %T.626 [symbolic]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness @T.as_type.as.B.impl.%B.impl_witness_table, @T.as_type.as.B.impl(%T.626) [symbolic]
// CHECK:STDOUT:   %B.facet.c04: %B.type = facet_value %T.as_type.3b8, (%B.impl_witness) [symbolic]
// CHECK:STDOUT:   %pattern_type.7b5: type = pattern_type %B.type [concrete]
// CHECK:STDOUT:   %T.patt.ca5: %pattern_type.7b5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.ed4: %B.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.4de: type = facet_access_type %T.ed4 [symbolic]
// CHECK:STDOUT:   %C.impl_witness.9e3: <witness> = impl_witness @T.as_type.as.C.impl.%C.impl_witness_table, @T.as_type.as.C.impl(%T.ed4) [symbolic]
// CHECK:STDOUT:   %C.facet.355: %C.type = facet_value %T.as_type.4de, (%C.impl_witness.9e3) [symbolic]
// CHECK:STDOUT:   %pattern_type.647: type = pattern_type %C.type [concrete]
// CHECK:STDOUT:   %T.patt.ece: %pattern_type.647 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.659: %C.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.c69: type = facet_access_type %T.659 [symbolic]
// CHECK:STDOUT:   %D.impl_witness.8a1: <witness> = impl_witness @T.as_type.as.D.impl.%D.impl_witness_table, @T.as_type.as.D.impl(%T.659) [symbolic]
// CHECK:STDOUT:   %D.facet.3d0: %D.type = facet_value %T.as_type.c69, (%D.impl_witness.8a1) [symbolic]
// CHECK:STDOUT:   %pattern_type.c89: type = pattern_type %T.as_type.4de [symbolic]
// CHECK:STDOUT:   %x.param_patt.1d2: %pattern_type.c89 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.3a5: %pattern_type.c89 = at_binding_pattern x, %x.param_patt.1d2 [symbolic]
// CHECK:STDOUT:   %RequiresB.type: type = fn_type @RequiresB [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %RequiresB: %RequiresB.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.346: <witness> = require_complete_type %T.as_type.4de [symbolic]
// CHECK:STDOUT:   %pattern_type.549: type = pattern_type %D.type [concrete]
// CHECK:STDOUT:   %T.patt.604: %pattern_type.549 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.3dd: %D.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.8cd: type = facet_access_type %T.3dd [symbolic]
// CHECK:STDOUT:   %pattern_type.86b: type = pattern_type %T.as_type.8cd [symbolic]
// CHECK:STDOUT:   %x.param_patt.2ab: %pattern_type.86b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.a712ca.1: %pattern_type.86b = at_binding_pattern x, %x.param_patt.2ab [symbolic]
// CHECK:STDOUT:   %x.patt.a712ca.2: %pattern_type.86b = at_binding_pattern x, %x.param_patt.2ab [symbolic]
// CHECK:STDOUT:   %RequiresD.type: type = fn_type @RequiresD [concrete]
// CHECK:STDOUT:   %RequiresD: %RequiresD.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.904: <witness> = require_complete_type %T.as_type.8cd [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %T.as_type.3b8 [symbolic]
// CHECK:STDOUT:   %x.param_patt.241: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.19b9f1.1: %pattern_type.a0e = at_binding_pattern x, %x.param_patt.241 [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.94f: <witness> = require_complete_type %T.as_type.3b8 [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.626, @B [symbolic]
// CHECK:STDOUT:   %.714: require_specific_def_type = require_specific_def @T.as_type.as.B.impl(%T.626) [symbolic]
// CHECK:STDOUT:   %B.facet.c85: %B.type = facet_value %T.as_type.3b8, (%B.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %x.patt.19b9f1.2: %pattern_type.a0e = at_binding_pattern x, %x.param_patt.241 [symbolic]
// CHECK:STDOUT:   %RequiresB.specific_fn: <specific function> = specific_function %RequiresB, @RequiresB(%B.facet.c85) [symbolic]
// CHECK:STDOUT:   %impls.be1 = observe_impls %T.626, %B.type [symbolic]
// CHECK:STDOUT:   %impls.86f = observe_impls %T.626, %C.type [symbolic]
// CHECK:STDOUT:   %D.lookup_impl_witness: <witness> = lookup_impl_witness %T.626, @D [symbolic]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %T.626, @C [symbolic]
// CHECK:STDOUT:   %C.impl_witness.f05: <witness> = impl_witness @T.as_type.as.C.impl.%C.impl_witness_table, @T.as_type.as.C.impl(%B.facet.c85) [symbolic]
// CHECK:STDOUT:   %C.facet.124: %C.type = facet_value %T.as_type.3b8, (%C.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %D.impl_witness.049: <witness> = impl_witness @T.as_type.as.D.impl.%D.impl_witness_table, @T.as_type.as.D.impl(%C.facet.124) [symbolic]
// CHECK:STDOUT:   %.f88: require_specific_def_type = require_specific_def @T.as_type.as.D.impl(%C.facet.124) [symbolic]
// CHECK:STDOUT:   %D.facet.5cf: %D.type = facet_value %T.as_type.3b8, (%D.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %RequiresD.specific_fn: <specific function> = specific_function %RequiresD, @RequiresD(%D.facet.5cf) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .RequiresB = %RequiresB.decl
// CHECK:STDOUT:     .RequiresD = %RequiresD.decl
// CHECK:STDOUT:     .RequiresA = %RequiresA.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   %D.decl: type = interface_decl @D [concrete = constants.%D.type] {} {}
// CHECK:STDOUT:   impl_decl @T.as_type.as.B.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt.a4b)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %A.type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T.626)]
// CHECK:STDOUT:     %T.as_type.loc8_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_21.2 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     %.loc8_21: type = converted %T.ref, %T.as_type.loc8_21.1 [symbolic = %T.as_type.loc8_21.2 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %.loc8_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T.626)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as_type.as.C.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc9_15.1: %pattern_type.7b5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt.ca5)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %B.type = name_ref T, %T.loc9_15.1 [symbolic = %T.loc9_15.2 (constants.%T.ed4)]
// CHECK:STDOUT:     %T.as_type.loc9_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_21.2 (constants.%T.as_type.4de)]
// CHECK:STDOUT:     %.loc9_21: type = converted %T.ref, %T.as_type.loc9_21.1 [symbolic = %T.as_type.loc9_21.2 (constants.%T.as_type.4de)]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     %.loc9_18: type = splice_block %B.ref [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_15.1: %B.type = symbolic_binding T, 0 [symbolic = %T.loc9_15.2 (constants.%T.ed4)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.as_type.as.D.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc10_15.1: %pattern_type.647 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_15.2 (constants.%T.patt.ece)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %C.type = name_ref T, %T.loc10_15.1 [symbolic = %T.loc10_15.2 (constants.%T.659)]
// CHECK:STDOUT:     %T.as_type.loc10_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_21.2 (constants.%T.as_type.c69)]
// CHECK:STDOUT:     %.loc10_21: type = converted %T.ref, %T.as_type.loc10_21.1 [symbolic = %T.as_type.loc10_21.2 (constants.%T.as_type.c69)]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D.type]
// CHECK:STDOUT:     %.loc10_18: type = splice_block %C.ref [concrete = constants.%C.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_15.1: %C.type = symbolic_binding T, 0 [symbolic = %T.loc10_15.2 (constants.%T.659)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RequiresB.decl: %RequiresB.type = fn_decl @RequiresB [concrete = constants.%RequiresB] {
// CHECK:STDOUT:     %T.patt.loc12_15.1: %pattern_type.7b5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt.ca5)]
// CHECK:STDOUT:     %x.param_patt.loc12_29.1: @RequiresB.%pattern_type (%pattern_type.c89) = value_param_pattern [symbolic = %x.param_patt.loc12_29.2 (constants.%x.param_patt.1d2)]
// CHECK:STDOUT:     %x.patt.loc12_29.1: @RequiresB.%pattern_type (%pattern_type.c89) = at_binding_pattern x, %x.param_patt.loc12_29.1 [symbolic = %x.patt.loc12_29.2 (constants.%x.patt.3a5)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_18: type = splice_block %B.ref [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_15.2: %B.type = symbolic_binding T, 0 [symbolic = %T.loc12_15.1 (constants.%T.ed4)]
// CHECK:STDOUT:     %x.param: @RequiresB.%T.as_type.loc12_31.1 (%T.as_type.4de) = value_param call_param0
// CHECK:STDOUT:     %.loc12_31.1: type = splice_block %.loc12_31.2 [symbolic = %T.as_type.loc12_31.1 (constants.%T.as_type.4de)] {
// CHECK:STDOUT:       %T.ref: %B.type = name_ref T, %T.loc12_15.2 [symbolic = %T.loc12_15.1 (constants.%T.ed4)]
// CHECK:STDOUT:       %T.as_type.loc12_31.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc12_31.1 (constants.%T.as_type.4de)]
// CHECK:STDOUT:       %.loc12_31.2: type = converted %T.ref, %T.as_type.loc12_31.2 [symbolic = %T.as_type.loc12_31.1 (constants.%T.as_type.4de)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresB.%T.as_type.loc12_31.1 (%T.as_type.4de) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RequiresD.decl: %RequiresD.type = fn_decl @RequiresD [concrete = constants.%RequiresD] {
// CHECK:STDOUT:     %T.patt.loc13_15.1: %pattern_type.549 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt.604)]
// CHECK:STDOUT:     %x.param_patt.loc13_29.1: @RequiresD.%pattern_type (%pattern_type.86b) = value_param_pattern [symbolic = %x.param_patt.loc13_29.2 (constants.%x.param_patt.2ab)]
// CHECK:STDOUT:     %x.patt.loc13_29.1: @RequiresD.%pattern_type (%pattern_type.86b) = at_binding_pattern x, %x.param_patt.loc13_29.1 [symbolic = %x.patt.loc13_29.2 (constants.%x.patt.a712ca.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_18: type = splice_block %D.ref [concrete = constants.%D.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_15.2: %D.type = symbolic_binding T, 0 [symbolic = %T.loc13_15.1 (constants.%T.3dd)]
// CHECK:STDOUT:     %x.param: @RequiresD.%T.as_type.loc13_31.1 (%T.as_type.8cd) = value_param call_param0
// CHECK:STDOUT:     %.loc13_31.1: type = splice_block %.loc13_31.2 [symbolic = %T.as_type.loc13_31.1 (constants.%T.as_type.8cd)] {
// CHECK:STDOUT:       %T.ref: %D.type = name_ref T, %T.loc13_15.2 [symbolic = %T.loc13_15.1 (constants.%T.3dd)]
// CHECK:STDOUT:       %T.as_type.loc13_31.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_31.1 (constants.%T.as_type.8cd)]
// CHECK:STDOUT:       %.loc13_31.2: type = converted %T.ref, %T.as_type.loc13_31.2 [symbolic = %T.as_type.loc13_31.1 (constants.%T.as_type.8cd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresD.%T.as_type.loc13_31.1 (%T.as_type.8cd) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt.loc15_15.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_15.2 (constants.%T.patt.a4b)]
// CHECK:STDOUT:     %x.param_patt.loc15_22.1: @RequiresA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc15_22.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:     %x.patt.loc15_22.1: @RequiresA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc15_22.1 [symbolic = %x.patt.loc15_22.2 (constants.%x.patt.19b9f1.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_15.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc15_15.1 (constants.%T.626)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.as_type.loc15_24.1 (%T.as_type.3b8) = value_param call_param0
// CHECK:STDOUT:     %.loc15_24.1: type = splice_block %.loc15_24.2 [symbolic = %T.as_type.loc15_24.1 (constants.%T.as_type.3b8)] {
// CHECK:STDOUT:       %T.ref.loc15: %A.type = name_ref T, %T.loc15_15.2 [symbolic = %T.loc15_15.1 (constants.%T.626)]
// CHECK:STDOUT:       %T.as_type.loc15_24.2: type = facet_access_type %T.ref.loc15 [symbolic = %T.as_type.loc15_24.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:       %.loc15_24.2: type = converted %T.ref.loc15, %T.as_type.loc15_24.2 [symbolic = %T.as_type.loc15_24.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.as_type.loc15_24.1 (%T.as_type.3b8) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0c]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b45]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.62a]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @D {
// CHECK:STDOUT:   %Self: %D.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e2d]
// CHECK:STDOUT:   %D.WithSelf.decl = interface_with_self_decl @D [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.B.impl(%T.loc8_15.1: %A.type) {
// CHECK:STDOUT:   %T.patt.loc8_15.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt.a4b)]
// CHECK:STDOUT:   %T.loc8_15.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T.626)]
// CHECK:STDOUT:   %T.as_type.loc8_21.2: type = facet_access_type %T.loc8_15.2 [symbolic = %T.as_type.loc8_21.2 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:   %B.impl_witness.loc8_28.2: <witness> = impl_witness %B.impl_witness_table, @T.as_type.as.B.impl(%T.loc8_15.2) [symbolic = %B.impl_witness.loc8_28.2 (constants.%B.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc8_21 as %B.ref {
// CHECK:STDOUT:     %B.impl_witness_table = impl_witness_table (), @T.as_type.as.B.impl [concrete]
// CHECK:STDOUT:     %B.impl_witness.loc8_28.1: <witness> = impl_witness %B.impl_witness_table, @T.as_type.as.B.impl(constants.%T.626) [symbolic = %B.impl_witness.loc8_28.2 (constants.%B.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:     witness = %B.impl_witness.loc8_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.C.impl(%T.loc9_15.1: %B.type) {
// CHECK:STDOUT:   %T.patt.loc9_15.2: %pattern_type.7b5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt.ca5)]
// CHECK:STDOUT:   %T.loc9_15.2: %B.type = symbolic_binding T, 0 [symbolic = %T.loc9_15.2 (constants.%T.ed4)]
// CHECK:STDOUT:   %T.as_type.loc9_21.2: type = facet_access_type %T.loc9_15.2 [symbolic = %T.as_type.loc9_21.2 (constants.%T.as_type.4de)]
// CHECK:STDOUT:   %C.impl_witness.loc9_28.2: <witness> = impl_witness %C.impl_witness_table, @T.as_type.as.C.impl(%T.loc9_15.2) [symbolic = %C.impl_witness.loc9_28.2 (constants.%C.impl_witness.9e3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc9_21 as %C.ref {
// CHECK:STDOUT:     %C.impl_witness_table = impl_witness_table (), @T.as_type.as.C.impl [concrete]
// CHECK:STDOUT:     %C.impl_witness.loc9_28.1: <witness> = impl_witness %C.impl_witness_table, @T.as_type.as.C.impl(constants.%T.ed4) [symbolic = %C.impl_witness.loc9_28.2 (constants.%C.impl_witness.9e3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %C.ref
// CHECK:STDOUT:     witness = %C.impl_witness.loc9_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as_type.as.D.impl(%T.loc10_15.1: %C.type) {
// CHECK:STDOUT:   %T.patt.loc10_15.2: %pattern_type.647 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_15.2 (constants.%T.patt.ece)]
// CHECK:STDOUT:   %T.loc10_15.2: %C.type = symbolic_binding T, 0 [symbolic = %T.loc10_15.2 (constants.%T.659)]
// CHECK:STDOUT:   %T.as_type.loc10_21.2: type = facet_access_type %T.loc10_15.2 [symbolic = %T.as_type.loc10_21.2 (constants.%T.as_type.c69)]
// CHECK:STDOUT:   %D.impl_witness.loc10_28.2: <witness> = impl_witness %D.impl_witness_table, @T.as_type.as.D.impl(%T.loc10_15.2) [symbolic = %D.impl_witness.loc10_28.2 (constants.%D.impl_witness.8a1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc10_21 as %D.ref {
// CHECK:STDOUT:     %D.impl_witness_table = impl_witness_table (), @T.as_type.as.D.impl [concrete]
// CHECK:STDOUT:     %D.impl_witness.loc10_28.1: <witness> = impl_witness %D.impl_witness_table, @T.as_type.as.D.impl(constants.%T.659) [symbolic = %D.impl_witness.loc10_28.2 (constants.%D.impl_witness.8a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %D.ref
// CHECK:STDOUT:     witness = %D.impl_witness.loc10_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresB(%T.loc12_15.2: %B.type) {
// CHECK:STDOUT:   %T.patt.loc12_15.2: %pattern_type.7b5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt.ca5)]
// CHECK:STDOUT:   %T.loc12_15.1: %B.type = symbolic_binding T, 0 [symbolic = %T.loc12_15.1 (constants.%T.ed4)]
// CHECK:STDOUT:   %T.as_type.loc12_31.1: type = facet_access_type %T.loc12_15.1 [symbolic = %T.as_type.loc12_31.1 (constants.%T.as_type.4de)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc12_31.1 [symbolic = %pattern_type (constants.%pattern_type.c89)]
// CHECK:STDOUT:   %x.param_patt.loc12_29.2: @RequiresB.%pattern_type (%pattern_type.c89) = value_param_pattern [symbolic = %x.param_patt.loc12_29.2 (constants.%x.param_patt.1d2)]
// CHECK:STDOUT:   %x.patt.loc12_29.2: @RequiresB.%pattern_type (%pattern_type.c89) = at_binding_pattern x, %x.param_patt.loc12_29.2 [symbolic = %x.patt.loc12_29.2 (constants.%x.patt.3a5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc12_31.1 [symbolic = %require_complete (constants.%require_complete.346)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresB.%T.as_type.loc12_31.1 (%T.as_type.4de)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresD(%T.loc13_15.2: %D.type) {
// CHECK:STDOUT:   %T.patt.loc13_15.2: %pattern_type.549 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt.604)]
// CHECK:STDOUT:   %T.loc13_15.1: %D.type = symbolic_binding T, 0 [symbolic = %T.loc13_15.1 (constants.%T.3dd)]
// CHECK:STDOUT:   %T.as_type.loc13_31.1: type = facet_access_type %T.loc13_15.1 [symbolic = %T.as_type.loc13_31.1 (constants.%T.as_type.8cd)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc13_31.1 [symbolic = %pattern_type (constants.%pattern_type.86b)]
// CHECK:STDOUT:   %x.param_patt.loc13_29.2: @RequiresD.%pattern_type (%pattern_type.86b) = value_param_pattern [symbolic = %x.param_patt.loc13_29.2 (constants.%x.param_patt.2ab)]
// CHECK:STDOUT:   %x.patt.loc13_29.2: @RequiresD.%pattern_type (%pattern_type.86b) = at_binding_pattern x, %x.param_patt.loc13_29.2 [symbolic = %x.patt.loc13_29.2 (constants.%x.patt.a712ca.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc13_31.1 [symbolic = %require_complete (constants.%require_complete.904)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresD.%T.as_type.loc13_31.1 (%T.as_type.8cd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc15_15.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc15_15.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_15.2 (constants.%T.patt.a4b)]
// CHECK:STDOUT:   %T.loc15_15.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc15_15.1 (constants.%T.626)]
// CHECK:STDOUT:   %T.as_type.loc15_24.1: type = facet_access_type %T.loc15_15.1 [symbolic = %T.as_type.loc15_24.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc15_24.1 [symbolic = %pattern_type (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %x.param_patt.loc15_22.2: @RequiresA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc15_22.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:   %x.patt.loc15_22.2: @RequiresA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc15_22.2 [symbolic = %x.patt.loc15_22.2 (constants.%x.patt.19b9f1.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc15_24.1 [symbolic = %require_complete (constants.%require_complete.94f)]
// CHECK:STDOUT:   %.loc16_14.3: require_specific_def_type = require_specific_def @T.as_type.as.B.impl(%T.loc15_15.1) [symbolic = %.loc16_14.3 (constants.%.714)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc15_15.1, @B [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness)]
// CHECK:STDOUT:   %B.facet.loc16_14.3: %B.type = facet_value %T.as_type.loc15_24.1, (%B.lookup_impl_witness) [symbolic = %B.facet.loc16_14.3 (constants.%B.facet.c85)]
// CHECK:STDOUT:   %RequiresB.specific_fn.loc16_3.2: <specific function> = specific_function constants.%RequiresB, @RequiresB(%B.facet.loc16_14.3) [symbolic = %RequiresB.specific_fn.loc16_3.2 (constants.%RequiresB.specific_fn)]
// CHECK:STDOUT:   %impls.loc18 = observe_impls %T.loc15_15.1, constants.%B.type [symbolic = %impls.loc18 (constants.%impls.be1)]
// CHECK:STDOUT:   %impls.loc19 = observe_impls %T.loc15_15.1, constants.%C.type [symbolic = %impls.loc19 (constants.%impls.86f)]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc15_15.1, @C [symbolic = %C.lookup_impl_witness (constants.%C.lookup_impl_witness)]
// CHECK:STDOUT:   %C.facet: %C.type = facet_value %T.as_type.loc15_24.1, (%C.lookup_impl_witness) [symbolic = %C.facet (constants.%C.facet.124)]
// CHECK:STDOUT:   %.loc20_14.3: require_specific_def_type = require_specific_def @T.as_type.as.D.impl(%C.facet) [symbolic = %.loc20_14.3 (constants.%.f88)]
// CHECK:STDOUT:   %D.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc15_15.1, @D [symbolic = %D.lookup_impl_witness (constants.%D.lookup_impl_witness)]
// CHECK:STDOUT:   %D.facet.loc20_14.3: %D.type = facet_value %T.as_type.loc15_24.1, (%D.lookup_impl_witness) [symbolic = %D.facet.loc20_14.3 (constants.%D.facet.5cf)]
// CHECK:STDOUT:   %RequiresD.specific_fn.loc20_3.2: <specific function> = specific_function constants.%RequiresD, @RequiresD(%D.facet.loc20_14.3) [symbolic = %RequiresD.specific_fn.loc20_3.2 (constants.%RequiresD.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.as_type.loc15_24.1 (%T.as_type.3b8)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RequiresB.ref: %RequiresB.type = name_ref RequiresB, file.%RequiresB.decl [concrete = constants.%RequiresB]
// CHECK:STDOUT:     %x.ref.loc16: @RequiresA.%T.as_type.loc15_24.1 (%T.as_type.3b8) = name_ref x, %x
// CHECK:STDOUT:     %B.facet.loc16_14.1: %B.type = facet_value constants.%T.as_type.3b8, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc16_14.3 (constants.%B.facet.c85)]
// CHECK:STDOUT:     %.loc16_14.1: %B.type = converted constants.%T.as_type.3b8, %B.facet.loc16_14.1 [symbolic = %B.facet.loc16_14.3 (constants.%B.facet.c85)]
// CHECK:STDOUT:     %B.facet.loc16_14.2: %B.type = facet_value constants.%T.as_type.3b8, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc16_14.3 (constants.%B.facet.c85)]
// CHECK:STDOUT:     %.loc16_14.2: %B.type = converted constants.%T.as_type.3b8, %B.facet.loc16_14.2 [symbolic = %B.facet.loc16_14.3 (constants.%B.facet.c85)]
// CHECK:STDOUT:     %RequiresB.specific_fn.loc16_3.1: <specific function> = specific_function %RequiresB.ref, @RequiresB(constants.%B.facet.c85) [symbolic = %RequiresB.specific_fn.loc16_3.2 (constants.%RequiresB.specific_fn)]
// CHECK:STDOUT:     %RequiresB.call: init %empty_tuple.type = call %RequiresB.specific_fn.loc16_3.1(%x.ref.loc16)
// CHECK:STDOUT:     %T.ref.loc18: %A.type = name_ref T, %T.loc15_15.2 [symbolic = %T.loc15_15.1 (constants.%T.626)]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %T.as_type.loc18: type = facet_access_type %T.ref.loc18 [symbolic = %T.as_type.loc15_24.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     %.loc18: type = converted %T.ref.loc18, %T.as_type.loc18 [symbolic = %T.as_type.loc15_24.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     %RequiresA.observe.decl.loc18 = observe_decl @RequiresA.observe.loc18 [concrete]
// CHECK:STDOUT:     %T.ref.loc19: %A.type = name_ref T, %T.loc15_15.2 [symbolic = %T.loc15_15.1 (constants.%T.626)]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     %T.as_type.loc19: type = facet_access_type %T.ref.loc19 [symbolic = %T.as_type.loc15_24.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     %.loc19: type = converted %T.ref.loc19, %T.as_type.loc19 [symbolic = %T.as_type.loc15_24.1 (constants.%T.as_type.3b8)]
// CHECK:STDOUT:     %RequiresA.observe.decl.loc19 = observe_decl @RequiresA.observe.loc19 [concrete]
// CHECK:STDOUT:     %RequiresD.ref: %RequiresD.type = name_ref RequiresD, file.%RequiresD.decl [concrete = constants.%RequiresD]
// CHECK:STDOUT:     %x.ref.loc20: @RequiresA.%T.as_type.loc15_24.1 (%T.as_type.3b8) = name_ref x, %x
// CHECK:STDOUT:     %D.facet.loc20_14.1: %D.type = facet_value constants.%T.as_type.3b8, (constants.%D.lookup_impl_witness) [symbolic = %D.facet.loc20_14.3 (constants.%D.facet.5cf)]
// CHECK:STDOUT:     %.loc20_14.1: %D.type = converted constants.%T.as_type.3b8, %D.facet.loc20_14.1 [symbolic = %D.facet.loc20_14.3 (constants.%D.facet.5cf)]
// CHECK:STDOUT:     %D.facet.loc20_14.2: %D.type = facet_value constants.%T.as_type.3b8, (constants.%D.lookup_impl_witness) [symbolic = %D.facet.loc20_14.3 (constants.%D.facet.5cf)]
// CHECK:STDOUT:     %.loc20_14.2: %D.type = converted constants.%T.as_type.3b8, %D.facet.loc20_14.2 [symbolic = %D.facet.loc20_14.3 (constants.%D.facet.5cf)]
// CHECK:STDOUT:     %RequiresD.specific_fn.loc20_3.1: <specific function> = specific_function %RequiresD.ref, @RequiresD(constants.%D.facet.5cf) [symbolic = %RequiresD.specific_fn.loc20_3.2 (constants.%RequiresD.specific_fn)]
// CHECK:STDOUT:     %RequiresD.call: init %empty_tuple.type = call %RequiresD.specific_fn.loc20_3.1(%x.ref.loc20)
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @RequiresA.observe.loc18 {
// CHECK:STDOUT:       observe constants.%T.626 impls constants.%B.type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     @RequiresA.observe.loc19 {
// CHECK:STDOUT:       observe constants.%T.626 impls constants.%C.type
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.d0c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.b45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.62a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.WithSelf(constants.%Self.e2d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.B.impl(constants.%T.626) {
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt.a4b
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T.626
// CHECK:STDOUT:   %T.as_type.loc8_21.2 => constants.%T.as_type.3b8
// CHECK:STDOUT:   %B.impl_witness.loc8_28.2 => constants.%B.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%B.facet.c04) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.C.impl(constants.%T.ed4) {
// CHECK:STDOUT:   %T.patt.loc9_15.2 => constants.%T.patt.ca5
// CHECK:STDOUT:   %T.loc9_15.2 => constants.%T.ed4
// CHECK:STDOUT:   %T.as_type.loc9_21.2 => constants.%T.as_type.4de
// CHECK:STDOUT:   %C.impl_witness.loc9_28.2 => constants.%C.impl_witness.9e3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%C.facet.355) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.D.impl(constants.%T.659) {
// CHECK:STDOUT:   %T.patt.loc10_15.2 => constants.%T.patt.ece
// CHECK:STDOUT:   %T.loc10_15.2 => constants.%T.659
// CHECK:STDOUT:   %T.as_type.loc10_21.2 => constants.%T.as_type.c69
// CHECK:STDOUT:   %D.impl_witness.loc10_28.2 => constants.%D.impl_witness.8a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.WithSelf(constants.%D.facet.3d0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresB(constants.%T.ed4) {
// CHECK:STDOUT:   %T.patt.loc12_15.2 => constants.%T.patt.ca5
// CHECK:STDOUT:   %T.loc12_15.1 => constants.%T.ed4
// CHECK:STDOUT:   %T.as_type.loc12_31.1 => constants.%T.as_type.4de
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c89
// CHECK:STDOUT:   %x.param_patt.loc12_29.2 => constants.%x.param_patt.1d2
// CHECK:STDOUT:   %x.patt.loc12_29.2 => constants.%x.patt.3a5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresD(constants.%T.3dd) {
// CHECK:STDOUT:   %T.patt.loc13_15.2 => constants.%T.patt.604
// CHECK:STDOUT:   %T.loc13_15.1 => constants.%T.3dd
// CHECK:STDOUT:   %T.as_type.loc13_31.1 => constants.%T.as_type.8cd
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.86b
// CHECK:STDOUT:   %x.param_patt.loc13_29.2 => constants.%x.param_patt.2ab
// CHECK:STDOUT:   %x.patt.loc13_29.2 => constants.%x.patt.a712ca.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T.626) {
// CHECK:STDOUT:   %T.patt.loc15_15.2 => constants.%T.patt.a4b
// CHECK:STDOUT:   %T.loc15_15.1 => constants.%T.626
// CHECK:STDOUT:   %T.as_type.loc15_24.1 => constants.%T.as_type.3b8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc15_22.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc15_22.2 => constants.%x.patt.19b9f1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresB(constants.%B.facet.c85) {
// CHECK:STDOUT:   %T.patt.loc12_15.2 => constants.%T.patt.ca5
// CHECK:STDOUT:   %T.loc12_15.1 => constants.%B.facet.c85
// CHECK:STDOUT:   %T.as_type.loc12_31.1 => constants.%T.as_type.3b8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc12_29.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc12_29.2 => constants.%x.patt.19b9f1.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.94f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.C.impl(constants.%B.facet.c85) {
// CHECK:STDOUT:   %T.patt.loc9_15.2 => constants.%T.patt.ca5
// CHECK:STDOUT:   %T.loc9_15.2 => constants.%B.facet.c85
// CHECK:STDOUT:   %T.as_type.loc9_21.2 => constants.%T.as_type.3b8
// CHECK:STDOUT:   %C.impl_witness.loc9_28.2 => constants.%C.impl_witness.f05
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as_type.as.D.impl(constants.%C.facet.124) {
// CHECK:STDOUT:   %T.patt.loc10_15.2 => constants.%T.patt.ece
// CHECK:STDOUT:   %T.loc10_15.2 => constants.%C.facet.124
// CHECK:STDOUT:   %T.as_type.loc10_21.2 => constants.%T.as_type.3b8
// CHECK:STDOUT:   %D.impl_witness.loc10_28.2 => constants.%D.impl_witness.049
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresD(constants.%D.facet.5cf) {
// CHECK:STDOUT:   %T.patt.loc13_15.2 => constants.%T.patt.604
// CHECK:STDOUT:   %T.loc13_15.1 => constants.%D.facet.5cf
// CHECK:STDOUT:   %T.as_type.loc13_31.1 => constants.%T.as_type.3b8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc13_29.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc13_29.2 => constants.%x.patt.19b9f1.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.94f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_functions_in_outer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.d0c: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.b45: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.62a: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.241: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.19b: %pattern_type.a0e = at_binding_pattern x, %x.param_patt.241 [symbolic]
// CHECK:STDOUT:   %TakesA.type: type = fn_type @TakesA [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesA: %TakesA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.94f: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.Self.frozen.f7e: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.263: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.8d7: %C.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.a25: %C.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %C_where.type: type = facet_type <@C where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.803: type = pattern_type %C_where.type [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic]
// CHECK:STDOUT:   %V: %C_where.type = symbolic_binding V, 2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %eq.bb6 = observe_equivalent %T, %U [symbolic]
// CHECK:STDOUT:   %V.as_type.38f: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %eq.51c = observe_equivalent %U, %V [symbolic]
// CHECK:STDOUT:   %pattern_type.0aa: type = pattern_type %V.as_type.38f [symbolic]
// CHECK:STDOUT:   %x.param_patt.d48: %pattern_type.0aa = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.764: %pattern_type.0aa = at_binding_pattern x, %x.param_patt.d48 [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF, @F(%T, %U, %V) [symbolic]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.269: <witness> = require_complete_type %V.as_type.38f [symbolic]
// CHECK:STDOUT:   %V.as_type.c9a: %A.type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %as_type: type = facet_access_type %V.as_type.c9a [symbolic]
// CHECK:STDOUT:   %pattern_type.bdf: type = pattern_type %as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.acb: %pattern_type.bdf = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.20d: %pattern_type.bdf = at_binding_pattern x, %x.param_patt.acb [symbolic]
// CHECK:STDOUT:   %TakesA.specific_fn: <specific function> = specific_function %TakesA, @TakesA(%V.as_type.c9a) [symbolic]
// CHECK:STDOUT:   %require_complete.379: <witness> = require_complete_type %as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TakesA = %TakesA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   %TakesA.decl: %TakesA.type = fn_decl @TakesA [concrete = constants.%TakesA] {
// CHECK:STDOUT:     %T.patt.loc7_12.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc7_26.1: @TakesA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_26.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:     %x.patt.loc7_26.1: @TakesA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_26.1 [symbolic = %x.patt.loc7_26.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_15: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_12.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesA.%T.as_type.loc7_28.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_28.1: type = splice_block %.loc7_28.2 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_12.2 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7_28.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_28.2: type = converted %T.ref, %T.as_type.loc7_28.2 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesA.%T.as_type.loc7_28.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc8_14.1: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt)]
// CHECK:STDOUT:     %V.patt.loc8_38.1: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic = %V.patt.loc8_38.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_19: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %base_facet_type.loc8_19 = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_25.1: %B.type = name_ref .Self, %.Self.frozen.loc8_19 [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %T.ref.loc8: %A.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %equiv.loc8_31.1 = requirement_equivalent %.Self.ref.loc8_25.1, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       %.Self.loc8_19: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.263]
// CHECK:STDOUT:       %.Self.ref.loc8_25.2: %B.type = name_ref .Self, %.Self.loc8_19 [symbolic_self = constants.%.Self.263]
// CHECK:STDOUT:       %equiv.loc8_31.2 = requirement_equivalent %.Self.ref.loc8_25.2, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         %base_facet_type.loc8_19 = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_31.2 = requirement_equivalent %.Self.ref.loc8_25.2, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_14.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:     %.loc8_43.1: type = splice_block %.loc8_43.2 [concrete = constants.%C_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_38: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_43: %C.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.8d7]
// CHECK:STDOUT:       %base_facet_type.loc8_43 = requirement_base_facet_type %C.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_49.1: %C.type = name_ref .Self, %.Self.frozen.loc8_43 [symbolic_self = constants.%.Self.frozen.8d7]
// CHECK:STDOUT:       %U.ref.loc8: %B_where.type = name_ref U, %U.loc8_14.2 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:       %equiv.loc8_55.1 = requirement_equivalent %.Self.ref.loc8_49.1, %U.ref.loc8 [concrete]
// CHECK:STDOUT:       %.Self.loc8_43: %C.type = symbolic_binding .Self [symbolic_self = constants.%.Self.a25]
// CHECK:STDOUT:       %.Self.ref.loc8_49.2: %C.type = name_ref .Self, %.Self.loc8_43 [symbolic_self = constants.%.Self.a25]
// CHECK:STDOUT:       %equiv.loc8_55.2 = requirement_equivalent %.Self.ref.loc8_49.2, %U.ref.loc8 [concrete]
// CHECK:STDOUT:       %.loc8_43.2: type = where_expr [concrete = constants.%C_where.type] {
// CHECK:STDOUT:         %base_facet_type.loc8_43 = requirement_base_facet_type %C.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_55.2 = requirement_equivalent %.Self.ref.loc8_49.2, %U.ref.loc8 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc8_38.2: %C_where.type = symbolic_binding V, 2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0c]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b45]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.62a]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesA(%T.loc7_12.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc7_12.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_12.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc7_28.1: type = facet_access_type %T.loc7_12.1 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_28.1 [symbolic = %pattern_type (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %x.param_patt.loc7_26.2: @TakesA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_26.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:   %x.patt.loc7_26.2: @TakesA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_26.2 [symbolic = %x.patt.loc7_26.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_28.1 [symbolic = %require_complete (constants.%require_complete.94f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TakesA.%T.as_type.loc7_28.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %A.type, %U.loc8_14.2: %B_where.type, %V.loc8_38.2: %C_where.type) {
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_7.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc8_14.2: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc8_14.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:   %V.patt.loc8_38.2: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic = %V.patt.loc8_38.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc8_38.1: %C_where.type = symbolic_binding V, 2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.as_type.loc9_16.3: type = facet_access_type %U.loc8_14.1 [symbolic = %U.as_type.loc9_16.3 (constants.%U.as_type)]
// CHECK:STDOUT:   %T.as_type.loc9_11.2: type = facet_access_type %T.loc8_7.1 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %eq.loc9_13 = observe_equivalent %T.loc8_7.1, %U.loc8_14.1 [symbolic = %eq.loc9_13 (constants.%eq.bb6)]
// CHECK:STDOUT:   %V.as_type.loc9_21.2: type = facet_access_type %V.loc8_38.1 [symbolic = %V.as_type.loc9_21.2 (constants.%V.as_type.38f)]
// CHECK:STDOUT:   %eq.loc9_18 = observe_equivalent %U.loc8_14.1, %V.loc8_38.1 [symbolic = %eq.loc9_18 (constants.%eq.51c)]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF, @F(%T.loc8_7.1, %U.loc8_14.1, %V.loc8_38.1) [symbolic = %InnerF.type (constants.%InnerF.type)]
// CHECK:STDOUT:   %InnerF: @F.%InnerF.type (%InnerF.type) = struct_value () [symbolic = %InnerF (constants.%InnerF)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc9: %A.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc9: %B_where.type = name_ref U, %U.loc8_14.2 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc9_16.1: type = facet_access_type %U.ref.loc9 [symbolic = %U.as_type.loc9_16.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc9_16.1: type = converted %U.ref.loc9, %U.as_type.loc9_16.1 [symbolic = %U.as_type.loc9_16.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %T.as_type.loc9_11.1: type = facet_access_type %T.ref.loc9 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_11: type = converted %T.ref.loc9, %T.as_type.loc9_11.1 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %V.ref: %C_where.type = name_ref V, %V.loc8_38.2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:     %V.as_type.loc9_21.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc9_21.2 (constants.%V.as_type.38f)]
// CHECK:STDOUT:     %.loc9_21: type = converted %V.ref, %V.as_type.loc9_21.1 [symbolic = %V.as_type.loc9_21.2 (constants.%V.as_type.38f)]
// CHECK:STDOUT:     %U.as_type.loc9_16.2: type = facet_access_type %U.ref.loc9 [symbolic = %U.as_type.loc9_16.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc9_16.2: type = converted %U.ref.loc9, %U.as_type.loc9_16.2 [symbolic = %U.as_type.loc9_16.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %F.observe.decl = observe_decl @F.observe [concrete]
// CHECK:STDOUT:     %InnerF.decl: @F.%InnerF.type (%InnerF.type) = fn_decl @InnerF [symbolic = @F.%InnerF (constants.%InnerF)] {
// CHECK:STDOUT:       %x.param_patt.loc11_14.1: @InnerF.%pattern_type (%pattern_type.0aa) = value_param_pattern [symbolic = %x.param_patt.loc11_14.2 (constants.%x.param_patt.d48)]
// CHECK:STDOUT:       %x.patt.loc11_14.1: @InnerF.%pattern_type (%pattern_type.0aa) = at_binding_pattern x, %x.param_patt.loc11_14.1 [symbolic = %x.patt.loc11_14.2 (constants.%x.patt.764)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%V.as_type.loc11_16.1 (%V.as_type.38f) = value_param call_param0
// CHECK:STDOUT:       %.loc11_16.1: type = splice_block %.loc11_16.2 [symbolic = %V.as_type.loc11_16.1 (constants.%V.as_type.38f)] {
// CHECK:STDOUT:         %V.ref: %C_where.type = name_ref V, @F.%V.loc8_38.2 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:         %V.as_type.loc11_16.2: type = facet_access_type %V.ref [symbolic = %V.as_type.loc11_16.1 (constants.%V.as_type.38f)]
// CHECK:STDOUT:         %.loc11_16.2: type = converted %V.ref, %V.as_type.loc11_16.2 [symbolic = %V.as_type.loc11_16.1 (constants.%V.as_type.38f)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%V.as_type.loc11_16.1 (%V.as_type.38f) = wrapper_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @F.observe {
// CHECK:STDOUT:       observe constants.%T == constants.%U == constants.%V
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc8_7.2: %A.type, @F.%U.loc8_14.2: %B_where.type, @F.%V.loc8_38.2: %C_where.type) {
// CHECK:STDOUT:   %V: %C_where.type = symbolic_binding V, 2 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc11_16.1: type = facet_access_type %V [symbolic = %V.as_type.loc11_16.1 (constants.%V.as_type.38f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %V.as_type.loc11_16.1 [symbolic = %pattern_type (constants.%pattern_type.0aa)]
// CHECK:STDOUT:   %x.param_patt.loc11_14.2: @InnerF.%pattern_type (%pattern_type.0aa) = value_param_pattern [symbolic = %x.param_patt.loc11_14.2 (constants.%x.param_patt.d48)]
// CHECK:STDOUT:   %x.patt.loc11_14.2: @InnerF.%pattern_type (%pattern_type.0aa) = at_binding_pattern x, %x.param_patt.loc11_14.2 [symbolic = %x.patt.loc11_14.2 (constants.%x.patt.764)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %V.as_type.loc11_16.1 [symbolic = %require_complete.loc11 (constants.%require_complete.269)]
// CHECK:STDOUT:   %V.as_type.loc12: %A.type = facet_access_type %V [symbolic = %V.as_type.loc12 (constants.%V.as_type.c9a)]
// CHECK:STDOUT:   %TakesA.specific_fn.loc12_5.2: <specific function> = specific_function constants.%TakesA, @TakesA(%V.as_type.loc12) [symbolic = %TakesA.specific_fn.loc12_5.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %V.as_type.loc12 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type %as_type [symbolic = %require_complete.loc12 (constants.%require_complete.379)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%V.as_type.loc11_16.1 (%V.as_type.38f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesA.ref: %TakesA.type = name_ref TakesA, file.%TakesA.decl [concrete = constants.%TakesA]
// CHECK:STDOUT:     %x.ref: @InnerF.%V.as_type.loc11_16.1 (%V.as_type.38f) = name_ref x, %x
// CHECK:STDOUT:     %.loc12_13.1: %A.type = as_compatible constants.%V.as_type.38f [symbolic = %V.as_type.loc12 (constants.%V.as_type.c9a)]
// CHECK:STDOUT:     %.loc12_13.2: %A.type = as_compatible constants.%V.as_type.38f [symbolic = %V.as_type.loc12 (constants.%V.as_type.c9a)]
// CHECK:STDOUT:     %TakesA.specific_fn.loc12_5.1: <specific function> = specific_function %TakesA.ref, @TakesA(constants.%V.as_type.c9a) [symbolic = %TakesA.specific_fn.loc12_5.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:     %.loc12_12: @InnerF.%as_type (%as_type) = as_compatible %x.ref
// CHECK:STDOUT:     %TakesA.call: init %empty_tuple.type = call %TakesA.specific_fn.loc12_5.1(%.loc12_12)
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.d0c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.b45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.62a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_12.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_12.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc7_26.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc7_26.2 => constants.%x.patt.19b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U, constants.%V) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc8_14.1 => constants.%U
// CHECK:STDOUT:   %V.patt.loc8_38.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc8_38.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U, constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %V.as_type.loc11_16.1 => constants.%V.as_type.38f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0aa
// CHECK:STDOUT:   %x.param_patt.loc11_14.2 => constants.%x.param_patt.d48
// CHECK:STDOUT:   %x.patt.loc11_14.2 => constants.%x.patt.764
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%V.as_type.c9a) {
// CHECK:STDOUT:   %T.patt.loc7_12.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_12.1 => constants.%V.as_type.c9a
// CHECK:STDOUT:   %T.as_type.loc7_28.1 => constants.%as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bdf
// CHECK:STDOUT:   %x.param_patt.loc7_26.2 => constants.%x.param_patt.acb
// CHECK:STDOUT:   %x.patt.loc7_26.2 => constants.%x.patt.20d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.379
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_functions_in_inner.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.d0c: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.b45: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.62a: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.241: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.19b: %pattern_type.a0e = at_binding_pattern x, %x.param_patt.241 [symbolic]
// CHECK:STDOUT:   %TakesA.type: type = fn_type @TakesA [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesA: %TakesA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.94f: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.Self.frozen.f7e: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.263: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.8d7: %C.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.a25: %C.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %C_where.type: type = facet_type <@C where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.803: type = pattern_type %C_where.type [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic]
// CHECK:STDOUT:   %V: %C_where.type = symbolic_binding V, 2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %V.as_type.38f: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %pattern_type.0aa: type = pattern_type %V.as_type.38f [symbolic]
// CHECK:STDOUT:   %x.param_patt.d48: %pattern_type.0aa = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.764: %pattern_type.0aa = at_binding_pattern x, %x.param_patt.d48 [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF, @F(%T, %U, %V) [symbolic]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.269: <witness> = require_complete_type %V.as_type.38f [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %eq.bb6 = observe_equivalent %T, %U [symbolic]
// CHECK:STDOUT:   %eq.51c = observe_equivalent %U, %V [symbolic]
// CHECK:STDOUT:   %V.as_type.c9a: %A.type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %as_type: type = facet_access_type %V.as_type.c9a [symbolic]
// CHECK:STDOUT:   %pattern_type.bdf: type = pattern_type %as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.acb: %pattern_type.bdf = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.20d: %pattern_type.bdf = at_binding_pattern x, %x.param_patt.acb [symbolic]
// CHECK:STDOUT:   %TakesA.specific_fn: <specific function> = specific_function %TakesA, @TakesA(%V.as_type.c9a) [symbolic]
// CHECK:STDOUT:   %require_complete.379: <witness> = require_complete_type %as_type [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TakesA = %TakesA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   %TakesA.decl: %TakesA.type = fn_decl @TakesA [concrete = constants.%TakesA] {
// CHECK:STDOUT:     %T.patt.loc7_12.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc7_26.1: @TakesA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_26.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:     %x.patt.loc7_26.1: @TakesA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_26.1 [symbolic = %x.patt.loc7_26.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_15: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_12.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesA.%T.as_type.loc7_28.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_28.1: type = splice_block %.loc7_28.2 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_12.2 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7_28.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_28.2: type = converted %T.ref, %T.as_type.loc7_28.2 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesA.%T.as_type.loc7_28.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc8_14.1: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt)]
// CHECK:STDOUT:     %V.patt.loc8_38.1: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic = %V.patt.loc8_38.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_19: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %base_facet_type.loc8_19 = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_25.1: %B.type = name_ref .Self, %.Self.frozen.loc8_19 [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %equiv.loc8_31.1 = requirement_equivalent %.Self.ref.loc8_25.1, %T.ref [concrete]
// CHECK:STDOUT:       %.Self.loc8_19: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.263]
// CHECK:STDOUT:       %.Self.ref.loc8_25.2: %B.type = name_ref .Self, %.Self.loc8_19 [symbolic_self = constants.%.Self.263]
// CHECK:STDOUT:       %equiv.loc8_31.2 = requirement_equivalent %.Self.ref.loc8_25.2, %T.ref [concrete]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         %base_facet_type.loc8_19 = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_31.2 = requirement_equivalent %.Self.ref.loc8_25.2, %T.ref [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_14.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:     %.loc8_43.1: type = splice_block %.loc8_43.2 [concrete = constants.%C_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_38: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_43: %C.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.8d7]
// CHECK:STDOUT:       %base_facet_type.loc8_43 = requirement_base_facet_type %C.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_49.1: %C.type = name_ref .Self, %.Self.frozen.loc8_43 [symbolic_self = constants.%.Self.frozen.8d7]
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc8_14.2 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:       %equiv.loc8_55.1 = requirement_equivalent %.Self.ref.loc8_49.1, %U.ref [concrete]
// CHECK:STDOUT:       %.Self.loc8_43: %C.type = symbolic_binding .Self [symbolic_self = constants.%.Self.a25]
// CHECK:STDOUT:       %.Self.ref.loc8_49.2: %C.type = name_ref .Self, %.Self.loc8_43 [symbolic_self = constants.%.Self.a25]
// CHECK:STDOUT:       %equiv.loc8_55.2 = requirement_equivalent %.Self.ref.loc8_49.2, %U.ref [concrete]
// CHECK:STDOUT:       %.loc8_43.2: type = where_expr [concrete = constants.%C_where.type] {
// CHECK:STDOUT:         %base_facet_type.loc8_43 = requirement_base_facet_type %C.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_55.2 = requirement_equivalent %.Self.ref.loc8_49.2, %U.ref [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc8_38.2: %C_where.type = symbolic_binding V, 2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0c]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b45]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.62a]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesA(%T.loc7_12.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc7_12.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_12.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc7_28.1: type = facet_access_type %T.loc7_12.1 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_28.1 [symbolic = %pattern_type (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %x.param_patt.loc7_26.2: @TakesA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_26.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:   %x.patt.loc7_26.2: @TakesA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_26.2 [symbolic = %x.patt.loc7_26.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_28.1 [symbolic = %require_complete (constants.%require_complete.94f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TakesA.%T.as_type.loc7_28.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %A.type, %U.loc8_14.2: %B_where.type, %V.loc8_38.2: %C_where.type) {
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_7.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc8_14.2: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc8_14.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:   %V.patt.loc8_38.2: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic = %V.patt.loc8_38.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc8_38.1: %C_where.type = symbolic_binding V, 2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF, @F(%T.loc8_7.1, %U.loc8_14.1, %V.loc8_38.1) [symbolic = %InnerF.type (constants.%InnerF.type)]
// CHECK:STDOUT:   %InnerF: @F.%InnerF.type (%InnerF.type) = struct_value () [symbolic = %InnerF (constants.%InnerF)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: @F.%InnerF.type (%InnerF.type) = fn_decl @InnerF [symbolic = @F.%InnerF (constants.%InnerF)] {
// CHECK:STDOUT:       %x.param_patt.loc10_14.1: @InnerF.%pattern_type (%pattern_type.0aa) = value_param_pattern [symbolic = %x.param_patt.loc10_14.2 (constants.%x.param_patt.d48)]
// CHECK:STDOUT:       %x.patt.loc10_14.1: @InnerF.%pattern_type (%pattern_type.0aa) = at_binding_pattern x, %x.param_patt.loc10_14.1 [symbolic = %x.patt.loc10_14.2 (constants.%x.patt.764)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%V.as_type.loc10_16.1 (%V.as_type.38f) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %.loc10_16.2 [symbolic = %V.as_type.loc10_16.1 (constants.%V.as_type.38f)] {
// CHECK:STDOUT:         %V.ref.loc10: %C_where.type = name_ref V, @F.%V.loc8_38.2 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:         %V.as_type.loc10_16.2: type = facet_access_type %V.ref.loc10 [symbolic = %V.as_type.loc10_16.1 (constants.%V.as_type.38f)]
// CHECK:STDOUT:         %.loc10_16.2: type = converted %V.ref.loc10, %V.as_type.loc10_16.2 [symbolic = %V.as_type.loc10_16.1 (constants.%V.as_type.38f)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%V.as_type.loc10_16.1 (%V.as_type.38f) = wrapper_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc8_7.2: %A.type, @F.%U.loc8_14.2: %B_where.type, @F.%V.loc8_38.2: %C_where.type) {
// CHECK:STDOUT:   %V: %C_where.type = symbolic_binding V, 2 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc10_16.1: type = facet_access_type %V [symbolic = %V.as_type.loc10_16.1 (constants.%V.as_type.38f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %V.as_type.loc10_16.1 [symbolic = %pattern_type (constants.%pattern_type.0aa)]
// CHECK:STDOUT:   %x.param_patt.loc10_14.2: @InnerF.%pattern_type (%pattern_type.0aa) = value_param_pattern [symbolic = %x.param_patt.loc10_14.2 (constants.%x.param_patt.d48)]
// CHECK:STDOUT:   %x.patt.loc10_14.2: @InnerF.%pattern_type (%pattern_type.0aa) = at_binding_pattern x, %x.param_patt.loc10_14.2 [symbolic = %x.patt.loc10_14.2 (constants.%x.patt.764)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %V.as_type.loc10_16.1 [symbolic = %require_complete.loc10 (constants.%require_complete.269)]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.as_type.loc11_20.3: type = facet_access_type %U [symbolic = %U.as_type.loc11_20.3 (constants.%U.as_type)]
// CHECK:STDOUT:   %T.as_type.loc11_15.2: type = facet_access_type %T [symbolic = %T.as_type.loc11_15.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %eq.loc11_17 = observe_equivalent %T, %U [symbolic = %eq.loc11_17 (constants.%eq.bb6)]
// CHECK:STDOUT:   %eq.loc11_22 = observe_equivalent %U, %V [symbolic = %eq.loc11_22 (constants.%eq.51c)]
// CHECK:STDOUT:   %V.as_type.loc13: %A.type = facet_access_type %V [symbolic = %V.as_type.loc13 (constants.%V.as_type.c9a)]
// CHECK:STDOUT:   %TakesA.specific_fn.loc13_5.2: <specific function> = specific_function constants.%TakesA, @TakesA(%V.as_type.loc13) [symbolic = %TakesA.specific_fn.loc13_5.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %V.as_type.loc13 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %as_type [symbolic = %require_complete.loc13 (constants.%require_complete.379)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%V.as_type.loc10_16.1 (%V.as_type.38f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type = name_ref T, @F.%T.loc8_7.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %U.ref: %B_where.type = name_ref U, @F.%U.loc8_14.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc11_20.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc11_20.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc11_20.1: type = converted %U.ref, %U.as_type.loc11_20.1 [symbolic = %U.as_type.loc11_20.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %T.as_type.loc11_15.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_15.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc11_15: type = converted %T.ref, %T.as_type.loc11_15.1 [symbolic = %T.as_type.loc11_15.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %V.ref.loc11: %C_where.type = name_ref V, @F.%V.loc8_38.2 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:     %V.as_type.loc11: type = facet_access_type %V.ref.loc11 [symbolic = %V.as_type.loc10_16.1 (constants.%V.as_type.38f)]
// CHECK:STDOUT:     %.loc11_25: type = converted %V.ref.loc11, %V.as_type.loc11 [symbolic = %V.as_type.loc10_16.1 (constants.%V.as_type.38f)]
// CHECK:STDOUT:     %U.as_type.loc11_20.2: type = facet_access_type %U.ref [symbolic = %U.as_type.loc11_20.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc11_20.2: type = converted %U.ref, %U.as_type.loc11_20.2 [symbolic = %U.as_type.loc11_20.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %InnerF.observe.decl = observe_decl @InnerF.observe [concrete]
// CHECK:STDOUT:     %TakesA.ref: %TakesA.type = name_ref TakesA, file.%TakesA.decl [concrete = constants.%TakesA]
// CHECK:STDOUT:     %x.ref: @InnerF.%V.as_type.loc10_16.1 (%V.as_type.38f) = name_ref x, %x
// CHECK:STDOUT:     %.loc13_13.1: %A.type = as_compatible constants.%V.as_type.38f [symbolic = %V.as_type.loc13 (constants.%V.as_type.c9a)]
// CHECK:STDOUT:     %.loc13_13.2: %A.type = as_compatible constants.%V.as_type.38f [symbolic = %V.as_type.loc13 (constants.%V.as_type.c9a)]
// CHECK:STDOUT:     %TakesA.specific_fn.loc13_5.1: <specific function> = specific_function %TakesA.ref, @TakesA(constants.%V.as_type.c9a) [symbolic = %TakesA.specific_fn.loc13_5.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:     %.loc13_12: @InnerF.%as_type (%as_type) = as_compatible %x.ref
// CHECK:STDOUT:     %TakesA.call: init %empty_tuple.type = call %TakesA.specific_fn.loc13_5.1(%.loc13_12)
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @InnerF.observe {
// CHECK:STDOUT:       observe constants.%T == constants.%U == constants.%V
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.d0c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.b45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.62a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_12.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_12.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc7_26.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc7_26.2 => constants.%x.patt.19b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U, constants.%V) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc8_14.1 => constants.%U
// CHECK:STDOUT:   %V.patt.loc8_38.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc8_38.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U, constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %V.as_type.loc10_16.1 => constants.%V.as_type.38f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0aa
// CHECK:STDOUT:   %x.param_patt.loc10_14.2 => constants.%x.param_patt.d48
// CHECK:STDOUT:   %x.patt.loc10_14.2 => constants.%x.patt.764
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%V.as_type.c9a) {
// CHECK:STDOUT:   %T.patt.loc7_12.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_12.1 => constants.%V.as_type.c9a
// CHECK:STDOUT:   %T.as_type.loc7_28.1 => constants.%as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bdf
// CHECK:STDOUT:   %x.param_patt.loc7_26.2 => constants.%x.param_patt.acb
// CHECK:STDOUT:   %x.patt.loc7_26.2 => constants.%x.patt.20d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.379
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested_functions_lexical_order.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.d0c: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.b45: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.62a: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.241: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.19b: %pattern_type.a0e = at_binding_pattern x, %x.param_patt.241 [symbolic]
// CHECK:STDOUT:   %TakesA.type: type = fn_type @TakesA [concrete]
// CHECK:STDOUT:   %TakesA: %TakesA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.94f: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.Self.frozen.f7e: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.263: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.8d7: %C.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.a25: %C.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %C_where.type: type = facet_type <@C where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.803: type = pattern_type %C_where.type [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic]
// CHECK:STDOUT:   %V: %C_where.type = symbolic_binding V, 2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %pattern_type.0aa: type = pattern_type %V.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.d48: %pattern_type.0aa = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.764: %pattern_type.0aa = at_binding_pattern x, %x.param_patt.d48 [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF, @F(%T, %U, %V) [symbolic]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.269: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %eq.bb6 = observe_equivalent %T, %U [symbolic]
// CHECK:STDOUT:   %eq.51c = observe_equivalent %U, %V [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TakesA = %TakesA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   %TakesA.decl: %TakesA.type = fn_decl @TakesA [concrete = constants.%TakesA] {
// CHECK:STDOUT:     %T.patt.loc7_12.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc7_26.1: @TakesA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_26.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:     %x.patt.loc7_26.1: @TakesA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_26.1 [symbolic = %x.patt.loc7_26.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_15: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_12.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesA.%T.as_type.loc7_28.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_28.1: type = splice_block %.loc7_28.2 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_12.2 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7_28.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_28.2: type = converted %T.ref, %T.as_type.loc7_28.2 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesA.%T.as_type.loc7_28.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc8_14.1: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt)]
// CHECK:STDOUT:     %V.patt.loc8_38.1: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic = %V.patt.loc8_38.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_19: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %base_facet_type.loc8_19 = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_25.1: %B.type = name_ref .Self, %.Self.frozen.loc8_19 [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %T.ref.loc8: %A.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %equiv.loc8_31.1 = requirement_equivalent %.Self.ref.loc8_25.1, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       %.Self.loc8_19: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.263]
// CHECK:STDOUT:       %.Self.ref.loc8_25.2: %B.type = name_ref .Self, %.Self.loc8_19 [symbolic_self = constants.%.Self.263]
// CHECK:STDOUT:       %equiv.loc8_31.2 = requirement_equivalent %.Self.ref.loc8_25.2, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         %base_facet_type.loc8_19 = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_31.2 = requirement_equivalent %.Self.ref.loc8_25.2, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_14.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:     %.loc8_43.1: type = splice_block %.loc8_43.2 [concrete = constants.%C_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_38: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_43: %C.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.8d7]
// CHECK:STDOUT:       %base_facet_type.loc8_43 = requirement_base_facet_type %C.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_49.1: %C.type = name_ref .Self, %.Self.frozen.loc8_43 [symbolic_self = constants.%.Self.frozen.8d7]
// CHECK:STDOUT:       %U.ref.loc8: %B_where.type = name_ref U, %U.loc8_14.2 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:       %equiv.loc8_55.1 = requirement_equivalent %.Self.ref.loc8_49.1, %U.ref.loc8 [concrete]
// CHECK:STDOUT:       %.Self.loc8_43: %C.type = symbolic_binding .Self [symbolic_self = constants.%.Self.a25]
// CHECK:STDOUT:       %.Self.ref.loc8_49.2: %C.type = name_ref .Self, %.Self.loc8_43 [symbolic_self = constants.%.Self.a25]
// CHECK:STDOUT:       %equiv.loc8_55.2 = requirement_equivalent %.Self.ref.loc8_49.2, %U.ref.loc8 [concrete]
// CHECK:STDOUT:       %.loc8_43.2: type = where_expr [concrete = constants.%C_where.type] {
// CHECK:STDOUT:         %base_facet_type.loc8_43 = requirement_base_facet_type %C.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_55.2 = requirement_equivalent %.Self.ref.loc8_49.2, %U.ref.loc8 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc8_38.2: %C_where.type = symbolic_binding V, 2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0c]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b45]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.62a]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesA(%T.loc7_12.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc7_12.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_12.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc7_28.1: type = facet_access_type %T.loc7_12.1 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_28.1 [symbolic = %pattern_type (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %x.param_patt.loc7_26.2: @TakesA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_26.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:   %x.patt.loc7_26.2: @TakesA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_26.2 [symbolic = %x.patt.loc7_26.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_28.1 [symbolic = %require_complete (constants.%require_complete.94f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TakesA.%T.as_type.loc7_28.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %A.type, %U.loc8_14.2: %B_where.type, %V.loc8_38.2: %C_where.type) {
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_7.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc8_14.2: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc8_14.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:   %V.patt.loc8_38.2: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic = %V.patt.loc8_38.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc8_38.1: %C_where.type = symbolic_binding V, 2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF, @F(%T.loc8_7.1, %U.loc8_14.1, %V.loc8_38.1) [symbolic = %InnerF.type (constants.%InnerF.type)]
// CHECK:STDOUT:   %InnerF: @F.%InnerF.type (%InnerF.type) = struct_value () [symbolic = %InnerF (constants.%InnerF)]
// CHECK:STDOUT:   %U.as_type.loc21_16.3: type = facet_access_type %U.loc8_14.1 [symbolic = %U.as_type.loc21_16.3 (constants.%U.as_type)]
// CHECK:STDOUT:   %T.as_type.loc21_11.2: type = facet_access_type %T.loc8_7.1 [symbolic = %T.as_type.loc21_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %eq.loc21_13 = observe_equivalent %T.loc8_7.1, %U.loc8_14.1 [symbolic = %eq.loc21_13 (constants.%eq.bb6)]
// CHECK:STDOUT:   %V.as_type.loc21_21.2: type = facet_access_type %V.loc8_38.1 [symbolic = %V.as_type.loc21_21.2 (constants.%V.as_type)]
// CHECK:STDOUT:   %eq.loc21_18 = observe_equivalent %U.loc8_14.1, %V.loc8_38.1 [symbolic = %eq.loc21_18 (constants.%eq.51c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: @F.%InnerF.type (%InnerF.type) = fn_decl @InnerF [symbolic = @F.%InnerF (constants.%InnerF)] {
// CHECK:STDOUT:       %x.param_patt.loc10_14.1: @InnerF.%pattern_type (%pattern_type.0aa) = value_param_pattern [symbolic = %x.param_patt.loc10_14.2 (constants.%x.param_patt.d48)]
// CHECK:STDOUT:       %x.patt.loc10_14.1: @InnerF.%pattern_type (%pattern_type.0aa) = at_binding_pattern x, %x.param_patt.loc10_14.1 [symbolic = %x.patt.loc10_14.2 (constants.%x.patt.764)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%V.as_type.loc10_16.1 (%V.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %.loc10_16.2 [symbolic = %V.as_type.loc10_16.1 (constants.%V.as_type)] {
// CHECK:STDOUT:         %V.ref: %C_where.type = name_ref V, @F.%V.loc8_38.2 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:         %V.as_type.loc10_16.2: type = facet_access_type %V.ref [symbolic = %V.as_type.loc10_16.1 (constants.%V.as_type)]
// CHECK:STDOUT:         %.loc10_16.2: type = converted %V.ref, %V.as_type.loc10_16.2 [symbolic = %V.as_type.loc10_16.1 (constants.%V.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%V.as_type.loc10_16.1 (%V.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref.loc21: %A.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc21: %B_where.type = name_ref U, %U.loc8_14.2 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc21_16.1: type = facet_access_type %U.ref.loc21 [symbolic = %U.as_type.loc21_16.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc21_16.1: type = converted %U.ref.loc21, %U.as_type.loc21_16.1 [symbolic = %U.as_type.loc21_16.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %T.as_type.loc21_11.1: type = facet_access_type %T.ref.loc21 [symbolic = %T.as_type.loc21_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc21_11: type = converted %T.ref.loc21, %T.as_type.loc21_11.1 [symbolic = %T.as_type.loc21_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %V.ref: %C_where.type = name_ref V, %V.loc8_38.2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:     %V.as_type.loc21_21.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc21_21.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %.loc21_21: type = converted %V.ref, %V.as_type.loc21_21.1 [symbolic = %V.as_type.loc21_21.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %U.as_type.loc21_16.2: type = facet_access_type %U.ref.loc21 [symbolic = %U.as_type.loc21_16.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc21_16.2: type = converted %U.ref.loc21, %U.as_type.loc21_16.2 [symbolic = %U.as_type.loc21_16.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %F.observe.decl = observe_decl @F.observe [concrete]
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @F.observe {
// CHECK:STDOUT:       observe constants.%T == constants.%U == constants.%V
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc8_7.2: %A.type, @F.%U.loc8_14.2: %B_where.type, @F.%V.loc8_38.2: %C_where.type) {
// CHECK:STDOUT:   %V: %C_where.type = symbolic_binding V, 2 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc10_16.1: type = facet_access_type %V [symbolic = %V.as_type.loc10_16.1 (constants.%V.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %V.as_type.loc10_16.1 [symbolic = %pattern_type (constants.%pattern_type.0aa)]
// CHECK:STDOUT:   %x.param_patt.loc10_14.2: @InnerF.%pattern_type (%pattern_type.0aa) = value_param_pattern [symbolic = %x.param_patt.loc10_14.2 (constants.%x.param_patt.d48)]
// CHECK:STDOUT:   %x.patt.loc10_14.2: @InnerF.%pattern_type (%pattern_type.0aa) = at_binding_pattern x, %x.param_patt.loc10_14.2 [symbolic = %x.patt.loc10_14.2 (constants.%x.patt.764)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %V.as_type.loc10_16.1 [symbolic = %require_complete (constants.%require_complete.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%V.as_type.loc10_16.1 (%V.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %TakesA.ref: %TakesA.type = name_ref TakesA, file.%TakesA.decl [concrete = constants.%TakesA]
// CHECK:STDOUT:     %x.ref: @InnerF.%V.as_type.loc10_16.1 (%V.as_type) = name_ref x, %x
// CHECK:STDOUT:     %C.type: type = facet_type <@C> [concrete = constants.%C.type]
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.d0c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.b45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.62a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_12.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_12.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc7_26.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc7_26.2 => constants.%x.patt.19b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U, constants.%V) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc8_14.1 => constants.%U
// CHECK:STDOUT:   %V.patt.loc8_38.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc8_38.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U, constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %V.as_type.loc10_16.1 => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0aa
// CHECK:STDOUT:   %x.param_patt.loc10_14.2 => constants.%x.param_patt.d48
// CHECK:STDOUT:   %x.patt.loc10_14.2 => constants.%x.patt.764
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested_functions_leak_upward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.d0c: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.b45: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.62a: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.241: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.19b: %pattern_type.a0e = at_binding_pattern x, %x.param_patt.241 [symbolic]
// CHECK:STDOUT:   %TakesA.type: type = fn_type @TakesA [concrete]
// CHECK:STDOUT:   %TakesA: %TakesA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.94f: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.Self.frozen.f7e: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.263: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.8d7: %C.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.a25: %C.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %C_where.type: type = facet_type <@C where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.803: type = pattern_type %C_where.type [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic]
// CHECK:STDOUT:   %V: %C_where.type = symbolic_binding V, 2 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %pattern_type.0aa: type = pattern_type %V.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.d48: %pattern_type.0aa = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.764: %pattern_type.0aa = at_binding_pattern x, %x.param_patt.d48 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.269: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF, @F(%T, %U, %V) [symbolic]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %eq.bb6 = observe_equivalent %T, %U [symbolic]
// CHECK:STDOUT:   %eq.51c = observe_equivalent %U, %V [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TakesA = %TakesA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   %TakesA.decl: %TakesA.type = fn_decl @TakesA [concrete = constants.%TakesA] {
// CHECK:STDOUT:     %T.patt.loc7_12.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc7_26.1: @TakesA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_26.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:     %x.patt.loc7_26.1: @TakesA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_26.1 [symbolic = %x.patt.loc7_26.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_15: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_12.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesA.%T.as_type.loc7_28.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_28.1: type = splice_block %.loc7_28.2 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_12.2 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7_28.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_28.2: type = converted %T.ref, %T.as_type.loc7_28.2 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesA.%T.as_type.loc7_28.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc8_14.1: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt)]
// CHECK:STDOUT:     %V.patt.loc8_38.1: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic = %V.patt.loc8_38.2 (constants.%V.patt)]
// CHECK:STDOUT:     %x.param_patt.loc8_62.1: @F.%pattern_type (%pattern_type.0aa) = value_param_pattern [symbolic = %x.param_patt.loc8_62.2 (constants.%x.param_patt.d48)]
// CHECK:STDOUT:     %x.patt.loc8_62.1: @F.%pattern_type (%pattern_type.0aa) = at_binding_pattern x, %x.param_patt.loc8_62.1 [symbolic = %x.patt.loc8_62.2 (constants.%x.patt.764)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_19: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %base_facet_type.loc8_19 = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_25.1: %B.type = name_ref .Self, %.Self.frozen.loc8_19 [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %equiv.loc8_31.1 = requirement_equivalent %.Self.ref.loc8_25.1, %T.ref [concrete]
// CHECK:STDOUT:       %.Self.loc8_19: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.263]
// CHECK:STDOUT:       %.Self.ref.loc8_25.2: %B.type = name_ref .Self, %.Self.loc8_19 [symbolic_self = constants.%.Self.263]
// CHECK:STDOUT:       %equiv.loc8_31.2 = requirement_equivalent %.Self.ref.loc8_25.2, %T.ref [concrete]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         %base_facet_type.loc8_19 = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_31.2 = requirement_equivalent %.Self.ref.loc8_25.2, %T.ref [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_14.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:     %.loc8_43.1: type = splice_block %.loc8_43.2 [concrete = constants.%C_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_38: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_43: %C.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.8d7]
// CHECK:STDOUT:       %base_facet_type.loc8_43 = requirement_base_facet_type %C.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_49.1: %C.type = name_ref .Self, %.Self.frozen.loc8_43 [symbolic_self = constants.%.Self.frozen.8d7]
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc8_14.2 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:       %equiv.loc8_55.1 = requirement_equivalent %.Self.ref.loc8_49.1, %U.ref [concrete]
// CHECK:STDOUT:       %.Self.loc8_43: %C.type = symbolic_binding .Self [symbolic_self = constants.%.Self.a25]
// CHECK:STDOUT:       %.Self.ref.loc8_49.2: %C.type = name_ref .Self, %.Self.loc8_43 [symbolic_self = constants.%.Self.a25]
// CHECK:STDOUT:       %equiv.loc8_55.2 = requirement_equivalent %.Self.ref.loc8_49.2, %U.ref [concrete]
// CHECK:STDOUT:       %.loc8_43.2: type = where_expr [concrete = constants.%C_where.type] {
// CHECK:STDOUT:         %base_facet_type.loc8_43 = requirement_base_facet_type %C.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_55.2 = requirement_equivalent %.Self.ref.loc8_49.2, %U.ref [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc8_38.2: %C_where.type = symbolic_binding V, 2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:     %x.param: @F.%V.as_type.loc8_64.1 (%V.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_64.1: type = splice_block %.loc8_64.2 [symbolic = %V.as_type.loc8_64.1 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref: %C_where.type = name_ref V, %V.loc8_38.2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc8_64.2: type = facet_access_type %V.ref [symbolic = %V.as_type.loc8_64.1 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc8_64.2: type = converted %V.ref, %V.as_type.loc8_64.2 [symbolic = %V.as_type.loc8_64.1 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%V.as_type.loc8_64.1 (%V.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0c]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b45]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.62a]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesA(%T.loc7_12.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc7_12.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_12.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc7_28.1: type = facet_access_type %T.loc7_12.1 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_28.1 [symbolic = %pattern_type (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %x.param_patt.loc7_26.2: @TakesA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_26.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:   %x.patt.loc7_26.2: @TakesA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_26.2 [symbolic = %x.patt.loc7_26.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_28.1 [symbolic = %require_complete (constants.%require_complete.94f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TakesA.%T.as_type.loc7_28.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %A.type, %U.loc8_14.2: %B_where.type, %V.loc8_38.2: %C_where.type) {
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_7.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc8_14.2: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc8_14.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:   %V.patt.loc8_38.2: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic = %V.patt.loc8_38.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc8_38.1: %C_where.type = symbolic_binding V, 2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc8_64.1: type = facet_access_type %V.loc8_38.1 [symbolic = %V.as_type.loc8_64.1 (constants.%V.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %V.as_type.loc8_64.1 [symbolic = %pattern_type (constants.%pattern_type.0aa)]
// CHECK:STDOUT:   %x.param_patt.loc8_62.2: @F.%pattern_type (%pattern_type.0aa) = value_param_pattern [symbolic = %x.param_patt.loc8_62.2 (constants.%x.param_patt.d48)]
// CHECK:STDOUT:   %x.patt.loc8_62.2: @F.%pattern_type (%pattern_type.0aa) = at_binding_pattern x, %x.param_patt.loc8_62.2 [symbolic = %x.patt.loc8_62.2 (constants.%x.patt.764)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %V.as_type.loc8_64.1 [symbolic = %require_complete (constants.%require_complete.269)]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF, @F(%T.loc8_7.1, %U.loc8_14.1, %V.loc8_38.1) [symbolic = %InnerF.type (constants.%InnerF.type)]
// CHECK:STDOUT:   %InnerF: @F.%InnerF.type (%InnerF.type) = struct_value () [symbolic = %InnerF (constants.%InnerF)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%V.as_type.loc8_64.1 (%V.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: @F.%InnerF.type (%InnerF.type) = fn_decl @InnerF [symbolic = @F.%InnerF (constants.%InnerF)] {} {}
// CHECK:STDOUT:     %TakesA.ref: %TakesA.type = name_ref TakesA, file.%TakesA.decl [concrete = constants.%TakesA]
// CHECK:STDOUT:     %x.ref: @F.%V.as_type.loc8_64.1 (%V.as_type) = name_ref x, %x
// CHECK:STDOUT:     %C.type: type = facet_type <@C> [concrete = constants.%C.type]
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc8_7.2: %A.type, @F.%U.loc8_14.2: %B_where.type, @F.%V.loc8_38.2: %C_where.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.as_type.loc11_18.3: type = facet_access_type %U [symbolic = %U.as_type.loc11_18.3 (constants.%U.as_type)]
// CHECK:STDOUT:   %T.as_type.loc11_13.2: type = facet_access_type %T [symbolic = %T.as_type.loc11_13.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %eq.loc11_15 = observe_equivalent %T, %U [symbolic = %eq.loc11_15 (constants.%eq.bb6)]
// CHECK:STDOUT:   %V: %C_where.type = symbolic_binding V, 2 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc11_23.2: type = facet_access_type %V [symbolic = %V.as_type.loc11_23.2 (constants.%V.as_type)]
// CHECK:STDOUT:   %eq.loc11_20 = observe_equivalent %U, %V [symbolic = %eq.loc11_20 (constants.%eq.51c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type = name_ref T, @F.%T.loc8_7.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %U.ref: %B_where.type = name_ref U, @F.%U.loc8_14.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc11_18.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc11_18.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc11_18.1: type = converted %U.ref, %U.as_type.loc11_18.1 [symbolic = %U.as_type.loc11_18.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %T.as_type.loc11_13.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_13.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc11_13: type = converted %T.ref, %T.as_type.loc11_13.1 [symbolic = %T.as_type.loc11_13.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %V.ref: %C_where.type = name_ref V, @F.%V.loc8_38.2 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:     %V.as_type.loc11_23.1: type = facet_access_type %V.ref [symbolic = %V.as_type.loc11_23.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %.loc11_23: type = converted %V.ref, %V.as_type.loc11_23.1 [symbolic = %V.as_type.loc11_23.2 (constants.%V.as_type)]
// CHECK:STDOUT:     %U.as_type.loc11_18.2: type = facet_access_type %U.ref [symbolic = %U.as_type.loc11_18.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc11_18.2: type = converted %U.ref, %U.as_type.loc11_18.2 [symbolic = %U.as_type.loc11_18.3 (constants.%U.as_type)]
// CHECK:STDOUT:     %InnerF.observe.decl = observe_decl @InnerF.observe [concrete]
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @InnerF.observe {
// CHECK:STDOUT:       observe constants.%T == constants.%U == constants.%V
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.d0c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.b45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.62a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_12.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_12.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc7_26.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc7_26.2 => constants.%x.patt.19b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U, constants.%V) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc8_14.1 => constants.%U
// CHECK:STDOUT:   %V.patt.loc8_38.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc8_38.1 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc8_64.1 => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0aa
// CHECK:STDOUT:   %x.param_patt.loc8_62.2 => constants.%x.param_patt.d48
// CHECK:STDOUT:   %x.patt.loc8_62.2 => constants.%x.patt.764
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U, constants.%V) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_observed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.d0c: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.b45: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.62a: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.241: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.19b: %pattern_type.a0e = at_binding_pattern x, %x.param_patt.241 [symbolic]
// CHECK:STDOUT:   %TakesA.type: type = fn_type @TakesA [concrete]
// CHECK:STDOUT:   %TakesA: %TakesA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.94f: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.Self.frozen.f7e: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.263: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.8d7: %C.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.a25: %C.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %C_where.type: type = facet_type <@C where TODO> [concrete]
// CHECK:STDOUT:   %pattern_type.803: type = pattern_type %C_where.type [concrete]
// CHECK:STDOUT:   %V.patt: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic]
// CHECK:STDOUT:   %V: %C_where.type = symbolic_binding V, 2 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %pattern_type.0aa: type = pattern_type %V.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.d48: %pattern_type.0aa = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.764: %pattern_type.0aa = at_binding_pattern x, %x.param_patt.d48 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.269: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %eq = observe_equivalent %T, %U [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TakesA = %TakesA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   %TakesA.decl: %TakesA.type = fn_decl @TakesA [concrete = constants.%TakesA] {
// CHECK:STDOUT:     %T.patt.loc7_12.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_12.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc7_26.1: @TakesA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_26.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:     %x.patt.loc7_26.1: @TakesA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_26.1 [symbolic = %x.patt.loc7_26.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_15: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_12.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesA.%T.as_type.loc7_28.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_28.1: type = splice_block %.loc7_28.2 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_12.2 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7_28.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_28.2: type = converted %T.ref, %T.as_type.loc7_28.2 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesA.%T.as_type.loc7_28.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_7.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc8_14.1: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt)]
// CHECK:STDOUT:     %V.patt.loc8_38.1: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic = %V.patt.loc8_38.2 (constants.%V.patt)]
// CHECK:STDOUT:     %x.param_patt.loc8_62.1: @F.%pattern_type (%pattern_type.0aa) = value_param_pattern [symbolic = %x.param_patt.loc8_62.2 (constants.%x.param_patt.d48)]
// CHECK:STDOUT:     %x.patt.loc8_62.1: @F.%pattern_type (%pattern_type.0aa) = at_binding_pattern x, %x.param_patt.loc8_62.1 [symbolic = %x.patt.loc8_62.2 (constants.%x.patt.764)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_19: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %base_facet_type.loc8_19 = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_25.1: %B.type = name_ref .Self, %.Self.frozen.loc8_19 [symbolic_self = constants.%.Self.frozen.f7e]
// CHECK:STDOUT:       %T.ref.loc8: %A.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:       %equiv.loc8_31.1 = requirement_equivalent %.Self.ref.loc8_25.1, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       %.Self.loc8_19: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.263]
// CHECK:STDOUT:       %.Self.ref.loc8_25.2: %B.type = name_ref .Self, %.Self.loc8_19 [symbolic_self = constants.%.Self.263]
// CHECK:STDOUT:       %equiv.loc8_31.2 = requirement_equivalent %.Self.ref.loc8_25.2, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         %base_facet_type.loc8_19 = requirement_base_facet_type %B.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_31.2 = requirement_equivalent %.Self.ref.loc8_25.2, %T.ref.loc8 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_14.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:     %.loc8_43.1: type = splice_block %.loc8_43.2 [concrete = constants.%C_where.type] {
// CHECK:STDOUT:       %.Self.frozen.loc8_38: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:       %.Self.frozen.loc8_43: %C.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.8d7]
// CHECK:STDOUT:       %base_facet_type.loc8_43 = requirement_base_facet_type %C.ref [concrete]
// CHECK:STDOUT:       %.Self.ref.loc8_49.1: %C.type = name_ref .Self, %.Self.frozen.loc8_43 [symbolic_self = constants.%.Self.frozen.8d7]
// CHECK:STDOUT:       %U.ref.loc8: %B_where.type = name_ref U, %U.loc8_14.2 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:       %equiv.loc8_55.1 = requirement_equivalent %.Self.ref.loc8_49.1, %U.ref.loc8 [concrete]
// CHECK:STDOUT:       %.Self.loc8_43: %C.type = symbolic_binding .Self [symbolic_self = constants.%.Self.a25]
// CHECK:STDOUT:       %.Self.ref.loc8_49.2: %C.type = name_ref .Self, %.Self.loc8_43 [symbolic_self = constants.%.Self.a25]
// CHECK:STDOUT:       %equiv.loc8_55.2 = requirement_equivalent %.Self.ref.loc8_49.2, %U.ref.loc8 [concrete]
// CHECK:STDOUT:       %.loc8_43.2: type = where_expr [concrete = constants.%C_where.type] {
// CHECK:STDOUT:         %base_facet_type.loc8_43 = requirement_base_facet_type %C.ref [concrete]
// CHECK:STDOUT:         %equiv.loc8_55.2 = requirement_equivalent %.Self.ref.loc8_49.2, %U.ref.loc8 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc8_38.2: %C_where.type = symbolic_binding V, 2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:     %x.param: @F.%V.as_type.loc8_64.1 (%V.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_64.1: type = splice_block %.loc8_64.2 [symbolic = %V.as_type.loc8_64.1 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref: %C_where.type = name_ref V, %V.loc8_38.2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc8_64.2: type = facet_access_type %V.ref [symbolic = %V.as_type.loc8_64.1 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc8_64.2: type = converted %V.ref, %V.as_type.loc8_64.2 [symbolic = %V.as_type.loc8_64.1 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%V.as_type.loc8_64.1 (%V.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0c]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b45]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.62a]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesA(%T.loc7_12.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc7_12.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_12.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_12.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_12.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc7_28.1: type = facet_access_type %T.loc7_12.1 [symbolic = %T.as_type.loc7_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_28.1 [symbolic = %pattern_type (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %x.param_patt.loc7_26.2: @TakesA.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %x.param_patt.loc7_26.2 (constants.%x.param_patt.241)]
// CHECK:STDOUT:   %x.patt.loc7_26.2: @TakesA.%pattern_type (%pattern_type.a0e) = at_binding_pattern x, %x.param_patt.loc7_26.2 [symbolic = %x.patt.loc7_26.2 (constants.%x.patt.19b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_28.1 [symbolic = %require_complete (constants.%require_complete.94f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TakesA.%T.as_type.loc7_28.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_7.2: %A.type, %U.loc8_14.2: %B_where.type, %V.loc8_38.2: %C_where.type) {
// CHECK:STDOUT:   %T.patt.loc8_7.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_7.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc8_14.2: %pattern_type.831 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc8_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc8_14.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:   %V.patt.loc8_38.2: %pattern_type.803 = symbolic_binding_pattern V, 2 [symbolic = %V.patt.loc8_38.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc8_38.1: %C_where.type = symbolic_binding V, 2 [symbolic = %V.loc8_38.1 (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc8_64.1: type = facet_access_type %V.loc8_38.1 [symbolic = %V.as_type.loc8_64.1 (constants.%V.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %V.as_type.loc8_64.1 [symbolic = %pattern_type (constants.%pattern_type.0aa)]
// CHECK:STDOUT:   %x.param_patt.loc8_62.2: @F.%pattern_type (%pattern_type.0aa) = value_param_pattern [symbolic = %x.param_patt.loc8_62.2 (constants.%x.param_patt.d48)]
// CHECK:STDOUT:   %x.patt.loc8_62.2: @F.%pattern_type (%pattern_type.0aa) = at_binding_pattern x, %x.param_patt.loc8_62.2 [symbolic = %x.patt.loc8_62.2 (constants.%x.patt.764)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %V.as_type.loc8_64.1 [symbolic = %require_complete (constants.%require_complete.269)]
// CHECK:STDOUT:   %U.as_type.loc9_16.2: type = facet_access_type %U.loc8_14.1 [symbolic = %U.as_type.loc9_16.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %T.as_type.loc9_11.2: type = facet_access_type %T.loc8_7.1 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %eq = observe_equivalent %T.loc8_7.1, %U.loc8_14.1 [symbolic = %eq (constants.%eq)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%V.as_type.loc8_64.1 (%V.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc9: %A.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc9: %B_where.type = name_ref U, %U.loc8_14.2 [symbolic = %U.loc8_14.1 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc9_16.1: type = facet_access_type %U.ref.loc9 [symbolic = %U.as_type.loc9_16.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc9_16: type = converted %U.ref.loc9, %U.as_type.loc9_16.1 [symbolic = %U.as_type.loc9_16.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %T.as_type.loc9_11.1: type = facet_access_type %T.ref.loc9 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_11: type = converted %T.ref.loc9, %T.as_type.loc9_11.1 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %F.observe.decl = observe_decl @F.observe [concrete]
// CHECK:STDOUT:     %TakesA.ref: %TakesA.type = name_ref TakesA, file.%TakesA.decl [concrete = constants.%TakesA]
// CHECK:STDOUT:     %x.ref: @F.%V.as_type.loc8_64.1 (%V.as_type) = name_ref x, %x
// CHECK:STDOUT:     %C.type: type = facet_type <@C> [concrete = constants.%C.type]
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @F.observe {
// CHECK:STDOUT:       observe constants.%T == constants.%U
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.d0c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.b45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.62a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_12.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_12.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %x.param_patt.loc7_26.2 => constants.%x.param_patt.241
// CHECK:STDOUT:   %x.patt.loc7_26.2 => constants.%x.patt.19b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U, constants.%V) {
// CHECK:STDOUT:   %T.patt.loc8_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc8_14.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc8_14.1 => constants.%U
// CHECK:STDOUT:   %V.patt.loc8_38.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc8_38.1 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc8_64.1 => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0aa
// CHECK:STDOUT:   %x.param_patt.loc8_62.2 => constants.%x.param_patt.d48
// CHECK:STDOUT:   %x.patt.loc8_62.2 => constants.%x.patt.764
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_invalid_equivalent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.d0c: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.b45: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7b5: type = pattern_type %B.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.7b5 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: %B.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %eq = observe_equivalent %T, %U [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc6_14.1: %pattern_type.7b5 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc6_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen.loc6_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_17: type = splice_block %B.ref [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self.frozen.loc6_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc6_14.2: %B.type = symbolic_binding U, 1 [symbolic = %U.loc6_14.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0c]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b45]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: %A.type, %U.loc6_14.2: %B.type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_7.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc6_14.2: %pattern_type.7b5 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc6_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc6_14.1: %B.type = symbolic_binding U, 1 [symbolic = %U.loc6_14.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.as_type.loc9_16.2: type = facet_access_type %U.loc6_14.1 [symbolic = %U.as_type.loc9_16.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %T.as_type.loc9_11.2: type = facet_access_type %T.loc6_7.1 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %eq = observe_equivalent %T.loc6_7.1, %U.loc6_14.1 [symbolic = %eq (constants.%eq)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref: %B.type = name_ref U, %U.loc6_14.2 [symbolic = %U.loc6_14.1 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc9_16.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc9_16.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc9_16: type = converted %U.ref, %U.as_type.loc9_16.1 [symbolic = %U.as_type.loc9_16.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %T.as_type.loc9_11.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_11: type = converted %T.ref, %T.as_type.loc9_11.1 [symbolic = %T.as_type.loc9_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %F.observe.decl = observe_decl @F.observe [concrete]
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @F.observe {
// CHECK:STDOUT:       observe constants.%T == constants.%U
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.d0c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.b45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc6_14.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc6_14.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_scope_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %pattern_type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T, %U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc3_10.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc3_20.1: %pattern_type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc3_20.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_13.1: type = splice_block %.loc3_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc3_10: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_10.1 (constants.%T)]
// CHECK:STDOUT:     %.loc3_23.1: type = splice_block %.loc3_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc3_20: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc3_20.2: type = symbolic_binding U, 1 [symbolic = %U.loc3_20.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc3_10.2: type, %U.loc3_20.2: type) {
// CHECK:STDOUT:   %T.patt.loc3_10.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc3_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_10.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc3_20.2: %pattern_type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc3_20.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc3_20.1: type = symbolic_binding U, 1 [symbolic = %U.loc3_20.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc3_10.2 [symbolic = %T.loc3_10.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc3_20.2 [symbolic = %U.loc3_20.1 (constants.%U)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.patt.loc3_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc3_10.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc3_20.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc3_20.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_scope_named_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %pattern_type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %C.type.eac: type = generic_named_constaint_type @C [concrete]
// CHECK:STDOUT:   %empty_struct: %C.type.eac = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.d68: type = facet_type <@C, @C(%T, %U)> [symbolic]
// CHECK:STDOUT:   %Self: %C.type.d68 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type.eac = constraint_decl @C [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt.loc3_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc3_25.1: %pattern_type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc3_25.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_18.1: type = splice_block %.loc3_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc3_15: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_15.1 (constants.%T)]
// CHECK:STDOUT:     %.loc3_28.1: type = splice_block %.loc3_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc3_25: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc3_25.2: type = symbolic_binding U, 1 [symbolic = %U.loc3_25.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @C(%T.loc3_15.2: type, %U.loc3_25.2: type) {
// CHECK:STDOUT:   %T.patt.loc3_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc3_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_15.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc3_25.2: %pattern_type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc3_25.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc3_25.1: type = symbolic_binding U, 1 [symbolic = %U.loc3_25.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.type: type = facet_type <@C, @C(%T.loc3_15.1, %U.loc3_25.1)> [symbolic = %C.type (constants.%C.type.d68)]
// CHECK:STDOUT:   %Self.loc3_34.2: @C.%C.type (%C.type.d68) = symbolic_binding Self, 2 [symbolic = %Self.loc3_34.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc3_34.1: @C.%C.type (%C.type.d68) = symbolic_binding Self, 2 [symbolic = %Self.loc3_34.2 (constants.%Self)]
// CHECK:STDOUT:     %C.WithSelf.decl = constraint_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %T.ref: type = name_ref T, @C.%T.loc3_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %U.ref: type = name_ref U, @C.%U.loc3_25.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_34.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.patt.loc3_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc3_15.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc3_25.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc3_25.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%T, constants.%U, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_facet_type_in_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %impls = observe_impls %T, %C [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_7.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc11_11.2: type = facet_access_type %T.loc6_7.1 [symbolic = %T.as_type.loc11_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %impls = observe_impls %T.loc6_7.1, constants.%C [symbolic = %impls (constants.%impls)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %A.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %T.as_type.loc11_11.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc11: type = converted %T.ref, %T.as_type.loc11_11.1 [symbolic = %T.as_type.loc11_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_observe.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.d0c: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.b45: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.62a: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.282: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.13d: %I.type.282 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type.208: type = facet_type <@B & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.877: %I.assoc_type = assoc_entity element0, @I.WithSelf.%U [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self.13d, @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %impl.elem0: %facet_type.208 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.392: type = facet_type <@C & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.WithSelf.%V [symbolic]
// CHECK:STDOUT:   %Self.as_type.363: type = facet_access_type %Self.13d [symbolic]
// CHECK:STDOUT:   %pattern_type.1c8: type = pattern_type %Self.as_type.363 [symbolic]
// CHECK:STDOUT:   %self.param_patt.4a9: %pattern_type.1c8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.452: %pattern_type.1c8 = at_binding_pattern self, %self.param_patt.4a9 [symbolic]
// CHECK:STDOUT:   %impl.elem1: %facet_type.392 = impl_witness_access %I.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %as_type.64d: type = facet_access_type %impl.elem1 [symbolic]
// CHECK:STDOUT:   %.04f: Core.Form = init_form %as_type.64d [symbolic]
// CHECK:STDOUT:   %pattern_type.a24: type = pattern_type %as_type.64d [symbolic]
// CHECK:STDOUT:   %return.param_patt.49a: %pattern_type.a24 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.4d9: %pattern_type.a24 = return_slot_pattern %return.param_patt.49a, %as_type.64d [symbolic]
// CHECK:STDOUT:   %I.WithSelf.GetV.type: type = fn_type @I.WithSelf.GetV, @I.WithSelf(%T, %Self.13d) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.GetV: %I.WithSelf.GetV.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc2: %I.assoc_type = assoc_entity element2, @I.WithSelf.%I.WithSelf.GetV.decl [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt: %pattern_type.a0e = at_binding_pattern t, %t.param_patt [symbolic]
// CHECK:STDOUT:   %I.WithSelf.TakesT.type: type = fn_type @I.WithSelf.TakesT, @I.WithSelf(%T, %Self.13d) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.TakesT: %I.WithSelf.TakesT.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc3: %I.assoc_type = assoc_entity element3, @I.WithSelf.%I.WithSelf.TakesT.decl [symbolic]
// CHECK:STDOUT:   %as_type.49a: type = facet_access_type %impl.elem0 [symbolic]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc6_14.1: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.197]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0c]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b45]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.62a]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc6_14.2: %A.type) {
// CHECK:STDOUT:   %T.patt.loc6_14.2: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc6_14.1)> [symbolic = %I.type (constants.%I.type.282)]
// CHECK:STDOUT:   %Self.loc6_20.2: @I.%I.type (%I.type.282) = symbolic_binding Self, 1 [symbolic = %Self.loc6_20.2 (constants.%Self.13d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc6_20.1: @I.%I.type (%I.type.282) = symbolic_binding Self, 1 [symbolic = %Self.loc6_20.2 (constants.%Self.13d)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %U: %facet_type.208 = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:       %assoc0: @I.WithSelf.%I.assoc_type (%I.assoc_type) = assoc_entity element0, @I.WithSelf.%U [symbolic = @I.WithSelf.%assoc0 (constants.%assoc0.877)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V: %facet_type.392 = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:       %assoc1: @I.WithSelf.%I.assoc_type (%I.assoc_type) = assoc_entity element1, @I.WithSelf.%V [symbolic = @I.WithSelf.%assoc1 (constants.%assoc1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.WithSelf.GetV.decl: @I.WithSelf.%I.WithSelf.GetV.type (%I.WithSelf.GetV.type) = fn_decl @I.WithSelf.GetV [symbolic = @I.WithSelf.%I.WithSelf.GetV (constants.%I.WithSelf.GetV)] {
// CHECK:STDOUT:       %self.param_patt.loc10_11.1: @I.WithSelf.GetV.%pattern_type.loc10_11 (%pattern_type.1c8) = value_param_pattern [symbolic = %self.param_patt.loc10_11.2 (constants.%self.param_patt.4a9)]
// CHECK:STDOUT:       %self.patt.loc10_11.1: @I.WithSelf.GetV.%pattern_type.loc10_11 (%pattern_type.1c8) = at_binding_pattern self, %self.param_patt.loc10_11.1 [symbolic = %self.patt.loc10_11.2 (constants.%self.patt.452)]
// CHECK:STDOUT:       %return.param_patt.loc10_20.1: @I.WithSelf.GetV.%pattern_type.loc10_20 (%pattern_type.a24) = out_param_pattern [symbolic = %return.param_patt.loc10_20.2 (constants.%return.param_patt.49a)]
// CHECK:STDOUT:       %return.patt.loc10_17.1: @I.WithSelf.GetV.%pattern_type.loc10_20 (%pattern_type.a24) = return_slot_pattern %return.param_patt.loc10_20.1, %.loc10_20.2 [symbolic = %return.patt.loc10_17.2 (constants.%return.patt.4d9)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %impl.elem1.loc10_20.2: %facet_type.392 = impl_witness_access constants.%I.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc10_20.1 (constants.%impl.elem1)]
// CHECK:STDOUT:       %V.ref: %facet_type.392 = name_ref V, %impl.elem1.loc10_20.2 [symbolic = %impl.elem1.loc10_20.1 (constants.%impl.elem1)]
// CHECK:STDOUT:       %V.as_type: type = facet_access_type %V.ref [symbolic = %as_type (constants.%as_type.64d)]
// CHECK:STDOUT:       %.loc10_20.2: type = converted %V.ref, %V.as_type [symbolic = %as_type (constants.%as_type.64d)]
// CHECK:STDOUT:       %.loc10_20.3: Core.Form = init_form %.loc10_20.2 [symbolic = %.loc10_20.1 (constants.%.04f)]
// CHECK:STDOUT:       %self.param: @I.WithSelf.GetV.%Self.as_type.loc10_11.1 (%Self.as_type.363) = value_param call_param0
// CHECK:STDOUT:       %.loc10_11.1: type = splice_block %.loc10_11.3 [symbolic = %Self.as_type.loc10_11.1 (constants.%Self.as_type.363)] {
// CHECK:STDOUT:         %.loc10_11.2: @I.WithSelf.GetV.%I.type (%I.type.282) = specific_constant @I.%Self.loc6_20.1, @I(constants.%T) [symbolic = %Self (constants.%Self.13d)]
// CHECK:STDOUT:         %Self.ref: @I.WithSelf.GetV.%I.type (%I.type.282) = name_ref Self, %.loc10_11.2 [symbolic = %Self (constants.%Self.13d)]
// CHECK:STDOUT:         %Self.as_type.loc10_11.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc10_11.1 (constants.%Self.as_type.363)]
// CHECK:STDOUT:         %.loc10_11.3: type = converted %Self.ref, %Self.as_type.loc10_11.2 [symbolic = %Self.as_type.loc10_11.1 (constants.%Self.as_type.363)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @I.WithSelf.GetV.%Self.as_type.loc10_11.1 (%Self.as_type.363) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @I.WithSelf.GetV.%as_type (%as_type.64d) = out_param call_param1
// CHECK:STDOUT:       %return: ref @I.WithSelf.GetV.%as_type (%as_type.64d) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc2.loc10_21.1: @I.WithSelf.%I.assoc_type (%I.assoc_type) = assoc_entity element2, %I.WithSelf.GetV.decl [symbolic = %assoc2.loc10_21.2 (constants.%assoc2)]
// CHECK:STDOUT:     %I.WithSelf.TakesT.decl: @I.WithSelf.%I.WithSelf.TakesT.type (%I.WithSelf.TakesT.type) = fn_decl @I.WithSelf.TakesT [symbolic = @I.WithSelf.%I.WithSelf.TakesT (constants.%I.WithSelf.TakesT)] {
// CHECK:STDOUT:       %self.param_patt.loc11_13.1: @I.WithSelf.TakesT.%pattern_type.loc11_13 (%pattern_type.1c8) = value_param_pattern [symbolic = %self.param_patt.loc11_13.2 (constants.%self.param_patt.4a9)]
// CHECK:STDOUT:       %self.patt.loc11_13.1: @I.WithSelf.TakesT.%pattern_type.loc11_13 (%pattern_type.1c8) = at_binding_pattern self, %self.param_patt.loc11_13.1 [symbolic = %self.patt.loc11_13.2 (constants.%self.patt.452)]
// CHECK:STDOUT:       %t.param_patt.loc11_20.1: @I.WithSelf.TakesT.%pattern_type.loc11_20 (%pattern_type.a0e) = value_param_pattern [symbolic = %t.param_patt.loc11_20.2 (constants.%t.param_patt)]
// CHECK:STDOUT:       %t.patt.loc11_20.1: @I.WithSelf.TakesT.%pattern_type.loc11_20 (%pattern_type.a0e) = at_binding_pattern t, %t.param_patt.loc11_20.1 [symbolic = %t.patt.loc11_20.2 (constants.%t.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @I.WithSelf.TakesT.%Self.as_type.loc11_13.1 (%Self.as_type.363) = value_param call_param0
// CHECK:STDOUT:       %.loc11_13.1: type = splice_block %.loc11_13.3 [symbolic = %Self.as_type.loc11_13.1 (constants.%Self.as_type.363)] {
// CHECK:STDOUT:         %.loc11_13.2: @I.WithSelf.TakesT.%I.type (%I.type.282) = specific_constant @I.%Self.loc6_20.1, @I(constants.%T) [symbolic = %Self (constants.%Self.13d)]
// CHECK:STDOUT:         %Self.ref: @I.WithSelf.TakesT.%I.type (%I.type.282) = name_ref Self, %.loc11_13.2 [symbolic = %Self (constants.%Self.13d)]
// CHECK:STDOUT:         %Self.as_type.loc11_13.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc11_13.1 (constants.%Self.as_type.363)]
// CHECK:STDOUT:         %.loc11_13.3: type = converted %Self.ref, %Self.as_type.loc11_13.2 [symbolic = %Self.as_type.loc11_13.1 (constants.%Self.as_type.363)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @I.WithSelf.TakesT.%Self.as_type.loc11_13.1 (%Self.as_type.363) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %t.param: @I.WithSelf.TakesT.%T.as_type.loc11_22.1 (%T.as_type) = value_param call_param1
// CHECK:STDOUT:       %.loc11_22.1: type = splice_block %.loc11_22.2 [symbolic = %T.as_type.loc11_22.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref: %A.type = name_ref T, @I.%T.loc6_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:         %T.as_type.loc11_22.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_22.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc11_22.2: type = converted %T.ref, %T.as_type.loc11_22.2 [symbolic = %T.as_type.loc11_22.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: @I.WithSelf.TakesT.%T.as_type.loc11_22.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc3.loc11_24.1: @I.WithSelf.%I.assoc_type (%I.assoc_type) = assoc_entity element3, %I.WithSelf.TakesT.decl [symbolic = %assoc3.loc11_24.2 (constants.%assoc3)]
// CHECK:STDOUT:     %T.ref: %A.type = name_ref T, @I.%T.loc6_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %impl.elem0.loc13: %facet_type.208 = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.ref: %facet_type.208 = name_ref U, %impl.elem0.loc13 [symbolic = %impl.elem0.loc8 (constants.%impl.elem0)]
// CHECK:STDOUT:     %U.as_type.loc13_16.1: type = facet_access_type %U.ref [symbolic = %as_type.loc13_16 (constants.%as_type.49a)]
// CHECK:STDOUT:     %.loc13_16.1: type = converted %U.ref, %U.as_type.loc13_16.1 [symbolic = %as_type.loc13_16 (constants.%as_type.49a)]
// CHECK:STDOUT:     %T.as_type.loc13_11.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc13_11: type = converted %T.ref, %T.as_type.loc13_11.1 [symbolic = %T.as_type.loc13_11.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem1.loc13_21.1: %facet_type.392 = impl_witness_access constants.%I.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc13_21.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %V.ref: %facet_type.392 = name_ref V, %impl.elem1.loc13_21.1 [symbolic = %impl.elem1.loc13_21.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %V.as_type: type = facet_access_type %V.ref [symbolic = %as_type.loc13_21 (constants.%as_type.64d)]
// CHECK:STDOUT:     %.loc13_21: type = converted %V.ref, %V.as_type [symbolic = %as_type.loc13_21 (constants.%as_type.64d)]
// CHECK:STDOUT:     %U.as_type.loc13_16.2: type = facet_access_type %U.ref [symbolic = %as_type.loc13_16 (constants.%as_type.49a)]
// CHECK:STDOUT:     %.loc13_16.2: type = converted %U.ref, %U.as_type.loc13_16.2 [symbolic = %as_type.loc13_16 (constants.%as_type.49a)]
// CHECK:STDOUT:     %I.observe.decl = observe_decl @I.observe [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc6_20.1
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .U = @U.%assoc0
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .V = @V.%assoc1
// CHECK:STDOUT:     .GetV = @I.WithSelf.%assoc2.loc10_21.1
// CHECK:STDOUT:     .TakesT = @I.WithSelf.%assoc3.loc11_24.1
// CHECK:STDOUT:     witness = (@I.WithSelf.%U, @I.WithSelf.%V, @I.WithSelf.%I.WithSelf.GetV.decl, @I.WithSelf.%I.WithSelf.TakesT.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @I.observe {
// CHECK:STDOUT:       observe constants.%T == constants.%impl.elem0 == constants.%impl.elem1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.GetV(@I.%T.loc6_14.2: %A.type, @I.%Self.loc6_20.1: @I.%I.type (%I.type.282)) {
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.282)]
// CHECK:STDOUT:   %Self: @I.WithSelf.GetV.%I.type (%I.type.282) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.13d)]
// CHECK:STDOUT:   %Self.as_type.loc10_11.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_11.1 (constants.%Self.as_type.363)]
// CHECK:STDOUT:   %pattern_type.loc10_11: type = pattern_type %Self.as_type.loc10_11.1 [symbolic = %pattern_type.loc10_11 (constants.%pattern_type.1c8)]
// CHECK:STDOUT:   %self.param_patt.loc10_11.2: @I.WithSelf.GetV.%pattern_type.loc10_11 (%pattern_type.1c8) = value_param_pattern [symbolic = %self.param_patt.loc10_11.2 (constants.%self.param_patt.4a9)]
// CHECK:STDOUT:   %self.patt.loc10_11.2: @I.WithSelf.GetV.%pattern_type.loc10_11 (%pattern_type.1c8) = at_binding_pattern self, %self.param_patt.loc10_11.2 [symbolic = %self.patt.loc10_11.2 (constants.%self.patt.452)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I, @I(%T) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem1.loc10_20.1: %facet_type.392 = impl_witness_access %I.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc10_20.1 (constants.%impl.elem1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem1.loc10_20.1 [symbolic = %as_type (constants.%as_type.64d)]
// CHECK:STDOUT:   %.loc10_20.1: Core.Form = init_form %as_type [symbolic = %.loc10_20.1 (constants.%.04f)]
// CHECK:STDOUT:   %pattern_type.loc10_20: type = pattern_type %as_type [symbolic = %pattern_type.loc10_20 (constants.%pattern_type.a24)]
// CHECK:STDOUT:   %return.param_patt.loc10_20.2: @I.WithSelf.GetV.%pattern_type.loc10_20 (%pattern_type.a24) = out_param_pattern [symbolic = %return.param_patt.loc10_20.2 (constants.%return.param_patt.49a)]
// CHECK:STDOUT:   %return.patt.loc10_17.2: @I.WithSelf.GetV.%pattern_type.loc10_20 (%pattern_type.a24) = return_slot_pattern %return.param_patt.loc10_20.2, %as_type [symbolic = %return.patt.loc10_17.2 (constants.%return.patt.4d9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.GetV.%Self.as_type.loc10_11.1 (%Self.as_type.363)) -> out %return.param: @I.WithSelf.GetV.%as_type (%as_type.64d);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.TakesT(@I.%T.loc6_14.2: %A.type, @I.%Self.loc6_20.1: @I.%I.type (%I.type.282)) {
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.282)]
// CHECK:STDOUT:   %Self: @I.WithSelf.TakesT.%I.type (%I.type.282) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.13d)]
// CHECK:STDOUT:   %Self.as_type.loc11_13.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc11_13.1 (constants.%Self.as_type.363)]
// CHECK:STDOUT:   %pattern_type.loc11_13: type = pattern_type %Self.as_type.loc11_13.1 [symbolic = %pattern_type.loc11_13 (constants.%pattern_type.1c8)]
// CHECK:STDOUT:   %self.param_patt.loc11_13.2: @I.WithSelf.TakesT.%pattern_type.loc11_13 (%pattern_type.1c8) = value_param_pattern [symbolic = %self.param_patt.loc11_13.2 (constants.%self.param_patt.4a9)]
// CHECK:STDOUT:   %self.patt.loc11_13.2: @I.WithSelf.TakesT.%pattern_type.loc11_13 (%pattern_type.1c8) = at_binding_pattern self, %self.param_patt.loc11_13.2 [symbolic = %self.patt.loc11_13.2 (constants.%self.patt.452)]
// CHECK:STDOUT:   %T.as_type.loc11_22.1: type = facet_access_type %T [symbolic = %T.as_type.loc11_22.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc11_20: type = pattern_type %T.as_type.loc11_22.1 [symbolic = %pattern_type.loc11_20 (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %t.param_patt.loc11_20.2: @I.WithSelf.TakesT.%pattern_type.loc11_20 (%pattern_type.a0e) = value_param_pattern [symbolic = %t.param_patt.loc11_20.2 (constants.%t.param_patt)]
// CHECK:STDOUT:   %t.patt.loc11_20.2: @I.WithSelf.TakesT.%pattern_type.loc11_20 (%pattern_type.a0e) = at_binding_pattern t, %t.param_patt.loc11_20.2 [symbolic = %t.patt.loc11_20.2 (constants.%t.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.TakesT.%Self.as_type.loc11_13.1 (%Self.as_type.363), %t.param: @I.WithSelf.TakesT.%T.as_type.loc11_22.1 (%T.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.d0c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.b45) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.62a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.13d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.GetV(constants.%T, constants.%Self.13d) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.282
// CHECK:STDOUT:   %Self => constants.%Self.13d
// CHECK:STDOUT:   %Self.as_type.loc10_11.1 => constants.%Self.as_type.363
// CHECK:STDOUT:   %pattern_type.loc10_11 => constants.%pattern_type.1c8
// CHECK:STDOUT:   %self.param_patt.loc10_11.2 => constants.%self.param_patt.4a9
// CHECK:STDOUT:   %self.patt.loc10_11.2 => constants.%self.patt.452
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem1.loc10_20.1 => constants.%impl.elem1
// CHECK:STDOUT:   %as_type => constants.%as_type.64d
// CHECK:STDOUT:   %.loc10_20.1 => constants.%.04f
// CHECK:STDOUT:   %pattern_type.loc10_20 => constants.%pattern_type.a24
// CHECK:STDOUT:   %return.param_patt.loc10_20.2 => constants.%return.param_patt.49a
// CHECK:STDOUT:   %return.patt.loc10_17.2 => constants.%return.patt.4d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.TakesT(constants.%T, constants.%Self.13d) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.282
// CHECK:STDOUT:   %Self => constants.%Self.13d
// CHECK:STDOUT:   %Self.as_type.loc11_13.1 => constants.%Self.as_type.363
// CHECK:STDOUT:   %pattern_type.loc11_13 => constants.%pattern_type.1c8
// CHECK:STDOUT:   %self.param_patt.loc11_13.2 => constants.%self.param_patt.4a9
// CHECK:STDOUT:   %self.patt.loc11_13.2 => constants.%self.patt.452
// CHECK:STDOUT:   %T.as_type.loc11_22.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc11_20 => constants.%pattern_type.a0e
// CHECK:STDOUT:   %t.param_patt.loc11_20.2 => constants.%t.param_patt
// CHECK:STDOUT:   %t.patt.loc11_20.2 => constants.%t.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.626: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %W.patt: %pattern_type.029 = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %W: %A.type = symbolic_binding W, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.2822e1.1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %T.patt: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.ce3d89.1: %I.type.2822e1.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.ee8ac3.1: <witness> = lookup_impl_witness %Self.ce3d89.1, @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.659: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %facet_type.079: type = facet_type <@Destroy & @C where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem1.5d5064.1: %facet_type.079 = impl_witness_access %I.lookup_impl_witness.ee8ac3.1, element1 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.ed4: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %facet_type.0e2: type = facet_type <@Destroy & @B where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem0.74f2f5.1: %facet_type.0e2 = impl_witness_access %I.lookup_impl_witness.ee8ac3.1, element0 [symbolic]
// CHECK:STDOUT:   %as_type.4496b9.1: type = facet_access_type %impl.elem1.5d5064.1 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %I.assoc_type.48d412.1: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.TakesT.type.998dca.1: type = fn_type @I.WithSelf.TakesT, @I.WithSelf(%T, %Self.ce3d89.1) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.TakesT.d84984.1: %I.WithSelf.TakesT.type.998dca.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.563: type = facet_access_type %Self.ce3d89.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.4ed4cb.1: type = pattern_type %Self.as_type.563 [symbolic]
// CHECK:STDOUT:   %self.param_patt.528759.1: %pattern_type.4ed4cb.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.49f9eb.1: %pattern_type.4ed4cb.1 = at_binding_pattern self, %self.param_patt.528759.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e098.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt.ab8b6c.1: %pattern_type.a0e098.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.e3ace8.1: %pattern_type.a0e098.1 = at_binding_pattern t, %t.param_patt.ab8b6c.1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.GetV.type.31c601.1: type = fn_type @I.WithSelf.GetV, @I.WithSelf(%T, %Self.ce3d89.1) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.GetV.9e4424.1: %I.WithSelf.GetV.type.31c601.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.bc8080.1: type = pattern_type %as_type.4496b9.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.ad58e0.1: %pattern_type.bc8080.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f17c51.1: %pattern_type.bc8080.1 = return_slot_pattern %return.param_patt.ad58e0.1, invalid [symbolic]
// CHECK:STDOUT:   %.503c64.1: Core.Form = init_form %as_type.4496b9.1 [symbolic]
// CHECK:STDOUT:   %I.type.2822e1.2: type = facet_type <@I, @I(%W)> [symbolic]
// CHECK:STDOUT:   %pattern_type.feb: type = pattern_type %I.type.2822e1.2 [symbolic]
// CHECK:STDOUT:   %X.patt: %pattern_type.feb = symbolic_binding_pattern X, 1 [symbolic]
// CHECK:STDOUT:   %X: %I.type.2822e1.2 = symbolic_binding X, 1 [symbolic]
// CHECK:STDOUT:   %Self.ce3d89.2: %I.type.2822e1.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.48d412.2: type = assoc_entity_type @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.7822c0.2: %I.assoc_type.48d412.2 = assoc_entity element0, imports.%Main.import_ref.4e9 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.ee8ac3.2: <witness> = lookup_impl_witness %Self.ce3d89.1, @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %impl.elem0.74f2f5.2: %facet_type.0e2 = impl_witness_access %I.lookup_impl_witness.ee8ac3.2, element0 [symbolic]
// CHECK:STDOUT:   %assoc1.b01cb8.2: %I.assoc_type.48d412.2 = assoc_entity element1, imports.%Main.import_ref.fef [symbolic]
// CHECK:STDOUT:   %I.WithSelf.GetV.type.31c601.2: type = fn_type @I.WithSelf.GetV, @I.WithSelf(%W, %Self.ce3d89.1) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.GetV.9e4424.2: %I.WithSelf.GetV.type.31c601.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc2.28c102.2: %I.assoc_type.48d412.2 = assoc_entity element2, imports.%Main.import_ref.01c [symbolic]
// CHECK:STDOUT:   %I.WithSelf.TakesT.type.998dca.2: type = fn_type @I.WithSelf.TakesT, @I.WithSelf(%W, %Self.ce3d89.1) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.TakesT.d84984.2: %I.WithSelf.TakesT.type.998dca.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc3.5d5782.2: %I.assoc_type.48d412.2 = assoc_entity element3, imports.%Main.import_ref.a6a [symbolic]
// CHECK:STDOUT:   %as_type.222b72.2: type = facet_access_type %impl.elem0.74f2f5.2 [symbolic]
// CHECK:STDOUT:   %W.as_type: type = facet_access_type %W [symbolic]
// CHECK:STDOUT:   %eq.cadb81.2 = observe_equivalent %W, %impl.elem0.74f2f5.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.5d5064.2: %facet_type.079 = impl_witness_access %I.lookup_impl_witness.ee8ac3.2, element1 [symbolic]
// CHECK:STDOUT:   %as_type.4496b9.2: type = facet_access_type %impl.elem1.5d5064.2 [symbolic]
// CHECK:STDOUT:   %eq.894cd4.2 = observe_equivalent %impl.elem0.74f2f5.2, %impl.elem1.5d5064.2 [symbolic]
// CHECK:STDOUT:   %X.as_type: type = facet_access_type %X [symbolic]
// CHECK:STDOUT:   %pattern_type.4ed4cb.2: type = pattern_type %X.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.4ed4cb.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.4ed4cb.2 = at_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.068: <witness> = require_complete_type %X.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.a4f: <witness> = require_complete_type %I.type.2822e1.2 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.ee8ac3.3: <witness> = lookup_impl_witness %X, @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %impl.elem0.74f2f5.3: %facet_type.0e2 = impl_witness_access %I.lookup_impl_witness.ee8ac3.3, element0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.GetV.type.31c601.3: type = fn_type @I.WithSelf.GetV, @I.WithSelf(%W, %X) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.GetV.9e4424.3: %I.WithSelf.GetV.type.31c601.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.WithSelf.TakesT.type.998dca.3: type = fn_type @I.WithSelf.TakesT, @I.WithSelf(%W, %X) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.TakesT.d84984.3: %I.WithSelf.TakesT.type.998dca.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %as_type.222b72.3: type = facet_access_type %impl.elem0.74f2f5.3 [symbolic]
// CHECK:STDOUT:   %eq.cadb81.3 = observe_equivalent %W, %impl.elem0.74f2f5.3 [symbolic]
// CHECK:STDOUT:   %impl.elem1.5d5064.3: %facet_type.079 = impl_witness_access %I.lookup_impl_witness.ee8ac3.3, element1 [symbolic]
// CHECK:STDOUT:   %as_type.4496b9.3: type = facet_access_type %impl.elem1.5d5064.3 [symbolic]
// CHECK:STDOUT:   %eq.894cd4.3 = observe_equivalent %impl.elem0.74f2f5.3, %impl.elem1.5d5064.3 [symbolic]
// CHECK:STDOUT:   %assoc1.37f: %I.assoc_type.48d412.1 = assoc_entity element1, imports.%Main.import_ref.c1b [symbolic]
// CHECK:STDOUT:   %require_complete.b29: <witness> = require_complete_type %as_type.4496b9.3 [symbolic]
// CHECK:STDOUT:   %pattern_type.bc8080.2: type = pattern_type %as_type.4496b9.3 [symbolic]
// CHECK:STDOUT:   %v.patt: %pattern_type.bc8080.2 = ref_binding_pattern v [symbolic]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.bc8080.2 = var_pattern %v.patt [symbolic]
// CHECK:STDOUT:   %assoc2.677: %I.assoc_type.48d412.1 = assoc_entity element2, imports.%Main.import_ref.4f2 [symbolic]
// CHECK:STDOUT:   %.21b: type = fn_type_with_self_type %I.WithSelf.GetV.type.31c601.3, %X [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.21b = impl_witness_access %I.lookup_impl_witness.ee8ac3.3, element2 [symbolic]
// CHECK:STDOUT:   %self.param_patt.528759.2: %pattern_type.4ed4cb.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.49f9eb.2: %pattern_type.4ed4cb.2 = at_binding_pattern self, %self.param_patt.528759.2 [symbolic]
// CHECK:STDOUT:   %.503c64.2: Core.Form = init_form %as_type.4496b9.3 [symbolic]
// CHECK:STDOUT:   %return.param_patt.ad58e0.2: %pattern_type.bc8080.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f17c51.2: %pattern_type.bc8080.2 = return_slot_pattern %return.param_patt.ad58e0.2, invalid [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4ac: <specific function> = specific_impl_function %impl.elem2, @I.WithSelf.GetV(%W, %X) [symbolic]
// CHECK:STDOUT:   %assoc3.c50: %I.assoc_type.48d412.1 = assoc_entity element3, imports.%Main.import_ref.778 [symbolic]
// CHECK:STDOUT:   %.a34: type = fn_type_with_self_type %I.WithSelf.TakesT.type.998dca.3, %X [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.a34 = impl_witness_access %I.lookup_impl_witness.ee8ac3.3, element3 [symbolic]
// CHECK:STDOUT:   %as_type.abb: %A.type = facet_access_type %impl.elem1.5d5064.3 [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e098.2: type = pattern_type %W.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt.ab8b6c.2: %pattern_type.a0e098.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.e3ace8.2: %pattern_type.a0e098.2 = at_binding_pattern t, %t.param_patt.ab8b6c.2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.33f: <specific function> = specific_impl_function %impl.elem3, @I.WithSelf.TakesT(%W, %X) [symbolic]
// CHECK:STDOUT:   %require_complete.94f: <witness> = require_complete_type %W.as_type [symbolic]
// CHECK:STDOUT:   %facet_type.9c1: type = facet_type <@Destroy & @C> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.5d5064.3, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.4496b9.3, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7d4: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.850: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7d4, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.983: %.850 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ad4: <specific function> = specific_impl_function %impl.elem0.983, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.A: type = import_ref Main//generic_observe, A, loaded [concrete = constants.%A.type]
// CHECK:STDOUT:   %Main.B = import_ref Main//generic_observe, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//generic_observe, C, unloaded
// CHECK:STDOUT:   %Main.I: %I.type.335 = import_ref Main//generic_observe, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Core.048aa3.1: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.0a1 = import_ref Main//generic_observe, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.874 = import_ref Main//generic_observe, loc7_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.249: @I.WithSelf.%I.assoc_type (%I.assoc_type.48d412.1) = import_ref Main//generic_observe, loc8_8, loaded [symbolic = @I.WithSelf.%assoc1 (constants.%assoc1.37f)]
// CHECK:STDOUT:   %Main.import_ref.299: @I.WithSelf.%I.assoc_type (%I.assoc_type.48d412.1) = import_ref Main//generic_observe, loc10_21, loaded [symbolic = @I.WithSelf.%assoc2 (constants.%assoc2.677)]
// CHECK:STDOUT:   %Main.import_ref.730: @I.WithSelf.%I.assoc_type (%I.assoc_type.48d412.1) = import_ref Main//generic_observe, loc11_24, loaded [symbolic = @I.WithSelf.%assoc3 (constants.%assoc3.c50)]
// CHECK:STDOUT:   %Main.U = import_ref Main//generic_observe, U, unloaded
// CHECK:STDOUT:   %Main.V = import_ref Main//generic_observe, V, unloaded
// CHECK:STDOUT:   %Main.GetV = import_ref Main//generic_observe, GetV, unloaded
// CHECK:STDOUT:   %Main.TakesT = import_ref Main//generic_observe, TakesT, unloaded
// CHECK:STDOUT:   %Main.import_ref.fb3 = import_ref Main//generic_observe, loc5_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c07 = import_ref Main//generic_observe, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.a6a: @I.WithSelf.%I.WithSelf.TakesT.type (%I.WithSelf.TakesT.type.998dca.1) = import_ref Main//generic_observe, loc11_24, loaded [symbolic = @I.WithSelf.%I.WithSelf.TakesT (constants.%I.WithSelf.TakesT.d84984.1)]
// CHECK:STDOUT:   %Main.import_ref.1b4054.2: %A.type = import_ref Main//generic_observe, loc6_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.33bbe5.2: @I.%I.type (%I.type.2822e1.1) = import_ref Main//generic_observe, loc6_20, loaded [symbolic = @I.%Self (constants.%Self.ce3d89.1)]
// CHECK:STDOUT:   %Main.import_ref.01c: @I.WithSelf.%I.WithSelf.GetV.type (%I.WithSelf.GetV.type.31c601.1) = import_ref Main//generic_observe, loc10_21, loaded [symbolic = @I.WithSelf.%I.WithSelf.GetV (constants.%I.WithSelf.GetV.9e4424.1)]
// CHECK:STDOUT:   %Main.import_ref.1b4054.3: %A.type = import_ref Main//generic_observe, loc6_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.33bbe5.3: @I.%I.type (%I.type.2822e1.1) = import_ref Main//generic_observe, loc6_20, loaded [symbolic = @I.%Self (constants.%Self.ce3d89.1)]
// CHECK:STDOUT:   %Main.import_ref.fef: %facet_type.079 = import_ref Main//generic_observe, loc8_8, loaded [concrete = %V]
// CHECK:STDOUT:   %Main.import_ref.4e9 = import_ref Main//generic_observe, loc7_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.e82 = import_ref Main//generic_observe, loc6_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.1b4054.4: %A.type = import_ref Main//generic_observe, loc6_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.c1b = import_ref Main//generic_observe, loc8_8, unloaded
// CHECK:STDOUT:   %V: %facet_type.079 = assoc_const_decl @V [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.4f2 = import_ref Main//generic_observe, loc10_21, unloaded
// CHECK:STDOUT:   %Main.import_ref.778 = import_ref Main//generic_observe, loc11_24, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .Core = imports.%Core.048aa3.1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %W.patt.loc5_7.1: %pattern_type.029 = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc5_7.2 (constants.%W.patt)]
// CHECK:STDOUT:     %X.patt.loc5_14.1: @F.%pattern_type.loc5_14 (%pattern_type.feb) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc5_14.2 (constants.%X.patt)]
// CHECK:STDOUT:     %x.param_patt.loc5_24.1: @F.%pattern_type.loc5_24 (%pattern_type.4ed4cb.2) = value_param_pattern [symbolic = %x.param_patt.loc5_24.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc5_24.1: @F.%pattern_type.loc5_24 (%pattern_type.4ed4cb.2) = at_binding_pattern x, %x.param_patt.loc5_24.1 [symbolic = %x.patt.loc5_24.2 (constants.%x.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen.loc5_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, imports.%Main.A [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc5_7.2: %A.type = symbolic_binding W, 0 [symbolic = %W.loc5_7.1 (constants.%W)]
// CHECK:STDOUT:     %.loc5_20: type = splice_block %I.type.loc5_20.2 [symbolic = %I.type.loc5_20.1 (constants.%I.type.2822e1.2)] {
// CHECK:STDOUT:       %.Self.frozen.loc5_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:       %W.ref: %A.type = name_ref W, %W.loc5_7.2 [symbolic = %W.loc5_7.1 (constants.%W)]
// CHECK:STDOUT:       %I.type.loc5_20.2: type = facet_type <@I, @I(constants.%W)> [symbolic = %I.type.loc5_20.1 (constants.%I.type.2822e1.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc5_14.2: @F.%I.type.loc5_20.1 (%I.type.2822e1.2) = symbolic_binding X, 1 [symbolic = %X.loc5_14.1 (constants.%X)]
// CHECK:STDOUT:     %x.param: @F.%X.as_type.loc5_26.1 (%X.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_26.1: type = splice_block %.loc5_26.2 [symbolic = %X.as_type.loc5_26.1 (constants.%X.as_type)] {
// CHECK:STDOUT:       %X.ref.loc5: @F.%I.type.loc5_20.1 (%I.type.2822e1.2) = name_ref X, %X.loc5_14.2 [symbolic = %X.loc5_14.1 (constants.%X)]
// CHECK:STDOUT:       %X.as_type.loc5_26.2: type = facet_access_type %X.ref.loc5 [symbolic = %X.as_type.loc5_26.1 (constants.%X.as_type)]
// CHECK:STDOUT:       %.loc5_26.2: type = converted %X.ref.loc5, %X.as_type.loc5_26.2 [symbolic = %X.as_type.loc5_26.1 (constants.%X.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%X.as_type.loc5_26.1 (%X.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A [from "generic_observe.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.0a1
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.1b4054.4: %A.type) [from "generic_observe.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.029 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2822e1.1)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.2822e1.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.ce3d89.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.e82
// CHECK:STDOUT:     .U = imports.%Main.import_ref.874
// CHECK:STDOUT:     .V = imports.%Main.import_ref.249
// CHECK:STDOUT:     .GetV = imports.%Main.import_ref.299
// CHECK:STDOUT:     .TakesT = imports.%Main.import_ref.730
// CHECK:STDOUT:     witness = (imports.%Main.U, imports.%Main.V, imports.%Main.GetV, imports.%Main.TakesT)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:     @I.observe {
// CHECK:STDOUT:       observe constants.%T == constants.%impl.elem0.74f2f5.1 == constants.%impl.elem1.5d5064.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C [from "generic_observe.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.fb3
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B [from "generic_observe.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.c07
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:
// CHECK:STDOUT: !observes:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.TakesT(imports.%Main.import_ref.1b4054.2: %A.type, imports.%Main.import_ref.33bbe5.2: @I.%I.type (%I.type.2822e1.1)) [from "generic_observe.carbon"] {
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2822e1.1)]
// CHECK:STDOUT:   %Self: @I.WithSelf.TakesT.%I.type (%I.type.2822e1.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.ce3d89.1)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.563)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.4ed4cb.1)]
// CHECK:STDOUT:   %self.param_patt: @I.WithSelf.TakesT.%pattern_type.1 (%pattern_type.4ed4cb.1) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.528759.1)]
// CHECK:STDOUT:   %self.patt: @I.WithSelf.TakesT.%pattern_type.1 (%pattern_type.4ed4cb.1) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.49f9eb.1)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T.as_type [symbolic = %pattern_type.2 (constants.%pattern_type.a0e098.1)]
// CHECK:STDOUT:   %t.param_patt: @I.WithSelf.TakesT.%pattern_type.2 (%pattern_type.a0e098.1) = value_param_pattern [symbolic = %t.param_patt (constants.%t.param_patt.ab8b6c.1)]
// CHECK:STDOUT:   %t.patt: @I.WithSelf.TakesT.%pattern_type.2 (%pattern_type.a0e098.1) = at_binding_pattern t, %t.param_patt [symbolic = %t.patt (constants.%t.patt.e3ace8.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.GetV(imports.%Main.import_ref.1b4054.3: %A.type, imports.%Main.import_ref.33bbe5.3: @I.%I.type (%I.type.2822e1.1)) [from "generic_observe.carbon"] {
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2822e1.1)]
// CHECK:STDOUT:   %Self: @I.WithSelf.GetV.%I.type (%I.type.2822e1.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.ce3d89.1)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.563)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.4ed4cb.1)]
// CHECK:STDOUT:   %self.param_patt: @I.WithSelf.GetV.%pattern_type.1 (%pattern_type.4ed4cb.1) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.528759.1)]
// CHECK:STDOUT:   %self.patt: @I.WithSelf.GetV.%pattern_type.1 (%pattern_type.4ed4cb.1) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.49f9eb.1)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I, @I(%T) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.ee8ac3.1)]
// CHECK:STDOUT:   %impl.elem1: %facet_type.079 = impl_witness_access %I.lookup_impl_witness, element1 [symbolic = %impl.elem1 (constants.%impl.elem1.5d5064.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem1 [symbolic = %as_type (constants.%as_type.4496b9.1)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %as_type [symbolic = %.1 (constants.%.503c64.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.bc8080.1)]
// CHECK:STDOUT:   %return.param_patt: @I.WithSelf.GetV.%pattern_type.2 (%pattern_type.bc8080.1) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.ad58e0.1)]
// CHECK:STDOUT:   %return.patt: @I.WithSelf.GetV.%pattern_type.2 (%pattern_type.bc8080.1) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.f17c51.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%W.loc5_7.2: %A.type, %X.loc5_14.2: @F.%I.type.loc5_20.1 (%I.type.2822e1.2)) {
// CHECK:STDOUT:   %W.patt.loc5_7.2: %pattern_type.029 = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc5_7.2 (constants.%W.patt)]
// CHECK:STDOUT:   %W.loc5_7.1: %A.type = symbolic_binding W, 0 [symbolic = %W.loc5_7.1 (constants.%W)]
// CHECK:STDOUT:   %I.type.loc5_20.1: type = facet_type <@I, @I(%W.loc5_7.1)> [symbolic = %I.type.loc5_20.1 (constants.%I.type.2822e1.2)]
// CHECK:STDOUT:   %pattern_type.loc5_14: type = pattern_type %I.type.loc5_20.1 [symbolic = %pattern_type.loc5_14 (constants.%pattern_type.feb)]
// CHECK:STDOUT:   %X.patt.loc5_14.2: @F.%pattern_type.loc5_14 (%pattern_type.feb) = symbolic_binding_pattern X, 1 [symbolic = %X.patt.loc5_14.2 (constants.%X.patt)]
// CHECK:STDOUT:   %X.loc5_14.1: @F.%I.type.loc5_20.1 (%I.type.2822e1.2) = symbolic_binding X, 1 [symbolic = %X.loc5_14.1 (constants.%X)]
// CHECK:STDOUT:   %X.as_type.loc5_26.1: type = facet_access_type %X.loc5_14.1 [symbolic = %X.as_type.loc5_26.1 (constants.%X.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5_24: type = pattern_type %X.as_type.loc5_26.1 [symbolic = %pattern_type.loc5_24 (constants.%pattern_type.4ed4cb.2)]
// CHECK:STDOUT:   %x.param_patt.loc5_24.2: @F.%pattern_type.loc5_24 (%pattern_type.4ed4cb.2) = value_param_pattern [symbolic = %x.param_patt.loc5_24.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc5_24.2: @F.%pattern_type.loc5_24 (%pattern_type.4ed4cb.2) = at_binding_pattern x, %x.param_patt.loc5_24.2 [symbolic = %x.patt.loc5_24.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5: <witness> = require_complete_type %X.as_type.loc5_26.1 [symbolic = %require_complete.loc5 (constants.%require_complete.068)]
// CHECK:STDOUT:   %require_complete.loc6_10: <witness> = require_complete_type %I.type.loc5_20.1 [symbolic = %require_complete.loc6_10 (constants.%require_complete.a4f)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%W.loc5_7.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.48d412.2)]
// CHECK:STDOUT:   %assoc1: @F.%I.assoc_type (%I.assoc_type.48d412.2) = assoc_entity element1, imports.%Main.import_ref.fef [symbolic = %assoc1 (constants.%assoc1.b01cb8.2)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %X.loc5_14.1, @I, @I(%W.loc5_7.1) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.ee8ac3.3)]
// CHECK:STDOUT:   %impl.elem1.loc6_11.2: %facet_type.079 = impl_witness_access %I.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc6_11.2 (constants.%impl.elem1.5d5064.3)]
// CHECK:STDOUT:   %as_type.loc6_11.2: type = facet_access_type %impl.elem1.loc6_11.2 [symbolic = %as_type.loc6_11.2 (constants.%as_type.4496b9.3)]
// CHECK:STDOUT:   %require_complete.loc6_11: <witness> = require_complete_type %as_type.loc6_11.2 [symbolic = %require_complete.loc6_11 (constants.%require_complete.b29)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %as_type.loc6_11.2 [symbolic = %pattern_type.loc6 (constants.%pattern_type.bc8080.2)]
// CHECK:STDOUT:   %v.patt.loc6_8.2: @F.%pattern_type.loc6 (%pattern_type.bc8080.2) = ref_binding_pattern v [symbolic = %v.patt.loc6_8.2 (constants.%v.patt)]
// CHECK:STDOUT:   %v.var_patt.loc6_3.2: @F.%pattern_type.loc6 (%pattern_type.bc8080.2) = var_pattern %v.patt.loc6_8.2 [symbolic = %v.var_patt.loc6_3.2 (constants.%v.var_patt)]
// CHECK:STDOUT:   %assoc2: @F.%I.assoc_type (%I.assoc_type.48d412.2) = assoc_entity element2, imports.%Main.import_ref.01c [symbolic = %assoc2 (constants.%assoc2.28c102.2)]
// CHECK:STDOUT:   %I.WithSelf.GetV.type: type = fn_type @I.WithSelf.GetV, @I.WithSelf(%W.loc5_7.1, %X.loc5_14.1) [symbolic = %I.WithSelf.GetV.type (constants.%I.WithSelf.GetV.type.31c601.3)]
// CHECK:STDOUT:   %.loc6_17.2: type = fn_type_with_self_type %I.WithSelf.GetV.type, %X.loc5_14.1 [symbolic = %.loc6_17.2 (constants.%.21b)]
// CHECK:STDOUT:   %impl.elem2.loc6_17.2: @F.%.loc6_17.2 (%.21b) = impl_witness_access %I.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc6_17.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_17.2: <specific function> = specific_impl_function %impl.elem2.loc6_17.2, @I.WithSelf.GetV(%W.loc5_7.1, %X.loc5_14.1) [symbolic = %specific_impl_fn.loc6_17.2 (constants.%specific_impl_fn.4ac)]
// CHECK:STDOUT:   %assoc3: @F.%I.assoc_type (%I.assoc_type.48d412.2) = assoc_entity element3, imports.%Main.import_ref.a6a [symbolic = %assoc3 (constants.%assoc3.5d5782.2)]
// CHECK:STDOUT:   %I.WithSelf.TakesT.type: type = fn_type @I.WithSelf.TakesT, @I.WithSelf(%W.loc5_7.1, %X.loc5_14.1) [symbolic = %I.WithSelf.TakesT.type (constants.%I.WithSelf.TakesT.type.998dca.3)]
// CHECK:STDOUT:   %.loc8_4.2: type = fn_type_with_self_type %I.WithSelf.TakesT.type, %X.loc5_14.1 [symbolic = %.loc8_4.2 (constants.%.a34)]
// CHECK:STDOUT:   %impl.elem3.loc8_4.2: @F.%.loc8_4.2 (%.a34) = impl_witness_access %I.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc8_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %as_type.loc8: %A.type = facet_access_type %impl.elem1.loc6_11.2 [symbolic = %as_type.loc8 (constants.%as_type.abb)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_4.2: <specific function> = specific_impl_function %impl.elem3.loc8_4.2, @I.WithSelf.TakesT(%W.loc5_7.1, %X.loc5_14.1) [symbolic = %specific_impl_fn.loc8_4.2 (constants.%specific_impl_fn.33f)]
// CHECK:STDOUT:   %W.as_type: type = facet_access_type %W.loc5_7.1 [symbolic = %W.as_type (constants.%W.as_type)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %W.as_type [symbolic = %require_complete.loc8 (constants.%require_complete.94f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.loc6_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet.loc6_3.3: %Destroy.type = facet_value %as_type.loc6_11.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc6_3.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.7d4)]
// CHECK:STDOUT:   %.loc6_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc6_3.3 [symbolic = %.loc6_3.4 (constants.%.850)]
// CHECK:STDOUT:   %impl.elem0.loc6_3.2: @F.%.loc6_3.4 (%.850) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.983)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_3.2: <specific function> = specific_impl_function %impl.elem0.loc6_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc6_3.3) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.ad4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%X.as_type.loc5_26.1 (%X.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @F.%as_type.loc6_11.2 (%as_type.4496b9.3) = var_storage %v.var_patt.loc6_3.1
// CHECK:STDOUT:     %x.ref.loc6: @F.%X.as_type.loc5_26.1 (%X.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_17.1: @F.%I.assoc_type (%I.assoc_type.48d412.2) = specific_constant imports.%Main.import_ref.299, @I.WithSelf(constants.%W, constants.%X) [symbolic = %assoc2 (constants.%assoc2.28c102.2)]
// CHECK:STDOUT:     %GetV.ref: @F.%I.assoc_type (%I.assoc_type.48d412.2) = name_ref GetV, %.loc6_17.1 [symbolic = %assoc2 (constants.%assoc2.28c102.2)]
// CHECK:STDOUT:     %impl.elem2.loc6_17.1: @F.%.loc6_17.2 (%.21b) = impl_witness_access constants.%I.lookup_impl_witness.ee8ac3.3, element2 [symbolic = %impl.elem2.loc6_17.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc6_17: <bound method> = bound_method %x.ref.loc6, %impl.elem2.loc6_17.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_17.1: <specific function> = specific_impl_function %impl.elem2.loc6_17.1, @I.WithSelf.GetV(constants.%W, constants.%X) [symbolic = %specific_impl_fn.loc6_17.2 (constants.%specific_impl_fn.4ac)]
// CHECK:STDOUT:     %bound_method.loc6_23: <bound method> = bound_method %x.ref.loc6, %specific_impl_fn.loc6_17.1
// CHECK:STDOUT:     %.loc6_3.1: ref @F.%as_type.loc6_11.2 (%as_type.4496b9.3) = splice_block %v.var {}
// CHECK:STDOUT:     %I.WithSelf.GetV.call: init @F.%as_type.loc6_11.2 (%as_type.4496b9.3) to %.loc6_3.1 = call %bound_method.loc6_23(%x.ref.loc6)
// CHECK:STDOUT:     assign %v.var, %I.WithSelf.GetV.call
// CHECK:STDOUT:     %.loc6_11.1: type = splice_block %.loc6_11.4 [symbolic = %as_type.loc6_11.2 (constants.%as_type.4496b9.3)] {
// CHECK:STDOUT:       %X.ref.loc6: @F.%I.type.loc5_20.1 (%I.type.2822e1.2) = name_ref X, %X.loc5_14.2 [symbolic = %X.loc5_14.1 (constants.%X)]
// CHECK:STDOUT:       %X.as_type.loc6: type = facet_access_type %X.ref.loc6 [symbolic = %X.as_type.loc5_26.1 (constants.%X.as_type)]
// CHECK:STDOUT:       %.loc6_11.2: type = converted %X.ref.loc6, %X.as_type.loc6 [symbolic = %X.as_type.loc5_26.1 (constants.%X.as_type)]
// CHECK:STDOUT:       %.loc6_11.3: @F.%I.assoc_type (%I.assoc_type.48d412.2) = specific_constant imports.%Main.import_ref.249, @I.WithSelf(constants.%W, constants.%X) [symbolic = %assoc1 (constants.%assoc1.b01cb8.2)]
// CHECK:STDOUT:       %V.ref: @F.%I.assoc_type (%I.assoc_type.48d412.2) = name_ref V, %.loc6_11.3 [symbolic = %assoc1 (constants.%assoc1.b01cb8.2)]
// CHECK:STDOUT:       %impl.elem1.loc6_11.1: %facet_type.079 = impl_witness_access constants.%I.lookup_impl_witness.ee8ac3.3, element1 [symbolic = %impl.elem1.loc6_11.2 (constants.%impl.elem1.5d5064.3)]
// CHECK:STDOUT:       %as_type.loc6_11.1: type = facet_access_type %impl.elem1.loc6_11.1 [symbolic = %as_type.loc6_11.2 (constants.%as_type.4496b9.3)]
// CHECK:STDOUT:       %.loc6_11.4: type = converted %impl.elem1.loc6_11.1, %as_type.loc6_11.1 [symbolic = %as_type.loc6_11.2 (constants.%as_type.4496b9.3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%as_type.loc6_11.2 (%as_type.4496b9.3) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc6_8.1: @F.%pattern_type.loc6 (%pattern_type.bc8080.2) = ref_binding_pattern v [symbolic = %v.patt.loc6_8.2 (constants.%v.patt)]
// CHECK:STDOUT:       %v.var_patt.loc6_3.1: @F.%pattern_type.loc6 (%pattern_type.bc8080.2) = var_pattern %v.patt.loc6_8.1 [symbolic = %v.var_patt.loc6_3.2 (constants.%v.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.ref.loc8: @F.%X.as_type.loc5_26.1 (%X.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc8_4.1: @F.%I.assoc_type (%I.assoc_type.48d412.2) = specific_constant imports.%Main.import_ref.730, @I.WithSelf(constants.%W, constants.%X) [symbolic = %assoc3 (constants.%assoc3.5d5782.2)]
// CHECK:STDOUT:     %TakesT.ref: @F.%I.assoc_type (%I.assoc_type.48d412.2) = name_ref TakesT, %.loc8_4.1 [symbolic = %assoc3 (constants.%assoc3.5d5782.2)]
// CHECK:STDOUT:     %impl.elem3.loc8_4.1: @F.%.loc8_4.2 (%.a34) = impl_witness_access constants.%I.lookup_impl_witness.ee8ac3.3, element3 [symbolic = %impl.elem3.loc8_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc8_4: <bound method> = bound_method %x.ref.loc8, %impl.elem3.loc8_4.1
// CHECK:STDOUT:     %v.ref: ref @F.%as_type.loc6_11.2 (%as_type.4496b9.3) = name_ref v, %v
// CHECK:STDOUT:     %.loc8_13.1: %A.type = as_compatible constants.%as_type.4496b9.3 [symbolic = %as_type.loc8 (constants.%as_type.abb)]
// CHECK:STDOUT:     %.loc8_13.2: %A.type = as_compatible constants.%as_type.4496b9.3 [symbolic = %as_type.loc8 (constants.%as_type.abb)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_4.1: <specific function> = specific_impl_function %impl.elem3.loc8_4.1, @I.WithSelf.TakesT(constants.%W, constants.%X) [symbolic = %specific_impl_fn.loc8_4.2 (constants.%specific_impl_fn.33f)]
// CHECK:STDOUT:     %bound_method.loc8_13: <bound method> = bound_method %x.ref.loc8, %specific_impl_fn.loc8_4.1
// CHECK:STDOUT:     %.loc8_12: ref @F.%W.as_type (%W.as_type) = as_compatible %v.ref
// CHECK:STDOUT:     %I.WithSelf.TakesT.call: init %empty_tuple.type = call %bound_method.loc8_13(%x.ref.loc8, %.loc8_12)
// CHECK:STDOUT:     %facet_type: type = facet_type <@Destroy & @C> [concrete = constants.%facet_type.9c1]
// CHECK:STDOUT:     %impl.elem0.loc6_3.1: @F.%.loc6_3.4 (%.850) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.983)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc6_3.1
// CHECK:STDOUT:     %Destroy.facet.loc6_3.1: %Destroy.type = facet_value constants.%as_type.4496b9.3, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc6_3.2: %Destroy.type = converted constants.%as_type.4496b9.3, %Destroy.facet.loc6_3.1 [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %Destroy.facet.loc6_3.2: %Destroy.type = facet_value constants.%as_type.4496b9.3, (constants.%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %.loc6_3.3: %Destroy.type = converted constants.%as_type.4496b9.3, %Destroy.facet.loc6_3.2 [symbolic = %Destroy.facet.loc6_3.3 (constants.%Destroy.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc6_3.1: <specific function> = specific_impl_function %impl.elem0.loc6_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.ad4)]
// CHECK:STDOUT:     %bound_method.loc6_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc6_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc6_3.2(%v.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:
// CHECK:STDOUT:   !observes:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.626) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.659) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.ed4) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.ce3d89.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.TakesT(constants.%T, constants.%Self.ce3d89.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2822e1.1
// CHECK:STDOUT:   %Self => constants.%Self.ce3d89.1
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.563
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.4ed4cb.1
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.528759.1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.49f9eb.1
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.a0e098.1
// CHECK:STDOUT:   %t.param_patt => constants.%t.param_patt.ab8b6c.1
// CHECK:STDOUT:   %t.patt => constants.%t.patt.e3ace8.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.GetV(constants.%T, constants.%Self.ce3d89.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2822e1.1
// CHECK:STDOUT:   %Self => constants.%Self.ce3d89.1
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.563
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.4ed4cb.1
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.528759.1
// CHECK:STDOUT:   %self.patt => constants.%self.patt.49f9eb.1
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.ee8ac3.1
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.5d5064.1
// CHECK:STDOUT:   %as_type => constants.%as_type.4496b9.1
// CHECK:STDOUT:   %.1 => constants.%.503c64.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.bc8080.1
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.ad58e0.1
// CHECK:STDOUT:   %return.patt => constants.%return.patt.f17c51.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%W) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.2822e1.2
// CHECK:STDOUT:   %Self => constants.%Self.ce3d89.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%W, constants.%Self.ce3d89.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.48d412.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7822c0.2
// CHECK:STDOUT:   %I.type => constants.%I.type.2822e1.2
// CHECK:STDOUT:   %Self => constants.%Self.ce3d89.1
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.ee8ac3.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.74f2f5.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b01cb8.2
// CHECK:STDOUT:   %I.WithSelf.GetV.type => constants.%I.WithSelf.GetV.type.31c601.2
// CHECK:STDOUT:   %I.WithSelf.GetV => constants.%I.WithSelf.GetV.9e4424.2
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.28c102.2
// CHECK:STDOUT:   %I.WithSelf.TakesT.type => constants.%I.WithSelf.TakesT.type.998dca.2
// CHECK:STDOUT:   %I.WithSelf.TakesT => constants.%I.WithSelf.TakesT.d84984.2
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.5d5782.2
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.222b72.2
// CHECK:STDOUT:   %T.as_type => constants.%W.as_type
// CHECK:STDOUT:   %eq.1 => constants.%eq.cadb81.2
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.5d5064.2
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.4496b9.2
// CHECK:STDOUT:   %eq.2 => constants.%eq.894cd4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%W, constants.%X) {
// CHECK:STDOUT:   %W.patt.loc5_7.2 => constants.%W.patt
// CHECK:STDOUT:   %W.loc5_7.1 => constants.%W
// CHECK:STDOUT:   %I.type.loc5_20.1 => constants.%I.type.2822e1.2
// CHECK:STDOUT:   %pattern_type.loc5_14 => constants.%pattern_type.feb
// CHECK:STDOUT:   %X.patt.loc5_14.2 => constants.%X.patt
// CHECK:STDOUT:   %X.loc5_14.1 => constants.%X
// CHECK:STDOUT:   %X.as_type.loc5_26.1 => constants.%X.as_type
// CHECK:STDOUT:   %pattern_type.loc5_24 => constants.%pattern_type.4ed4cb.2
// CHECK:STDOUT:   %x.param_patt.loc5_24.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc5_24.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%W, constants.%X) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.48d412.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7822c0.2
// CHECK:STDOUT:   %I.type => constants.%I.type.2822e1.2
// CHECK:STDOUT:   %Self => constants.%X
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.ee8ac3.3
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.74f2f5.3
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b01cb8.2
// CHECK:STDOUT:   %I.WithSelf.GetV.type => constants.%I.WithSelf.GetV.type.31c601.3
// CHECK:STDOUT:   %I.WithSelf.GetV => constants.%I.WithSelf.GetV.9e4424.3
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.28c102.2
// CHECK:STDOUT:   %I.WithSelf.TakesT.type => constants.%I.WithSelf.TakesT.type.998dca.3
// CHECK:STDOUT:   %I.WithSelf.TakesT => constants.%I.WithSelf.TakesT.d84984.3
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.5d5782.2
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.222b72.3
// CHECK:STDOUT:   %T.as_type => constants.%W.as_type
// CHECK:STDOUT:   %eq.1 => constants.%eq.cadb81.3
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.5d5064.3
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.4496b9.3
// CHECK:STDOUT:   %eq.2 => constants.%eq.894cd4.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.GetV(constants.%W, constants.%X) {
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %I.type => constants.%I.type.2822e1.2
// CHECK:STDOUT:   %Self => constants.%X
// CHECK:STDOUT:   %Self.as_type => constants.%X.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.4ed4cb.2
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.528759.2
// CHECK:STDOUT:   %self.patt => constants.%self.patt.49f9eb.2
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.ee8ac3.3
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.5d5064.3
// CHECK:STDOUT:   %as_type => constants.%as_type.4496b9.3
// CHECK:STDOUT:   %.1 => constants.%.503c64.2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.bc8080.2
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.ad58e0.2
// CHECK:STDOUT:   %return.patt => constants.%return.patt.f17c51.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.TakesT(constants.%W, constants.%X) {
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:   %I.type => constants.%I.type.2822e1.2
// CHECK:STDOUT:   %Self => constants.%X
// CHECK:STDOUT:   %Self.as_type => constants.%X.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.4ed4cb.2
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.528759.2
// CHECK:STDOUT:   %self.patt => constants.%self.patt.49f9eb.2
// CHECK:STDOUT:   %T.as_type => constants.%W.as_type
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.a0e098.2
// CHECK:STDOUT:   %t.param_patt => constants.%t.param_patt.ab8b6c.2
// CHECK:STDOUT:   %t.patt => constants.%t.patt.e3ace8.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

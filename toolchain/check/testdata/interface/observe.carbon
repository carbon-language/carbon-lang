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

interface P { fn InP[self: Self](); }
interface Q { fn InQ[self: Self](); }
interface R { fn InR[self: Self](); }

interface Transitive {
  let A:! P & Core.Destroy;
  let B:! (Q where .Self == A) & Core.Destroy;
  let C:! (R where .Self == B) & Core.Destroy;

  fn GetA[self: Self]() -> A;
  fn TakesC[self: Self](c: C);

  observe A == B == C;
}

// --- equivalent.carbon
library "[[@TEST_NAME]]";

import library "transitive";

fn F[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  var b: T.B = a;
  b.InQ();

  a.(T.C.InR)();
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

fn RequiresA[T:! A](unused x: T) {}
fn F[T:! A, U:! B where .Self == T]() {
  observe T == U;

  fn InnerF(x: U) {
    RequiresA(x);
  }
}

// --- nested_functions_in_inner.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T:! A](unused x: T) {}
fn F[T:! A, U:! B where .Self == T]() {

  fn InnerF(x: U) {
    observe T == U;

    RequiresA(x);
  }
}

// --- fail_nested_functions_lexical_order.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T:! A](unused x: T) {}
fn F[T:! A, U:! B where .Self == T]() {

  fn InnerF(x: U) {
    // CHECK:STDERR: fail_nested_functions_lexical_order.carbon:[[@LINE+7]]:5: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     RequiresA(x);
    // CHECK:STDERR:     ^~~~~~~~~~~~
    // CHECK:STDERR: fail_nested_functions_lexical_order.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR: fn RequiresA[T:! A](unused x: T) {}
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    RequiresA(x);
  }

  observe T == U;
}

// --- fail_nested_functions_leak_upward.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T:! A](unused x: T) {}
fn F[T:! A, U:! B where .Self == T](x: U) {

  fn InnerF(unused x: U) {
    observe T == U;
  }

  // CHECK:STDERR: fail_nested_functions_leak_upward.carbon:[[@LINE+7]]:3: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_nested_functions_leak_upward.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T:! A](unused x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(x);
}

// --- fail_not_observed.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T:! A](unused x: T) {}
fn F[T:! A, U:! B where .Self == T, V:! B](x: U) {
  observe T == V;

  // CHECK:STDERR: fail_not_observed.carbon:[[@LINE+7]]:3: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_not_observed.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T:! A](unused x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(x);
}

// --- todo_fail_invalid_equivalent.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn TakesA[T:! A](unused x: T) {}
fn F[T:! A, U:! B](x: U) {
  observe T == U;
  TakesA(x);
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

// CHECK:STDOUT: --- transitive.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.d21: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.6c8: type = symbolic_binding_type Self, 0, %Self.d21 [symbolic]
// CHECK:STDOUT:   %pattern_type.2eb: type = pattern_type %Self.binding.as_type.6c8 [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P.WithSelf(%Self.d21) [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %P.assoc_type: type = assoc_entity_type @P [concrete]
// CHECK:STDOUT:   %assoc0.4ec: %P.assoc_type = assoc_entity element0, @P.WithSelf.%P.WithSelf.InP.decl [concrete]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.f64: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.1b2: type = symbolic_binding_type Self, 0, %Self.f64 [symbolic]
// CHECK:STDOUT:   %pattern_type.992: type = pattern_type %Self.binding.as_type.1b2 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Self.f64) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ: %Q.WithSelf.InQ.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.038: %Q.assoc_type = assoc_entity element0, @Q.WithSelf.%Q.WithSelf.InQ.decl [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.bea: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.296: type = symbolic_binding_type Self, 0, %Self.bea [symbolic]
// CHECK:STDOUT:   %pattern_type.351: type = pattern_type %Self.binding.as_type.296 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R.WithSelf(%Self.bea) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR: %R.WithSelf.InR.type = struct_value () [symbolic]
// CHECK:STDOUT:   %R.assoc_type: type = assoc_entity_type @R [concrete]
// CHECK:STDOUT:   %assoc0.dce: %R.assoc_type = assoc_entity element0, @R.WithSelf.%R.WithSelf.InR.decl [concrete]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.e2a: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type.de1: type = facet_type <@P & @Destroy> [concrete]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.5e0: %Transitive.assoc_type = assoc_entity element0, @Transitive.WithSelf.%A [concrete]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self.e2a, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0: %facet_type.de1 = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.7d8: type = facet_type <@Q & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.WithSelf.%B [concrete]
// CHECK:STDOUT:   %facet_type.37d: type = facet_type <@R & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.WithSelf.%C [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.e5b: type = symbolic_binding_type Self, 0, %Self.e2a [symbolic]
// CHECK:STDOUT:   %pattern_type.b0d: type = pattern_type %Self.binding.as_type.e5b [symbolic]
// CHECK:STDOUT:   %as_type.91d: type = facet_access_type %impl.elem0 [symbolic]
// CHECK:STDOUT:   %.a36: Core.Form = init_form %as_type.91d, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.485: type = pattern_type %as_type.91d [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%Self.e2a) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA: %Transitive.WithSelf.GetA.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, @Transitive.WithSelf.%Transitive.WithSelf.GetA.decl [concrete]
// CHECK:STDOUT:   %impl.elem2: %facet_type.37d = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %as_type.d58: type = facet_access_type %impl.elem2 [symbolic]
// CHECK:STDOUT:   %pattern_type.219: type = pattern_type %as_type.d58 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%Self.e2a) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC: %Transitive.WithSelf.TakesC.type = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, @Transitive.WithSelf.%Transitive.WithSelf.TakesC.decl [concrete]
// CHECK:STDOUT:   %Transitive.WithSelf.observe.decl = observe_decl @Transitive.WithSelf.observe [concrete]
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
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
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
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d21]
// CHECK:STDOUT:   %P.WithSelf.decl = interface_with_self_decl @P [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %P.WithSelf.InP.decl: @P.WithSelf.%P.WithSelf.InP.type (%P.WithSelf.InP.type) = fn_decl @P.WithSelf.InP [symbolic = @P.WithSelf.%P.WithSelf.InP (constants.%P.WithSelf.InP)] {
// CHECK:STDOUT:     %self.patt: @P.WithSelf.InP.%pattern_type (%pattern_type.2eb) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @P.WithSelf.InP.%pattern_type (%pattern_type.2eb) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @P.WithSelf.InP.%Self.binding.as_type (%Self.binding.as_type.6c8) = value_param call_param0
// CHECK:STDOUT:     %.loc3_28.1: type = splice_block %.loc3_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)] {
// CHECK:STDOUT:       %Self.ref: %P.type = name_ref Self, @P.%Self [symbolic = %Self (constants.%Self.d21)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:       %.loc3_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @P.WithSelf.InP.%Self.binding.as_type (%Self.binding.as_type.6c8) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %P.assoc_type = assoc_entity element0, %P.WithSelf.InP.decl [concrete = constants.%assoc0.4ec]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InP = @P.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@P.WithSelf.%P.WithSelf.InP.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f64]
// CHECK:STDOUT:   %Q.WithSelf.decl = interface_with_self_decl @Q [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Q.WithSelf.InQ.decl: @Q.WithSelf.%Q.WithSelf.InQ.type (%Q.WithSelf.InQ.type) = fn_decl @Q.WithSelf.InQ [symbolic = @Q.WithSelf.%Q.WithSelf.InQ (constants.%Q.WithSelf.InQ)] {
// CHECK:STDOUT:     %self.patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.992) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.992) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Q.WithSelf.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2) = value_param call_param0
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)] {
// CHECK:STDOUT:       %Self.ref: %Q.type = name_ref Self, @Q.%Self [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:       %.loc4_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Q.WithSelf.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Q.assoc_type = assoc_entity element0, %Q.WithSelf.InQ.decl [concrete = constants.%assoc0.038]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InQ = @Q.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@Q.WithSelf.%Q.WithSelf.InQ.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = constants.%Self.bea]
// CHECK:STDOUT:   %R.WithSelf.decl = interface_with_self_decl @R [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %R.WithSelf.InR.decl: @R.WithSelf.%R.WithSelf.InR.type (%R.WithSelf.InR.type) = fn_decl @R.WithSelf.InR [symbolic = @R.WithSelf.%R.WithSelf.InR (constants.%R.WithSelf.InR)] {
// CHECK:STDOUT:     %self.patt: @R.WithSelf.InR.%pattern_type (%pattern_type.351) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @R.WithSelf.InR.%pattern_type (%pattern_type.351) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @R.WithSelf.InR.%Self.binding.as_type (%Self.binding.as_type.296) = value_param call_param0
// CHECK:STDOUT:     %.loc5_28.1: type = splice_block %.loc5_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)] {
// CHECK:STDOUT:       %Self.ref: %R.type = name_ref Self, @R.%Self [symbolic = %Self (constants.%Self.bea)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:       %.loc5_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @R.WithSelf.InR.%Self.binding.as_type (%Self.binding.as_type.296) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %R.assoc_type = assoc_entity element0, %R.WithSelf.InR.decl [concrete = constants.%assoc0.dce]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InR = @R.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@R.WithSelf.%R.WithSelf.InR.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e2a]
// CHECK:STDOUT:   %Transitive.WithSelf.decl = interface_with_self_decl @Transitive [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A: %facet_type.de1 = assoc_const_decl @A [concrete] {
// CHECK:STDOUT:     %assoc0: %Transitive.assoc_type = assoc_entity element0, @Transitive.WithSelf.%A [concrete = constants.%assoc0.5e0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: %facet_type.7d8 = assoc_const_decl @B [concrete] {
// CHECK:STDOUT:     %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.WithSelf.%B [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: %facet_type.37d = assoc_const_decl @C [concrete] {
// CHECK:STDOUT:     %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.WithSelf.%C [concrete = constants.%assoc2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.decl: @Transitive.WithSelf.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type) = fn_decl @Transitive.WithSelf.GetA [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA)] {
// CHECK:STDOUT:     %self.patt: @Transitive.WithSelf.GetA.%pattern_type.loc12_11 (%pattern_type.b0d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Transitive.WithSelf.GetA.%pattern_type.loc12_11 (%pattern_type.b0d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Transitive.WithSelf.GetA.%pattern_type.loc12_25 (%pattern_type.485) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Transitive.WithSelf.GetA.%pattern_type.loc12_25 (%pattern_type.485) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc12_28.2: %facet_type.de1 = impl_witness_access constants.%Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_28.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %A.ref: %facet_type.de1 = name_ref A, %impl.elem0.loc12_28.2 [symbolic = %impl.elem0.loc12_28.1 (constants.%impl.elem0)]
// CHECK:STDOUT:     %A.as_type: type = facet_access_type %A.ref [symbolic = %as_type (constants.%as_type.91d)]
// CHECK:STDOUT:     %.loc12_28.2: type = converted %A.ref, %A.as_type [symbolic = %as_type (constants.%as_type.91d)]
// CHECK:STDOUT:     %.loc12_28.3: Core.Form = init_form %.loc12_28.2, call_param1 [symbolic = %.loc12_28.1 (constants.%.a36)]
// CHECK:STDOUT:     %self.param: @Transitive.WithSelf.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_param call_param0
// CHECK:STDOUT:     %.loc12_17.1: type = splice_block %.loc12_17.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:       %.loc12_17.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.WithSelf.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Transitive.WithSelf.GetA.%as_type (%as_type.91d) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Transitive.WithSelf.GetA.%as_type (%as_type.91d) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, %Transitive.WithSelf.GetA.decl [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.decl: @Transitive.WithSelf.%Transitive.WithSelf.TakesC.type (%Transitive.WithSelf.TakesC.type) = fn_decl @Transitive.WithSelf.TakesC [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.TakesC (constants.%Transitive.WithSelf.TakesC)] {
// CHECK:STDOUT:     %self.patt: @Transitive.WithSelf.TakesC.%pattern_type.loc13_13 (%pattern_type.b0d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.loc13_13 (%pattern_type.b0d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %c.patt: @Transitive.WithSelf.TakesC.%pattern_type.loc13_25 (%pattern_type.219) = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.loc13_25 (%pattern_type.219) = value_param_pattern %c.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Transitive.WithSelf.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_param call_param0
// CHECK:STDOUT:     %.loc13_19.1: type = splice_block %.loc13_19.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:       %.loc13_19.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.WithSelf.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_binding self, %self.param
// CHECK:STDOUT:     %c.param: @Transitive.WithSelf.TakesC.%as_type (%as_type.d58) = value_param call_param1
// CHECK:STDOUT:     %.loc13_28.1: type = splice_block %.loc13_28.2 [symbolic = %as_type (constants.%as_type.d58)] {
// CHECK:STDOUT:       %impl.elem2.loc13_28.2: %facet_type.37d = impl_witness_access constants.%Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc13_28.1 (constants.%impl.elem2)]
// CHECK:STDOUT:       %C.ref: %facet_type.37d = name_ref C, %impl.elem2.loc13_28.2 [symbolic = %impl.elem2.loc13_28.1 (constants.%impl.elem2)]
// CHECK:STDOUT:       %C.as_type: type = facet_access_type %C.ref [symbolic = %as_type (constants.%as_type.d58)]
// CHECK:STDOUT:       %.loc13_28.2: type = converted %C.ref, %C.as_type [symbolic = %as_type (constants.%as_type.d58)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @Transitive.WithSelf.TakesC.%as_type (%as_type.d58) = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, %Transitive.WithSelf.TakesC.decl [concrete = constants.%assoc4]
// CHECK:STDOUT:   %Transitive.WithSelf.observe.decl = observe_decl @Transitive.WithSelf.observe [concrete = constants.%Transitive.WithSelf.observe.decl]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.WithSelf.InP(@P.%Self: %P.type) {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d21)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @P.WithSelf.InP.%Self.binding.as_type (%Self.binding.as_type.6c8));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(@Q.%Self: %Q.type) {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.992)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Q.WithSelf.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(@R.%Self: %R.type) {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.bea)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.351)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @R.WithSelf.InR.%Self.binding.as_type (%Self.binding.as_type.296));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:   %pattern_type.loc12_11: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc12_11 (constants.%pattern_type.b0d)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc12_28.1: %facet_type.de1 = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc12_28.1 (constants.%impl.elem0)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc12_28.1 [symbolic = %as_type (constants.%as_type.91d)]
// CHECK:STDOUT:   %.loc12_28.1: Core.Form = init_form %as_type, call_param1 [symbolic = %.loc12_28.1 (constants.%.a36)]
// CHECK:STDOUT:   %pattern_type.loc12_25: type = pattern_type %as_type [symbolic = %pattern_type.loc12_25 (constants.%pattern_type.485)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.WithSelf.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b)) -> out %return.param: @Transitive.WithSelf.GetA.%as_type (%as_type.91d);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:   %pattern_type.loc13_13: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc13_13 (constants.%pattern_type.b0d)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem2.loc13_28.1: %facet_type.37d = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc13_28.1 (constants.%impl.elem2)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2.loc13_28.1 [symbolic = %as_type (constants.%as_type.d58)]
// CHECK:STDOUT:   %pattern_type.loc13_25: type = pattern_type %as_type [symbolic = %pattern_type.loc13_25 (constants.%pattern_type.219)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.WithSelf.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b), %c.param: @Transitive.WithSelf.TakesC.%as_type (%as_type.d58));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf(constants.%Self.d21) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.d21
// CHECK:STDOUT:   %P.WithSelf.InP.type => constants.%P.WithSelf.InP.type
// CHECK:STDOUT:   %P.WithSelf.InP => constants.%P.WithSelf.InP
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf.InP(constants.%Self.d21) {
// CHECK:STDOUT:   %Self => constants.%Self.d21
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.6c8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Self.f64) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.f64
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.f64) {
// CHECK:STDOUT:   %Self => constants.%Self.f64
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.1b2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%Self.bea) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.bea
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%Self.bea) {
// CHECK:STDOUT:   %Self => constants.%Self.bea
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.296
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.351
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%Self.e2a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e5b
// CHECK:STDOUT:   %pattern_type.loc12_11 => constants.%pattern_type.b0d
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem0.loc12_28.1 => constants.%impl.elem0
// CHECK:STDOUT:   %as_type => constants.%as_type.91d
// CHECK:STDOUT:   %.loc12_28.1 => constants.%.a36
// CHECK:STDOUT:   %pattern_type.loc12_25 => constants.%pattern_type.485
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e5b
// CHECK:STDOUT:   %pattern_type.loc13_13 => constants.%pattern_type.b0d
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness
// CHECK:STDOUT:   %impl.elem2.loc13_28.1 => constants.%impl.elem2
// CHECK:STDOUT:   %as_type => constants.%as_type.d58
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- equivalent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.232: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.6d8f6f.1: <witness> = lookup_impl_witness %Self.232, @Transitive [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.9ed: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type.4f4: type = fn_type @R.WithSelf.InR, @R.WithSelf(%Self.9ed) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.b83: %R.WithSelf.InR.type.4f4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.05c: type = symbolic_binding_type Self, 0, %Self.9ed [symbolic]
// CHECK:STDOUT:   %pattern_type.2d7: type = pattern_type %Self.binding.as_type.05c [symbolic]
// CHECK:STDOUT:   %facet_type.9e9: type = facet_type <@Destroy & @R where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem2.709ce6.1: %facet_type.9e9 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.1, element2 [symbolic]
// CHECK:STDOUT:   %as_type.5c1916.1: type = facet_access_type %impl.elem2.709ce6.1 [symbolic]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.82c: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.8c7: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Self.82c) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.7e9: %Q.WithSelf.InQ.type.8c7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.d77: type = symbolic_binding_type Self, 0, %Self.82c [symbolic]
// CHECK:STDOUT:   %pattern_type.eb5: type = pattern_type %Self.binding.as_type.d77 [symbolic]
// CHECK:STDOUT:   %facet_type.89e: type = facet_type <@Destroy & @Q where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem1.0fcb86.1: %facet_type.89e = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.1, element1 [symbolic]
// CHECK:STDOUT:   %as_type.3b9245.1: type = facet_access_type %impl.elem1.0fcb86.1 [symbolic]
// CHECK:STDOUT:   %eq.24bd63.1 = observe_equivalent %as_type.3b9245.1, %as_type.5c1916.1 [symbolic]
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.321: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P.WithSelf(%Self.321) [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.9df: type = symbolic_binding_type Self, 0, %Self.321 [symbolic]
// CHECK:STDOUT:   %pattern_type.d71: type = pattern_type %Self.binding.as_type.9df [symbolic]
// CHECK:STDOUT:   %facet_type.3f8: type = facet_type <@Destroy & @P> [concrete]
// CHECK:STDOUT:   %impl.elem0.4a2c11.1: %facet_type.3f8 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.1, element0 [symbolic]
// CHECK:STDOUT:   %as_type.4576dc.1: type = facet_access_type %impl.elem0.4a2c11.1 [symbolic]
// CHECK:STDOUT:   %eq.c5a5a7.1 = observe_equivalent %as_type.4576dc.1, %as_type.3b9245.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.eda37f.1: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%Self.232) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.67da40.1: %Transitive.WithSelf.TakesC.type.eda37f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.104: type = symbolic_binding_type Self, 0, %Self.232 [symbolic]
// CHECK:STDOUT:   %pattern_type.993a5c.1: type = pattern_type %Self.binding.as_type.104 [symbolic]
// CHECK:STDOUT:   %pattern_type.28dd0e.1: type = pattern_type %as_type.5c1916.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.dc6981.1: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%Self.232) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.69dcdb.1: %Transitive.WithSelf.GetA.type.dc6981.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.fb24f5.1: type = pattern_type %as_type.4576dc.1 [symbolic]
// CHECK:STDOUT:   %.833f3b.1: Core.Form = init_form %as_type.4576dc.1, call_param1 [symbolic]
// CHECK:STDOUT:   %T: %Transitive.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %Transitive.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.993a5c.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1c0: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.6d8f6f.2: <witness> = lookup_impl_witness %T, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.4a2c11.2: %facet_type.3f8 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.0fcb86.2: %facet_type.89e = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.dc6981.2: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.69dcdb.2: %Transitive.WithSelf.GetA.type.dc6981.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.eda37f.2: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.67da40.2: %Transitive.WithSelf.TakesC.type.eda37f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %as_type.3b9245.2: type = facet_access_type %impl.elem1.0fcb86.2 [symbolic]
// CHECK:STDOUT:   %as_type.4576dc.2: type = facet_access_type %impl.elem0.4a2c11.2 [symbolic]
// CHECK:STDOUT:   %eq.c5a5a7.2 = observe_equivalent %as_type.4576dc.2, %as_type.3b9245.2 [symbolic]
// CHECK:STDOUT:   %impl.elem2.709ce6.2: %facet_type.9e9 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element2 [symbolic]
// CHECK:STDOUT:   %as_type.5c1916.2: type = facet_access_type %impl.elem2.709ce6.2 [symbolic]
// CHECK:STDOUT:   %eq.24bd63.2 = observe_equivalent %as_type.3b9245.2, %as_type.5c1916.2 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.eef: %Transitive.assoc_type = assoc_entity element0, imports.%Main.import_ref.663 [concrete]
// CHECK:STDOUT:   %require_complete.b40: <witness> = require_complete_type %as_type.4576dc.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.fb24f5.2: type = pattern_type %as_type.4576dc.2 [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, imports.%Main.import_ref.11f [concrete]
// CHECK:STDOUT:   %.c0a: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type.dc6981.2, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.c0a = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element3 [symbolic]
// CHECK:STDOUT:   %.833f3b.2: Core.Form = init_form %as_type.4576dc.2, call_param1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a8a: <specific function> = specific_impl_function %impl.elem3, @Transitive.WithSelf.GetA(%T) [symbolic]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, imports.%Main.import_ref.37e [concrete]
// CHECK:STDOUT:   %require_complete.599: <witness> = require_complete_type %as_type.3b9245.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.31a: type = pattern_type %as_type.3b9245.2 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.88d: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%impl.elem1.0fcb86.2) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.9f9: %Q.WithSelf.InQ.type.88d = struct_value () [symbolic]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.8b2: %Q.assoc_type = assoc_entity element0, imports.%Main.import_ref.37b [concrete]
// CHECK:STDOUT:   %Q.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.0fcb86.2, @Q [symbolic]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.3b9245.2, (%Q.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.2e1: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Q.facet) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.6b9: %Q.WithSelf.InQ.type.2e1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.436: type = fn_type_with_self_type %Q.WithSelf.InQ.type.2e1, %Q.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.a4a: %.436 = impl_witness_access %Q.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f2c: <specific function> = specific_impl_function %impl.elem0.a4a, @Q.WithSelf.InQ(%Q.facet) [symbolic]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, imports.%Main.import_ref.18e [concrete]
// CHECK:STDOUT:   %R.WithSelf.InR.type.d28: type = fn_type @R.WithSelf.InR, @R.WithSelf(%impl.elem2.709ce6.2) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.649: %R.WithSelf.InR.type.d28 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.assoc_type: type = assoc_entity_type @R [concrete]
// CHECK:STDOUT:   %assoc0.931: %R.assoc_type = assoc_entity element0, imports.%Main.import_ref.8dc [concrete]
// CHECK:STDOUT:   %R.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem2.709ce6.2, @R [symbolic]
// CHECK:STDOUT:   %R.facet: %R.type = facet_value %as_type.5c1916.2, (%R.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type.b94: type = fn_type @R.WithSelf.InR, @R.WithSelf(%R.facet) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.70c: %R.WithSelf.InR.type.b94 = struct_value () [symbolic]
// CHECK:STDOUT:   %.d19: type = fn_type_with_self_type %R.WithSelf.InR.type.b94, %R.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.3d8: %.d19 = impl_witness_access %R.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.28dd0e.2: type = pattern_type %as_type.5c1916.2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c8e: <specific function> = specific_impl_function %impl.elem0.3d8, @R.WithSelf.InR(%R.facet) [symbolic]
// CHECK:STDOUT:   %require_complete.b24: <witness> = require_complete_type %as_type.5c1916.2 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.4e8: <witness> = lookup_impl_witness %impl.elem1.0fcb86.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.bb9: %Destroy.type = facet_value %as_type.3b9245.2, (%Destroy.lookup_impl_witness.4e8) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5d3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.bb9) [symbolic]
// CHECK:STDOUT:   %.64f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5d3, %Destroy.facet.bb9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.12c: %.64f = impl_witness_access %Destroy.lookup_impl_witness.4e8, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e61: <specific function> = specific_impl_function %impl.elem0.12c, @Destroy.WithSelf.Op(%Destroy.facet.bb9) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.c44: <witness> = lookup_impl_witness %impl.elem0.4a2c11.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e93: %Destroy.type = facet_value %as_type.4576dc.2, (%Destroy.lookup_impl_witness.c44) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.bb3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e93) [symbolic]
// CHECK:STDOUT:   %.b45: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.bb3, %Destroy.facet.e93 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1b1: %.b45 = impl_witness_access %Destroy.lookup_impl_witness.c44, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9da: <specific function> = specific_impl_function %impl.elem0.1b1, @Destroy.WithSelf.Op(%Destroy.facet.e93) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.P = import_ref Main//transitive, P, unloaded
// CHECK:STDOUT:   %Main.Q = import_ref Main//transitive, Q, unloaded
// CHECK:STDOUT:   %Main.R = import_ref Main//transitive, R, unloaded
// CHECK:STDOUT:   %Main.Transitive: type = import_ref Main//transitive, Transitive, loaded [concrete = constants.%Transitive.type]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.059: %Transitive.assoc_type = import_ref Main//transitive, loc8_8, loaded [concrete = constants.%assoc0.eef]
// CHECK:STDOUT:   %Main.import_ref.772: %Transitive.assoc_type = import_ref Main//transitive, loc9_8, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Main.import_ref.bdf: %Transitive.assoc_type = import_ref Main//transitive, loc10_8, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Main.import_ref.06e: %Transitive.assoc_type = import_ref Main//transitive, loc12_29, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Main.import_ref.a70 = import_ref Main//transitive, loc13_30, unloaded
// CHECK:STDOUT:   %Main.A = import_ref Main//transitive, A, unloaded
// CHECK:STDOUT:   %Main.B = import_ref Main//transitive, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//transitive, C, unloaded
// CHECK:STDOUT:   %Main.GetA = import_ref Main//transitive, GetA, unloaded
// CHECK:STDOUT:   %Main.TakesC = import_ref Main//transitive, TakesC, unloaded
// CHECK:STDOUT:   %Main.import_ref.d9d: %R.assoc_type = import_ref Main//transitive, loc5_35, loaded [concrete = constants.%assoc0.931]
// CHECK:STDOUT:   %Main.InR = import_ref Main//transitive, InR, unloaded
// CHECK:STDOUT:   %Main.import_ref.ac9d5a.2: %R.type = import_ref Main//transitive, loc5_13, loaded [symbolic = constants.%Self.9ed]
// CHECK:STDOUT:   %Main.import_ref.fb9 = import_ref Main//transitive, loc5_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.b1e: %Q.assoc_type = import_ref Main//transitive, loc4_35, loaded [concrete = constants.%assoc0.8b2]
// CHECK:STDOUT:   %Main.InQ = import_ref Main//transitive, InQ, unloaded
// CHECK:STDOUT:   %Main.import_ref.60b85f.2: %Q.type = import_ref Main//transitive, loc4_13, loaded [symbolic = constants.%Self.82c]
// CHECK:STDOUT:   %Main.import_ref.7a1 = import_ref Main//transitive, loc4_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.d29 = import_ref Main//transitive, loc3_35, unloaded
// CHECK:STDOUT:   %Main.InP = import_ref Main//transitive, InP, unloaded
// CHECK:STDOUT:   %Main.import_ref.e19f0c.2: %P.type = import_ref Main//transitive, loc3_13, loaded [symbolic = constants.%Self.321]
// CHECK:STDOUT:   %Main.import_ref.761 = import_ref Main//transitive, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.abd70f.2: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.232]
// CHECK:STDOUT:   %Main.import_ref.abd70f.3: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.232]
// CHECK:STDOUT:   %Main.import_ref.92e = import_ref Main//transitive, loc7_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.663: %facet_type.3f8 = import_ref Main//transitive, loc8_8, loaded [concrete = %A]
// CHECK:STDOUT:   %A: %facet_type.3f8 = assoc_const_decl @A [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.11f: @Transitive.WithSelf.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type.dc6981.1) = import_ref Main//transitive, loc12_29, loaded [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA.69dcdb.1)]
// CHECK:STDOUT:   %Main.import_ref.37e: %facet_type.89e = import_ref Main//transitive, loc9_8, loaded [concrete = %B]
// CHECK:STDOUT:   %B: %facet_type.89e = assoc_const_decl @B [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.37b: @Q.WithSelf.%Q.WithSelf.InQ.type (%Q.WithSelf.InQ.type.8c7) = import_ref Main//transitive, loc4_35, loaded [symbolic = @Q.WithSelf.%Q.WithSelf.InQ (constants.%Q.WithSelf.InQ.7e9)]
// CHECK:STDOUT:   %Main.import_ref.18e: %facet_type.9e9 = import_ref Main//transitive, loc10_8, loaded [concrete = %C]
// CHECK:STDOUT:   %C: %facet_type.9e9 = assoc_const_decl @C [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.8dc: @R.WithSelf.%R.WithSelf.InR.type (%R.WithSelf.InR.type.4f4) = import_ref Main//transitive, loc5_35, loaded [symbolic = @R.WithSelf.%R.WithSelf.InR (constants.%R.WithSelf.InR.b83)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .P = imports.%Main.P
// CHECK:STDOUT:     .Q = imports.%Main.Q
// CHECK:STDOUT:     .R = imports.%Main.R
// CHECK:STDOUT:     .Transitive = imports.%Main.Transitive
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.18b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc5 (%pattern_type.993a5c.2) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc5 (%pattern_type.993a5c.2) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_10: type = splice_block %Transitive.ref [concrete = constants.%Transitive.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Transitive.ref: type = name_ref Transitive, imports.%Main.Transitive [concrete = constants.%Transitive.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_6.2: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc5_6.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_25.1: type = splice_block %.loc5_25.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc5: %Transitive.type = name_ref T, %T.loc5_6.2 [symbolic = %T.loc5_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc5: type = facet_access_type %T.ref.loc5 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc5_25.2: type = converted %T.ref.loc5, %T.as_type.loc5 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.92e
// CHECK:STDOUT:   .A = imports.%Main.import_ref.059
// CHECK:STDOUT:   .B = imports.%Main.import_ref.772
// CHECK:STDOUT:   .C = imports.%Main.import_ref.bdf
// CHECK:STDOUT:   .GetA = imports.%Main.import_ref.06e
// CHECK:STDOUT:   .TakesC = imports.%Main.import_ref.a70
// CHECK:STDOUT:   witness = (imports.%Main.A, imports.%Main.B, imports.%Main.C, imports.%Main.GetA, imports.%Main.TakesC)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.fb9
// CHECK:STDOUT:   .InR = imports.%Main.import_ref.d9d
// CHECK:STDOUT:   witness = (imports.%Main.InR)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.7a1
// CHECK:STDOUT:   .InQ = imports.%Main.import_ref.b1e
// CHECK:STDOUT:   witness = (imports.%Main.InQ)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @P [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.761
// CHECK:STDOUT:   .InP = imports.%Main.import_ref.d29
// CHECK:STDOUT:   witness = (imports.%Main.InP)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(imports.%Main.import_ref.ac9d5a.2: %R.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9ed)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.05c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(imports.%Main.import_ref.60b85f.2: %Q.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.82c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d77)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.eb5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.WithSelf.InP(imports.%Main.import_ref.e19f0c.2: %P.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.321)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.9df)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.d71)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(imports.%Main.import_ref.abd70f.2: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.232)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.104)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.993a5c.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8f6f.1)]
// CHECK:STDOUT:   %impl.elem2: %facet_type.9e9 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2.709ce6.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2 [symbolic = %as_type (constants.%as_type.5c1916.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.28dd0e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(imports.%Main.import_ref.abd70f.3: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.232)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.104)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.993a5c.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8f6f.1)]
// CHECK:STDOUT:   %impl.elem0: %facet_type.3f8 = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.4a2c11.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic = %as_type (constants.%as_type.4576dc.1)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %as_type, call_param1 [symbolic = %.1 (constants.%.833f3b.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.fb24f5.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_6.2: %Transitive.type) {
// CHECK:STDOUT:   %T.loc5_6.1: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc5_6.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc5_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc5 (constants.%pattern_type.993a5c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc5 (constants.%require_complete.1c0)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc5_6.1, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8f6f.2)]
// CHECK:STDOUT:   %impl.elem0.loc6_11.2: %facet_type.3f8 = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_11.2 (constants.%impl.elem0.4a2c11.2)]
// CHECK:STDOUT:   %as_type.loc6_11.2: type = facet_access_type %impl.elem0.loc6_11.2 [symbolic = %as_type.loc6_11.2 (constants.%as_type.4576dc.2)]
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %as_type.loc6_11.2 [symbolic = %require_complete.loc6 (constants.%require_complete.b40)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %as_type.loc6_11.2 [symbolic = %pattern_type.loc6 (constants.%pattern_type.fb24f5.2)]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T.loc5_6.1) [symbolic = %Transitive.WithSelf.GetA.type (constants.%Transitive.WithSelf.GetA.type.dc6981.2)]
// CHECK:STDOUT:   %.loc6_17: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type, %T.loc5_6.1 [symbolic = %.loc6_17 (constants.%.c0a)]
// CHECK:STDOUT:   %impl.elem3.loc6_17.2: @F.%.loc6_17 (%.c0a) = impl_witness_access %Transitive.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc6_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_17.2: <specific function> = specific_impl_function %impl.elem3.loc6_17.2, @Transitive.WithSelf.GetA(%T.loc5_6.1) [symbolic = %specific_impl_fn.loc6_17.2 (constants.%specific_impl_fn.a8a)]
// CHECK:STDOUT:   %impl.elem1.loc8_11.2: %facet_type.89e = impl_witness_access %Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc8_11.2 (constants.%impl.elem1.0fcb86.2)]
// CHECK:STDOUT:   %as_type.loc8_11.2: type = facet_access_type %impl.elem1.loc8_11.2 [symbolic = %as_type.loc8_11.2 (constants.%as_type.3b9245.2)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %as_type.loc8_11.2 [symbolic = %require_complete.loc8 (constants.%require_complete.599)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %as_type.loc8_11.2 [symbolic = %pattern_type.loc8 (constants.%pattern_type.31a)]
// CHECK:STDOUT:   %Q.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.loc8_11.2, @Q [symbolic = %Q.lookup_impl_witness (constants.%Q.lookup_impl_witness)]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.loc8_11.2, (%Q.lookup_impl_witness) [symbolic = %Q.facet (constants.%Q.facet)]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Q.facet) [symbolic = %Q.WithSelf.InQ.type (constants.%Q.WithSelf.InQ.type.2e1)]
// CHECK:STDOUT:   %.loc9_4: type = fn_type_with_self_type %Q.WithSelf.InQ.type, %Q.facet [symbolic = %.loc9_4 (constants.%.436)]
// CHECK:STDOUT:   %impl.elem0.loc9_4.2: @F.%.loc9_4 (%.436) = impl_witness_access %Q.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0.a4a)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_4.2: <specific function> = specific_impl_function %impl.elem0.loc9_4.2, @Q.WithSelf.InQ(%Q.facet) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn.f2c)]
// CHECK:STDOUT:   %impl.elem2.loc11_7.2: %facet_type.9e9 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc11_7.2 (constants.%impl.elem2.709ce6.2)]
// CHECK:STDOUT:   %as_type.loc11_9.2: type = facet_access_type %impl.elem2.loc11_7.2 [symbolic = %as_type.loc11_9.2 (constants.%as_type.5c1916.2)]
// CHECK:STDOUT:   %R.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem2.loc11_7.2, @R [symbolic = %R.lookup_impl_witness (constants.%R.lookup_impl_witness)]
// CHECK:STDOUT:   %R.facet: %R.type = facet_value %as_type.loc11_9.2, (%R.lookup_impl_witness) [symbolic = %R.facet (constants.%R.facet)]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R.WithSelf(%R.facet) [symbolic = %R.WithSelf.InR.type (constants.%R.WithSelf.InR.type.b94)]
// CHECK:STDOUT:   %.loc11_9.2: type = fn_type_with_self_type %R.WithSelf.InR.type, %R.facet [symbolic = %.loc11_9.2 (constants.%.d19)]
// CHECK:STDOUT:   %impl.elem0.loc11_9.2: @F.%.loc11_9.2 (%.d19) = impl_witness_access %R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_9.2 (constants.%impl.elem0.3d8)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_9.2: <specific function> = specific_impl_function %impl.elem0.loc11_9.2, @R.WithSelf.InR(%R.facet) [symbolic = %specific_impl_fn.loc11_9.2 (constants.%specific_impl_fn.c8e)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %as_type.loc11_9.2 [symbolic = %require_complete.loc11 (constants.%require_complete.b24)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc8: <witness> = lookup_impl_witness %impl.elem1.loc8_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc8 (constants.%Destroy.lookup_impl_witness.4e8)]
// CHECK:STDOUT:   %Destroy.facet.loc8: %Destroy.type = facet_value %as_type.loc8_11.2, (%Destroy.lookup_impl_witness.loc8) [symbolic = %Destroy.facet.loc8 (constants.%Destroy.facet.bb9)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc8) [symbolic = %Destroy.WithSelf.Op.type.loc8 (constants.%Destroy.WithSelf.Op.type.5d3)]
// CHECK:STDOUT:   %.loc8_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc8, %Destroy.facet.loc8 [symbolic = %.loc8_3.4 (constants.%.64f)]
// CHECK:STDOUT:   %impl.elem0.loc8_3.2: @F.%.loc8_3.4 (%.64f) = impl_witness_access %Destroy.lookup_impl_witness.loc8, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.12c)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_3.2: <specific function> = specific_impl_function %impl.elem0.loc8_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc8) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.e61)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc6: <witness> = lookup_impl_witness %impl.elem0.loc6_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc6 (constants.%Destroy.lookup_impl_witness.c44)]
// CHECK:STDOUT:   %Destroy.facet.loc6: %Destroy.type = facet_value %as_type.loc6_11.2, (%Destroy.lookup_impl_witness.loc6) [symbolic = %Destroy.facet.loc6 (constants.%Destroy.facet.e93)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc6) [symbolic = %Destroy.WithSelf.Op.type.loc6 (constants.%Destroy.WithSelf.Op.type.bb3)]
// CHECK:STDOUT:   %.loc6_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc6, %Destroy.facet.loc6 [symbolic = %.loc6_3.4 (constants.%.b45)]
// CHECK:STDOUT:   %impl.elem0.loc6_3.2: @F.%.loc6_3.4 (%.b45) = impl_witness_access %Destroy.lookup_impl_witness.loc6, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.1b1)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_3.2: <specific function> = specific_impl_function %impl.elem0.loc6_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc6) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.9da)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type.loc6 (%pattern_type.fb24f5.2) = ref_binding_pattern a [concrete]
// CHECK:STDOUT:       %a.var_patt: @F.%pattern_type.loc6 (%pattern_type.fb24f5.2) = var_pattern %a.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.var: ref @F.%as_type.loc6_11.2 (%as_type.4576dc.2) = var %a.var_patt
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %GetA.ref: %Transitive.assoc_type = name_ref GetA, imports.%Main.import_ref.06e [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc6_17.1: @F.%.loc6_17 (%.c0a) = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8f6f.2, element3 [symbolic = %impl.elem3.loc6_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc6_17: <bound method> = bound_method %t.ref, %impl.elem3.loc6_17.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_17.1: <specific function> = specific_impl_function %impl.elem3.loc6_17.1, @Transitive.WithSelf.GetA(constants.%T) [symbolic = %specific_impl_fn.loc6_17.2 (constants.%specific_impl_fn.a8a)]
// CHECK:STDOUT:     %bound_method.loc6_23: <bound method> = bound_method %t.ref, %specific_impl_fn.loc6_17.1
// CHECK:STDOUT:     %.loc6_23: ref @F.%as_type.loc6_11.2 (%as_type.4576dc.2) = temporary_storage
// CHECK:STDOUT:     %Transitive.WithSelf.GetA.call: init @F.%as_type.loc6_11.2 (%as_type.4576dc.2) to %.loc6_23 = call %bound_method.loc6_23(%t.ref)
// CHECK:STDOUT:     %.loc6_3.1: init @F.%as_type.loc6_11.2 (%as_type.4576dc.2) = as_compatible %Transitive.WithSelf.GetA.call
// CHECK:STDOUT:     assign %a.var, %.loc6_3.1
// CHECK:STDOUT:     %.loc6_11.1: type = splice_block %.loc6_11.3 [symbolic = %as_type.loc6_11.2 (constants.%as_type.4576dc.2)] {
// CHECK:STDOUT:       %T.ref.loc6: %Transitive.type = name_ref T, %T.loc5_6.2 [symbolic = %T.loc5_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6: type = facet_access_type %T.ref.loc6 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_11.2: type = converted %T.ref.loc6, %T.as_type.loc6 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %A.ref: %Transitive.assoc_type = name_ref A, imports.%Main.import_ref.059 [concrete = constants.%assoc0.eef]
// CHECK:STDOUT:       %impl.elem0.loc6_11.1: %facet_type.3f8 = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8f6f.2, element0 [symbolic = %impl.elem0.loc6_11.2 (constants.%impl.elem0.4a2c11.2)]
// CHECK:STDOUT:       %as_type.loc6_11.1: type = facet_access_type %impl.elem0.loc6_11.1 [symbolic = %as_type.loc6_11.2 (constants.%as_type.4576dc.2)]
// CHECK:STDOUT:       %.loc6_11.3: type = converted %impl.elem0.loc6_11.1, %as_type.loc6_11.1 [symbolic = %as_type.loc6_11.2 (constants.%as_type.4576dc.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc6_3.2: ref @F.%as_type.loc6_11.2 (%as_type.4576dc.2) = as_compatible %a.var
// CHECK:STDOUT:     %a: ref @F.%as_type.loc6_11.2 (%as_type.4576dc.2) = ref_binding a, %.loc6_3.2
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %b.patt: @F.%pattern_type.loc8 (%pattern_type.31a) = ref_binding_pattern b [concrete]
// CHECK:STDOUT:       %b.var_patt: @F.%pattern_type.loc8 (%pattern_type.31a) = var_pattern %b.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.var: ref @F.%as_type.loc8_11.2 (%as_type.3b9245.2) = var %b.var_patt
// CHECK:STDOUT:     %a.ref.loc8: ref @F.%as_type.loc6_11.2 (%as_type.4576dc.2) = name_ref a, %a
// CHECK:STDOUT:     %.loc8_3.1: ref @F.%as_type.loc8_11.2 (%as_type.3b9245.2) = as_compatible %a.ref.loc8
// CHECK:STDOUT:     assign %b.var, %.loc8_3.1
// CHECK:STDOUT:     %.loc8_11.1: type = splice_block %.loc8_11.3 [symbolic = %as_type.loc8_11.2 (constants.%as_type.3b9245.2)] {
// CHECK:STDOUT:       %T.ref.loc8: %Transitive.type = name_ref T, %T.loc5_6.2 [symbolic = %T.loc5_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8: type = facet_access_type %T.ref.loc8 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc8_11.2: type = converted %T.ref.loc8, %T.as_type.loc8 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %B.ref: %Transitive.assoc_type = name_ref B, imports.%Main.import_ref.772 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %impl.elem1.loc8_11.1: %facet_type.89e = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8f6f.2, element1 [symbolic = %impl.elem1.loc8_11.2 (constants.%impl.elem1.0fcb86.2)]
// CHECK:STDOUT:       %as_type.loc8_11.1: type = facet_access_type %impl.elem1.loc8_11.1 [symbolic = %as_type.loc8_11.2 (constants.%as_type.3b9245.2)]
// CHECK:STDOUT:       %.loc8_11.3: type = converted %impl.elem1.loc8_11.1, %as_type.loc8_11.1 [symbolic = %as_type.loc8_11.2 (constants.%as_type.3b9245.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc8_3.2: ref @F.%as_type.loc8_11.2 (%as_type.3b9245.2) = as_compatible %b.var
// CHECK:STDOUT:     %b: ref @F.%as_type.loc8_11.2 (%as_type.3b9245.2) = ref_binding b, %.loc8_3.2
// CHECK:STDOUT:     %b.ref: ref @F.%as_type.loc8_11.2 (%as_type.3b9245.2) = name_ref b, %b
// CHECK:STDOUT:     %.loc9_3.1: ref @F.%as_type.loc8_11.2 (%as_type.3b9245.2) = as_compatible %b.ref
// CHECK:STDOUT:     %InQ.ref: %Q.assoc_type = name_ref InQ, imports.%Main.import_ref.b1e [concrete = constants.%assoc0.8b2]
// CHECK:STDOUT:     %impl.elem0.loc9_4.1: @F.%.loc9_4 (%.436) = impl_witness_access constants.%Q.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0.a4a)]
// CHECK:STDOUT:     %bound_method.loc9_4: <bound method> = bound_method %.loc9_3.1, %impl.elem0.loc9_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_4.1: <specific function> = specific_impl_function %impl.elem0.loc9_4.1, @Q.WithSelf.InQ(constants.%Q.facet) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn.f2c)]
// CHECK:STDOUT:     %bound_method.loc9_9: <bound method> = bound_method %.loc9_3.1, %specific_impl_fn.loc9_4.1
// CHECK:STDOUT:     %.loc9_3.2: ref @F.%as_type.loc8_11.2 (%as_type.3b9245.2) = as_compatible %.loc9_3.1
// CHECK:STDOUT:     %Q.WithSelf.InQ.call: init %empty_tuple.type = call %bound_method.loc9_9(%.loc9_3.2)
// CHECK:STDOUT:     %a.ref.loc11: ref @F.%as_type.loc6_11.2 (%as_type.4576dc.2) = name_ref a, %a
// CHECK:STDOUT:     %T.ref.loc11: %Transitive.type = name_ref T, %T.loc5_6.2 [symbolic = %T.loc5_6.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc11: type = facet_access_type %T.ref.loc11 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc11_7: type = converted %T.ref.loc11, %T.as_type.loc11 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %C.ref.loc11: %Transitive.assoc_type = name_ref C, imports.%Main.import_ref.bdf [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc11_7.1: %facet_type.9e9 = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8f6f.2, element2 [symbolic = %impl.elem2.loc11_7.2 (constants.%impl.elem2.709ce6.2)]
// CHECK:STDOUT:     %as_type.loc11_9.1: type = facet_access_type %impl.elem2.loc11_7.1 [symbolic = %as_type.loc11_9.2 (constants.%as_type.5c1916.2)]
// CHECK:STDOUT:     %.loc11_9.1: type = converted %impl.elem2.loc11_7.1, %as_type.loc11_9.1 [symbolic = %as_type.loc11_9.2 (constants.%as_type.5c1916.2)]
// CHECK:STDOUT:     %InR.ref.loc11: %R.assoc_type = name_ref InR, imports.%Main.import_ref.d9d [concrete = constants.%assoc0.931]
// CHECK:STDOUT:     %impl.elem0.loc11_9.1: @F.%.loc11_9.2 (%.d19) = impl_witness_access constants.%R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_9.2 (constants.%impl.elem0.3d8)]
// CHECK:STDOUT:     %bound_method.loc11_4: <bound method> = bound_method %a.ref.loc11, %impl.elem0.loc11_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_9.1: <specific function> = specific_impl_function %impl.elem0.loc11_9.1, @R.WithSelf.InR(constants.%R.facet) [symbolic = %specific_impl_fn.loc11_9.2 (constants.%specific_impl_fn.c8e)]
// CHECK:STDOUT:     %bound_method.loc11_15: <bound method> = bound_method %a.ref.loc11, %specific_impl_fn.loc11_9.1
// CHECK:STDOUT:     %.loc11_3: ref @F.%as_type.loc11_9.2 (%as_type.5c1916.2) = as_compatible %a.ref.loc11
// CHECK:STDOUT:     %R.WithSelf.InR.call.loc11: init %empty_tuple.type = call %bound_method.loc11_15(%.loc11_3)
// CHECK:STDOUT:     %a.ref.loc12: ref @F.%as_type.loc6_11.2 (%as_type.4576dc.2) = name_ref a, %a
// CHECK:STDOUT:     %T.ref.loc12: %Transitive.type = name_ref T, %T.loc5_6.2 [symbolic = %T.loc5_6.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc12: type = facet_access_type %T.ref.loc12 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc12_10.1: type = converted %T.ref.loc12, %T.as_type.loc12 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %C.ref.loc12: %Transitive.assoc_type = name_ref C, imports.%Main.import_ref.bdf [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc12: %facet_type.9e9 = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8f6f.2, element2 [symbolic = %impl.elem2.loc11_7.2 (constants.%impl.elem2.709ce6.2)]
// CHECK:STDOUT:     %as_type.loc12: type = facet_access_type %impl.elem2.loc12 [symbolic = %as_type.loc11_9.2 (constants.%as_type.5c1916.2)]
// CHECK:STDOUT:     %.loc12_10.2: type = converted %impl.elem2.loc12, %as_type.loc12 [symbolic = %as_type.loc11_9.2 (constants.%as_type.5c1916.2)]
// CHECK:STDOUT:     %.loc12_6.1: ref @F.%as_type.loc11_9.2 (%as_type.5c1916.2) = as_compatible %a.ref.loc12
// CHECK:STDOUT:     %.loc12_6.2: ref @F.%as_type.loc11_9.2 (%as_type.5c1916.2) = as_compatible %.loc12_6.1
// CHECK:STDOUT:     %InR.ref.loc12: %R.assoc_type = name_ref InR, imports.%Main.import_ref.d9d [concrete = constants.%assoc0.931]
// CHECK:STDOUT:     %impl.elem0.loc12: @F.%.loc11_9.2 (%.d19) = impl_witness_access constants.%R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_9.2 (constants.%impl.elem0.3d8)]
// CHECK:STDOUT:     %bound_method.loc12_13: <bound method> = bound_method %.loc12_6.2, %impl.elem0.loc12
// CHECK:STDOUT:     %specific_impl_fn.loc12: <specific function> = specific_impl_function %impl.elem0.loc12, @R.WithSelf.InR(constants.%R.facet) [symbolic = %specific_impl_fn.loc11_9.2 (constants.%specific_impl_fn.c8e)]
// CHECK:STDOUT:     %bound_method.loc12_18: <bound method> = bound_method %.loc12_6.2, %specific_impl_fn.loc12
// CHECK:STDOUT:     %.loc12_6.3: ref @F.%as_type.loc11_9.2 (%as_type.5c1916.2) = as_compatible %.loc12_6.2
// CHECK:STDOUT:     %R.WithSelf.InR.call.loc12: init %empty_tuple.type = call %bound_method.loc12_18(%.loc12_6.3)
// CHECK:STDOUT:     %impl.elem0.loc8_3.1: @F.%.loc8_3.4 (%.64f) = impl_witness_access constants.%Destroy.lookup_impl_witness.4e8, element0 [symbolic = %impl.elem0.loc8_3.2 (constants.%impl.elem0.12c)]
// CHECK:STDOUT:     %bound_method.loc8_3.1: <bound method> = bound_method %b.var, %impl.elem0.loc8_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc8_3.1: <specific function> = specific_impl_function %impl.elem0.loc8_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.bb9) [symbolic = %specific_impl_fn.loc8_3.2 (constants.%specific_impl_fn.e61)]
// CHECK:STDOUT:     %bound_method.loc8_3.2: <bound method> = bound_method %b.var, %specific_impl_fn.loc8_3.1
// CHECK:STDOUT:     %.loc8_3.3: ref @F.%as_type.loc8_11.2 (%as_type.3b9245.2) = as_compatible %b.var
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8_3.2(%.loc8_3.3)
// CHECK:STDOUT:     %impl.elem0.loc6_3.1: @F.%.loc6_3.4 (%.b45) = impl_witness_access constants.%Destroy.lookup_impl_witness.c44, element0 [symbolic = %impl.elem0.loc6_3.2 (constants.%impl.elem0.1b1)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc6_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_3.1: <specific function> = specific_impl_function %impl.elem0.loc6_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.e93) [symbolic = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn.9da)]
// CHECK:STDOUT:     %bound_method.loc6_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc6_3.1
// CHECK:STDOUT:     %.loc6_3.3: ref @F.%as_type.loc6_11.2 (%as_type.4576dc.2) = as_compatible %a.var
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc6: init %empty_tuple.type = call %bound_method.loc6_3.2(%.loc6_3.3)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%Self.9ed) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.9ed
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type.4f4
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR.b83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%Self.9ed) {
// CHECK:STDOUT:   %Self => constants.%Self.9ed
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.05c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Self.82c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.82c
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.8c7
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.7e9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.82c) {
// CHECK:STDOUT:   %Self => constants.%Self.82c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.d77
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eb5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf(constants.%Self.321) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.321
// CHECK:STDOUT:   %P.WithSelf.InP.type => constants.%P.WithSelf.InP.type
// CHECK:STDOUT:   %P.WithSelf.InP => constants.%P.WithSelf.InP
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf.InP(constants.%Self.321) {
// CHECK:STDOUT:   %Self => constants.%Self.321
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.9df
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%Self.232) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.232
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.4a2c11.1
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.0fcb86.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.dc6981.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.69dcdb.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.eda37f.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.67da40.1
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.3b9245.1
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.4576dc.1
// CHECK:STDOUT:   %eq.1 => constants.%eq.c5a5a7.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.709ce6.1
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.5c1916.1
// CHECK:STDOUT:   %eq.2 => constants.%eq.24bd63.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.232) {
// CHECK:STDOUT:   %Self => constants.%Self.232
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.104
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.993a5c.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.709ce6.1
// CHECK:STDOUT:   %as_type => constants.%as_type.5c1916.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.28dd0e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.232) {
// CHECK:STDOUT:   %Self => constants.%Self.232
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.104
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.993a5c.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.4a2c11.1
// CHECK:STDOUT:   %as_type => constants.%as_type.4576dc.1
// CHECK:STDOUT:   %.1 => constants.%.833f3b.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.fb24f5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc5_6.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc5 => constants.%pattern_type.993a5c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.4a2c11.2
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.0fcb86.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.dc6981.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.69dcdb.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.eda37f.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.67da40.2
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.3b9245.2
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.4576dc.2
// CHECK:STDOUT:   %eq.1 => constants.%eq.c5a5a7.2
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.709ce6.2
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.5c1916.2
// CHECK:STDOUT:   %eq.2 => constants.%eq.24bd63.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.993a5c.2
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.4a2c11.2
// CHECK:STDOUT:   %as_type => constants.%as_type.4576dc.2
// CHECK:STDOUT:   %.1 => constants.%.833f3b.2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.fb24f5.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%impl.elem1.0fcb86.2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%impl.elem1.0fcb86.2
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.88d
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.9f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Q.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Q.facet
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.2e1
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.6b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Q.facet) {
// CHECK:STDOUT:   %Self => constants.%Q.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%as_type.3b9245.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.31a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%impl.elem2.709ce6.2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%impl.elem2.709ce6.2
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type.d28
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR.649
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%R.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%R.facet
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type.b94
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR.70c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%R.facet) {
// CHECK:STDOUT:   %Self => constants.%R.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%as_type.5c1916.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.28dd0e.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.8ee: type = symbolic_binding_type Self, 0, %Self.d0b [symbolic]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.b7c: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.89a: type = symbolic_binding_type Self, 0, %Self.b7c [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.5a3: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.0b3: type = symbolic_binding_type T, 0, %T.5a3 [symbolic]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %T.binding.as_type.0b3 [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1a7: <witness> = require_complete_type %T.binding.as_type.0b3 [symbolic]
// CHECK:STDOUT:   %T.94e: %C.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7f1: type = pattern_type %C.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.d92: type = symbolic_binding_type T, 0, %T.94e [symbolic]
// CHECK:STDOUT:   %pattern_type.901: type = pattern_type %T.binding.as_type.d92 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.791: <witness> = require_complete_type %T.binding.as_type.d92 [symbolic]
// CHECK:STDOUT:   %impls = observe_impls %T.binding.as_type.d92, %B.type [symbolic]
// CHECK:STDOUT:   %observe.decl = observe_decl @observe [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresA.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresA.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = value_param call_param0
// CHECK:STDOUT:     %.loc7_31.1: type = splice_block %.loc7_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:       %.loc7_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7f1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.901) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.901) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %C.ref [concrete = constants.%C.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %C.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T.94e)]
// CHECK:STDOUT:     %x.param: @F.%T.binding.as_type (%T.binding.as_type.d92) = value_param call_param0
// CHECK:STDOUT:     %.loc8_16.1: type = splice_block %.loc8_16.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)] {
// CHECK:STDOUT:       %T.ref: %C.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T.94e)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:       %.loc8_16.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.binding.as_type (%T.binding.as_type.d92) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %B.WithSelf.Self.binding.as_type.impls.A.type.require.decl = require_decl @B.WithSelf.Self.binding.as_type.impls.A.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %A.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @B.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.8ee)]
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
// CHECK:STDOUT:   @B.WithSelf.Self.binding.as_type.impls.A.type.require {
// CHECK:STDOUT:     require @B.WithSelf.Self.binding.as_type.impls.A.type.require.%Self.as_type impls @B.WithSelf.Self.binding.as_type.impls.A.type.require.%A.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b7c]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %C.WithSelf.Self.binding.as_type.impls.B.type.require.decl = require_decl @C.WithSelf.Self.binding.as_type.impls.B.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %B.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @C.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.89a)]
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
// CHECK:STDOUT:   @C.WithSelf.Self.binding.as_type.impls.B.type.require {
// CHECK:STDOUT:     require @C.WithSelf.Self.binding.as_type.impls.B.type.require.%Self.as_type impls @C.WithSelf.Self.binding.as_type.impls.B.type.require.%B.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @B.WithSelf.Self.binding.as_type.impls.A.type.require(@B.%Self: %B.type) {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d0b)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.8ee)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @C.WithSelf.Self.binding.as_type.impls.B.type.require(@C.%Self: %C.type) {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.b7c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.89a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc7_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc7_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.057)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1a7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %C.type) {
// CHECK:STDOUT:   %T.loc8_6.1: %C.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T.94e)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.901)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.791)]
// CHECK:STDOUT:   %impls = observe_impls %T.binding.as_type, constants.%B.type [symbolic = %impls (constants.%impls)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.binding.as_type (%T.binding.as_type.d92)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe [concrete = constants.%observe.decl]
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @F.%T.binding.as_type (%T.binding.as_type.d92) = name_ref x, %x
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.d0b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf.Self.binding.as_type.impls.A.type.require(constants.%Self.d0b) {
// CHECK:STDOUT:   %Self => constants.%Self.d0b
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.8ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.b7c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf.Self.binding.as_type.impls.B.type.require(constants.%Self.b7c) {
// CHECK:STDOUT:   %Self => constants.%Self.b7c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.89a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T.5a3) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%T.5a3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.94e) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T.94e
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.d92
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.901
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_equivalent_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.321: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P.WithSelf(%Self.321) [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.9df: type = symbolic_binding_type Self, 0, %Self.321 [symbolic]
// CHECK:STDOUT:   %pattern_type.d71: type = pattern_type %Self.binding.as_type.9df [symbolic]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.82c: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Self.82c) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ: %Q.WithSelf.InQ.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.d77: type = symbolic_binding_type Self, 0, %Self.82c [symbolic]
// CHECK:STDOUT:   %pattern_type.eb5: type = pattern_type %Self.binding.as_type.d77 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4bd: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4bd, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.aef: <bound method> = bound_method %P.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.dec: type = facet_type <@P & @Q> [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.9ed: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R.WithSelf(%Self.9ed) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR: %R.WithSelf.InR.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.05c: type = symbolic_binding_type Self, 0, %Self.9ed [symbolic]
// CHECK:STDOUT:   %pattern_type.2d7: type = pattern_type %Self.binding.as_type.05c [symbolic]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.cf2: <bound method> = bound_method %facet_type.dec, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.7a2: type = facet_type <@P & @Q & @R> [concrete]
// CHECK:STDOUT:   %U: %facet_type.7a2 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3de: type = pattern_type %facet_type.7a2 [concrete]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.885: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %TakesPQR.type: type = fn_type @TakesPQR [concrete]
// CHECK:STDOUT:   %TakesPQR: %TakesPQR.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.0b3: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.232: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.6d8f6f.1: <witness> = lookup_impl_witness %Self.232, @Transitive [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type.381: type = facet_type <@R & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem2.889845.1: %facet_type.381 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.1, element2 [symbolic]
// CHECK:STDOUT:   %as_type.6edb67.1: type = facet_access_type %impl.elem2.889845.1 [symbolic]
// CHECK:STDOUT:   %facet_type.f69: type = facet_type <@Q & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem1.5edc4a.1: %facet_type.f69 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.1, element1 [symbolic]
// CHECK:STDOUT:   %as_type.734f04.1: type = facet_access_type %impl.elem1.5edc4a.1 [symbolic]
// CHECK:STDOUT:   %eq.3ada47.1 = observe_equivalent %as_type.734f04.1, %as_type.6edb67.1 [symbolic]
// CHECK:STDOUT:   %facet_type.a2c: type = facet_type <@P & @Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem0.8bc875.1: %facet_type.a2c = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.1, element0 [symbolic]
// CHECK:STDOUT:   %as_type.871755.1: type = facet_access_type %impl.elem0.8bc875.1 [symbolic]
// CHECK:STDOUT:   %eq.075fd0.1 = observe_equivalent %as_type.871755.1, %as_type.734f04.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.eda37f.1: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%Self.232) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.67da40.1: %Transitive.WithSelf.TakesC.type.eda37f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.104: type = symbolic_binding_type Self, 0, %Self.232 [symbolic]
// CHECK:STDOUT:   %pattern_type.993a5c.1: type = pattern_type %Self.binding.as_type.104 [symbolic]
// CHECK:STDOUT:   %pattern_type.2ef: type = pattern_type %as_type.6edb67.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.dc6981.1: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%Self.232) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.69dcdb.1: %Transitive.WithSelf.GetA.type.dc6981.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.23b964.1: type = pattern_type %as_type.871755.1 [symbolic]
// CHECK:STDOUT:   %.079496.1: Core.Form = init_form %as_type.871755.1, call_param1 [symbolic]
// CHECK:STDOUT:   %T: %Transitive.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %Transitive.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.993a5c.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1c0: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.6d8f6f.2: <witness> = lookup_impl_witness %T, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.8bc875.2: %facet_type.a2c = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.5edc4a.2: %facet_type.f69 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.dc6981.2: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.69dcdb.2: %Transitive.WithSelf.GetA.type.dc6981.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.eda37f.2: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.67da40.2: %Transitive.WithSelf.TakesC.type.eda37f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %as_type.734f04.2: type = facet_access_type %impl.elem1.5edc4a.2 [symbolic]
// CHECK:STDOUT:   %as_type.871755.2: type = facet_access_type %impl.elem0.8bc875.2 [symbolic]
// CHECK:STDOUT:   %eq.075fd0.2 = observe_equivalent %as_type.871755.2, %as_type.734f04.2 [symbolic]
// CHECK:STDOUT:   %impl.elem2.889845.2: %facet_type.381 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element2 [symbolic]
// CHECK:STDOUT:   %as_type.6edb67.2: type = facet_access_type %impl.elem2.889845.2 [symbolic]
// CHECK:STDOUT:   %eq.3ada47.2 = observe_equivalent %as_type.734f04.2, %as_type.6edb67.2 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.983: %Transitive.assoc_type = assoc_entity element0, imports.%Main.import_ref.606 [concrete]
// CHECK:STDOUT:   %require_complete.f6c: <witness> = require_complete_type %as_type.871755.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.23b964.2: type = pattern_type %as_type.871755.2 [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, imports.%Main.import_ref.11f [concrete]
// CHECK:STDOUT:   %.c0a: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type.dc6981.2, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.c0a = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element3 [symbolic]
// CHECK:STDOUT:   %.079496.2: Core.Form = init_form %as_type.871755.2, call_param1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a8a: <specific function> = specific_impl_function %impl.elem3, @Transitive.WithSelf.GetA(%T) [symbolic]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, imports.%Main.import_ref.5d7 [concrete]
// CHECK:STDOUT:   %impls.f83 = observe_impls %as_type.734f04.2, %Q.type [symbolic]
// CHECK:STDOUT:   %observe.decl.9f0 = observe_decl @observe.loc9 [concrete]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.8b2: %Q.assoc_type = assoc_entity element0, imports.%Main.import_ref.37b [concrete]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, imports.%Main.import_ref.efa [concrete]
// CHECK:STDOUT:   %eq.43e = observe_equivalent %as_type.871755.2, %as_type.6edb67.2 [symbolic]
// CHECK:STDOUT:   %impls.a2f = observe_impls %as_type.6edb67.2, %R.type [symbolic]
// CHECK:STDOUT:   %observe.decl.508 = observe_decl @observe.loc16 [concrete]
// CHECK:STDOUT:   %P.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.8bc875.2, @P [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.8bc875.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.871755.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.017: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.526: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.017, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.ea6: %.526 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f1e: <specific function> = specific_impl_function %impl.elem0.ea6, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.P: type = import_ref Main//transitive, P, loaded [concrete = constants.%P.type]
// CHECK:STDOUT:   %Main.Q: type = import_ref Main//transitive, Q, loaded [concrete = constants.%Q.type]
// CHECK:STDOUT:   %Main.R: type = import_ref Main//transitive, R, loaded [concrete = constants.%R.type]
// CHECK:STDOUT:   %Main.Transitive: type = import_ref Main//transitive, Transitive, loaded [concrete = constants.%Transitive.type]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.d29 = import_ref Main//transitive, loc3_35, unloaded
// CHECK:STDOUT:   %Main.InP = import_ref Main//transitive, InP, unloaded
// CHECK:STDOUT:   %Main.import_ref.e19f0c.2: %P.type = import_ref Main//transitive, loc3_13, loaded [symbolic = constants.%Self.321]
// CHECK:STDOUT:   %Main.import_ref.761 = import_ref Main//transitive, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.b1e: %Q.assoc_type = import_ref Main//transitive, loc4_35, loaded [concrete = constants.%assoc0.8b2]
// CHECK:STDOUT:   %Main.InQ = import_ref Main//transitive, InQ, unloaded
// CHECK:STDOUT:   %Main.import_ref.60b85f.2: %Q.type = import_ref Main//transitive, loc4_13, loaded [symbolic = constants.%Self.82c]
// CHECK:STDOUT:   %Main.import_ref.7a1 = import_ref Main//transitive, loc4_13, unloaded
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.8d3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.8d3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.ea8 = import_ref Main//transitive, loc5_35, unloaded
// CHECK:STDOUT:   %Main.InR = import_ref Main//transitive, InR, unloaded
// CHECK:STDOUT:   %Main.import_ref.ac9d5a.2: %R.type = import_ref Main//transitive, loc5_13, loaded [symbolic = constants.%Self.9ed]
// CHECK:STDOUT:   %Main.import_ref.fb9 = import_ref Main//transitive, loc5_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.059: %Transitive.assoc_type = import_ref Main//transitive, loc8_8, loaded [concrete = constants.%assoc0.983]
// CHECK:STDOUT:   %Main.import_ref.772: %Transitive.assoc_type = import_ref Main//transitive, loc9_8, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Main.import_ref.bdf: %Transitive.assoc_type = import_ref Main//transitive, loc10_8, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Main.import_ref.06e: %Transitive.assoc_type = import_ref Main//transitive, loc12_29, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Main.import_ref.a70 = import_ref Main//transitive, loc13_30, unloaded
// CHECK:STDOUT:   %Main.A = import_ref Main//transitive, A, unloaded
// CHECK:STDOUT:   %Main.B = import_ref Main//transitive, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//transitive, C, unloaded
// CHECK:STDOUT:   %Main.GetA = import_ref Main//transitive, GetA, unloaded
// CHECK:STDOUT:   %Main.TakesC = import_ref Main//transitive, TakesC, unloaded
// CHECK:STDOUT:   %Main.import_ref.abd70f.2: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.232]
// CHECK:STDOUT:   %Main.import_ref.abd70f.3: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.232]
// CHECK:STDOUT:   %Main.import_ref.92e = import_ref Main//transitive, loc7_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.606: %facet_type.a2c = import_ref Main//transitive, loc8_8, loaded [concrete = %A]
// CHECK:STDOUT:   %A: %facet_type.a2c = assoc_const_decl @A [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.11f: @Transitive.WithSelf.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type.dc6981.1) = import_ref Main//transitive, loc12_29, loaded [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA.69dcdb.1)]
// CHECK:STDOUT:   %Main.import_ref.5d7: %facet_type.f69 = import_ref Main//transitive, loc9_8, loaded [concrete = %B]
// CHECK:STDOUT:   %B: %facet_type.f69 = assoc_const_decl @B [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.37b: @Q.WithSelf.%Q.WithSelf.InQ.type (%Q.WithSelf.InQ.type) = import_ref Main//transitive, loc4_35, loaded [symbolic = @Q.WithSelf.%Q.WithSelf.InQ (constants.%Q.WithSelf.InQ)]
// CHECK:STDOUT:   %Main.import_ref.efa: %facet_type.381 = import_ref Main//transitive, loc10_8, loaded [concrete = %C]
// CHECK:STDOUT:   %C: %facet_type.381 = assoc_const_decl @C [concrete] {}
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .P = imports.%Main.P
// CHECK:STDOUT:     .Q = imports.%Main.Q
// CHECK:STDOUT:     .R = imports.%Main.R
// CHECK:STDOUT:     .Transitive = imports.%Main.Transitive
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .TakesPQR = %TakesPQR.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %TakesPQR.decl: %TakesPQR.type = fn_decl @TakesPQR [concrete = constants.%TakesPQR] {
// CHECK:STDOUT:     %U.patt: %pattern_type.3de = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %u.patt: @TakesPQR.%pattern_type (%pattern_type.885) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @TakesPQR.%pattern_type (%pattern_type.885) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_23.1: type = splice_block %.loc5_23.3 [concrete = constants.%facet_type.7a2] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %P.ref: type = name_ref P, imports.%Main.P [concrete = constants.%P.type]
// CHECK:STDOUT:       %Q.ref: type = name_ref Q, imports.%Main.Q [concrete = constants.%Q.type]
// CHECK:STDOUT:       %impl.elem0.loc5_19: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc5_19: <bound method> = bound_method %P.ref, %impl.elem0.loc5_19 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.aef]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc5_19: init type = call %bound_method.loc5_19(%P.ref, %Q.ref) [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %R.ref: type = name_ref R, imports.%Main.R [concrete = constants.%R.type]
// CHECK:STDOUT:       %impl.elem0.loc5_23: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc5_23: <bound method> = bound_method %type.as.BitAndWith.impl.Op.call.loc5_19, %impl.elem0.loc5_23 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.cf2]
// CHECK:STDOUT:       %.loc5_19.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc5_19 [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %.loc5_19.2: type = converted %type.as.BitAndWith.impl.Op.call.loc5_19, %.loc5_19.1 [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc5_23: init type = call %bound_method.loc5_23(%.loc5_19.2, %R.ref) [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:       %.loc5_23.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc5_23 [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:       %.loc5_23.3: type = converted %type.as.BitAndWith.impl.Op.call.loc5_23, %.loc5_23.2 [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc5_13.2: %facet_type.7a2 = symbolic_binding U, 0 [symbolic = %U.loc5_13.1 (constants.%U)]
// CHECK:STDOUT:     %u.param: @TakesPQR.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_38.1: type = splice_block %.loc5_38.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %facet_type.7a2 = name_ref U, %U.loc5_13.2 [symbolic = %U.loc5_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc5_38.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @TakesPQR.%U.binding.as_type (%U.binding.as_type) = value_binding u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.18b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @G.%pattern_type.loc6 (%pattern_type.993a5c.2) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @G.%pattern_type.loc6 (%pattern_type.993a5c.2) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_10: type = splice_block %Transitive.ref [concrete = constants.%Transitive.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Transitive.ref: type = name_ref Transitive, imports.%Main.Transitive [concrete = constants.%Transitive.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_6.2: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @G.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_25.1: type = splice_block %.loc6_25.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc6: %Transitive.type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6: type = facet_access_type %T.ref.loc6 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_25.2: type = converted %T.ref.loc6, %T.as_type.loc6 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @P [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.761
// CHECK:STDOUT:   .InP = imports.%Main.import_ref.d29
// CHECK:STDOUT:   witness = (imports.%Main.InP)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.7a1
// CHECK:STDOUT:   .InQ = imports.%Main.import_ref.b1e
// CHECK:STDOUT:   witness = (imports.%Main.InQ)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.fb9
// CHECK:STDOUT:   .InR = imports.%Main.import_ref.ea8
// CHECK:STDOUT:   witness = (imports.%Main.InR)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.92e
// CHECK:STDOUT:   .A = imports.%Main.import_ref.059
// CHECK:STDOUT:   .B = imports.%Main.import_ref.772
// CHECK:STDOUT:   .C = imports.%Main.import_ref.bdf
// CHECK:STDOUT:   .GetA = imports.%Main.import_ref.06e
// CHECK:STDOUT:   .TakesC = imports.%Main.import_ref.a70
// CHECK:STDOUT:   witness = (imports.%Main.A, imports.%Main.B, imports.%Main.C, imports.%Main.GetA, imports.%Main.TakesC)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.WithSelf.InP(imports.%Main.import_ref.e19f0c.2: %P.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.321)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.9df)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.d71)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(imports.%Main.import_ref.60b85f.2: %Q.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.82c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d77)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.eb5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(imports.%Main.import_ref.ac9d5a.2: %R.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9ed)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.05c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesPQR(%U.loc5_13.2: %facet_type.7a2) {
// CHECK:STDOUT:   %U.loc5_13.1: %facet_type.7a2 = symbolic_binding U, 0 [symbolic = %U.loc5_13.1 (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc5_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.885)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.0b3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @TakesPQR.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(imports.%Main.import_ref.abd70f.2: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.232)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.104)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.993a5c.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8f6f.1)]
// CHECK:STDOUT:   %impl.elem2: %facet_type.381 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2.889845.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2 [symbolic = %as_type (constants.%as_type.6edb67.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.2ef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(imports.%Main.import_ref.abd70f.3: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.232)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.104)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.993a5c.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8f6f.1)]
// CHECK:STDOUT:   %impl.elem0: %facet_type.a2c = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.8bc875.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic = %as_type (constants.%as_type.871755.1)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %as_type, call_param1 [symbolic = %.1 (constants.%.079496.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.23b964.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc6_6.2: %Transitive.type) {
// CHECK:STDOUT:   %T.loc6_6.1: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc6_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc6 (constants.%pattern_type.993a5c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc6 (constants.%require_complete.1c0)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc6_6.1, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8f6f.2)]
// CHECK:STDOUT:   %impl.elem0.loc7_11.2: %facet_type.a2c = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_11.2 (constants.%impl.elem0.8bc875.2)]
// CHECK:STDOUT:   %as_type.loc7_11.2: type = facet_access_type %impl.elem0.loc7_11.2 [symbolic = %as_type.loc7_11.2 (constants.%as_type.871755.2)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %as_type.loc7_11.2 [symbolic = %require_complete.loc7 (constants.%require_complete.f6c)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %as_type.loc7_11.2 [symbolic = %pattern_type.loc7 (constants.%pattern_type.23b964.2)]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T.loc6_6.1) [symbolic = %Transitive.WithSelf.GetA.type (constants.%Transitive.WithSelf.GetA.type.dc6981.2)]
// CHECK:STDOUT:   %.loc7_17: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type, %T.loc6_6.1 [symbolic = %.loc7_17 (constants.%.c0a)]
// CHECK:STDOUT:   %impl.elem3.loc7_17.2: @G.%.loc7_17 (%.c0a) = impl_witness_access %Transitive.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc7_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_17.2: <specific function> = specific_impl_function %impl.elem3.loc7_17.2, @Transitive.WithSelf.GetA(%T.loc6_6.1) [symbolic = %specific_impl_fn.loc7_17.2 (constants.%specific_impl_fn.a8a)]
// CHECK:STDOUT:   %impl.elem1: %facet_type.f69 = impl_witness_access %Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1 (constants.%impl.elem1.5edc4a.2)]
// CHECK:STDOUT:   %as_type.loc9: type = facet_access_type %impl.elem1 [symbolic = %as_type.loc9 (constants.%as_type.734f04.2)]
// CHECK:STDOUT:   %eq.loc9 = observe_equivalent %as_type.loc7_11.2, %as_type.loc9 [symbolic = %eq.loc9 (constants.%eq.075fd0.2)]
// CHECK:STDOUT:   %impls.loc9 = observe_impls %as_type.loc9, constants.%Q.type [symbolic = %impls.loc9 (constants.%impls.f83)]
// CHECK:STDOUT:   %impl.elem2: %facet_type.381 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2.889845.2)]
// CHECK:STDOUT:   %as_type.loc16: type = facet_access_type %impl.elem2 [symbolic = %as_type.loc16 (constants.%as_type.6edb67.2)]
// CHECK:STDOUT:   %eq.loc16 = observe_equivalent %as_type.loc7_11.2, %as_type.loc16 [symbolic = %eq.loc16 (constants.%eq.43e)]
// CHECK:STDOUT:   %impls.loc16 = observe_impls %as_type.loc16, constants.%R.type [symbolic = %impls.loc16 (constants.%impls.a2f)]
// CHECK:STDOUT:   %P.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc7_11.2, @P [symbolic = %P.lookup_impl_witness (constants.%P.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc7_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.loc7_11.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.017)]
// CHECK:STDOUT:   %.loc7_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc7_3.4 (constants.%.526)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @G.%.loc7_3.4 (%.526) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.ea6)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.f1e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @G.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @G.%pattern_type.loc7 (%pattern_type.23b964.2) = ref_binding_pattern a [concrete]
// CHECK:STDOUT:       %a.var_patt: @G.%pattern_type.loc7 (%pattern_type.23b964.2) = var_pattern %a.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.var: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = var %a.var_patt
// CHECK:STDOUT:     %t.ref: @G.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %GetA.ref: %Transitive.assoc_type = name_ref GetA, imports.%Main.import_ref.06e [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc7_17.1: @G.%.loc7_17 (%.c0a) = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8f6f.2, element3 [symbolic = %impl.elem3.loc7_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc7_17: <bound method> = bound_method %t.ref, %impl.elem3.loc7_17.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_17.1: <specific function> = specific_impl_function %impl.elem3.loc7_17.1, @Transitive.WithSelf.GetA(constants.%T) [symbolic = %specific_impl_fn.loc7_17.2 (constants.%specific_impl_fn.a8a)]
// CHECK:STDOUT:     %bound_method.loc7_23: <bound method> = bound_method %t.ref, %specific_impl_fn.loc7_17.1
// CHECK:STDOUT:     %.loc7_23: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = temporary_storage
// CHECK:STDOUT:     %Transitive.WithSelf.GetA.call: init @G.%as_type.loc7_11.2 (%as_type.871755.2) to %.loc7_23 = call %bound_method.loc7_23(%t.ref)
// CHECK:STDOUT:     %.loc7_3.1: init @G.%as_type.loc7_11.2 (%as_type.871755.2) = as_compatible %Transitive.WithSelf.GetA.call
// CHECK:STDOUT:     assign %a.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_11.1: type = splice_block %.loc7_11.3 [symbolic = %as_type.loc7_11.2 (constants.%as_type.871755.2)] {
// CHECK:STDOUT:       %T.ref.loc7: %Transitive.type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7: type = facet_access_type %T.ref.loc7 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_11.2: type = converted %T.ref.loc7, %T.as_type.loc7 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %A.ref: %Transitive.assoc_type = name_ref A, imports.%Main.import_ref.059 [concrete = constants.%assoc0.983]
// CHECK:STDOUT:       %impl.elem0.loc7_11.1: %facet_type.a2c = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8f6f.2, element0 [symbolic = %impl.elem0.loc7_11.2 (constants.%impl.elem0.8bc875.2)]
// CHECK:STDOUT:       %as_type.loc7_11.1: type = facet_access_type %impl.elem0.loc7_11.1 [symbolic = %as_type.loc7_11.2 (constants.%as_type.871755.2)]
// CHECK:STDOUT:       %.loc7_11.3: type = converted %impl.elem0.loc7_11.1, %as_type.loc7_11.1 [symbolic = %as_type.loc7_11.2 (constants.%as_type.871755.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_3.2: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = as_compatible %a.var
// CHECK:STDOUT:     %a: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = ref_binding a, %.loc7_3.2
// CHECK:STDOUT:     %observe.decl.loc9 = observe_decl @observe.loc9 [concrete = constants.%observe.decl.9f0]
// CHECK:STDOUT:     %a.ref.loc14: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = name_ref a, %a
// CHECK:STDOUT:     %Q.ref: type = name_ref Q, imports.%Main.Q [concrete = constants.%Q.type]
// CHECK:STDOUT:     %InQ.ref: %Q.assoc_type = name_ref InQ, imports.%Main.import_ref.b1e [concrete = constants.%assoc0.8b2]
// CHECK:STDOUT:     %observe.decl.loc16 = observe_decl @observe.loc16 [concrete = constants.%observe.decl.508]
// CHECK:STDOUT:     %TakesPQR.ref: %TakesPQR.type = name_ref TakesPQR, file.%TakesPQR.decl [concrete = constants.%TakesPQR]
// CHECK:STDOUT:     %a.ref.loc24: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = name_ref a, %a
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @G.%.loc7_3.4 (%.526) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.ea6)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.f1e)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %.loc7_3.3: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = as_compatible %a.var
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%.loc7_3.3)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf(constants.%Self.321) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.321
// CHECK:STDOUT:   %P.WithSelf.InP.type => constants.%P.WithSelf.InP.type
// CHECK:STDOUT:   %P.WithSelf.InP => constants.%P.WithSelf.InP
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf.InP(constants.%Self.321) {
// CHECK:STDOUT:   %Self => constants.%Self.321
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.9df
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Self.82c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.82c
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.82c) {
// CHECK:STDOUT:   %Self => constants.%Self.82c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.d77
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eb5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%Self.9ed) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.9ed
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%Self.9ed) {
// CHECK:STDOUT:   %Self => constants.%Self.9ed
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.05c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesPQR(constants.%U) {
// CHECK:STDOUT:   %U.loc5_13.1 => constants.%U
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.885
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%Self.232) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.232
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8bc875.1
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.5edc4a.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.dc6981.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.69dcdb.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.eda37f.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.67da40.1
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.734f04.1
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.871755.1
// CHECK:STDOUT:   %eq.1 => constants.%eq.075fd0.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.889845.1
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.6edb67.1
// CHECK:STDOUT:   %eq.2 => constants.%eq.3ada47.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.232) {
// CHECK:STDOUT:   %Self => constants.%Self.232
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.104
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.993a5c.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.889845.1
// CHECK:STDOUT:   %as_type => constants.%as_type.6edb67.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.2ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.232) {
// CHECK:STDOUT:   %Self => constants.%Self.232
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.104
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.993a5c.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8bc875.1
// CHECK:STDOUT:   %as_type => constants.%as_type.871755.1
// CHECK:STDOUT:   %.1 => constants.%.079496.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.23b964.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc6 => constants.%pattern_type.993a5c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8bc875.2
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.5edc4a.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.dc6981.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.69dcdb.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.eda37f.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.67da40.2
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.734f04.2
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.871755.2
// CHECK:STDOUT:   %eq.1 => constants.%eq.075fd0.2
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.889845.2
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.6edb67.2
// CHECK:STDOUT:   %eq.2 => constants.%eq.3ada47.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.993a5c.2
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8bc875.2
// CHECK:STDOUT:   %as_type => constants.%as_type.871755.2
// CHECK:STDOUT:   %.1 => constants.%.079496.2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.23b964.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_equivalent_impls_without_observe.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.321: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P.WithSelf(%Self.321) [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.9df: type = symbolic_binding_type Self, 0, %Self.321 [symbolic]
// CHECK:STDOUT:   %pattern_type.d71: type = pattern_type %Self.binding.as_type.9df [symbolic]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.82c: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q.WithSelf(%Self.82c) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ: %Q.WithSelf.InQ.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.d77: type = symbolic_binding_type Self, 0, %Self.82c [symbolic]
// CHECK:STDOUT:   %pattern_type.eb5: type = pattern_type %Self.binding.as_type.d77 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4bd: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4bd, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.aef: <bound method> = bound_method %P.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.dec: type = facet_type <@P & @Q> [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.9ed: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R.WithSelf(%Self.9ed) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR: %R.WithSelf.InR.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.05c: type = symbolic_binding_type Self, 0, %Self.9ed [symbolic]
// CHECK:STDOUT:   %pattern_type.2d7: type = pattern_type %Self.binding.as_type.05c [symbolic]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.cf2: <bound method> = bound_method %facet_type.dec, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.7a2: type = facet_type <@P & @Q & @R> [concrete]
// CHECK:STDOUT:   %U: %facet_type.7a2 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3de: type = pattern_type %facet_type.7a2 [concrete]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.885: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %TakesPQR.type: type = fn_type @TakesPQR [concrete]
// CHECK:STDOUT:   %TakesPQR: %TakesPQR.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.0b3: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.232: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.6d8f6f.1: <witness> = lookup_impl_witness %Self.232, @Transitive [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type.381: type = facet_type <@R & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem2.889845.1: %facet_type.381 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.1, element2 [symbolic]
// CHECK:STDOUT:   %as_type.6edb67.1: type = facet_access_type %impl.elem2.889845.1 [symbolic]
// CHECK:STDOUT:   %facet_type.f69: type = facet_type <@Q & @Destroy where TODO> [concrete]
// CHECK:STDOUT:   %impl.elem1.5edc4a.1: %facet_type.f69 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.1, element1 [symbolic]
// CHECK:STDOUT:   %as_type.734f04.1: type = facet_access_type %impl.elem1.5edc4a.1 [symbolic]
// CHECK:STDOUT:   %eq.3ada47.1 = observe_equivalent %as_type.734f04.1, %as_type.6edb67.1 [symbolic]
// CHECK:STDOUT:   %facet_type.a2c: type = facet_type <@P & @Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem0.8bc875.1: %facet_type.a2c = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.1, element0 [symbolic]
// CHECK:STDOUT:   %as_type.871755.1: type = facet_access_type %impl.elem0.8bc875.1 [symbolic]
// CHECK:STDOUT:   %eq.075fd0.1 = observe_equivalent %as_type.871755.1, %as_type.734f04.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.eda37f.1: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%Self.232) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.67da40.1: %Transitive.WithSelf.TakesC.type.eda37f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.104: type = symbolic_binding_type Self, 0, %Self.232 [symbolic]
// CHECK:STDOUT:   %pattern_type.993a5c.1: type = pattern_type %Self.binding.as_type.104 [symbolic]
// CHECK:STDOUT:   %pattern_type.2ef: type = pattern_type %as_type.6edb67.1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.dc6981.1: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%Self.232) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.69dcdb.1: %Transitive.WithSelf.GetA.type.dc6981.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.23b964.1: type = pattern_type %as_type.871755.1 [symbolic]
// CHECK:STDOUT:   %.079496.1: Core.Form = init_form %as_type.871755.1, call_param1 [symbolic]
// CHECK:STDOUT:   %T: %Transitive.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %Transitive.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.993a5c.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1c0: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.6d8f6f.2: <witness> = lookup_impl_witness %T, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.8bc875.2: %facet_type.a2c = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.5edc4a.2: %facet_type.f69 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.dc6981.2: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.69dcdb.2: %Transitive.WithSelf.GetA.type.dc6981.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.eda37f.2: type = fn_type @Transitive.WithSelf.TakesC, @Transitive.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.67da40.2: %Transitive.WithSelf.TakesC.type.eda37f.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %as_type.734f04.2: type = facet_access_type %impl.elem1.5edc4a.2 [symbolic]
// CHECK:STDOUT:   %as_type.871755.2: type = facet_access_type %impl.elem0.8bc875.2 [symbolic]
// CHECK:STDOUT:   %eq.075fd0.2 = observe_equivalent %as_type.871755.2, %as_type.734f04.2 [symbolic]
// CHECK:STDOUT:   %impl.elem2.889845.2: %facet_type.381 = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element2 [symbolic]
// CHECK:STDOUT:   %as_type.6edb67.2: type = facet_access_type %impl.elem2.889845.2 [symbolic]
// CHECK:STDOUT:   %eq.3ada47.2 = observe_equivalent %as_type.734f04.2, %as_type.6edb67.2 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.983: %Transitive.assoc_type = assoc_entity element0, imports.%Main.import_ref.606 [concrete]
// CHECK:STDOUT:   %require_complete.f6c: <witness> = require_complete_type %as_type.871755.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.23b964.2: type = pattern_type %as_type.871755.2 [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, imports.%Main.import_ref.11f [concrete]
// CHECK:STDOUT:   %.c0a: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type.dc6981.2, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.c0a = impl_witness_access %Transitive.lookup_impl_witness.6d8f6f.2, element3 [symbolic]
// CHECK:STDOUT:   %.079496.2: Core.Form = init_form %as_type.871755.2, call_param1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a8a: <specific function> = specific_impl_function %impl.elem3, @Transitive.WithSelf.GetA(%T) [symbolic]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.8b2: %Q.assoc_type = assoc_entity element0, imports.%Main.import_ref.37b [concrete]
// CHECK:STDOUT:   %P.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.8bc875.2, @P [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.8bc875.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.871755.2, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.017: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.526: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.017, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.ea6: %.526 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f1e: <specific function> = specific_impl_function %impl.elem0.ea6, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.P: type = import_ref Main//transitive, P, loaded [concrete = constants.%P.type]
// CHECK:STDOUT:   %Main.Q: type = import_ref Main//transitive, Q, loaded [concrete = constants.%Q.type]
// CHECK:STDOUT:   %Main.R: type = import_ref Main//transitive, R, loaded [concrete = constants.%R.type]
// CHECK:STDOUT:   %Main.Transitive: type = import_ref Main//transitive, Transitive, loaded [concrete = constants.%Transitive.type]
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.d29 = import_ref Main//transitive, loc3_35, unloaded
// CHECK:STDOUT:   %Main.InP = import_ref Main//transitive, InP, unloaded
// CHECK:STDOUT:   %Main.import_ref.e19f0c.2: %P.type = import_ref Main//transitive, loc3_13, loaded [symbolic = constants.%Self.321]
// CHECK:STDOUT:   %Main.import_ref.761 = import_ref Main//transitive, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.b1e: %Q.assoc_type = import_ref Main//transitive, loc4_35, loaded [concrete = constants.%assoc0.8b2]
// CHECK:STDOUT:   %Main.InQ = import_ref Main//transitive, InQ, unloaded
// CHECK:STDOUT:   %Main.import_ref.60b85f.2: %Q.type = import_ref Main//transitive, loc4_13, loaded [symbolic = constants.%Self.82c]
// CHECK:STDOUT:   %Main.import_ref.7a1 = import_ref Main//transitive, loc4_13, unloaded
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.8d3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.8d3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.ea8 = import_ref Main//transitive, loc5_35, unloaded
// CHECK:STDOUT:   %Main.InR = import_ref Main//transitive, InR, unloaded
// CHECK:STDOUT:   %Main.import_ref.ac9d5a.2: %R.type = import_ref Main//transitive, loc5_13, loaded [symbolic = constants.%Self.9ed]
// CHECK:STDOUT:   %Main.import_ref.fb9 = import_ref Main//transitive, loc5_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.059: %Transitive.assoc_type = import_ref Main//transitive, loc8_8, loaded [concrete = constants.%assoc0.983]
// CHECK:STDOUT:   %Main.import_ref.0dc = import_ref Main//transitive, loc9_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.d19 = import_ref Main//transitive, loc10_8, unloaded
// CHECK:STDOUT:   %Main.import_ref.06e: %Transitive.assoc_type = import_ref Main//transitive, loc12_29, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Main.import_ref.a70 = import_ref Main//transitive, loc13_30, unloaded
// CHECK:STDOUT:   %Main.A = import_ref Main//transitive, A, unloaded
// CHECK:STDOUT:   %Main.B = import_ref Main//transitive, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//transitive, C, unloaded
// CHECK:STDOUT:   %Main.GetA = import_ref Main//transitive, GetA, unloaded
// CHECK:STDOUT:   %Main.TakesC = import_ref Main//transitive, TakesC, unloaded
// CHECK:STDOUT:   %Main.import_ref.abd70f.2: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.232]
// CHECK:STDOUT:   %Main.import_ref.abd70f.3: %Transitive.type = import_ref Main//transitive, loc7_22, loaded [symbolic = constants.%Self.232]
// CHECK:STDOUT:   %Main.import_ref.92e = import_ref Main//transitive, loc7_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.606: %facet_type.a2c = import_ref Main//transitive, loc8_8, loaded [concrete = %A]
// CHECK:STDOUT:   %A: %facet_type.a2c = assoc_const_decl @A [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.11f: @Transitive.WithSelf.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type.dc6981.1) = import_ref Main//transitive, loc12_29, loaded [symbolic = @Transitive.WithSelf.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA.69dcdb.1)]
// CHECK:STDOUT:   %Main.import_ref.37b: @Q.WithSelf.%Q.WithSelf.InQ.type (%Q.WithSelf.InQ.type) = import_ref Main//transitive, loc4_35, loaded [symbolic = @Q.WithSelf.%Q.WithSelf.InQ (constants.%Q.WithSelf.InQ)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .P = imports.%Main.P
// CHECK:STDOUT:     .Q = imports.%Main.Q
// CHECK:STDOUT:     .R = imports.%Main.R
// CHECK:STDOUT:     .Transitive = imports.%Main.Transitive
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .TakesPQR = %TakesPQR.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %TakesPQR.decl: %TakesPQR.type = fn_decl @TakesPQR [concrete = constants.%TakesPQR] {
// CHECK:STDOUT:     %U.patt: %pattern_type.3de = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %u.patt: @TakesPQR.%pattern_type (%pattern_type.885) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @TakesPQR.%pattern_type (%pattern_type.885) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_23.1: type = splice_block %.loc5_23.3 [concrete = constants.%facet_type.7a2] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %P.ref: type = name_ref P, imports.%Main.P [concrete = constants.%P.type]
// CHECK:STDOUT:       %Q.ref: type = name_ref Q, imports.%Main.Q [concrete = constants.%Q.type]
// CHECK:STDOUT:       %impl.elem0.loc5_19: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc5_19: <bound method> = bound_method %P.ref, %impl.elem0.loc5_19 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.aef]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc5_19: init type = call %bound_method.loc5_19(%P.ref, %Q.ref) [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %R.ref: type = name_ref R, imports.%Main.R [concrete = constants.%R.type]
// CHECK:STDOUT:       %impl.elem0.loc5_23: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc5_23: <bound method> = bound_method %type.as.BitAndWith.impl.Op.call.loc5_19, %impl.elem0.loc5_23 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.cf2]
// CHECK:STDOUT:       %.loc5_19.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc5_19 [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %.loc5_19.2: type = converted %type.as.BitAndWith.impl.Op.call.loc5_19, %.loc5_19.1 [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc5_23: init type = call %bound_method.loc5_23(%.loc5_19.2, %R.ref) [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:       %.loc5_23.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc5_23 [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:       %.loc5_23.3: type = converted %type.as.BitAndWith.impl.Op.call.loc5_23, %.loc5_23.2 [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc5_13.2: %facet_type.7a2 = symbolic_binding U, 0 [symbolic = %U.loc5_13.1 (constants.%U)]
// CHECK:STDOUT:     %u.param: @TakesPQR.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_38.1: type = splice_block %.loc5_38.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %facet_type.7a2 = name_ref U, %U.loc5_13.2 [symbolic = %U.loc5_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc5_38.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @TakesPQR.%U.binding.as_type (%U.binding.as_type) = value_binding u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.18b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @G.%pattern_type.loc6 (%pattern_type.993a5c.2) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @G.%pattern_type.loc6 (%pattern_type.993a5c.2) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_10: type = splice_block %Transitive.ref [concrete = constants.%Transitive.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Transitive.ref: type = name_ref Transitive, imports.%Main.Transitive [concrete = constants.%Transitive.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_6.2: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @G.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_25.1: type = splice_block %.loc6_25.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc6: %Transitive.type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6: type = facet_access_type %T.ref.loc6 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_25.2: type = converted %T.ref.loc6, %T.as_type.loc6 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @P [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.761
// CHECK:STDOUT:   .InP = imports.%Main.import_ref.d29
// CHECK:STDOUT:   witness = (imports.%Main.InP)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.7a1
// CHECK:STDOUT:   .InQ = imports.%Main.import_ref.b1e
// CHECK:STDOUT:   witness = (imports.%Main.InQ)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.fb9
// CHECK:STDOUT:   .InR = imports.%Main.import_ref.ea8
// CHECK:STDOUT:   witness = (imports.%Main.InR)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive [from "transitive.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.92e
// CHECK:STDOUT:   .A = imports.%Main.import_ref.059
// CHECK:STDOUT:   .B = imports.%Main.import_ref.0dc
// CHECK:STDOUT:   .C = imports.%Main.import_ref.d19
// CHECK:STDOUT:   .GetA = imports.%Main.import_ref.06e
// CHECK:STDOUT:   .TakesC = imports.%Main.import_ref.a70
// CHECK:STDOUT:   witness = (imports.%Main.A, imports.%Main.B, imports.%Main.C, imports.%Main.GetA, imports.%Main.TakesC)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.WithSelf.InP(imports.%Main.import_ref.e19f0c.2: %P.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.321)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.9df)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.d71)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(imports.%Main.import_ref.60b85f.2: %Q.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.82c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d77)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.eb5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(imports.%Main.import_ref.ac9d5a.2: %R.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.9ed)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.05c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesPQR(%U.loc5_13.2: %facet_type.7a2) {
// CHECK:STDOUT:   %U.loc5_13.1: %facet_type.7a2 = symbolic_binding U, 0 [symbolic = %U.loc5_13.1 (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc5_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.885)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.0b3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @TakesPQR.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(imports.%Main.import_ref.abd70f.2: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.232)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.104)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.993a5c.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8f6f.1)]
// CHECK:STDOUT:   %impl.elem2: %facet_type.381 = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2.889845.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2 [symbolic = %as_type (constants.%as_type.6edb67.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.2ef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(imports.%Main.import_ref.abd70f.3: %Transitive.type) [from "transitive.carbon"] {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.232)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.104)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.993a5c.1)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8f6f.1)]
// CHECK:STDOUT:   %impl.elem0: %facet_type.a2c = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.8bc875.1)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0 [symbolic = %as_type (constants.%as_type.871755.1)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %as_type, call_param1 [symbolic = %.1 (constants.%.079496.1)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %as_type [symbolic = %pattern_type.2 (constants.%pattern_type.23b964.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc6_6.2: %Transitive.type) {
// CHECK:STDOUT:   %T.loc6_6.1: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc6_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc6 (constants.%pattern_type.993a5c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc6 (constants.%require_complete.1c0)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc6_6.1, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8f6f.2)]
// CHECK:STDOUT:   %impl.elem0.loc7_11.2: %facet_type.a2c = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_11.2 (constants.%impl.elem0.8bc875.2)]
// CHECK:STDOUT:   %as_type.loc7_11.2: type = facet_access_type %impl.elem0.loc7_11.2 [symbolic = %as_type.loc7_11.2 (constants.%as_type.871755.2)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %as_type.loc7_11.2 [symbolic = %require_complete.loc7 (constants.%require_complete.f6c)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %as_type.loc7_11.2 [symbolic = %pattern_type.loc7 (constants.%pattern_type.23b964.2)]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive.WithSelf(%T.loc6_6.1) [symbolic = %Transitive.WithSelf.GetA.type (constants.%Transitive.WithSelf.GetA.type.dc6981.2)]
// CHECK:STDOUT:   %.loc7_17: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type, %T.loc6_6.1 [symbolic = %.loc7_17 (constants.%.c0a)]
// CHECK:STDOUT:   %impl.elem3.loc7_17.2: @G.%.loc7_17 (%.c0a) = impl_witness_access %Transitive.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc7_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_17.2: <specific function> = specific_impl_function %impl.elem3.loc7_17.2, @Transitive.WithSelf.GetA(%T.loc6_6.1) [symbolic = %specific_impl_fn.loc7_17.2 (constants.%specific_impl_fn.a8a)]
// CHECK:STDOUT:   %P.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc7_11.2, @P [symbolic = %P.lookup_impl_witness (constants.%P.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc7_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.loc7_11.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.017)]
// CHECK:STDOUT:   %.loc7_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc7_3.4 (constants.%.526)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @G.%.loc7_3.4 (%.526) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.ea6)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.f1e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @G.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @G.%pattern_type.loc7 (%pattern_type.23b964.2) = ref_binding_pattern a [concrete]
// CHECK:STDOUT:       %a.var_patt: @G.%pattern_type.loc7 (%pattern_type.23b964.2) = var_pattern %a.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.var: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = var %a.var_patt
// CHECK:STDOUT:     %t.ref: @G.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %GetA.ref: %Transitive.assoc_type = name_ref GetA, imports.%Main.import_ref.06e [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc7_17.1: @G.%.loc7_17 (%.c0a) = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8f6f.2, element3 [symbolic = %impl.elem3.loc7_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc7_17: <bound method> = bound_method %t.ref, %impl.elem3.loc7_17.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_17.1: <specific function> = specific_impl_function %impl.elem3.loc7_17.1, @Transitive.WithSelf.GetA(constants.%T) [symbolic = %specific_impl_fn.loc7_17.2 (constants.%specific_impl_fn.a8a)]
// CHECK:STDOUT:     %bound_method.loc7_23: <bound method> = bound_method %t.ref, %specific_impl_fn.loc7_17.1
// CHECK:STDOUT:     %.loc7_23: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = temporary_storage
// CHECK:STDOUT:     %Transitive.WithSelf.GetA.call: init @G.%as_type.loc7_11.2 (%as_type.871755.2) to %.loc7_23 = call %bound_method.loc7_23(%t.ref)
// CHECK:STDOUT:     %.loc7_3.1: init @G.%as_type.loc7_11.2 (%as_type.871755.2) = as_compatible %Transitive.WithSelf.GetA.call
// CHECK:STDOUT:     assign %a.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_11.1: type = splice_block %.loc7_11.3 [symbolic = %as_type.loc7_11.2 (constants.%as_type.871755.2)] {
// CHECK:STDOUT:       %T.ref.loc7: %Transitive.type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc7: type = facet_access_type %T.ref.loc7 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_11.2: type = converted %T.ref.loc7, %T.as_type.loc7 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %A.ref: %Transitive.assoc_type = name_ref A, imports.%Main.import_ref.059 [concrete = constants.%assoc0.983]
// CHECK:STDOUT:       %impl.elem0.loc7_11.1: %facet_type.a2c = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8f6f.2, element0 [symbolic = %impl.elem0.loc7_11.2 (constants.%impl.elem0.8bc875.2)]
// CHECK:STDOUT:       %as_type.loc7_11.1: type = facet_access_type %impl.elem0.loc7_11.1 [symbolic = %as_type.loc7_11.2 (constants.%as_type.871755.2)]
// CHECK:STDOUT:       %.loc7_11.3: type = converted %impl.elem0.loc7_11.1, %as_type.loc7_11.1 [symbolic = %as_type.loc7_11.2 (constants.%as_type.871755.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_3.2: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = as_compatible %a.var
// CHECK:STDOUT:     %a: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = ref_binding a, %.loc7_3.2
// CHECK:STDOUT:     %a.ref.loc13: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = name_ref a, %a
// CHECK:STDOUT:     %Q.ref: type = name_ref Q, imports.%Main.Q [concrete = constants.%Q.type]
// CHECK:STDOUT:     %InQ.ref: %Q.assoc_type = name_ref InQ, imports.%Main.import_ref.b1e [concrete = constants.%assoc0.8b2]
// CHECK:STDOUT:     %TakesPQR.ref: %TakesPQR.type = name_ref TakesPQR, file.%TakesPQR.decl [concrete = constants.%TakesPQR]
// CHECK:STDOUT:     %a.ref.loc22: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = name_ref a, %a
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @G.%.loc7_3.4 (%.526) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0.ea6)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn.f1e)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %.loc7_3.3: ref @G.%as_type.loc7_11.2 (%as_type.871755.2) = as_compatible %a.var
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%.loc7_3.3)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf(constants.%Self.321) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.321
// CHECK:STDOUT:   %P.WithSelf.InP.type => constants.%P.WithSelf.InP.type
// CHECK:STDOUT:   %P.WithSelf.InP => constants.%P.WithSelf.InP
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.WithSelf.InP(constants.%Self.321) {
// CHECK:STDOUT:   %Self => constants.%Self.321
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.9df
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf(constants.%Self.82c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.82c
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.82c) {
// CHECK:STDOUT:   %Self => constants.%Self.82c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.d77
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eb5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf(constants.%Self.9ed) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.9ed
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%Self.9ed) {
// CHECK:STDOUT:   %Self => constants.%Self.9ed
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.05c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesPQR(constants.%U) {
// CHECK:STDOUT:   %U.loc5_13.1 => constants.%U
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.885
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%Self.232) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.232
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8bc875.1
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.5edc4a.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.dc6981.1
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.69dcdb.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.eda37f.1
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.67da40.1
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.734f04.1
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.871755.1
// CHECK:STDOUT:   %eq.1 => constants.%eq.075fd0.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.889845.1
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.6edb67.1
// CHECK:STDOUT:   %eq.2 => constants.%eq.3ada47.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.232) {
// CHECK:STDOUT:   %Self => constants.%Self.232
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.104
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.993a5c.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.1
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.889845.1
// CHECK:STDOUT:   %as_type => constants.%as_type.6edb67.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.2ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.232) {
// CHECK:STDOUT:   %Self => constants.%Self.232
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.104
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.993a5c.1
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.1
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8bc875.1
// CHECK:STDOUT:   %as_type => constants.%as_type.871755.1
// CHECK:STDOUT:   %.1 => constants.%.079496.1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.23b964.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc6 => constants.%pattern_type.993a5c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8bc875.2
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.5edc4a.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.dc6981.2
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.69dcdb.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.eda37f.2
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.67da40.2
// CHECK:STDOUT:   %as_type.1 => constants.%as_type.734f04.2
// CHECK:STDOUT:   %as_type.2 => constants.%as_type.871755.2
// CHECK:STDOUT:   %eq.1 => constants.%eq.075fd0.2
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.889845.2
// CHECK:STDOUT:   %as_type.3 => constants.%as_type.6edb67.2
// CHECK:STDOUT:   %eq.2 => constants.%eq.3ada47.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.993a5c.2
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8f6f.2
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.8bc875.2
// CHECK:STDOUT:   %as_type => constants.%as_type.871755.2
// CHECK:STDOUT:   %.1 => constants.%.079496.2
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.23b964.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- blanket_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self.b7c: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %D.type: type = facet_type <@D> [concrete]
// CHECK:STDOUT:   %Self.b8b: %D.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.5a3: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.0b3: type = symbolic_binding_type T, 0, %T.5a3 [symbolic]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness @T.binding.as_type.as.B.impl.%B.impl_witness_table, @T.binding.as_type.as.B.impl(%T.5a3) [symbolic]
// CHECK:STDOUT:   %B.facet.234: %B.type = facet_value %T.binding.as_type.0b3, (%B.impl_witness) [symbolic]
// CHECK:STDOUT:   %T.5ee: %B.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.035: type = pattern_type %B.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.856: type = symbolic_binding_type T, 0, %T.5ee [symbolic]
// CHECK:STDOUT:   %C.impl_witness.c1a: <witness> = impl_witness @T.binding.as_type.as.C.impl.%C.impl_witness_table, @T.binding.as_type.as.C.impl(%T.5ee) [symbolic]
// CHECK:STDOUT:   %C.facet.50a: %C.type = facet_value %T.binding.as_type.856, (%C.impl_witness.c1a) [symbolic]
// CHECK:STDOUT:   %T.94e: %C.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7f1: type = pattern_type %C.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.d92: type = symbolic_binding_type T, 0, %T.94e [symbolic]
// CHECK:STDOUT:   %D.impl_witness.c16: <witness> = impl_witness @T.binding.as_type.as.D.impl.%D.impl_witness_table, @T.binding.as_type.as.D.impl(%T.94e) [symbolic]
// CHECK:STDOUT:   %D.facet.d95: %D.type = facet_value %T.binding.as_type.d92, (%D.impl_witness.c16) [symbolic]
// CHECK:STDOUT:   %pattern_type.fae: type = pattern_type %T.binding.as_type.856 [symbolic]
// CHECK:STDOUT:   %RequiresB.type: type = fn_type @RequiresB [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %RequiresB: %RequiresB.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.f6b: <witness> = require_complete_type %T.binding.as_type.856 [symbolic]
// CHECK:STDOUT:   %T.233: %D.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f05: type = pattern_type %D.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.f5f: type = symbolic_binding_type T, 0, %T.233 [symbolic]
// CHECK:STDOUT:   %pattern_type.d60: type = pattern_type %T.binding.as_type.f5f [symbolic]
// CHECK:STDOUT:   %RequiresD.type: type = fn_type @RequiresD [concrete]
// CHECK:STDOUT:   %RequiresD: %RequiresD.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ee0: <witness> = require_complete_type %T.binding.as_type.f5f [symbolic]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %T.binding.as_type.0b3 [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1a7: <witness> = require_complete_type %T.binding.as_type.0b3 [symbolic]
// CHECK:STDOUT:   %.7ae: require_specific_def_type = require_specific_def @T.binding.as_type.as.B.impl(%T.5a3) [symbolic]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.5a3, @B [symbolic]
// CHECK:STDOUT:   %B.facet.83f: %B.type = facet_value %T.binding.as_type.0b3, (%B.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %RequiresB.specific_fn: <specific function> = specific_function %RequiresB, @RequiresB(%B.facet.83f) [symbolic]
// CHECK:STDOUT:   %impls.c88 = observe_impls %T.binding.as_type.0b3, %B.type [symbolic]
// CHECK:STDOUT:   %observe.decl.565 = observe_decl @observe.loc18 [concrete]
// CHECK:STDOUT:   %impls.2b3 = observe_impls %T.binding.as_type.0b3, %C.type [symbolic]
// CHECK:STDOUT:   %observe.decl.3f6 = observe_decl @observe.loc19 [concrete]
// CHECK:STDOUT:   %C.impl_witness.257: <witness> = impl_witness @T.binding.as_type.as.C.impl.%C.impl_witness_table, @T.binding.as_type.as.C.impl(%B.facet.83f) [symbolic]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %T.5a3, @C [symbolic]
// CHECK:STDOUT:   %C.facet.9cd: %C.type = facet_value %T.binding.as_type.0b3, (%C.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %D.impl_witness.555: <witness> = impl_witness @T.binding.as_type.as.D.impl.%D.impl_witness_table, @T.binding.as_type.as.D.impl(%C.facet.9cd) [symbolic]
// CHECK:STDOUT:   %.c21: require_specific_def_type = require_specific_def @T.binding.as_type.as.D.impl(%C.facet.9cd) [symbolic]
// CHECK:STDOUT:   %D.lookup_impl_witness: <witness> = lookup_impl_witness %T.5a3, @D [symbolic]
// CHECK:STDOUT:   %D.facet.514: %D.type = facet_value %T.binding.as_type.0b3, (%D.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %RequiresD.specific_fn: <specific function> = specific_function %RequiresD, @RequiresD(%D.facet.514) [symbolic]
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
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.B.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %A.type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T.5a3)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:     %.loc8_21: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %.loc8_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_14.2 (constants.%T.5a3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.C.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.035 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %B.type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.5ee)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:     %.loc9_21: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     %.loc9_18: type = splice_block %B.ref [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_14.1: %B.type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T.5ee)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.D.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7f1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %C.type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.94e)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:     %.loc10_21: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D.type]
// CHECK:STDOUT:     %.loc10_18: type = splice_block %C.ref [concrete = constants.%C.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_14.1: %C.type = symbolic_binding T, 0 [symbolic = %T.loc10_14.2 (constants.%T.94e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RequiresB.decl: %RequiresB.type = fn_decl @RequiresB [concrete = constants.%RequiresB] {
// CHECK:STDOUT:     %T.patt: %pattern_type.035 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresB.%pattern_type (%pattern_type.fae) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresB.%pattern_type (%pattern_type.fae) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_18: type = splice_block %B.ref [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_14.2: %B.type = symbolic_binding T, 0 [symbolic = %T.loc12_14.1 (constants.%T.5ee)]
// CHECK:STDOUT:     %x.param: @RequiresB.%T.binding.as_type (%T.binding.as_type.856) = value_param call_param0
// CHECK:STDOUT:     %.loc12_31.1: type = splice_block %.loc12_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)] {
// CHECK:STDOUT:       %T.ref: %B.type = name_ref T, %T.loc12_14.2 [symbolic = %T.loc12_14.1 (constants.%T.5ee)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:       %.loc12_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresB.%T.binding.as_type (%T.binding.as_type.856) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RequiresD.decl: %RequiresD.type = fn_decl @RequiresD [concrete = constants.%RequiresD] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f05 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresD.%pattern_type (%pattern_type.d60) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresD.%pattern_type (%pattern_type.d60) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_18: type = splice_block %D.ref [concrete = constants.%D.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_14.2: %D.type = symbolic_binding T, 0 [symbolic = %T.loc13_14.1 (constants.%T.233)]
// CHECK:STDOUT:     %x.param: @RequiresD.%T.binding.as_type (%T.binding.as_type.f5f) = value_param call_param0
// CHECK:STDOUT:     %.loc13_31.1: type = splice_block %.loc13_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.f5f)] {
// CHECK:STDOUT:       %T.ref: %D.type = name_ref T, %T.loc13_14.2 [symbolic = %T.loc13_14.1 (constants.%T.233)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.f5f)]
// CHECK:STDOUT:       %.loc13_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.f5f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresD.%T.binding.as_type (%T.binding.as_type.f5f) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresA.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresA.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc15_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = value_param call_param0
// CHECK:STDOUT:     %.loc15_24.1: type = splice_block %.loc15_24.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc15_14.2 [symbolic = %T.loc15_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:       %.loc15_24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b7c]
// CHECK:STDOUT:   %C.WithSelf.decl = interface_with_self_decl @C [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @D {
// CHECK:STDOUT:   %Self: %D.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b8b]
// CHECK:STDOUT:   %D.WithSelf.decl = interface_with_self_decl @D [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.B.impl(%T.loc8_14.1: %A.type) {
// CHECK:STDOUT:   %T.loc8_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_14.2 (constants.%T.5a3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:   %B.impl_witness.loc8_28.2: <witness> = impl_witness %B.impl_witness_table, @T.binding.as_type.as.B.impl(%T.loc8_14.2) [symbolic = %B.impl_witness.loc8_28.2 (constants.%B.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc8_21 as %B.ref {
// CHECK:STDOUT:     %B.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.B.impl [concrete]
// CHECK:STDOUT:     %B.impl_witness.loc8_28.1: <witness> = impl_witness %B.impl_witness_table, @T.binding.as_type.as.B.impl(constants.%T.5a3) [symbolic = %B.impl_witness.loc8_28.2 (constants.%B.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %B.impl_witness.loc8_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.C.impl(%T.loc9_14.1: %B.type) {
// CHECK:STDOUT:   %T.loc9_14.2: %B.type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T.5ee)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:   %C.impl_witness.loc9_28.2: <witness> = impl_witness %C.impl_witness_table, @T.binding.as_type.as.C.impl(%T.loc9_14.2) [symbolic = %C.impl_witness.loc9_28.2 (constants.%C.impl_witness.c1a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc9_21 as %C.ref {
// CHECK:STDOUT:     %C.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.C.impl [concrete]
// CHECK:STDOUT:     %C.impl_witness.loc9_28.1: <witness> = impl_witness %C.impl_witness_table, @T.binding.as_type.as.C.impl(constants.%T.5ee) [symbolic = %C.impl_witness.loc9_28.2 (constants.%C.impl_witness.c1a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %C.impl_witness.loc9_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.D.impl(%T.loc10_14.1: %C.type) {
// CHECK:STDOUT:   %T.loc10_14.2: %C.type = symbolic_binding T, 0 [symbolic = %T.loc10_14.2 (constants.%T.94e)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc10_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:   %D.impl_witness.loc10_28.2: <witness> = impl_witness %D.impl_witness_table, @T.binding.as_type.as.D.impl(%T.loc10_14.2) [symbolic = %D.impl_witness.loc10_28.2 (constants.%D.impl_witness.c16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc10_21 as %D.ref {
// CHECK:STDOUT:     %D.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.D.impl [concrete]
// CHECK:STDOUT:     %D.impl_witness.loc10_28.1: <witness> = impl_witness %D.impl_witness_table, @T.binding.as_type.as.D.impl(constants.%T.94e) [symbolic = %D.impl_witness.loc10_28.2 (constants.%D.impl_witness.c16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %D.impl_witness.loc10_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresB(%T.loc12_14.2: %B.type) {
// CHECK:STDOUT:   %T.loc12_14.1: %B.type = symbolic_binding T, 0 [symbolic = %T.loc12_14.1 (constants.%T.5ee)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc12_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.fae)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.f6b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresB.%T.binding.as_type (%T.binding.as_type.856)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresD(%T.loc13_14.2: %D.type) {
// CHECK:STDOUT:   %T.loc13_14.1: %D.type = symbolic_binding T, 0 [symbolic = %T.loc13_14.1 (constants.%T.233)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc13_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.f5f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.d60)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.ee0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresD.%T.binding.as_type (%T.binding.as_type.f5f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc15_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc15_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc15_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc15_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.057)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1a7)]
// CHECK:STDOUT:   %.loc16_14.3: require_specific_def_type = require_specific_def @T.binding.as_type.as.B.impl(%T.loc15_14.1) [symbolic = %.loc16_14.3 (constants.%.7ae)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc15_14.1, @B [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness)]
// CHECK:STDOUT:   %B.facet.loc16_14.3: %B.type = facet_value %T.binding.as_type, (%B.lookup_impl_witness) [symbolic = %B.facet.loc16_14.3 (constants.%B.facet.83f)]
// CHECK:STDOUT:   %RequiresB.specific_fn.loc16_3.2: <specific function> = specific_function constants.%RequiresB, @RequiresB(%B.facet.loc16_14.3) [symbolic = %RequiresB.specific_fn.loc16_3.2 (constants.%RequiresB.specific_fn)]
// CHECK:STDOUT:   %impls.loc18 = observe_impls %T.binding.as_type, constants.%B.type [symbolic = %impls.loc18 (constants.%impls.c88)]
// CHECK:STDOUT:   %impls.loc19 = observe_impls %T.binding.as_type, constants.%C.type [symbolic = %impls.loc19 (constants.%impls.2b3)]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc15_14.1, @C [symbolic = %C.lookup_impl_witness (constants.%C.lookup_impl_witness)]
// CHECK:STDOUT:   %C.facet: %C.type = facet_value %T.binding.as_type, (%C.lookup_impl_witness) [symbolic = %C.facet (constants.%C.facet.9cd)]
// CHECK:STDOUT:   %.loc20_14.3: require_specific_def_type = require_specific_def @T.binding.as_type.as.D.impl(%C.facet) [symbolic = %.loc20_14.3 (constants.%.c21)]
// CHECK:STDOUT:   %D.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc15_14.1, @D [symbolic = %D.lookup_impl_witness (constants.%D.lookup_impl_witness)]
// CHECK:STDOUT:   %D.facet.loc20_14.3: %D.type = facet_value %T.binding.as_type, (%D.lookup_impl_witness) [symbolic = %D.facet.loc20_14.3 (constants.%D.facet.514)]
// CHECK:STDOUT:   %RequiresD.specific_fn.loc20_3.2: <specific function> = specific_function constants.%RequiresD, @RequiresD(%D.facet.loc20_14.3) [symbolic = %RequiresD.specific_fn.loc20_3.2 (constants.%RequiresD.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RequiresB.ref: %RequiresB.type = name_ref RequiresB, file.%RequiresB.decl [concrete = constants.%RequiresB]
// CHECK:STDOUT:     %x.ref.loc16: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = name_ref x, %x
// CHECK:STDOUT:     %B.facet.loc16_14.1: %B.type = facet_value constants.%T.binding.as_type.0b3, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc16_14.3 (constants.%B.facet.83f)]
// CHECK:STDOUT:     %.loc16_14.1: %B.type = converted constants.%T.binding.as_type.0b3, %B.facet.loc16_14.1 [symbolic = %B.facet.loc16_14.3 (constants.%B.facet.83f)]
// CHECK:STDOUT:     %B.facet.loc16_14.2: %B.type = facet_value constants.%T.binding.as_type.0b3, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc16_14.3 (constants.%B.facet.83f)]
// CHECK:STDOUT:     %.loc16_14.2: %B.type = converted constants.%T.binding.as_type.0b3, %B.facet.loc16_14.2 [symbolic = %B.facet.loc16_14.3 (constants.%B.facet.83f)]
// CHECK:STDOUT:     %RequiresB.specific_fn.loc16_3.1: <specific function> = specific_function %RequiresB.ref, @RequiresB(constants.%B.facet.83f) [symbolic = %RequiresB.specific_fn.loc16_3.2 (constants.%RequiresB.specific_fn)]
// CHECK:STDOUT:     %RequiresB.call: init %empty_tuple.type = call %RequiresB.specific_fn.loc16_3.1(%x.ref.loc16)
// CHECK:STDOUT:     %observe.decl.loc18 = observe_decl @observe.loc18 [concrete = constants.%observe.decl.565]
// CHECK:STDOUT:     %observe.decl.loc19 = observe_decl @observe.loc19 [concrete = constants.%observe.decl.3f6]
// CHECK:STDOUT:     %RequiresD.ref: %RequiresD.type = name_ref RequiresD, file.%RequiresD.decl [concrete = constants.%RequiresD]
// CHECK:STDOUT:     %x.ref.loc20: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = name_ref x, %x
// CHECK:STDOUT:     %D.facet.loc20_14.1: %D.type = facet_value constants.%T.binding.as_type.0b3, (constants.%D.lookup_impl_witness) [symbolic = %D.facet.loc20_14.3 (constants.%D.facet.514)]
// CHECK:STDOUT:     %.loc20_14.1: %D.type = converted constants.%T.binding.as_type.0b3, %D.facet.loc20_14.1 [symbolic = %D.facet.loc20_14.3 (constants.%D.facet.514)]
// CHECK:STDOUT:     %D.facet.loc20_14.2: %D.type = facet_value constants.%T.binding.as_type.0b3, (constants.%D.lookup_impl_witness) [symbolic = %D.facet.loc20_14.3 (constants.%D.facet.514)]
// CHECK:STDOUT:     %.loc20_14.2: %D.type = converted constants.%T.binding.as_type.0b3, %D.facet.loc20_14.2 [symbolic = %D.facet.loc20_14.3 (constants.%D.facet.514)]
// CHECK:STDOUT:     %RequiresD.specific_fn.loc20_3.1: <specific function> = specific_function %RequiresD.ref, @RequiresD(constants.%D.facet.514) [symbolic = %RequiresD.specific_fn.loc20_3.2 (constants.%RequiresD.specific_fn)]
// CHECK:STDOUT:     %.loc20_13: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = as_compatible %x.ref.loc20
// CHECK:STDOUT:     %RequiresD.call: init %empty_tuple.type = call %RequiresD.specific_fn.loc20_3.1(%.loc20_13)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self.b7c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.WithSelf(constants.%Self.b8b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.B.impl(constants.%T.5a3) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T.5a3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %B.impl_witness.loc8_28.2 => constants.%B.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%B.facet.234) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.C.impl(constants.%T.5ee) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.5ee
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.856
// CHECK:STDOUT:   %C.impl_witness.loc9_28.2 => constants.%C.impl_witness.c1a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%C.facet.50a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.D.impl(constants.%T.94e) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T.94e
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.d92
// CHECK:STDOUT:   %D.impl_witness.loc10_28.2 => constants.%D.impl_witness.c16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.WithSelf(constants.%D.facet.d95) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresB(constants.%T.5ee) {
// CHECK:STDOUT:   %T.loc12_14.1 => constants.%T.5ee
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.856
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresD(constants.%T.233) {
// CHECK:STDOUT:   %T.loc13_14.1 => constants.%T.233
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.f5f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T.5a3) {
// CHECK:STDOUT:   %T.loc15_14.1 => constants.%T.5a3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresB(constants.%B.facet.83f) {
// CHECK:STDOUT:   %T.loc12_14.1 => constants.%B.facet.83f
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1a7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.C.impl(constants.%B.facet.83f) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%B.facet.83f
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %C.impl_witness.loc9_28.2 => constants.%C.impl_witness.257
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.D.impl(constants.%C.facet.9cd) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%C.facet.9cd
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %D.impl_witness.loc10_28.2 => constants.%D.impl_witness.555
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresD(constants.%D.facet.514) {
// CHECK:STDOUT:   %T.loc13_14.1 => constants.%D.facet.514
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1a7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_functions_in_outer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1a7: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.Self.d66: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.14b: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.binding.as_type.e57: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %U.binding.as_type.e57 [symbolic]
// CHECK:STDOUT:   %observe.decl = observe_decl @observe [concrete]
// CHECK:STDOUT:   %pattern_type.2d9: type = pattern_type %U.binding.as_type.e57 [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF [concrete]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a2d: <witness> = require_complete_type %U.binding.as_type.e57 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.89d: %A.type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %as_type: type = facet_access_type %U.binding.as_type.89d [symbolic]
// CHECK:STDOUT:   %pattern_type.562: type = pattern_type %as_type [symbolic]
// CHECK:STDOUT:   %RequiresA.specific_fn: <specific function> = specific_function %RequiresA, @RequiresA(%U.binding.as_type.89d) [symbolic]
// CHECK:STDOUT:   %require_complete.d94: <witness> = require_complete_type %as_type [symbolic]
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
// CHECK:STDOUT:     .RequiresA = %RequiresA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresA.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresA.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_31.1: type = splice_block %.loc6_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc6_14.2 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.2: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %.Self.ref: %B.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc7_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc6_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc6_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc6_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.057)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1a7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: %A.type, %U.loc7_13.2: %B_where.type) {
// CHECK:STDOUT:   %T.loc7_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc7_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc7_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %U.binding.as_type [symbolic = %eq (constants.%eq)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe [concrete = constants.%observe.decl]
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %x.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%U.binding.as_type.loc10 (%U.binding.as_type.e57) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %.loc10_16.2 [symbolic = %U.binding.as_type.loc10 (constants.%U.binding.as_type.e57)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc7_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type.loc10 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:         %.loc10_16.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type.loc10 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%U.binding.as_type.loc10 (%U.binding.as_type.e57) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc7_6.2: %A.type, @F.%U.loc7_13.2: %B_where.type) {
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type.loc10: type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type.loc10 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type.loc10 [symbolic = %pattern_type (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %U.binding.as_type.loc10 [symbolic = %require_complete.loc10 (constants.%require_complete.a2d)]
// CHECK:STDOUT:   %U.binding.as_type.loc11: %A.type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type.loc11 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:   %RequiresA.specific_fn.loc11_5.2: <specific function> = specific_function constants.%RequiresA, @RequiresA(%U.binding.as_type.loc11) [symbolic = %RequiresA.specific_fn.loc11_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %U.binding.as_type.loc11 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %as_type [symbolic = %require_complete.loc11 (constants.%require_complete.d94)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%U.binding.as_type.loc10 (%U.binding.as_type.e57)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @InnerF.%U.binding.as_type.loc10 (%U.binding.as_type.e57) = name_ref x, %x
// CHECK:STDOUT:     %.loc11_16.1: %A.type = as_compatible constants.%U.binding.as_type.e57 [symbolic = %U.binding.as_type.loc11 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:     %.loc11_16.2: %A.type = as_compatible constants.%U.binding.as_type.e57 [symbolic = %U.binding.as_type.loc11 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:     %RequiresA.specific_fn.loc11_5.1: <specific function> = specific_function %RequiresA.ref, @RequiresA(constants.%U.binding.as_type.89d) [symbolic = %RequiresA.specific_fn.loc11_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:     %.loc11_15: @InnerF.%as_type (%as_type) = as_compatible %x.ref
// CHECK:STDOUT:     %RequiresA.call: init %empty_tuple.type = call %RequiresA.specific_fn.loc11_5.1(%.loc11_15)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.loc6_14.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc7_13.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.binding.as_type.loc10 => constants.%U.binding.as_type.e57
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%U.binding.as_type.89d) {
// CHECK:STDOUT:   %T.loc6_14.1 => constants.%U.binding.as_type.89d
// CHECK:STDOUT:   %T.binding.as_type => constants.%as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.562
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_functions_in_inner.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1a7: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.Self.d66: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.14b: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.binding.as_type.e57: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.2d9: type = pattern_type %U.binding.as_type.e57 [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF [concrete]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a2d: <witness> = require_complete_type %U.binding.as_type.e57 [symbolic]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %U.binding.as_type.e57 [symbolic]
// CHECK:STDOUT:   %observe.decl = observe_decl @observe [concrete]
// CHECK:STDOUT:   %U.binding.as_type.89d: %A.type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %as_type: type = facet_access_type %U.binding.as_type.89d [symbolic]
// CHECK:STDOUT:   %pattern_type.562: type = pattern_type %as_type [symbolic]
// CHECK:STDOUT:   %RequiresA.specific_fn: <specific function> = specific_function %RequiresA, @RequiresA(%U.binding.as_type.89d) [symbolic]
// CHECK:STDOUT:   %require_complete.d94: <witness> = require_complete_type %as_type [symbolic]
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
// CHECK:STDOUT:     .RequiresA = %RequiresA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresA.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresA.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_31.1: type = splice_block %.loc6_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc6_14.2 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.2: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %.Self.ref: %B.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc7_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc6_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc6_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc6_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.057)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1a7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: %A.type, %U.loc7_13.2: %B_where.type) {
// CHECK:STDOUT:   %T.loc7_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc7_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %x.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%U.binding.as_type.loc9 (%U.binding.as_type.e57) = value_param call_param0
// CHECK:STDOUT:       %.loc9_16.1: type = splice_block %.loc9_16.2 [symbolic = %U.binding.as_type.loc9 (constants.%U.binding.as_type.e57)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc7_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type.loc9 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:         %.loc9_16.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type.loc9 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%U.binding.as_type.loc9 (%U.binding.as_type.e57) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc7_6.2: %A.type, @F.%U.loc7_13.2: %B_where.type) {
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type.loc9: type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type.loc9 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type.loc9 [symbolic = %pattern_type (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %U.binding.as_type.loc9 [symbolic = %require_complete.loc9 (constants.%require_complete.a2d)]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %U.binding.as_type.loc9 [symbolic = %eq (constants.%eq)]
// CHECK:STDOUT:   %U.binding.as_type.loc12: %A.type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type.loc12 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:   %RequiresA.specific_fn.loc12_5.2: <specific function> = specific_function constants.%RequiresA, @RequiresA(%U.binding.as_type.loc12) [symbolic = %RequiresA.specific_fn.loc12_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %U.binding.as_type.loc12 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type %as_type [symbolic = %require_complete.loc12 (constants.%require_complete.d94)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%U.binding.as_type.loc9 (%U.binding.as_type.e57)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe [concrete = constants.%observe.decl]
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @InnerF.%U.binding.as_type.loc9 (%U.binding.as_type.e57) = name_ref x, %x
// CHECK:STDOUT:     %.loc12_16.1: %A.type = as_compatible constants.%U.binding.as_type.e57 [symbolic = %U.binding.as_type.loc12 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:     %.loc12_16.2: %A.type = as_compatible constants.%U.binding.as_type.e57 [symbolic = %U.binding.as_type.loc12 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:     %RequiresA.specific_fn.loc12_5.1: <specific function> = specific_function %RequiresA.ref, @RequiresA(constants.%U.binding.as_type.89d) [symbolic = %RequiresA.specific_fn.loc12_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:     %.loc12_15: @InnerF.%as_type (%as_type) = as_compatible %x.ref
// CHECK:STDOUT:     %RequiresA.call: init %empty_tuple.type = call %RequiresA.specific_fn.loc12_5.1(%.loc12_15)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.loc6_14.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc7_13.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.binding.as_type.loc9 => constants.%U.binding.as_type.e57
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%U.binding.as_type.89d) {
// CHECK:STDOUT:   %T.loc6_14.1 => constants.%U.binding.as_type.89d
// CHECK:STDOUT:   %T.binding.as_type => constants.%as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.562
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested_functions_lexical_order.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1a7: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.Self.d66: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.14b: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.2d9: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF [concrete]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a2d: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %observe.decl = observe_decl @observe [concrete]
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
// CHECK:STDOUT:     .RequiresA = %RequiresA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresA.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresA.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_31.1: type = splice_block %.loc6_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc6_14.2 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.2: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %.Self.ref: %B.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc7_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc6_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc6_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc6_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.057)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1a7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: %A.type, %U.loc7_13.2: %B_where.type) {
// CHECK:STDOUT:   %T.loc7_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc7_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc7_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %U.binding.as_type [symbolic = %eq (constants.%eq)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %x.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_16.1: type = splice_block %.loc9_16.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc7_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:         %.loc9_16.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %observe.decl = observe_decl @observe [concrete = constants.%observe.decl]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc7_6.2: %A.type, @F.%U.loc7_13.2: %B_where.type) {
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.a2d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @InnerF.%U.binding.as_type (%U.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.loc6_14.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc7_13.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested_functions_leak_upward.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1a7: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.Self.d66: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.14b: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.2d9: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a2d: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF [concrete]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [concrete]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %observe.decl = observe_decl @observe [concrete]
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
// CHECK:STDOUT:     .RequiresA = %RequiresA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresA.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresA.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_31.1: type = splice_block %.loc6_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc6_14.2 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.2d9) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.2d9) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.2: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %.Self.ref: %B.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc7_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:     %x.param: @F.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_40.1: type = splice_block %.loc7_40.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc7_13.2 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc7_40.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%U.binding.as_type (%U.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc6_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc6_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc6_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.057)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1a7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: %A.type, %U.loc7_13.2: %B_where.type) {
// CHECK:STDOUT:   %T.loc7_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc7_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc7_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.a2d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %x.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_23.1: type = splice_block %.loc9_23.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc7_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:         %.loc9_23.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @F.%U.binding.as_type (%U.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc7_6.2: %A.type, @F.%U.loc7_13.2: %B_where.type) {
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.a2d)]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %U.binding.as_type [symbolic = %eq (constants.%eq)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe [concrete = constants.%observe.decl]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.loc6_14.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc7_13.1 => constants.%U
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_observed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1a7: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.Self.d66: %B.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B_where.type: type = facet_type <@B where TODO> [concrete]
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.14b: type = pattern_type %B_where.type [concrete]
// CHECK:STDOUT:   %V: %B.type = symbolic_binding V, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.035: type = pattern_type %B.type [concrete]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.2d9: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a2d: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 2, %V [symbolic]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %V.binding.as_type [symbolic]
// CHECK:STDOUT:   %observe.decl = observe_decl @observe [concrete]
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
// CHECK:STDOUT:     .RequiresA = %RequiresA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresA.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresA.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_31.1: type = splice_block %.loc6_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc6_14.2 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.035 = symbolic_binding_pattern V, 2 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.2d9) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.2d9) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.4: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref.loc7_17: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.3: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %.Self.ref: %B.type = name_ref .Self, %.Self.3 [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc7_19.2: type = where_expr %.Self.3 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:     %.loc7_41: type = splice_block %B.ref.loc7_41 [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref.loc7_41: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc7_37.2: %B.type = symbolic_binding V, 2 [symbolic = %V.loc7_37.1 (constants.%V)]
// CHECK:STDOUT:     %x.param: @F.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_47.1: type = splice_block %.loc7_47.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc7_13.2 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc7_47.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%U.binding.as_type (%U.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc6_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc6_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_14.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc6_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.057)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1a7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: %A.type, %U.loc7_13.2: %B_where.type, %V.loc7_37.2: %B.type) {
// CHECK:STDOUT:   %T.loc7_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc7_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc7_37.1: %B.type = symbolic_binding V, 2 [symbolic = %V.loc7_37.1 (constants.%V)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc7_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.a2d)]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 2, %V.loc7_37.1 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %V.binding.as_type [symbolic = %eq (constants.%eq)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe [concrete = constants.%observe.decl]
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @F.%U.binding.as_type (%U.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.loc6_14.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U, constants.%V) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc7_13.1 => constants.%U
// CHECK:STDOUT:   %V.loc7_37.1 => constants.%V
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_invalid_equivalent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %B.type: type = facet_type <@B> [concrete]
// CHECK:STDOUT:   %Self.d0b: %B.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %TakesA.type: type = fn_type @TakesA [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakesA: %TakesA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1a7: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %U: %B.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.035: type = pattern_type %B.type [concrete]
// CHECK:STDOUT:   %U.binding.as_type.121: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.72f: type = pattern_type %U.binding.as_type.121 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.f78: <witness> = require_complete_type %U.binding.as_type.121 [symbolic]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %U.binding.as_type.121 [symbolic]
// CHECK:STDOUT:   %observe.decl = observe_decl @observe [concrete]
// CHECK:STDOUT:   %U.binding.as_type.ddb: %A.type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %as_type: type = facet_access_type %U.binding.as_type.ddb [symbolic]
// CHECK:STDOUT:   %pattern_type.97b: type = pattern_type %as_type [symbolic]
// CHECK:STDOUT:   %TakesA.specific_fn: <specific function> = specific_function %TakesA, @TakesA(%U.binding.as_type.ddb) [symbolic]
// CHECK:STDOUT:   %require_complete.a22: <witness> = require_complete_type %as_type [symbolic]
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
// CHECK:STDOUT:     .TakesA = %TakesA.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %TakesA.decl: %TakesA.type = fn_decl @TakesA [concrete = constants.%TakesA] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @TakesA.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @TakesA.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_15: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_11.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_11.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TakesA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_28.1: type = splice_block %.loc6_28.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc6_11.2 [symbolic = %T.loc6_11.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_28.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TakesA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.035 = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.72f) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.72f) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_17: type = splice_block %B.ref [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_13.2: %B.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:     %x.param: @F.%U.binding.as_type.loc7 (%U.binding.as_type.121) = value_param call_param0
// CHECK:STDOUT:     %.loc7_23.1: type = splice_block %.loc7_23.2 [symbolic = %U.binding.as_type.loc7 (constants.%U.binding.as_type.121)] {
// CHECK:STDOUT:       %U.ref: %B.type = name_ref U, %U.loc7_13.2 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type.loc7 (constants.%U.binding.as_type.121)]
// CHECK:STDOUT:       %.loc7_23.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type.loc7 (constants.%U.binding.as_type.121)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%U.binding.as_type.loc7 (%U.binding.as_type.121) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @B {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d0b]
// CHECK:STDOUT:   %B.WithSelf.decl = interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesA(%T.loc6_11.2: %A.type) {
// CHECK:STDOUT:   %T.loc6_11.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_11.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc6_11.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.057)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1a7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TakesA.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: %A.type, %U.loc7_13.2: %B.type) {
// CHECK:STDOUT:   %T.loc7_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc7_13.1: %B.type = symbolic_binding U, 1 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type.loc7: type = symbolic_binding_type U, 1, %U.loc7_13.1 [symbolic = %U.binding.as_type.loc7 (constants.%U.binding.as_type.121)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type.loc7 [symbolic = %pattern_type (constants.%pattern_type.72f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %U.binding.as_type.loc7 [symbolic = %require_complete.loc7 (constants.%require_complete.f78)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %eq = observe_equivalent %T.binding.as_type, %U.binding.as_type.loc7 [symbolic = %eq (constants.%eq)]
// CHECK:STDOUT:   %U.binding.as_type.loc9: %A.type = symbolic_binding_type U, 1, %U.loc7_13.1 [symbolic = %U.binding.as_type.loc9 (constants.%U.binding.as_type.ddb)]
// CHECK:STDOUT:   %TakesA.specific_fn.loc9_3.2: <specific function> = specific_function constants.%TakesA, @TakesA(%U.binding.as_type.loc9) [symbolic = %TakesA.specific_fn.loc9_3.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %U.binding.as_type.loc9 [symbolic = %as_type (constants.%as_type)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %as_type [symbolic = %require_complete.loc9 (constants.%require_complete.a22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%U.binding.as_type.loc7 (%U.binding.as_type.121)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe [concrete = constants.%observe.decl]
// CHECK:STDOUT:     %TakesA.ref: %TakesA.type = name_ref TakesA, file.%TakesA.decl [concrete = constants.%TakesA]
// CHECK:STDOUT:     %x.ref: @F.%U.binding.as_type.loc7 (%U.binding.as_type.121) = name_ref x, %x
// CHECK:STDOUT:     %.loc9_11.1: %A.type = as_compatible constants.%U.binding.as_type.121 [symbolic = %U.binding.as_type.loc9 (constants.%U.binding.as_type.ddb)]
// CHECK:STDOUT:     %.loc9_11.2: %A.type = as_compatible constants.%U.binding.as_type.121 [symbolic = %U.binding.as_type.loc9 (constants.%U.binding.as_type.ddb)]
// CHECK:STDOUT:     %TakesA.specific_fn.loc9_3.1: <specific function> = specific_function %TakesA.ref, @TakesA(constants.%U.binding.as_type.ddb) [symbolic = %TakesA.specific_fn.loc9_3.2 (constants.%TakesA.specific_fn)]
// CHECK:STDOUT:     %.loc9_10: @F.%as_type (%as_type) = as_compatible %x.ref
// CHECK:STDOUT:     %TakesA.call: init %empty_tuple.type = call %TakesA.specific_fn.loc9_3.1(%.loc9_10)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%T) {
// CHECK:STDOUT:   %T.loc6_11.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc7_13.1 => constants.%U
// CHECK:STDOUT:   %U.binding.as_type.loc7 => constants.%U.binding.as_type.121
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.72f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesA(constants.%U.binding.as_type.ddb) {
// CHECK:STDOUT:   %T.loc6_11.1 => constants.%U.binding.as_type.ddb
// CHECK:STDOUT:   %T.binding.as_type => constants.%as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.97b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.a22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_scope_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_13.1: type = splice_block %.loc3_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc3_23.1: type = splice_block %.loc3_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc3_19.2: type = symbolic_binding U, 1 [symbolic = %U.loc3_19.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc3_9.2: type, %U.loc3_19.2: type) {
// CHECK:STDOUT:   %T.loc3_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_9.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc3_19.1: type = symbolic_binding U, 1 [symbolic = %U.loc3_19.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
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
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%T
// CHECK:STDOUT:   %U.loc3_19.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_scope_named_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %C.type.016: type = generic_named_constaint_type @C [concrete]
// CHECK:STDOUT:   %empty_struct: %C.type.016 = struct_value () [concrete]
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
// CHECK:STDOUT:   %C.decl: %C.type.016 = constraint_decl @C [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_18.1: type = splice_block %.loc3_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T)]
// CHECK:STDOUT:     %.loc3_28.1: type = splice_block %.loc3_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc3_24.2: type = symbolic_binding U, 1 [symbolic = %U.loc3_24.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @C(%T.loc3_14.2: type, %U.loc3_24.2: type) {
// CHECK:STDOUT:   %T.loc3_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc3_24.1: type = symbolic_binding U, 1 [symbolic = %U.loc3_24.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.type: type = facet_type <@C, @C(%T.loc3_14.1, %U.loc3_24.1)> [symbolic = %C.type (constants.%C.type.d68)]
// CHECK:STDOUT:   %Self.loc3_34.2: @C.%C.type (%C.type.d68) = symbolic_binding Self, 2 [symbolic = %Self.loc3_34.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc3_34.1: @C.%C.type (%C.type.d68) = symbolic_binding Self, 2 [symbolic = %Self.loc3_34.2 (constants.%Self)]
// CHECK:STDOUT:     %C.WithSelf.decl = constraint_with_self_decl @C [concrete]
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
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%T
// CHECK:STDOUT:   %U.loc3_24.1 => constants.%U
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
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %impls = observe_impls %T.binding.as_type, %C [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
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
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: %A.type) {
// CHECK:STDOUT:   %T.loc6_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc6_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %impls = observe_impls %T.binding.as_type, constants.%C [symbolic = %impls (constants.%impls)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:

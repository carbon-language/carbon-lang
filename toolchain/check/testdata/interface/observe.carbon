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

// --- equivalent.carbon

library "[[@TEST_NAME]]";

interface P { fn InP[self: Self](); }
interface Q { fn InQ[self: Self](); }
interface R { fn InR[self: Self](); }

interface Transitive {
  let A:! P;
  let B:! Q where .Self == A;
  let C:! R where .Self == B;

  fn GetA[self: Self]() -> A;
  fn TakesC[self: Self](c: C);

  observe A == B == C;
}

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

// --- equivalent_impls.carbon

library "[[@TEST_NAME]]";

interface P { fn InP[self: Self](); }
interface Q { fn InQ[self: Self](); }
interface R { fn InR[self: Self](); }

interface Transitive {
  let A:! P;
  let B:! Q where .Self == A;
  let C:! R where .Self == B;

  fn GetA[self: Self]() -> A;
  fn TakesC[self: Self](c: C);

  observe A == B == C;
}

fn TakesPQR[U:! P & Q & R](unused u: U) {}

fn G[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  observe T.A == T.B impls Q;
  a.(Q.InQ)();

  observe T.A == T.C impls R;
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

// --- fail_wrong_scope.carbon

library "[[@TEST_NAME]]";

class C(T:! type, U:! type) {
  // CHECK:STDERR: fail_wrong_scope.carbon:[[@LINE+4]]:3: error: `observe` can only be used in an `interface` or `function` [ObserveInWrongScope]
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

// CHECK:STDOUT: --- equivalent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.d21: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.6c8: type = symbolic_binding_type Self, 0, %Self.d21 [symbolic]
// CHECK:STDOUT:   %pattern_type.2eb: type = pattern_type %Self.binding.as_type.6c8 [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P(%Self.d21) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %P.assoc_type: type = assoc_entity_type @P [concrete]
// CHECK:STDOUT:   %assoc0.4ec: %P.assoc_type = assoc_entity element0, @P.%P.WithSelf.InP.decl [concrete]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.f64: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.1b2: type = symbolic_binding_type Self, 0, %Self.f64 [symbolic]
// CHECK:STDOUT:   %pattern_type.992: type = pattern_type %Self.binding.as_type.1b2 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.377: type = fn_type @Q.WithSelf.InQ, @Q(%Self.f64) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.2bf: %Q.WithSelf.InQ.type.377 = struct_value () [symbolic]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.038: %Q.assoc_type = assoc_entity element0, @Q.%Q.WithSelf.InQ.decl [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.bea: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.296: type = symbolic_binding_type Self, 0, %Self.bea [symbolic]
// CHECK:STDOUT:   %pattern_type.351: type = pattern_type %Self.binding.as_type.296 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type.f8a: type = fn_type @R.WithSelf.InR, @R(%Self.bea) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.b79: %R.WithSelf.InR.type.f8a = struct_value () [symbolic]
// CHECK:STDOUT:   %R.assoc_type: type = assoc_entity_type @R [concrete]
// CHECK:STDOUT:   %assoc0.dce: %R.assoc_type = assoc_entity element0, @R.%R.WithSelf.InR.decl [concrete]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.e2a: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.be3: %Transitive.assoc_type = assoc_entity element0, @Transitive.%A [concrete]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.804: <witness> = lookup_impl_witness %Self.e2a, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.9bd: %P.type = impl_witness_access %Transitive.lookup_impl_witness.804, element0 [symbolic]
// CHECK:STDOUT:   %Q_where.type: type = facet_type <@Q where TODO> [concrete]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.%B [concrete]
// CHECK:STDOUT:   %impl.elem1.4e4: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.804, element1 [symbolic]
// CHECK:STDOUT:   %R_where.type: type = facet_type <@R where TODO> [concrete]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.%C [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.e5b: type = symbolic_binding_type Self, 0, %Self.e2a [symbolic]
// CHECK:STDOUT:   %pattern_type.b0d: type = pattern_type %Self.binding.as_type.e5b [symbolic]
// CHECK:STDOUT:   %as_type.0ea: type = facet_access_type %impl.elem0.9bd [symbolic]
// CHECK:STDOUT:   %.7a7: Core.Form = init_form %as_type.0ea, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.bff: type = pattern_type %as_type.0ea [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.bde: type = fn_type @Transitive.WithSelf.GetA, @Transitive(%Self.e2a) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.e53: %Transitive.WithSelf.GetA.type.bde = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, @Transitive.%Transitive.WithSelf.GetA.decl [concrete]
// CHECK:STDOUT:   %impl.elem2.8f7: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.804, element2 [symbolic]
// CHECK:STDOUT:   %as_type.ecc: type = facet_access_type %impl.elem2.8f7 [symbolic]
// CHECK:STDOUT:   %pattern_type.364: type = pattern_type %as_type.ecc [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.1ad: type = fn_type @Transitive.WithSelf.TakesC, @Transitive(%Self.e2a) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.9f1: %Transitive.WithSelf.TakesC.type.1ad = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, @Transitive.%Transitive.WithSelf.TakesC.decl [concrete]
// CHECK:STDOUT:   %as_type.79f: type = facet_access_type %impl.elem1.4e4 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %Transitive.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %Transitive.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.993: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1c0: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.6d8: <witness> = lookup_impl_witness %T, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.b09: %P.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.32a: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.dc6: type = fn_type @Transitive.WithSelf.GetA, @Transitive(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.69d: %Transitive.WithSelf.GetA.type.dc6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.eda: type = fn_type @Transitive.WithSelf.TakesC, @Transitive(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.67d: %Transitive.WithSelf.TakesC.type.eda = struct_value () [symbolic]
// CHECK:STDOUT:   %as_type.846: type = facet_access_type %impl.elem1.32a [symbolic]
// CHECK:STDOUT:   %as_type.e31: type = facet_access_type %impl.elem0.b09 [symbolic]
// CHECK:STDOUT:   %impl.elem2.458: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element2 [symbolic]
// CHECK:STDOUT:   %as_type.c65: type = facet_access_type %impl.elem2.458 [symbolic]
// CHECK:STDOUT:   %require_complete.b46: <witness> = require_complete_type %as_type.e31 [symbolic]
// CHECK:STDOUT:   %pattern_type.05e: type = pattern_type %as_type.e31 [symbolic]
// CHECK:STDOUT:   %.c0a: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type.dc6, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.c0a = impl_witness_access %Transitive.lookup_impl_witness.6d8, element3 [symbolic]
// CHECK:STDOUT:   %.1b9: Core.Form = init_form %as_type.e31, call_param1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c79: <specific function> = specific_impl_function %impl.elem3, @Transitive.WithSelf.GetA(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.8ba: <witness> = require_complete_type %as_type.846 [symbolic]
// CHECK:STDOUT:   %pattern_type.a21: type = pattern_type %as_type.846 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.031: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.738: %ImplicitAs.type.031 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b3a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%Dest, %Self.738) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.1de: %ImplicitAs.WithSelf.Convert.type.b3a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d5a: type = facet_type <@ImplicitAs, @ImplicitAs(%as_type.846)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.b4b: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%as_type.846) [symbolic]
// CHECK:STDOUT:   %assoc0.242: %ImplicitAs.assoc_type.b4b = assoc_entity element0, imports.%Core.import_ref.201 [symbolic]
// CHECK:STDOUT:   %require_complete.abb: <witness> = require_complete_type %ImplicitAs.type.d5a [symbolic]
// CHECK:STDOUT:   %assoc0.843: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.cc1 [symbolic]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.223: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.2d0: %As.type.223 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.assoc_type.752: type = assoc_entity_type @As, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.bb7: type = fn_type @As.WithSelf.Convert, @As(%Dest, %Self.2d0) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.923: %As.WithSelf.Convert.type.bb7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %as_type.b6a: %Q_where.type = facet_access_type %impl.elem0.b09 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.facet.746: %ImplicitAs.type.d5a = facet_value %as_type.e31, (%as_type.b6a) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e4d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%as_type.846, %ImplicitAs.facet.746) [symbolic]
// CHECK:STDOUT:   %.db6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e4d, %ImplicitAs.facet.746 [symbolic]
// CHECK:STDOUT:   %impl.elem0.9e8: %.db6 = impl_witness_access %as_type.b6a, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.060: <specific function> = specific_impl_function %impl.elem0.9e8, @ImplicitAs.WithSelf.Convert(%as_type.846, %ImplicitAs.facet.746) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.240: type = fn_type @Q.WithSelf.InQ, @Q(%impl.elem1.32a) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.5ef: %Q.WithSelf.InQ.type.240 = struct_value () [symbolic]
// CHECK:STDOUT:   %Q.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.32a, @Q [symbolic]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.846, (%Q.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.a66: type = fn_type @Q.WithSelf.InQ, @Q(%Q.facet) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.bcd: %Q.WithSelf.InQ.type.a66 = struct_value () [symbolic]
// CHECK:STDOUT:   %.803: type = fn_type_with_self_type %Q.WithSelf.InQ.type.a66, %Q.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.9f0: %.803 = impl_witness_access %Q.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.8e9: <specific function> = specific_impl_function %impl.elem0.9f0, @Q.WithSelf.InQ(%Q.facet) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type.fbb: type = fn_type @R.WithSelf.InR, @R(%impl.elem2.458) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.b14: %R.WithSelf.InR.type.fbb = struct_value () [symbolic]
// CHECK:STDOUT:   %R.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem2.458, @R [symbolic]
// CHECK:STDOUT:   %R.facet: %R.type = facet_value %as_type.c65, (%R.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type.63c: type = fn_type @R.WithSelf.InR, @R(%R.facet) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.1a9: %R.WithSelf.InR.type.63c = struct_value () [symbolic]
// CHECK:STDOUT:   %.e11: type = fn_type_with_self_type %R.WithSelf.InR.type.63c, %R.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.df8: %.e11 = impl_witness_access %R.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7db: type = pattern_type %as_type.c65 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2c6: <specific function> = specific_impl_function %impl.elem0.df8, @R.WithSelf.InR(%R.facet) [symbolic]
// CHECK:STDOUT:   %require_complete.60d: <witness> = require_complete_type %as_type.c65 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.ac2: type = facet_type <@ImplicitAs, @ImplicitAs(%as_type.c65)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.eeb: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%as_type.c65) [symbolic]
// CHECK:STDOUT:   %assoc0.14e: %ImplicitAs.assoc_type.eeb = assoc_entity element0, imports.%Core.import_ref.201 [symbolic]
// CHECK:STDOUT:   %require_complete.49e: <witness> = require_complete_type %ImplicitAs.type.ac2 [symbolic]
// CHECK:STDOUT:   %as_type.3e7: %R_where.type = facet_access_type %impl.elem0.b09 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.facet.21e: %ImplicitAs.type.ac2 = facet_value %as_type.e31, (%as_type.3e7) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c6b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%as_type.c65, %ImplicitAs.facet.21e) [symbolic]
// CHECK:STDOUT:   %.92e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c6b, %ImplicitAs.facet.21e [symbolic]
// CHECK:STDOUT:   %impl.elem0.c85: %.92e = impl_witness_access %as_type.3e7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.80c: <specific function> = specific_impl_function %impl.elem0.c85, @ImplicitAs.WithSelf.Convert(%as_type.c65, %ImplicitAs.facet.21e) [symbolic]
// CHECK:STDOUT:   %As.type.ae5: type = facet_type <@As, @As(%as_type.c65)> [symbolic]
// CHECK:STDOUT:   %As.assoc_type.69a: type = assoc_entity_type @As, @As(%as_type.c65) [symbolic]
// CHECK:STDOUT:   %assoc0.f5c: %As.assoc_type.69a = assoc_entity element0, imports.%Core.import_ref.0ec [symbolic]
// CHECK:STDOUT:   %require_complete.ea4: <witness> = require_complete_type %As.type.ae5 [symbolic]
// CHECK:STDOUT:   %assoc0.2f9: %As.assoc_type.752 = assoc_entity element0, imports.%Core.import_ref.4c6 [symbolic]
// CHECK:STDOUT:   %As.facet: %As.type.ae5 = facet_value %as_type.e31, (%as_type.3e7) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.cfa: type = fn_type @As.WithSelf.Convert, @As(%as_type.c65, %As.facet) [symbolic]
// CHECK:STDOUT:   %.d45: type = fn_type_with_self_type %As.WithSelf.Convert.type.cfa, %As.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.594: %.d45 = impl_witness_access %as_type.3e7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b76: <specific function> = specific_impl_function %impl.elem0.594, @As.WithSelf.Convert(%as_type.c65, %As.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.688: %Destroy.type = facet_value %as_type.c65, (%as_type.c65) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.f68: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet.688) [symbolic]
// CHECK:STDOUT:   %.682: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.f68, %Destroy.facet.688 [symbolic]
// CHECK:STDOUT:   %impl.elem0.7c5: %.682 = impl_witness_access %as_type.c65, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.089: <specific function> = specific_impl_function %impl.elem0.7c5, @Destroy.WithSelf.Op(%Destroy.facet.688) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.f83: %Destroy.type = facet_value %as_type.846, (%as_type.846) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d07: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet.f83) [symbolic]
// CHECK:STDOUT:   %.11d: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.d07, %Destroy.facet.f83 [symbolic]
// CHECK:STDOUT:   %impl.elem0.342: %.11d = impl_witness_access %as_type.846, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.97f: <specific function> = specific_impl_function %impl.elem0.342, @Destroy.WithSelf.Op(%Destroy.facet.f83) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.54e: %Destroy.type = facet_value %as_type.e31, (%as_type.e31) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7ab: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet.54e) [symbolic]
// CHECK:STDOUT:   %.4d9: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7ab, %Destroy.facet.54e [symbolic]
// CHECK:STDOUT:   %impl.elem0.e40: %.4d9 = impl_witness_access %as_type.e31, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdf: <specific function> = specific_impl_function %impl.elem0.e40, @Destroy.WithSelf.Op(%Destroy.facet.54e) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.178: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.843)]
// CHECK:STDOUT:   %Core.import_ref.201: @ImplicitAs.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.b3a) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.1de)]
// CHECK:STDOUT:   %Core.import_ref.cc1 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.8db: @As.%As.assoc_type (%As.assoc_type.752) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @As.%assoc0 (constants.%assoc0.2f9)]
// CHECK:STDOUT:   %Core.import_ref.0ec: @As.%As.WithSelf.Convert.type (%As.WithSelf.Convert.type.bb7) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @As.%As.WithSelf.Convert (constants.%As.WithSelf.Convert.923)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.4c6 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .P = %P.decl
// CHECK:STDOUT:     .Q = %Q.decl
// CHECK:STDOUT:     .R = %R.decl
// CHECK:STDOUT:     .Transitive = %Transitive.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %P.decl: type = interface_decl @P [concrete = constants.%P.type] {} {}
// CHECK:STDOUT:   %Q.decl: type = interface_decl @Q [concrete = constants.%Q.type] {} {}
// CHECK:STDOUT:   %R.decl: type = interface_decl @R [concrete = constants.%R.type] {} {}
// CHECK:STDOUT:   %Transitive.decl: type = interface_decl @Transitive [concrete = constants.%Transitive.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.18b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @F.%pattern_type.loc19 (%pattern_type.993) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @F.%pattern_type.loc19 (%pattern_type.993) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19_10: type = splice_block %Transitive.ref [concrete = constants.%Transitive.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Transitive.ref: type = name_ref Transitive, file.%Transitive.decl [concrete = constants.%Transitive.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc19_6.2: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc19_25.1: type = splice_block %.loc19_25.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc19: %Transitive.type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc19: type = facet_access_type %T.ref.loc19 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc19_25.2: type = converted %T.ref.loc19, %T.as_type.loc19 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @F.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @P {
// CHECK:STDOUT:   %Self.loc4_13: %P.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d21]
// CHECK:STDOUT:   interface_with_self_decl @P [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %P.WithSelf.InP.decl: @P.%P.WithSelf.InP.type (%P.WithSelf.InP.type) = fn_decl @P.WithSelf.InP [symbolic = @P.%P.WithSelf.InP (constants.%P.WithSelf.InP)] {
// CHECK:STDOUT:     %self.patt: @P.WithSelf.InP.%pattern_type (%pattern_type.2eb) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @P.WithSelf.InP.%pattern_type (%pattern_type.2eb) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @P.WithSelf.InP.%Self.binding.as_type (%Self.binding.as_type.6c8) = value_param call_param0
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)] {
// CHECK:STDOUT:       %Self.ref: %P.type = name_ref Self, @P.%Self.loc4_13 [symbolic = %Self (constants.%Self.d21)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:       %.loc4_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @P.WithSelf.InP.%Self.binding.as_type (%Self.binding.as_type.6c8) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %P.assoc_type = assoc_entity element0, %P.WithSelf.InP.decl [concrete = constants.%assoc0.4ec]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc4_13
// CHECK:STDOUT:   .InP = %assoc0
// CHECK:STDOUT:   witness = (%P.WithSelf.InP.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q {
// CHECK:STDOUT:   %Self.loc5_13: %Q.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f64]
// CHECK:STDOUT:   interface_with_self_decl @Q [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Q.WithSelf.InQ.decl: @Q.%Q.WithSelf.InQ.type (%Q.WithSelf.InQ.type.377) = fn_decl @Q.WithSelf.InQ [symbolic = @Q.%Q.WithSelf.InQ (constants.%Q.WithSelf.InQ.2bf)] {
// CHECK:STDOUT:     %self.patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.992) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.992) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Q.WithSelf.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2) = value_param call_param0
// CHECK:STDOUT:     %.loc5_28.1: type = splice_block %.loc5_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)] {
// CHECK:STDOUT:       %Self.ref: %Q.type = name_ref Self, @Q.%Self.loc5_13 [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:       %.loc5_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Q.WithSelf.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Q.assoc_type = assoc_entity element0, %Q.WithSelf.InQ.decl [concrete = constants.%assoc0.038]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc5_13
// CHECK:STDOUT:   .InQ = %assoc0
// CHECK:STDOUT:   witness = (%Q.WithSelf.InQ.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R {
// CHECK:STDOUT:   %Self.loc6_13: %R.type = symbolic_binding Self, 0 [symbolic = constants.%Self.bea]
// CHECK:STDOUT:   interface_with_self_decl @R [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %R.WithSelf.InR.decl: @R.%R.WithSelf.InR.type (%R.WithSelf.InR.type.f8a) = fn_decl @R.WithSelf.InR [symbolic = @R.%R.WithSelf.InR (constants.%R.WithSelf.InR.b79)] {
// CHECK:STDOUT:     %self.patt: @R.WithSelf.InR.%pattern_type (%pattern_type.351) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @R.WithSelf.InR.%pattern_type (%pattern_type.351) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @R.WithSelf.InR.%Self.binding.as_type (%Self.binding.as_type.296) = value_param call_param0
// CHECK:STDOUT:     %.loc6_28.1: type = splice_block %.loc6_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)] {
// CHECK:STDOUT:       %Self.ref: %R.type = name_ref Self, @R.%Self.loc6_13 [symbolic = %Self (constants.%Self.bea)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:       %.loc6_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @R.WithSelf.InR.%Self.binding.as_type (%Self.binding.as_type.296) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %R.assoc_type = assoc_entity element0, %R.WithSelf.InR.decl [concrete = constants.%assoc0.dce]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc6_13
// CHECK:STDOUT:   .InR = %assoc0
// CHECK:STDOUT:   witness = (%R.WithSelf.InR.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive {
// CHECK:STDOUT:   %Self.loc8: %Transitive.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e2a]
// CHECK:STDOUT:   interface_with_self_decl @Transitive [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A: %P.type = assoc_const_decl @A [concrete] {
// CHECK:STDOUT:     %assoc0: %Transitive.assoc_type = assoc_entity element0, @Transitive.%A [concrete = constants.%assoc0.be3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: %Q_where.type = assoc_const_decl @B [concrete] {
// CHECK:STDOUT:     %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.%B [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: %R_where.type = assoc_const_decl @C [concrete] {
// CHECK:STDOUT:     %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.%C [concrete = constants.%assoc2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.decl: @Transitive.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type.bde) = fn_decl @Transitive.WithSelf.GetA [symbolic = @Transitive.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA.e53)] {
// CHECK:STDOUT:     %self.patt: @Transitive.WithSelf.GetA.%pattern_type.loc13_11 (%pattern_type.b0d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Transitive.WithSelf.GetA.%pattern_type.loc13_11 (%pattern_type.b0d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Transitive.WithSelf.GetA.%pattern_type.loc13_25 (%pattern_type.bff) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Transitive.WithSelf.GetA.%pattern_type.loc13_25 (%pattern_type.bff) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc13_28.2: %P.type = impl_witness_access constants.%Transitive.lookup_impl_witness.804, element0 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %A.ref: %P.type = name_ref A, %impl.elem0.loc13_28.2 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %A.as_type: type = facet_access_type %A.ref [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:     %.loc13_28.2: type = converted %A.ref, %A.as_type [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:     %.loc13_28.3: Core.Form = init_form %.loc13_28.2, call_param1 [symbolic = %.loc13_28.1 (constants.%.7a7)]
// CHECK:STDOUT:     %self.param: @Transitive.WithSelf.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_param call_param0
// CHECK:STDOUT:     %.loc13_17.1: type = splice_block %.loc13_17.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self.loc8 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:       %.loc13_17.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.WithSelf.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Transitive.WithSelf.GetA.%as_type (%as_type.0ea) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Transitive.WithSelf.GetA.%as_type (%as_type.0ea) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, %Transitive.WithSelf.GetA.decl [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.decl: @Transitive.%Transitive.WithSelf.TakesC.type (%Transitive.WithSelf.TakesC.type.1ad) = fn_decl @Transitive.WithSelf.TakesC [symbolic = @Transitive.%Transitive.WithSelf.TakesC (constants.%Transitive.WithSelf.TakesC.9f1)] {
// CHECK:STDOUT:     %self.patt: @Transitive.WithSelf.TakesC.%pattern_type.loc14_13 (%pattern_type.b0d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.loc14_13 (%pattern_type.b0d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %c.patt: @Transitive.WithSelf.TakesC.%pattern_type.loc14_25 (%pattern_type.364) = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.loc14_25 (%pattern_type.364) = value_param_pattern %c.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Transitive.WithSelf.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self.loc8 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:       %.loc14_19.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.WithSelf.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_binding self, %self.param
// CHECK:STDOUT:     %c.param: @Transitive.WithSelf.TakesC.%as_type (%as_type.ecc) = value_param call_param1
// CHECK:STDOUT:     %.loc14_28.1: type = splice_block %.loc14_28.2 [symbolic = %as_type (constants.%as_type.ecc)] {
// CHECK:STDOUT:       %impl.elem2.loc14_28.2: %R_where.type = impl_witness_access constants.%Transitive.lookup_impl_witness.804, element2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:       %C.ref: %R_where.type = name_ref C, %impl.elem2.loc14_28.2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:       %C.as_type: type = facet_access_type %C.ref [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:       %.loc14_28.2: type = converted %C.ref, %C.as_type [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @Transitive.WithSelf.TakesC.%as_type (%as_type.ecc) = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, %Transitive.WithSelf.TakesC.decl [concrete = constants.%assoc4]
// CHECK:STDOUT:   %Transitive.WithSelf.observe.decl = observe_decl @Transitive.WithSelf.observe, (@Transitive.WithSelf.observe.%impl.elem0, @Transitive.WithSelf.observe.%A.ref, @Transitive.WithSelf.observe.%impl.elem1, @Transitive.WithSelf.observe.%B.ref, @Transitive.WithSelf.observe.%B.as_type.loc16_16.1, @Transitive.WithSelf.observe.%.loc16_16.1, @Transitive.WithSelf.observe.%A.as_type, @Transitive.WithSelf.observe.%.loc16_11, @Transitive.WithSelf.observe.%eq.loc16_13, @Transitive.WithSelf.observe.%impl.elem2, @Transitive.WithSelf.observe.%C.ref, @Transitive.WithSelf.observe.%C.as_type, @Transitive.WithSelf.observe.%.loc16_21, @Transitive.WithSelf.observe.%B.as_type.loc16_16.2, @Transitive.WithSelf.observe.%.loc16_16.2, @Transitive.WithSelf.observe.%eq.loc16_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc8
// CHECK:STDOUT:   .P = <poisoned>
// CHECK:STDOUT:   .Q = <poisoned>
// CHECK:STDOUT:   .R = <poisoned>
// CHECK:STDOUT:   .P = <poisoned>
// CHECK:STDOUT:   .A = @A.%assoc0
// CHECK:STDOUT:   .Q = <poisoned>
// CHECK:STDOUT:   .B = @B.%assoc1
// CHECK:STDOUT:   .R = <poisoned>
// CHECK:STDOUT:   .C = @C.%assoc2
// CHECK:STDOUT:   .GetA = %assoc3
// CHECK:STDOUT:   .TakesC = %assoc4
// CHECK:STDOUT:   witness = (%A, %B, %C, %Transitive.WithSelf.GetA.decl, %Transitive.WithSelf.TakesC.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.WithSelf.InP(@P.%Self.loc4_13: %P.type) {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d21)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @P.WithSelf.InP.%Self.binding.as_type (%Self.binding.as_type.6c8));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(@Q.%Self.loc5_13: %Q.type) {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.992)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Q.WithSelf.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(@R.%Self.loc6_13: %R.type) {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.bea)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.351)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @R.WithSelf.InR.%Self.binding.as_type (%Self.binding.as_type.296));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(@Transitive.%Self.loc8: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:   %pattern_type.loc13_11: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc13_11 (constants.%pattern_type.b0d)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem0.loc13_28.1: %P.type = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc13_28.1 [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:   %.loc13_28.1: Core.Form = init_form %as_type, call_param1 [symbolic = %.loc13_28.1 (constants.%.7a7)]
// CHECK:STDOUT:   %pattern_type.loc13_25: type = pattern_type %as_type [symbolic = %pattern_type.loc13_25 (constants.%pattern_type.bff)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.WithSelf.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b)) -> out %return.param: @Transitive.WithSelf.GetA.%as_type (%as_type.0ea);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(@Transitive.%Self.loc8: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:   %pattern_type.loc14_13: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc14_13 (constants.%pattern_type.b0d)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem2.loc14_28.1: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2.loc14_28.1 [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:   %pattern_type.loc14_25: type = pattern_type %as_type [symbolic = %pattern_type.loc14_25 (constants.%pattern_type.364)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.WithSelf.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b), %c.param: @Transitive.WithSelf.TakesC.%as_type (%as_type.ecc));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc19_6.2: %Transitive.type) {
// CHECK:STDOUT:   %T.loc19_6.1: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc19_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc19: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc19 (constants.%pattern_type.993)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc19: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc19 (constants.%require_complete.1c0)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc19_6.1, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8)]
// CHECK:STDOUT:   %impl.elem0.loc20_11.2: %P.type = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc20_11.2 (constants.%impl.elem0.b09)]
// CHECK:STDOUT:   %as_type.loc20_11.2: type = facet_access_type %impl.elem0.loc20_11.2 [symbolic = %as_type.loc20_11.2 (constants.%as_type.e31)]
// CHECK:STDOUT:   %require_complete.loc20: <witness> = require_complete_type %as_type.loc20_11.2 [symbolic = %require_complete.loc20 (constants.%require_complete.b46)]
// CHECK:STDOUT:   %pattern_type.loc20: type = pattern_type %as_type.loc20_11.2 [symbolic = %pattern_type.loc20 (constants.%pattern_type.05e)]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive(%T.loc19_6.1) [symbolic = %Transitive.WithSelf.GetA.type (constants.%Transitive.WithSelf.GetA.type.dc6)]
// CHECK:STDOUT:   %.loc20_17: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type, %T.loc19_6.1 [symbolic = %.loc20_17 (constants.%.c0a)]
// CHECK:STDOUT:   %impl.elem3.loc20_17.2: @F.%.loc20_17 (%.c0a) = impl_witness_access %Transitive.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc20_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_17.2: <specific function> = specific_impl_function %impl.elem3.loc20_17.2, @Transitive.WithSelf.GetA(%T.loc19_6.1) [symbolic = %specific_impl_fn.loc20_17.2 (constants.%specific_impl_fn.c79)]
// CHECK:STDOUT:   %impl.elem1.loc22_11.2: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc22_11.2 (constants.%impl.elem1.32a)]
// CHECK:STDOUT:   %as_type.loc22_11.2: type = facet_access_type %impl.elem1.loc22_11.2 [symbolic = %as_type.loc22_11.2 (constants.%as_type.846)]
// CHECK:STDOUT:   %require_complete.loc22_11: <witness> = require_complete_type %as_type.loc22_11.2 [symbolic = %require_complete.loc22_11 (constants.%require_complete.8ba)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %as_type.loc22_11.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.a21)]
// CHECK:STDOUT:   %ImplicitAs.type.loc22_3.2: type = facet_type <@ImplicitAs, @ImplicitAs(%as_type.loc22_11.2)> [symbolic = %ImplicitAs.type.loc22_3.2 (constants.%ImplicitAs.type.d5a)]
// CHECK:STDOUT:   %require_complete.loc22_3: <witness> = require_complete_type %ImplicitAs.type.loc22_3.2 [symbolic = %require_complete.loc22_3 (constants.%require_complete.abb)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.loc22: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%as_type.loc22_11.2) [symbolic = %ImplicitAs.assoc_type.loc22 (constants.%ImplicitAs.assoc_type.b4b)]
// CHECK:STDOUT:   %assoc0.loc22: @F.%ImplicitAs.assoc_type.loc22 (%ImplicitAs.assoc_type.b4b) = assoc_entity element0, imports.%Core.import_ref.201 [symbolic = %assoc0.loc22 (constants.%assoc0.242)]
// CHECK:STDOUT:   %as_type.loc22_3: %Q_where.type = facet_access_type %impl.elem0.loc20_11.2 [symbolic = %as_type.loc22_3 (constants.%as_type.b6a)]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22: @F.%ImplicitAs.type.loc22_3.2 (%ImplicitAs.type.d5a) = facet_value %as_type.loc20_11.2, (%as_type.loc22_3) [symbolic = %ImplicitAs.facet.loc22 (constants.%ImplicitAs.facet.746)]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.loc22: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%as_type.loc22_11.2, %ImplicitAs.facet.loc22) [symbolic = %ImplicitAs.WithSelf.Convert.type.loc22 (constants.%ImplicitAs.WithSelf.Convert.type.e4d)]
// CHECK:STDOUT:   %.loc22_3.5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.loc22, %ImplicitAs.facet.loc22 [symbolic = %.loc22_3.5 (constants.%.db6)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.3: @F.%.loc22_3.5 (%.db6) = impl_witness_access %as_type.loc22_3, element0 [symbolic = %impl.elem0.loc22_3.3 (constants.%impl.elem0.9e8)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_3.3: <specific function> = specific_impl_function %impl.elem0.loc22_3.3, @ImplicitAs.WithSelf.Convert(%as_type.loc22_11.2, %ImplicitAs.facet.loc22) [symbolic = %specific_impl_fn.loc22_3.3 (constants.%specific_impl_fn.060)]
// CHECK:STDOUT:   %Q.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.loc22_11.2, @Q [symbolic = %Q.lookup_impl_witness (constants.%Q.lookup_impl_witness)]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.loc22_11.2, (%Q.lookup_impl_witness) [symbolic = %Q.facet (constants.%Q.facet)]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q(%Q.facet) [symbolic = %Q.WithSelf.InQ.type (constants.%Q.WithSelf.InQ.type.a66)]
// CHECK:STDOUT:   %.loc23_4: type = fn_type_with_self_type %Q.WithSelf.InQ.type, %Q.facet [symbolic = %.loc23_4 (constants.%.803)]
// CHECK:STDOUT:   %impl.elem0.loc23_4.2: @F.%.loc23_4 (%.803) = impl_witness_access %Q.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc23_4.2 (constants.%impl.elem0.9f0)]
// CHECK:STDOUT:   %specific_impl_fn.loc23_4.2: <specific function> = specific_impl_function %impl.elem0.loc23_4.2, @Q.WithSelf.InQ(%Q.facet) [symbolic = %specific_impl_fn.loc23_4.2 (constants.%specific_impl_fn.8e9)]
// CHECK:STDOUT:   %impl.elem2.loc25_7.2: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc25_7.2 (constants.%impl.elem2.458)]
// CHECK:STDOUT:   %as_type.loc25_9.2: type = facet_access_type %impl.elem2.loc25_7.2 [symbolic = %as_type.loc25_9.2 (constants.%as_type.c65)]
// CHECK:STDOUT:   %R.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem2.loc25_7.2, @R [symbolic = %R.lookup_impl_witness (constants.%R.lookup_impl_witness)]
// CHECK:STDOUT:   %R.facet: %R.type = facet_value %as_type.loc25_9.2, (%R.lookup_impl_witness) [symbolic = %R.facet (constants.%R.facet)]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R(%R.facet) [symbolic = %R.WithSelf.InR.type (constants.%R.WithSelf.InR.type.63c)]
// CHECK:STDOUT:   %.loc25_9.2: type = fn_type_with_self_type %R.WithSelf.InR.type, %R.facet [symbolic = %.loc25_9.2 (constants.%.e11)]
// CHECK:STDOUT:   %impl.elem0.loc25_9.2: @F.%.loc25_9.2 (%.e11) = impl_witness_access %R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_9.2 (constants.%impl.elem0.df8)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_9.2: <specific function> = specific_impl_function %impl.elem0.loc25_9.2, @R.WithSelf.InR(%R.facet) [symbolic = %specific_impl_fn.loc25_9.2 (constants.%specific_impl_fn.2c6)]
// CHECK:STDOUT:   %require_complete.loc25_3.1: <witness> = require_complete_type %as_type.loc25_9.2 [symbolic = %require_complete.loc25_3.1 (constants.%require_complete.60d)]
// CHECK:STDOUT:   %ImplicitAs.type.loc25_3.2: type = facet_type <@ImplicitAs, @ImplicitAs(%as_type.loc25_9.2)> [symbolic = %ImplicitAs.type.loc25_3.2 (constants.%ImplicitAs.type.ac2)]
// CHECK:STDOUT:   %require_complete.loc25_3.2: <witness> = require_complete_type %ImplicitAs.type.loc25_3.2 [symbolic = %require_complete.loc25_3.2 (constants.%require_complete.49e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.loc25: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%as_type.loc25_9.2) [symbolic = %ImplicitAs.assoc_type.loc25 (constants.%ImplicitAs.assoc_type.eeb)]
// CHECK:STDOUT:   %assoc0.loc25: @F.%ImplicitAs.assoc_type.loc25 (%ImplicitAs.assoc_type.eeb) = assoc_entity element0, imports.%Core.import_ref.201 [symbolic = %assoc0.loc25 (constants.%assoc0.14e)]
// CHECK:STDOUT:   %as_type.loc25_3: %R_where.type = facet_access_type %impl.elem0.loc20_11.2 [symbolic = %as_type.loc25_3 (constants.%as_type.3e7)]
// CHECK:STDOUT:   %ImplicitAs.facet.loc25: @F.%ImplicitAs.type.loc25_3.2 (%ImplicitAs.type.ac2) = facet_value %as_type.loc20_11.2, (%as_type.loc25_3) [symbolic = %ImplicitAs.facet.loc25 (constants.%ImplicitAs.facet.21e)]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.loc25: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%as_type.loc25_9.2, %ImplicitAs.facet.loc25) [symbolic = %ImplicitAs.WithSelf.Convert.type.loc25 (constants.%ImplicitAs.WithSelf.Convert.type.c6b)]
// CHECK:STDOUT:   %.loc25_3.8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.loc25, %ImplicitAs.facet.loc25 [symbolic = %.loc25_3.8 (constants.%.92e)]
// CHECK:STDOUT:   %impl.elem0.loc25_3.3: @F.%.loc25_3.8 (%.92e) = impl_witness_access %as_type.loc25_3, element0 [symbolic = %impl.elem0.loc25_3.3 (constants.%impl.elem0.c85)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_3.3: <specific function> = specific_impl_function %impl.elem0.loc25_3.3, @ImplicitAs.WithSelf.Convert(%as_type.loc25_9.2, %ImplicitAs.facet.loc25) [symbolic = %specific_impl_fn.loc25_3.3 (constants.%specific_impl_fn.80c)]
// CHECK:STDOUT:   %As.type.loc26_6.2: type = facet_type <@As, @As(%as_type.loc25_9.2)> [symbolic = %As.type.loc26_6.2 (constants.%As.type.ae5)]
// CHECK:STDOUT:   %require_complete.loc26: <witness> = require_complete_type %As.type.loc26_6.2 [symbolic = %require_complete.loc26 (constants.%require_complete.ea4)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%as_type.loc25_9.2) [symbolic = %As.assoc_type (constants.%As.assoc_type.69a)]
// CHECK:STDOUT:   %assoc0.loc26: @F.%As.assoc_type (%As.assoc_type.69a) = assoc_entity element0, imports.%Core.import_ref.0ec [symbolic = %assoc0.loc26 (constants.%assoc0.f5c)]
// CHECK:STDOUT:   %As.facet: @F.%As.type.loc26_6.2 (%As.type.ae5) = facet_value %as_type.loc20_11.2, (%as_type.loc25_3) [symbolic = %As.facet (constants.%As.facet)]
// CHECK:STDOUT:   %As.WithSelf.Convert.type: type = fn_type @As.WithSelf.Convert, @As(%as_type.loc25_9.2, %As.facet) [symbolic = %As.WithSelf.Convert.type (constants.%As.WithSelf.Convert.type.cfa)]
// CHECK:STDOUT:   %.loc26_6.7: type = fn_type_with_self_type %As.WithSelf.Convert.type, %As.facet [symbolic = %.loc26_6.7 (constants.%.d45)]
// CHECK:STDOUT:   %impl.elem0.loc26_6.3: @F.%.loc26_6.7 (%.d45) = impl_witness_access %as_type.loc25_3, element0 [symbolic = %impl.elem0.loc26_6.3 (constants.%impl.elem0.594)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_6.3: <specific function> = specific_impl_function %impl.elem0.loc26_6.3, @As.WithSelf.Convert(%as_type.loc25_9.2, %As.facet) [symbolic = %specific_impl_fn.loc26_6.3 (constants.%specific_impl_fn.b76)]
// CHECK:STDOUT:   %Destroy.facet.loc26: %Destroy.type = facet_value %as_type.loc25_9.2, (%as_type.loc25_9.2) [symbolic = %Destroy.facet.loc26 (constants.%Destroy.facet.688)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc26: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet.loc26) [symbolic = %Destroy.WithSelf.Op.type.loc26 (constants.%Destroy.WithSelf.Op.type.f68)]
// CHECK:STDOUT:   %.loc26_6.8: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc26, %Destroy.facet.loc26 [symbolic = %.loc26_6.8 (constants.%.682)]
// CHECK:STDOUT:   %impl.elem0.loc26_6.4: @F.%.loc26_6.8 (%.682) = impl_witness_access %as_type.loc25_9.2, element0 [symbolic = %impl.elem0.loc26_6.4 (constants.%impl.elem0.7c5)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_6.4: <specific function> = specific_impl_function %impl.elem0.loc26_6.4, @Destroy.WithSelf.Op(%Destroy.facet.loc26) [symbolic = %specific_impl_fn.loc26_6.4 (constants.%specific_impl_fn.089)]
// CHECK:STDOUT:   %Destroy.facet.loc22: %Destroy.type = facet_value %as_type.loc22_11.2, (%as_type.loc22_11.2) [symbolic = %Destroy.facet.loc22 (constants.%Destroy.facet.f83)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc22: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet.loc22) [symbolic = %Destroy.WithSelf.Op.type.loc22 (constants.%Destroy.WithSelf.Op.type.d07)]
// CHECK:STDOUT:   %.loc22_3.6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc22, %Destroy.facet.loc22 [symbolic = %.loc22_3.6 (constants.%.11d)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.4: @F.%.loc22_3.6 (%.11d) = impl_witness_access %as_type.loc22_11.2, element0 [symbolic = %impl.elem0.loc22_3.4 (constants.%impl.elem0.342)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_3.4: <specific function> = specific_impl_function %impl.elem0.loc22_3.4, @Destroy.WithSelf.Op(%Destroy.facet.loc22) [symbolic = %specific_impl_fn.loc22_3.4 (constants.%specific_impl_fn.97f)]
// CHECK:STDOUT:   %Destroy.facet.loc20: %Destroy.type = facet_value %as_type.loc20_11.2, (%as_type.loc20_11.2) [symbolic = %Destroy.facet.loc20 (constants.%Destroy.facet.54e)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc20: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet.loc20) [symbolic = %Destroy.WithSelf.Op.type.loc20 (constants.%Destroy.WithSelf.Op.type.7ab)]
// CHECK:STDOUT:   %.loc20_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc20, %Destroy.facet.loc20 [symbolic = %.loc20_3.2 (constants.%.4d9)]
// CHECK:STDOUT:   %impl.elem0.loc20_3.2: @F.%.loc20_3.2 (%.4d9) = impl_witness_access %as_type.loc20_11.2, element0 [symbolic = %impl.elem0.loc20_3.2 (constants.%impl.elem0.e40)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_3.2: <specific function> = specific_impl_function %impl.elem0.loc20_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc20) [symbolic = %specific_impl_fn.loc20_3.2 (constants.%specific_impl_fn.bdf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @F.%pattern_type.loc20 (%pattern_type.05e) = ref_binding_pattern a [concrete]
// CHECK:STDOUT:       %a.var_patt: @F.%pattern_type.loc20 (%pattern_type.05e) = var_pattern %a.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.var: ref @F.%as_type.loc20_11.2 (%as_type.e31) = var %a.var_patt
// CHECK:STDOUT:     %t.ref: @F.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %GetA.ref: %Transitive.assoc_type = name_ref GetA, @Transitive.%assoc3 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc20_17.1: @F.%.loc20_17 (%.c0a) = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element3 [symbolic = %impl.elem3.loc20_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc20_17: <bound method> = bound_method %t.ref, %impl.elem3.loc20_17.1
// CHECK:STDOUT:     %specific_impl_fn.loc20_17.1: <specific function> = specific_impl_function %impl.elem3.loc20_17.1, @Transitive.WithSelf.GetA(constants.%T) [symbolic = %specific_impl_fn.loc20_17.2 (constants.%specific_impl_fn.c79)]
// CHECK:STDOUT:     %bound_method.loc20_23: <bound method> = bound_method %t.ref, %specific_impl_fn.loc20_17.1
// CHECK:STDOUT:     %.loc20_3.1: ref @F.%as_type.loc20_11.2 (%as_type.e31) = splice_block %a.var {}
// CHECK:STDOUT:     %Transitive.WithSelf.GetA.call: init @F.%as_type.loc20_11.2 (%as_type.e31) to %.loc20_3.1 = call %bound_method.loc20_23(%t.ref)
// CHECK:STDOUT:     assign %a.var, %Transitive.WithSelf.GetA.call
// CHECK:STDOUT:     %.loc20_11.1: type = splice_block %.loc20_11.3 [symbolic = %as_type.loc20_11.2 (constants.%as_type.e31)] {
// CHECK:STDOUT:       %T.ref.loc20: %Transitive.type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc20: type = facet_access_type %T.ref.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc20_11.2: type = converted %T.ref.loc20, %T.as_type.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %A.ref: %Transitive.assoc_type = name_ref A, @A.%assoc0 [concrete = constants.%assoc0.be3]
// CHECK:STDOUT:       %impl.elem0.loc20_11.1: %P.type = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element0 [symbolic = %impl.elem0.loc20_11.2 (constants.%impl.elem0.b09)]
// CHECK:STDOUT:       %as_type.loc20_11.1: type = facet_access_type %impl.elem0.loc20_11.1 [symbolic = %as_type.loc20_11.2 (constants.%as_type.e31)]
// CHECK:STDOUT:       %.loc20_11.3: type = converted %impl.elem0.loc20_11.1, %as_type.loc20_11.1 [symbolic = %as_type.loc20_11.2 (constants.%as_type.e31)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: ref @F.%as_type.loc20_11.2 (%as_type.e31) = ref_binding a, %a.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %b.patt: @F.%pattern_type.loc22 (%pattern_type.a21) = ref_binding_pattern b [concrete]
// CHECK:STDOUT:       %b.var_patt: @F.%pattern_type.loc22 (%pattern_type.a21) = var_pattern %b.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.var: ref @F.%as_type.loc22_11.2 (%as_type.846) = var %b.var_patt
// CHECK:STDOUT:     %a.ref.loc22: ref @F.%as_type.loc20_11.2 (%as_type.e31) = name_ref a, %a
// CHECK:STDOUT:     %ImplicitAs.type.loc22_3.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%as_type.846)> [symbolic = %ImplicitAs.type.loc22_3.2 (constants.%ImplicitAs.type.d5a)]
// CHECK:STDOUT:     %.loc22_3.1: @F.%ImplicitAs.assoc_type.loc22 (%ImplicitAs.assoc_type.b4b) = specific_constant imports.%Core.import_ref.178, @ImplicitAs(constants.%as_type.846, constants.%Self.738) [symbolic = %assoc0.loc22 (constants.%assoc0.242)]
// CHECK:STDOUT:     %Convert.ref.loc22: @F.%ImplicitAs.assoc_type.loc22 (%ImplicitAs.assoc_type.b4b) = name_ref Convert, %.loc22_3.1 [symbolic = %assoc0.loc22 (constants.%assoc0.242)]
// CHECK:STDOUT:     %.loc22_3.2: %Q_where.type = as_compatible constants.%as_type.e31 [symbolic = %as_type.loc22_3 (constants.%as_type.b6a)]
// CHECK:STDOUT:     %impl.elem0.loc22_3.1: @F.%.loc22_3.5 (%.db6) = impl_witness_access %.loc22_3.2, element0 [symbolic = %impl.elem0.loc22_3.3 (constants.%impl.elem0.9e8)]
// CHECK:STDOUT:     %bound_method.loc22_3.1: <bound method> = bound_method %a.ref.loc22, %impl.elem0.loc22_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_3.1: <specific function> = specific_impl_function %impl.elem0.loc22_3.1, @ImplicitAs.WithSelf.Convert(constants.%as_type.846, constants.%ImplicitAs.facet.746) [symbolic = %specific_impl_fn.loc22_3.3 (constants.%specific_impl_fn.060)]
// CHECK:STDOUT:     %bound_method.loc22_3.2: <bound method> = bound_method %a.ref.loc22, %specific_impl_fn.loc22_3.1
// CHECK:STDOUT:     %.loc22_3.3: ref @F.%as_type.loc22_11.2 (%as_type.846) = splice_block %b.var {}
// CHECK:STDOUT:     %.loc22_16: @F.%as_type.loc20_11.2 (%as_type.e31) = acquire_value %a.ref.loc22
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.call.loc22: init @F.%as_type.loc22_11.2 (%as_type.846) to %.loc22_3.3 = call %bound_method.loc22_3.2(%.loc22_16)
// CHECK:STDOUT:     %.loc22_3.4: init @F.%as_type.loc22_11.2 (%as_type.846) = converted %a.ref.loc22, %ImplicitAs.WithSelf.Convert.call.loc22
// CHECK:STDOUT:     assign %b.var, %.loc22_3.4
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %.loc22_11.3 [symbolic = %as_type.loc22_11.2 (constants.%as_type.846)] {
// CHECK:STDOUT:       %T.ref.loc22: %Transitive.type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc22_11.2: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %B.ref: %Transitive.assoc_type = name_ref B, @B.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:       %impl.elem1.loc22_11.1: %Q_where.type = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element1 [symbolic = %impl.elem1.loc22_11.2 (constants.%impl.elem1.32a)]
// CHECK:STDOUT:       %as_type.loc22_11.1: type = facet_access_type %impl.elem1.loc22_11.1 [symbolic = %as_type.loc22_11.2 (constants.%as_type.846)]
// CHECK:STDOUT:       %.loc22_11.3: type = converted %impl.elem1.loc22_11.1, %as_type.loc22_11.1 [symbolic = %as_type.loc22_11.2 (constants.%as_type.846)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: ref @F.%as_type.loc22_11.2 (%as_type.846) = ref_binding b, %b.var
// CHECK:STDOUT:     %b.ref: ref @F.%as_type.loc22_11.2 (%as_type.846) = name_ref b, %b
// CHECK:STDOUT:     %InQ.ref: %Q.assoc_type = name_ref InQ, @Q.%assoc0 [concrete = constants.%assoc0.038]
// CHECK:STDOUT:     %impl.elem0.loc23_4.1: @F.%.loc23_4 (%.803) = impl_witness_access constants.%Q.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc23_4.2 (constants.%impl.elem0.9f0)]
// CHECK:STDOUT:     %bound_method.loc23_4: <bound method> = bound_method %b.ref, %impl.elem0.loc23_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc23_4.1: <specific function> = specific_impl_function %impl.elem0.loc23_4.1, @Q.WithSelf.InQ(constants.%Q.facet) [symbolic = %specific_impl_fn.loc23_4.2 (constants.%specific_impl_fn.8e9)]
// CHECK:STDOUT:     %bound_method.loc23_9: <bound method> = bound_method %b.ref, %specific_impl_fn.loc23_4.1
// CHECK:STDOUT:     %.loc23_3: @F.%as_type.loc22_11.2 (%as_type.846) = acquire_value %b.ref
// CHECK:STDOUT:     %Q.WithSelf.InQ.call: init %empty_tuple.type = call %bound_method.loc23_9(%.loc23_3)
// CHECK:STDOUT:     %a.ref.loc25: ref @F.%as_type.loc20_11.2 (%as_type.e31) = name_ref a, %a
// CHECK:STDOUT:     %T.ref.loc25: %Transitive.type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc25: type = facet_access_type %T.ref.loc25 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc25_7: type = converted %T.ref.loc25, %T.as_type.loc25 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %C.ref.loc25: %Transitive.assoc_type = name_ref C, @C.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc25_7.1: %R_where.type = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element2 [symbolic = %impl.elem2.loc25_7.2 (constants.%impl.elem2.458)]
// CHECK:STDOUT:     %as_type.loc25_9.1: type = facet_access_type %impl.elem2.loc25_7.1 [symbolic = %as_type.loc25_9.2 (constants.%as_type.c65)]
// CHECK:STDOUT:     %.loc25_9.1: type = converted %impl.elem2.loc25_7.1, %as_type.loc25_9.1 [symbolic = %as_type.loc25_9.2 (constants.%as_type.c65)]
// CHECK:STDOUT:     %InR.ref.loc25: %R.assoc_type = name_ref InR, @R.%assoc0 [concrete = constants.%assoc0.dce]
// CHECK:STDOUT:     %impl.elem0.loc25_9.1: @F.%.loc25_9.2 (%.e11) = impl_witness_access constants.%R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_9.2 (constants.%impl.elem0.df8)]
// CHECK:STDOUT:     %bound_method.loc25_4: <bound method> = bound_method %a.ref.loc25, %impl.elem0.loc25_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_9.1: <specific function> = specific_impl_function %impl.elem0.loc25_9.1, @R.WithSelf.InR(constants.%R.facet) [symbolic = %specific_impl_fn.loc25_9.2 (constants.%specific_impl_fn.2c6)]
// CHECK:STDOUT:     %bound_method.loc25_15: <bound method> = bound_method %a.ref.loc25, %specific_impl_fn.loc25_9.1
// CHECK:STDOUT:     %ImplicitAs.type.loc25_3.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%as_type.c65)> [symbolic = %ImplicitAs.type.loc25_3.2 (constants.%ImplicitAs.type.ac2)]
// CHECK:STDOUT:     %.loc25_3.1: @F.%ImplicitAs.assoc_type.loc25 (%ImplicitAs.assoc_type.eeb) = specific_constant imports.%Core.import_ref.178, @ImplicitAs(constants.%as_type.c65, constants.%Self.738) [symbolic = %assoc0.loc25 (constants.%assoc0.14e)]
// CHECK:STDOUT:     %Convert.ref.loc25: @F.%ImplicitAs.assoc_type.loc25 (%ImplicitAs.assoc_type.eeb) = name_ref Convert, %.loc25_3.1 [symbolic = %assoc0.loc25 (constants.%assoc0.14e)]
// CHECK:STDOUT:     %.loc25_3.2: %R_where.type = as_compatible constants.%as_type.e31 [symbolic = %as_type.loc25_3 (constants.%as_type.3e7)]
// CHECK:STDOUT:     %impl.elem0.loc25_3.1: @F.%.loc25_3.8 (%.92e) = impl_witness_access %.loc25_3.2, element0 [symbolic = %impl.elem0.loc25_3.3 (constants.%impl.elem0.c85)]
// CHECK:STDOUT:     %bound_method.loc25_3.1: <bound method> = bound_method %a.ref.loc25, %impl.elem0.loc25_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_3.1: <specific function> = specific_impl_function %impl.elem0.loc25_3.1, @ImplicitAs.WithSelf.Convert(constants.%as_type.c65, constants.%ImplicitAs.facet.21e) [symbolic = %specific_impl_fn.loc25_3.3 (constants.%specific_impl_fn.80c)]
// CHECK:STDOUT:     %bound_method.loc25_3.2: <bound method> = bound_method %a.ref.loc25, %specific_impl_fn.loc25_3.1
// CHECK:STDOUT:     %.loc25_3.3: ref @F.%as_type.loc25_9.2 (%as_type.c65) = temporary_storage
// CHECK:STDOUT:     %.loc25_3.4: @F.%as_type.loc20_11.2 (%as_type.e31) = acquire_value %a.ref.loc25
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.call.loc25: init @F.%as_type.loc25_9.2 (%as_type.c65) to %.loc25_3.3 = call %bound_method.loc25_3.2(%.loc25_3.4)
// CHECK:STDOUT:     %.loc25_3.5: init @F.%as_type.loc25_9.2 (%as_type.c65) = converted %a.ref.loc25, %ImplicitAs.WithSelf.Convert.call.loc25
// CHECK:STDOUT:     %.loc25_3.6: ref @F.%as_type.loc25_9.2 (%as_type.c65) = temporary %.loc25_3.3, %.loc25_3.5
// CHECK:STDOUT:     %.loc25_3.7: @F.%as_type.loc25_9.2 (%as_type.c65) = acquire_value %.loc25_3.6
// CHECK:STDOUT:     %R.WithSelf.InR.call.loc25: init %empty_tuple.type = call %bound_method.loc25_15(%.loc25_3.7)
// CHECK:STDOUT:     %a.ref.loc26: ref @F.%as_type.loc20_11.2 (%as_type.e31) = name_ref a, %a
// CHECK:STDOUT:     %T.ref.loc26: %Transitive.type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc26: type = facet_access_type %T.ref.loc26 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc26_10.1: type = converted %T.ref.loc26, %T.as_type.loc26 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %C.ref.loc26: %Transitive.assoc_type = name_ref C, @C.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc26: %R_where.type = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element2 [symbolic = %impl.elem2.loc25_7.2 (constants.%impl.elem2.458)]
// CHECK:STDOUT:     %as_type.loc26: type = facet_access_type %impl.elem2.loc26 [symbolic = %as_type.loc25_9.2 (constants.%as_type.c65)]
// CHECK:STDOUT:     %.loc26_10.2: type = converted %impl.elem2.loc26, %as_type.loc26 [symbolic = %as_type.loc25_9.2 (constants.%as_type.c65)]
// CHECK:STDOUT:     %As.type.loc26_6.1: type = facet_type <@As, @As(constants.%as_type.c65)> [symbolic = %As.type.loc26_6.2 (constants.%As.type.ae5)]
// CHECK:STDOUT:     %.loc26_6.1: @F.%As.assoc_type (%As.assoc_type.69a) = specific_constant imports.%Core.import_ref.8db, @As(constants.%as_type.c65, constants.%Self.2d0) [symbolic = %assoc0.loc26 (constants.%assoc0.f5c)]
// CHECK:STDOUT:     %Convert.ref.loc26: @F.%As.assoc_type (%As.assoc_type.69a) = name_ref Convert, %.loc26_6.1 [symbolic = %assoc0.loc26 (constants.%assoc0.f5c)]
// CHECK:STDOUT:     %.loc26_6.2: %R_where.type = as_compatible constants.%as_type.e31 [symbolic = %as_type.loc25_3 (constants.%as_type.3e7)]
// CHECK:STDOUT:     %impl.elem0.loc26_6.1: @F.%.loc26_6.7 (%.d45) = impl_witness_access %.loc26_6.2, element0 [symbolic = %impl.elem0.loc26_6.3 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %bound_method.loc26_6.1: <bound method> = bound_method %a.ref.loc26, %impl.elem0.loc26_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc26_6.1: <specific function> = specific_impl_function %impl.elem0.loc26_6.1, @As.WithSelf.Convert(constants.%as_type.c65, constants.%As.facet) [symbolic = %specific_impl_fn.loc26_6.3 (constants.%specific_impl_fn.b76)]
// CHECK:STDOUT:     %bound_method.loc26_6.2: <bound method> = bound_method %a.ref.loc26, %specific_impl_fn.loc26_6.1
// CHECK:STDOUT:     %.loc26_6.3: ref @F.%as_type.loc25_9.2 (%as_type.c65) = temporary_storage
// CHECK:STDOUT:     %.loc26_4: @F.%as_type.loc20_11.2 (%as_type.e31) = acquire_value %a.ref.loc26
// CHECK:STDOUT:     %As.WithSelf.Convert.call: init @F.%as_type.loc25_9.2 (%as_type.c65) to %.loc26_6.3 = call %bound_method.loc26_6.2(%.loc26_4)
// CHECK:STDOUT:     %.loc26_6.4: init @F.%as_type.loc25_9.2 (%as_type.c65) = converted %a.ref.loc26, %As.WithSelf.Convert.call
// CHECK:STDOUT:     %.loc26_6.5: ref @F.%as_type.loc25_9.2 (%as_type.c65) = temporary %.loc26_6.3, %.loc26_6.4
// CHECK:STDOUT:     %InR.ref.loc26: %R.assoc_type = name_ref InR, @R.%assoc0 [concrete = constants.%assoc0.dce]
// CHECK:STDOUT:     %impl.elem0.loc26_13: @F.%.loc25_9.2 (%.e11) = impl_witness_access constants.%R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_9.2 (constants.%impl.elem0.df8)]
// CHECK:STDOUT:     %bound_method.loc26_13: <bound method> = bound_method %.loc26_6.5, %impl.elem0.loc26_13
// CHECK:STDOUT:     %specific_impl_fn.loc26_13: <specific function> = specific_impl_function %impl.elem0.loc26_13, @R.WithSelf.InR(constants.%R.facet) [symbolic = %specific_impl_fn.loc25_9.2 (constants.%specific_impl_fn.2c6)]
// CHECK:STDOUT:     %bound_method.loc26_18: <bound method> = bound_method %.loc26_6.5, %specific_impl_fn.loc26_13
// CHECK:STDOUT:     %.loc26_6.6: @F.%as_type.loc25_9.2 (%as_type.c65) = acquire_value %.loc26_6.5
// CHECK:STDOUT:     %R.WithSelf.InR.call.loc26: init %empty_tuple.type = call %bound_method.loc26_18(%.loc26_6.6)
// CHECK:STDOUT:     %impl.elem0.loc26_6.2: @F.%.loc26_6.8 (%.682) = impl_witness_access constants.%as_type.c65, element0 [symbolic = %impl.elem0.loc26_6.4 (constants.%impl.elem0.7c5)]
// CHECK:STDOUT:     %bound_method.loc26_6.3: <bound method> = bound_method %.loc26_6.5, %impl.elem0.loc26_6.2
// CHECK:STDOUT:     %specific_impl_fn.loc26_6.2: <specific function> = specific_impl_function %impl.elem0.loc26_6.2, @Destroy.WithSelf.Op(constants.%Destroy.facet.688) [symbolic = %specific_impl_fn.loc26_6.4 (constants.%specific_impl_fn.089)]
// CHECK:STDOUT:     %bound_method.loc26_6.4: <bound method> = bound_method %.loc26_6.5, %specific_impl_fn.loc26_6.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc26: init %empty_tuple.type = call %bound_method.loc26_6.4(%.loc26_6.5)
// CHECK:STDOUT:     %impl.elem0.loc25_3.2: @F.%.loc26_6.8 (%.682) = impl_witness_access constants.%as_type.c65, element0 [symbolic = %impl.elem0.loc26_6.4 (constants.%impl.elem0.7c5)]
// CHECK:STDOUT:     %bound_method.loc25_3.3: <bound method> = bound_method %.loc25_3.6, %impl.elem0.loc25_3.2
// CHECK:STDOUT:     %specific_impl_fn.loc25_3.2: <specific function> = specific_impl_function %impl.elem0.loc25_3.2, @Destroy.WithSelf.Op(constants.%Destroy.facet.688) [symbolic = %specific_impl_fn.loc26_6.4 (constants.%specific_impl_fn.089)]
// CHECK:STDOUT:     %bound_method.loc25_3.4: <bound method> = bound_method %.loc25_3.6, %specific_impl_fn.loc25_3.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc25: init %empty_tuple.type = call %bound_method.loc25_3.4(%.loc25_3.6)
// CHECK:STDOUT:     %impl.elem0.loc22_3.2: @F.%.loc22_3.6 (%.11d) = impl_witness_access constants.%as_type.846, element0 [symbolic = %impl.elem0.loc22_3.4 (constants.%impl.elem0.342)]
// CHECK:STDOUT:     %bound_method.loc22_3.3: <bound method> = bound_method %b.var, %impl.elem0.loc22_3.2
// CHECK:STDOUT:     %specific_impl_fn.loc22_3.2: <specific function> = specific_impl_function %impl.elem0.loc22_3.2, @Destroy.WithSelf.Op(constants.%Destroy.facet.f83) [symbolic = %specific_impl_fn.loc22_3.4 (constants.%specific_impl_fn.97f)]
// CHECK:STDOUT:     %bound_method.loc22_3.4: <bound method> = bound_method %b.var, %specific_impl_fn.loc22_3.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22_3.4(%b.var)
// CHECK:STDOUT:     %impl.elem0.loc20_3.1: @F.%.loc20_3.2 (%.4d9) = impl_witness_access constants.%as_type.e31, element0 [symbolic = %impl.elem0.loc20_3.2 (constants.%impl.elem0.e40)]
// CHECK:STDOUT:     %bound_method.loc20_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc20_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc20_3.1: <specific function> = specific_impl_function %impl.elem0.loc20_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.54e) [symbolic = %specific_impl_fn.loc20_3.2 (constants.%specific_impl_fn.bdf)]
// CHECK:STDOUT:     %bound_method.loc20_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc20_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc20: init %empty_tuple.type = call %bound_method.loc20_3.2(%a.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P(constants.%Self.d21) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc4_35 => constants.%Self.d21
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
// CHECK:STDOUT: specific @Q(constants.%Self.f64) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5_35 => constants.%Self.f64
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.377
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.2bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.f64) {
// CHECK:STDOUT:   %Self => constants.%Self.f64
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.1b2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R(constants.%Self.bea) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc6_35 => constants.%Self.bea
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type.f8a
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR.b79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%Self.bea) {
// CHECK:STDOUT:   %Self => constants.%Self.bea
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.296
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.351
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive(constants.%Self.e2a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc10 => constants.%Self.e2a
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.9bd
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.4e4
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.bde
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.e53
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.1ad
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.9f1
// CHECK:STDOUT:   %as_type.loc16_16 => constants.%as_type.79f
// CHECK:STDOUT:   %as_type.loc16_11 => constants.%as_type.0ea
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.8f7
// CHECK:STDOUT:   %as_type.loc16_21 => constants.%as_type.ecc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e5b
// CHECK:STDOUT:   %pattern_type.loc13_11 => constants.%pattern_type.b0d
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem0.loc13_28.1 => constants.%impl.elem0.9bd
// CHECK:STDOUT:   %as_type => constants.%as_type.0ea
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.7a7
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.bff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e5b
// CHECK:STDOUT:   %pattern_type.loc14_13 => constants.%pattern_type.b0d
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem2.loc14_28.1 => constants.%impl.elem2.8f7
// CHECK:STDOUT:   %as_type => constants.%as_type.ecc
// CHECK:STDOUT:   %pattern_type.loc14_25 => constants.%pattern_type.364
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc19_6.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc19 => constants.%pattern_type.993
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc10 => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.b09
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.32a
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.dc6
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.69d
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.eda
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.67d
// CHECK:STDOUT:   %as_type.loc16_16 => constants.%as_type.846
// CHECK:STDOUT:   %as_type.loc16_11 => constants.%as_type.e31
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.458
// CHECK:STDOUT:   %as_type.loc16_21 => constants.%as_type.c65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc13_11 => constants.%pattern_type.993
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8
// CHECK:STDOUT:   %impl.elem0.loc13_28.1 => constants.%impl.elem0.b09
// CHECK:STDOUT:   %as_type => constants.%as_type.e31
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.1b9
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.05e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q(constants.%impl.elem1.32a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5_35 => constants.%impl.elem1.32a
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.240
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.5ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q(constants.%Q.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5_35 => constants.%Q.facet
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.a66
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.bcd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Q.facet) {
// CHECK:STDOUT:   %Self => constants.%Q.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%as_type.846
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R(constants.%impl.elem2.458) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc6_35 => constants.%impl.elem2.458
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type.fbb
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR.b14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R(constants.%R.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc6_35 => constants.%R.facet
// CHECK:STDOUT:   %R.WithSelf.InR.type => constants.%R.WithSelf.InR.type.63c
// CHECK:STDOUT:   %R.WithSelf.InR => constants.%R.WithSelf.InR.1a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.WithSelf.InR(constants.%R.facet) {
// CHECK:STDOUT:   %Self => constants.%R.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%as_type.c65
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7db
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %.loc8_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = value_param call_param0
// CHECK:STDOUT:     %.loc8_31.1: type = splice_block %.loc8_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc8_14.2 [symbolic = %T.loc8_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:       %.loc8_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7f1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.901) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.901) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_10: type = splice_block %C.ref [concrete = constants.%C.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %C.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.94e)]
// CHECK:STDOUT:     %x.param: @F.%T.binding.as_type (%T.binding.as_type.d92) = value_param call_param0
// CHECK:STDOUT:     %.loc9_16.1: type = splice_block %.loc9_16.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)] {
// CHECK:STDOUT:       %T.ref: %C.type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T.94e)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:       %.loc9_16.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.binding.as_type (%T.binding.as_type.d92) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   interface_with_self_decl @A [concrete]
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
// CHECK:STDOUT:   interface_with_self_decl @B [concrete]
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
// CHECK:STDOUT:   interface_with_self_decl @C [concrete]
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
// CHECK:STDOUT: generic fn @RequiresA(%T.loc8_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc8_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
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
// CHECK:STDOUT: generic fn @F(%T.loc9_6.2: %C.type) {
// CHECK:STDOUT:   %T.loc9_6.1: %C.type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T.94e)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.901)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.791)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.binding.as_type (%T.binding.as_type.d92)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe, (@observe.%T.ref, @observe.%B.ref, @observe.%T.as_type, @observe.%.loc10, @observe.%impls)
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @F.%T.binding.as_type (%T.binding.as_type.d92) = name_ref x, %x
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self.d0b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.WithSelf.Self.binding.as_type.impls.A.type.require(constants.%Self.d0b) {
// CHECK:STDOUT:   %Self => constants.%Self.d0b
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.8ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self.b7c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf.Self.binding.as_type.impls.B.type.require(constants.%Self.b7c) {
// CHECK:STDOUT:   %Self => constants.%Self.b7c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.89a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T.5a3) {
// CHECK:STDOUT:   %T.loc8_14.1 => constants.%T.5a3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.94e) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T.94e
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.d92
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.901
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- equivalent_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %P.type: type = facet_type <@P> [concrete]
// CHECK:STDOUT:   %Self.d21: %P.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.6c8: type = symbolic_binding_type Self, 0, %Self.d21 [symbolic]
// CHECK:STDOUT:   %pattern_type.2eb: type = pattern_type %Self.binding.as_type.6c8 [symbolic]
// CHECK:STDOUT:   %P.WithSelf.InP.type: type = fn_type @P.WithSelf.InP, @P(%Self.d21) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %P.WithSelf.InP: %P.WithSelf.InP.type = struct_value () [symbolic]
// CHECK:STDOUT:   %P.assoc_type: type = assoc_entity_type @P [concrete]
// CHECK:STDOUT:   %assoc0.4ec: %P.assoc_type = assoc_entity element0, @P.%P.WithSelf.InP.decl [concrete]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.f64: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.1b2: type = symbolic_binding_type Self, 0, %Self.f64 [symbolic]
// CHECK:STDOUT:   %pattern_type.992: type = pattern_type %Self.binding.as_type.1b2 [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.377: type = fn_type @Q.WithSelf.InQ, @Q(%Self.f64) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.2bf: %Q.WithSelf.InQ.type.377 = struct_value () [symbolic]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.038: %Q.assoc_type = assoc_entity element0, @Q.%Q.WithSelf.InQ.decl [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.bea: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.296: type = symbolic_binding_type Self, 0, %Self.bea [symbolic]
// CHECK:STDOUT:   %pattern_type.351: type = pattern_type %Self.binding.as_type.296 [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR.type: type = fn_type @R.WithSelf.InR, @R(%Self.bea) [symbolic]
// CHECK:STDOUT:   %R.WithSelf.InR: %R.WithSelf.InR.type = struct_value () [symbolic]
// CHECK:STDOUT:   %R.assoc_type: type = assoc_entity_type @R [concrete]
// CHECK:STDOUT:   %assoc0.dce: %R.assoc_type = assoc_entity element0, @R.%R.WithSelf.InR.decl [concrete]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.e2a: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.be3: %Transitive.assoc_type = assoc_entity element0, @Transitive.%A [concrete]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.804: <witness> = lookup_impl_witness %Self.e2a, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.9bd: %P.type = impl_witness_access %Transitive.lookup_impl_witness.804, element0 [symbolic]
// CHECK:STDOUT:   %Q_where.type: type = facet_type <@Q where TODO> [concrete]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.%B [concrete]
// CHECK:STDOUT:   %impl.elem1.4e4: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.804, element1 [symbolic]
// CHECK:STDOUT:   %R_where.type: type = facet_type <@R where TODO> [concrete]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.%C [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.e5b: type = symbolic_binding_type Self, 0, %Self.e2a [symbolic]
// CHECK:STDOUT:   %pattern_type.b0d: type = pattern_type %Self.binding.as_type.e5b [symbolic]
// CHECK:STDOUT:   %as_type.0ea: type = facet_access_type %impl.elem0.9bd [symbolic]
// CHECK:STDOUT:   %.7a7: Core.Form = init_form %as_type.0ea, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.bff: type = pattern_type %as_type.0ea [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.bde: type = fn_type @Transitive.WithSelf.GetA, @Transitive(%Self.e2a) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.e53: %Transitive.WithSelf.GetA.type.bde = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, @Transitive.%Transitive.WithSelf.GetA.decl [concrete]
// CHECK:STDOUT:   %impl.elem2.8f7: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.804, element2 [symbolic]
// CHECK:STDOUT:   %as_type.ecc: type = facet_access_type %impl.elem2.8f7 [symbolic]
// CHECK:STDOUT:   %pattern_type.364: type = pattern_type %as_type.ecc [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.1ad: type = fn_type @Transitive.WithSelf.TakesC, @Transitive(%Self.e2a) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.9f1: %Transitive.WithSelf.TakesC.type.1ad = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, @Transitive.%Transitive.WithSelf.TakesC.decl [concrete]
// CHECK:STDOUT:   %as_type.79f: type = facet_access_type %impl.elem1.4e4 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4bd: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4bd, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.aef: <bound method> = bound_method %P.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.dec: type = facet_type <@P & @Q> [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound.cf2: <bound method> = bound_method %facet_type.dec, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.7a2: type = facet_type <@P & @Q & @R> [concrete]
// CHECK:STDOUT:   %U: %facet_type.7a2 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3de: type = pattern_type %facet_type.7a2 [concrete]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.885: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %TakesPQR.type: type = fn_type @TakesPQR [concrete]
// CHECK:STDOUT:   %TakesPQR: %TakesPQR.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.0b3: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %T: %Transitive.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.18b: type = pattern_type %Transitive.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.993: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1c0: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.6d8: <witness> = lookup_impl_witness %T, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.b09: %P.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.32a: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element1 [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type.dc6: type = fn_type @Transitive.WithSelf.GetA, @Transitive(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.69d: %Transitive.WithSelf.GetA.type.dc6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type.eda: type = fn_type @Transitive.WithSelf.TakesC, @Transitive(%T) [symbolic]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.67d: %Transitive.WithSelf.TakesC.type.eda = struct_value () [symbolic]
// CHECK:STDOUT:   %as_type.846: type = facet_access_type %impl.elem1.32a [symbolic]
// CHECK:STDOUT:   %as_type.e31: type = facet_access_type %impl.elem0.b09 [symbolic]
// CHECK:STDOUT:   %impl.elem2.458: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element2 [symbolic]
// CHECK:STDOUT:   %as_type.c65: type = facet_access_type %impl.elem2.458 [symbolic]
// CHECK:STDOUT:   %require_complete.b46: <witness> = require_complete_type %as_type.e31 [symbolic]
// CHECK:STDOUT:   %pattern_type.05e: type = pattern_type %as_type.e31 [symbolic]
// CHECK:STDOUT:   %.c0a: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type.dc6, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.c0a = impl_witness_access %Transitive.lookup_impl_witness.6d8, element3 [symbolic]
// CHECK:STDOUT:   %.1b9: Core.Form = init_form %as_type.e31, call_param1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c79: <specific function> = specific_impl_function %impl.elem3, @Transitive.WithSelf.GetA(%T) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %as_type.b6a: %Q_where.type = facet_access_type %impl.elem0.b09 [symbolic]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.e31, (%as_type.b6a) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type.177: type = fn_type @Q.WithSelf.InQ, @Q(%Q.facet) [symbolic]
// CHECK:STDOUT:   %Q.WithSelf.InQ.24a: %Q.WithSelf.InQ.type.177 = struct_value () [symbolic]
// CHECK:STDOUT:   %.d5e: type = fn_type_with_self_type %Q.WithSelf.InQ.type.177, %Q.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.43b: %.d5e = impl_witness_access %as_type.b6a, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.909: <specific function> = specific_impl_function %impl.elem0.43b, @Q.WithSelf.InQ(%Q.facet) [symbolic]
// CHECK:STDOUT:   %P.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.b09, @P [symbolic]
// CHECK:STDOUT:   %as_type.3e7: %R_where.type = facet_access_type %impl.elem0.b09 [symbolic]
// CHECK:STDOUT:   %facet_value: %facet_type.7a2 = facet_value %as_type.e31, (%P.lookup_impl_witness, %as_type.b6a, %as_type.3e7) [symbolic]
// CHECK:STDOUT:   %TakesPQR.specific_fn: <specific function> = specific_function %TakesPQR, @TakesPQR(%facet_value) [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.e31, (%as_type.e31) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7ab: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.4d9: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7ab, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.e40: %.4d9 = impl_witness_access %as_type.e31, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdf: <specific function> = specific_impl_function %impl.elem0.e40, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.8d3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.8d3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .P = %P.decl
// CHECK:STDOUT:     .Q = %Q.decl
// CHECK:STDOUT:     .R = %R.decl
// CHECK:STDOUT:     .Transitive = %Transitive.decl
// CHECK:STDOUT:     .TakesPQR = %TakesPQR.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %P.decl: type = interface_decl @P [concrete = constants.%P.type] {} {}
// CHECK:STDOUT:   %Q.decl: type = interface_decl @Q [concrete = constants.%Q.type] {} {}
// CHECK:STDOUT:   %R.decl: type = interface_decl @R [concrete = constants.%R.type] {} {}
// CHECK:STDOUT:   %Transitive.decl: type = interface_decl @Transitive [concrete = constants.%Transitive.type] {} {}
// CHECK:STDOUT:   %TakesPQR.decl: %TakesPQR.type = fn_decl @TakesPQR [concrete = constants.%TakesPQR] {
// CHECK:STDOUT:     %U.patt: %pattern_type.3de = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %u.patt: @TakesPQR.%pattern_type (%pattern_type.885) = value_binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: @TakesPQR.%pattern_type (%pattern_type.885) = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19_23.1: type = splice_block %.loc19_23.3 [concrete = constants.%facet_type.7a2] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %P.ref: type = name_ref P, file.%P.decl [concrete = constants.%P.type]
// CHECK:STDOUT:       %Q.ref: type = name_ref Q, file.%Q.decl [concrete = constants.%Q.type]
// CHECK:STDOUT:       %impl.elem0.loc19_19: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc19_19: <bound method> = bound_method %P.ref, %impl.elem0.loc19_19 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.aef]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc19_19: init type = call %bound_method.loc19_19(%P.ref, %Q.ref) [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %R.ref: type = name_ref R, file.%R.decl [concrete = constants.%R.type]
// CHECK:STDOUT:       %impl.elem0.loc19_23: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc19_23: <bound method> = bound_method %type.as.BitAndWith.impl.Op.call.loc19_19, %impl.elem0.loc19_23 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.cf2]
// CHECK:STDOUT:       %.loc19_19.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc19_19 [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %.loc19_19.2: type = converted %type.as.BitAndWith.impl.Op.call.loc19_19, %.loc19_19.1 [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc19_23: init type = call %bound_method.loc19_23(%.loc19_19.2, %R.ref) [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:       %.loc19_23.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc19_23 [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:       %.loc19_23.3: type = converted %type.as.BitAndWith.impl.Op.call.loc19_23, %.loc19_23.2 [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc19_13.2: %facet_type.7a2 = symbolic_binding U, 0 [symbolic = %U.loc19_13.1 (constants.%U)]
// CHECK:STDOUT:     %u.param: @TakesPQR.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc19_38.1: type = splice_block %.loc19_38.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %facet_type.7a2 = name_ref U, %U.loc19_13.2 [symbolic = %U.loc19_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc19_38.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @TakesPQR.%U.binding.as_type (%U.binding.as_type) = value_binding u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.18b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %t.patt: @G.%pattern_type.loc21 (%pattern_type.993) = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: @G.%pattern_type.loc21 (%pattern_type.993) = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_10: type = splice_block %Transitive.ref [concrete = constants.%Transitive.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %Transitive.ref: type = name_ref Transitive, file.%Transitive.decl [concrete = constants.%Transitive.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_6.2: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @G.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc21_25.1: type = splice_block %.loc21_25.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc21: %Transitive.type = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc21: type = facet_access_type %T.ref.loc21 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc21_25.2: type = converted %T.ref.loc21, %T.as_type.loc21 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @G.%T.binding.as_type (%T.binding.as_type) = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @P {
// CHECK:STDOUT:   %Self.loc4_13: %P.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d21]
// CHECK:STDOUT:   interface_with_self_decl @P [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %P.WithSelf.InP.decl: @P.%P.WithSelf.InP.type (%P.WithSelf.InP.type) = fn_decl @P.WithSelf.InP [symbolic = @P.%P.WithSelf.InP (constants.%P.WithSelf.InP)] {
// CHECK:STDOUT:     %self.patt: @P.WithSelf.InP.%pattern_type (%pattern_type.2eb) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @P.WithSelf.InP.%pattern_type (%pattern_type.2eb) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @P.WithSelf.InP.%Self.binding.as_type (%Self.binding.as_type.6c8) = value_param call_param0
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)] {
// CHECK:STDOUT:       %Self.ref: %P.type = name_ref Self, @P.%Self.loc4_13 [symbolic = %Self (constants.%Self.d21)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:       %.loc4_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @P.WithSelf.InP.%Self.binding.as_type (%Self.binding.as_type.6c8) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %P.assoc_type = assoc_entity element0, %P.WithSelf.InP.decl [concrete = constants.%assoc0.4ec]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc4_13
// CHECK:STDOUT:   .InP = %assoc0
// CHECK:STDOUT:   witness = (%P.WithSelf.InP.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q {
// CHECK:STDOUT:   %Self.loc5_13: %Q.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f64]
// CHECK:STDOUT:   interface_with_self_decl @Q [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Q.WithSelf.InQ.decl: @Q.%Q.WithSelf.InQ.type (%Q.WithSelf.InQ.type.377) = fn_decl @Q.WithSelf.InQ [symbolic = @Q.%Q.WithSelf.InQ (constants.%Q.WithSelf.InQ.2bf)] {
// CHECK:STDOUT:     %self.patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.992) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Q.WithSelf.InQ.%pattern_type (%pattern_type.992) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Q.WithSelf.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2) = value_param call_param0
// CHECK:STDOUT:     %.loc5_28.1: type = splice_block %.loc5_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)] {
// CHECK:STDOUT:       %Self.ref: %Q.type = name_ref Self, @Q.%Self.loc5_13 [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:       %.loc5_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Q.WithSelf.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Q.assoc_type = assoc_entity element0, %Q.WithSelf.InQ.decl [concrete = constants.%assoc0.038]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc5_13
// CHECK:STDOUT:   .InQ = %assoc0
// CHECK:STDOUT:   witness = (%Q.WithSelf.InQ.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R {
// CHECK:STDOUT:   %Self.loc6_13: %R.type = symbolic_binding Self, 0 [symbolic = constants.%Self.bea]
// CHECK:STDOUT:   interface_with_self_decl @R [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %R.WithSelf.InR.decl: @R.%R.WithSelf.InR.type (%R.WithSelf.InR.type) = fn_decl @R.WithSelf.InR [symbolic = @R.%R.WithSelf.InR (constants.%R.WithSelf.InR)] {
// CHECK:STDOUT:     %self.patt: @R.WithSelf.InR.%pattern_type (%pattern_type.351) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @R.WithSelf.InR.%pattern_type (%pattern_type.351) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @R.WithSelf.InR.%Self.binding.as_type (%Self.binding.as_type.296) = value_param call_param0
// CHECK:STDOUT:     %.loc6_28.1: type = splice_block %.loc6_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)] {
// CHECK:STDOUT:       %Self.ref: %R.type = name_ref Self, @R.%Self.loc6_13 [symbolic = %Self (constants.%Self.bea)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:       %.loc6_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @R.WithSelf.InR.%Self.binding.as_type (%Self.binding.as_type.296) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %R.assoc_type = assoc_entity element0, %R.WithSelf.InR.decl [concrete = constants.%assoc0.dce]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc6_13
// CHECK:STDOUT:   .InR = %assoc0
// CHECK:STDOUT:   witness = (%R.WithSelf.InR.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive {
// CHECK:STDOUT:   %Self.loc8: %Transitive.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e2a]
// CHECK:STDOUT:   interface_with_self_decl @Transitive [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A: %P.type = assoc_const_decl @A [concrete] {
// CHECK:STDOUT:     %assoc0: %Transitive.assoc_type = assoc_entity element0, @Transitive.%A [concrete = constants.%assoc0.be3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: %Q_where.type = assoc_const_decl @B [concrete] {
// CHECK:STDOUT:     %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.%B [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: %R_where.type = assoc_const_decl @C [concrete] {
// CHECK:STDOUT:     %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.%C [concrete = constants.%assoc2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.decl: @Transitive.%Transitive.WithSelf.GetA.type (%Transitive.WithSelf.GetA.type.bde) = fn_decl @Transitive.WithSelf.GetA [symbolic = @Transitive.%Transitive.WithSelf.GetA (constants.%Transitive.WithSelf.GetA.e53)] {
// CHECK:STDOUT:     %self.patt: @Transitive.WithSelf.GetA.%pattern_type.loc13_11 (%pattern_type.b0d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Transitive.WithSelf.GetA.%pattern_type.loc13_11 (%pattern_type.b0d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Transitive.WithSelf.GetA.%pattern_type.loc13_25 (%pattern_type.bff) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Transitive.WithSelf.GetA.%pattern_type.loc13_25 (%pattern_type.bff) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc13_28.2: %P.type = impl_witness_access constants.%Transitive.lookup_impl_witness.804, element0 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %A.ref: %P.type = name_ref A, %impl.elem0.loc13_28.2 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %A.as_type: type = facet_access_type %A.ref [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:     %.loc13_28.2: type = converted %A.ref, %A.as_type [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:     %.loc13_28.3: Core.Form = init_form %.loc13_28.2, call_param1 [symbolic = %.loc13_28.1 (constants.%.7a7)]
// CHECK:STDOUT:     %self.param: @Transitive.WithSelf.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_param call_param0
// CHECK:STDOUT:     %.loc13_17.1: type = splice_block %.loc13_17.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self.loc8 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:       %.loc13_17.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.WithSelf.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Transitive.WithSelf.GetA.%as_type (%as_type.0ea) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Transitive.WithSelf.GetA.%as_type (%as_type.0ea) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, %Transitive.WithSelf.GetA.decl [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.decl: @Transitive.%Transitive.WithSelf.TakesC.type (%Transitive.WithSelf.TakesC.type.1ad) = fn_decl @Transitive.WithSelf.TakesC [symbolic = @Transitive.%Transitive.WithSelf.TakesC (constants.%Transitive.WithSelf.TakesC.9f1)] {
// CHECK:STDOUT:     %self.patt: @Transitive.WithSelf.TakesC.%pattern_type.loc14_13 (%pattern_type.b0d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.loc14_13 (%pattern_type.b0d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %c.patt: @Transitive.WithSelf.TakesC.%pattern_type.loc14_25 (%pattern_type.364) = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @Transitive.WithSelf.TakesC.%pattern_type.loc14_25 (%pattern_type.364) = value_param_pattern %c.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Transitive.WithSelf.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self.loc8 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:       %.loc14_19.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.WithSelf.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_binding self, %self.param
// CHECK:STDOUT:     %c.param: @Transitive.WithSelf.TakesC.%as_type (%as_type.ecc) = value_param call_param1
// CHECK:STDOUT:     %.loc14_28.1: type = splice_block %.loc14_28.2 [symbolic = %as_type (constants.%as_type.ecc)] {
// CHECK:STDOUT:       %impl.elem2.loc14_28.2: %R_where.type = impl_witness_access constants.%Transitive.lookup_impl_witness.804, element2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:       %C.ref: %R_where.type = name_ref C, %impl.elem2.loc14_28.2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:       %C.as_type: type = facet_access_type %C.ref [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:       %.loc14_28.2: type = converted %C.ref, %C.as_type [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @Transitive.WithSelf.TakesC.%as_type (%as_type.ecc) = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, %Transitive.WithSelf.TakesC.decl [concrete = constants.%assoc4]
// CHECK:STDOUT:   %Transitive.WithSelf.observe.decl = observe_decl @Transitive.WithSelf.observe, (@Transitive.WithSelf.observe.%impl.elem0, @Transitive.WithSelf.observe.%A.ref, @Transitive.WithSelf.observe.%impl.elem1, @Transitive.WithSelf.observe.%B.ref, @Transitive.WithSelf.observe.%B.as_type.loc16_16.1, @Transitive.WithSelf.observe.%.loc16_16.1, @Transitive.WithSelf.observe.%A.as_type, @Transitive.WithSelf.observe.%.loc16_11, @Transitive.WithSelf.observe.%eq.loc16_13, @Transitive.WithSelf.observe.%impl.elem2, @Transitive.WithSelf.observe.%C.ref, @Transitive.WithSelf.observe.%C.as_type, @Transitive.WithSelf.observe.%.loc16_21, @Transitive.WithSelf.observe.%B.as_type.loc16_16.2, @Transitive.WithSelf.observe.%.loc16_16.2, @Transitive.WithSelf.observe.%eq.loc16_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc8
// CHECK:STDOUT:   .P = <poisoned>
// CHECK:STDOUT:   .Q = <poisoned>
// CHECK:STDOUT:   .R = <poisoned>
// CHECK:STDOUT:   .P = <poisoned>
// CHECK:STDOUT:   .A = @A.%assoc0
// CHECK:STDOUT:   .Q = <poisoned>
// CHECK:STDOUT:   .B = @B.%assoc1
// CHECK:STDOUT:   .R = <poisoned>
// CHECK:STDOUT:   .C = @C.%assoc2
// CHECK:STDOUT:   .GetA = %assoc3
// CHECK:STDOUT:   .TakesC = %assoc4
// CHECK:STDOUT:   witness = (%A, %B, %C, %Transitive.WithSelf.GetA.decl, %Transitive.WithSelf.TakesC.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.WithSelf.InP(@P.%Self.loc4_13: %P.type) {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d21)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @P.WithSelf.InP.%Self.binding.as_type (%Self.binding.as_type.6c8));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.WithSelf.InQ(@Q.%Self.loc5_13: %Q.type) {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.992)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Q.WithSelf.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.WithSelf.InR(@R.%Self.loc6_13: %R.type) {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.bea)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.351)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @R.WithSelf.InR.%Self.binding.as_type (%Self.binding.as_type.296));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.GetA(@Transitive.%Self.loc8: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:   %pattern_type.loc13_11: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc13_11 (constants.%pattern_type.b0d)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem0.loc13_28.1: %P.type = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc13_28.1 [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:   %.loc13_28.1: Core.Form = init_form %as_type, call_param1 [symbolic = %.loc13_28.1 (constants.%.7a7)]
// CHECK:STDOUT:   %pattern_type.loc13_25: type = pattern_type %as_type [symbolic = %pattern_type.loc13_25 (constants.%pattern_type.bff)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.WithSelf.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b)) -> out %return.param: @Transitive.WithSelf.GetA.%as_type (%as_type.0ea);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.WithSelf.TakesC(@Transitive.%Self.loc8: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:   %pattern_type.loc14_13: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc14_13 (constants.%pattern_type.b0d)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem2.loc14_28.1: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2.loc14_28.1 [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:   %pattern_type.loc14_25: type = pattern_type %as_type [symbolic = %pattern_type.loc14_25 (constants.%pattern_type.364)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.WithSelf.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b), %c.param: @Transitive.WithSelf.TakesC.%as_type (%as_type.ecc));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TakesPQR(%U.loc19_13.2: %facet_type.7a2) {
// CHECK:STDOUT:   %U.loc19_13.1: %facet_type.7a2 = symbolic_binding U, 0 [symbolic = %U.loc19_13.1 (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc19_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
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
// CHECK:STDOUT: generic fn @G(%T.loc21_6.2: %Transitive.type) {
// CHECK:STDOUT:   %T.loc21_6.1: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc21_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc21: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc21 (constants.%pattern_type.993)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc21 (constants.%require_complete.1c0)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc21_6.1, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.6d8)]
// CHECK:STDOUT:   %impl.elem0.loc22_11.2: %P.type = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.b09)]
// CHECK:STDOUT:   %as_type.loc22_11.2: type = facet_access_type %impl.elem0.loc22_11.2 [symbolic = %as_type.loc22_11.2 (constants.%as_type.e31)]
// CHECK:STDOUT:   %require_complete.loc22: <witness> = require_complete_type %as_type.loc22_11.2 [symbolic = %require_complete.loc22 (constants.%require_complete.b46)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %as_type.loc22_11.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.05e)]
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type: type = fn_type @Transitive.WithSelf.GetA, @Transitive(%T.loc21_6.1) [symbolic = %Transitive.WithSelf.GetA.type (constants.%Transitive.WithSelf.GetA.type.dc6)]
// CHECK:STDOUT:   %.loc22_17: type = fn_type_with_self_type %Transitive.WithSelf.GetA.type, %T.loc21_6.1 [symbolic = %.loc22_17 (constants.%.c0a)]
// CHECK:STDOUT:   %impl.elem3.loc22_17.2: @G.%.loc22_17 (%.c0a) = impl_witness_access %Transitive.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc22_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_17.2: <specific function> = specific_impl_function %impl.elem3.loc22_17.2, @Transitive.WithSelf.GetA(%T.loc21_6.1) [symbolic = %specific_impl_fn.loc22_17.2 (constants.%specific_impl_fn.c79)]
// CHECK:STDOUT:   %impl.elem1: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1 (constants.%impl.elem1.32a)]
// CHECK:STDOUT:   %as_type.loc24: type = facet_access_type %impl.elem1 [symbolic = %as_type.loc24 (constants.%as_type.846)]
// CHECK:STDOUT:   %as_type.loc25: %Q_where.type = facet_access_type %impl.elem0.loc22_11.2 [symbolic = %as_type.loc25 (constants.%as_type.b6a)]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.loc22_11.2, (%as_type.loc25) [symbolic = %Q.facet (constants.%Q.facet)]
// CHECK:STDOUT:   %Q.WithSelf.InQ.type: type = fn_type @Q.WithSelf.InQ, @Q(%Q.facet) [symbolic = %Q.WithSelf.InQ.type (constants.%Q.WithSelf.InQ.type.177)]
// CHECK:STDOUT:   %.loc25_4.2: type = fn_type_with_self_type %Q.WithSelf.InQ.type, %Q.facet [symbolic = %.loc25_4.2 (constants.%.d5e)]
// CHECK:STDOUT:   %impl.elem0.loc25_4.2: @G.%.loc25_4.2 (%.d5e) = impl_witness_access %as_type.loc25, element0 [symbolic = %impl.elem0.loc25_4.2 (constants.%impl.elem0.43b)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_4.2: <specific function> = specific_impl_function %impl.elem0.loc25_4.2, @Q.WithSelf.InQ(%Q.facet) [symbolic = %specific_impl_fn.loc25_4.2 (constants.%specific_impl_fn.909)]
// CHECK:STDOUT:   %impl.elem2: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2 (constants.%impl.elem2.458)]
// CHECK:STDOUT:   %as_type.loc27: type = facet_access_type %impl.elem2 [symbolic = %as_type.loc27 (constants.%as_type.c65)]
// CHECK:STDOUT:   %P.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc22_11.2, @P [symbolic = %P.lookup_impl_witness (constants.%P.lookup_impl_witness)]
// CHECK:STDOUT:   %as_type.loc28: %R_where.type = facet_access_type %impl.elem0.loc22_11.2 [symbolic = %as_type.loc28 (constants.%as_type.3e7)]
// CHECK:STDOUT:   %facet_value.loc28_13.3: %facet_type.7a2 = facet_value %as_type.loc22_11.2, (%P.lookup_impl_witness, %as_type.loc25, %as_type.loc28) [symbolic = %facet_value.loc28_13.3 (constants.%facet_value)]
// CHECK:STDOUT:   %TakesPQR.specific_fn.loc28_3.2: <specific function> = specific_function constants.%TakesPQR, @TakesPQR(%facet_value.loc28_13.3) [symbolic = %TakesPQR.specific_fn.loc28_3.2 (constants.%TakesPQR.specific_fn)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.loc22_11.2, (%as_type.loc22_11.2) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.7ab)]
// CHECK:STDOUT:   %.loc22_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc22_3.2 (constants.%.4d9)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.2: @G.%.loc22_3.2 (%.4d9) = impl_witness_access %as_type.loc22_11.2, element0 [symbolic = %impl.elem0.loc22_3.2 (constants.%impl.elem0.e40)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_3.2: <specific function> = specific_impl_function %impl.elem0.loc22_3.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc22_3.2 (constants.%specific_impl_fn.bdf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @G.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: @G.%pattern_type.loc22 (%pattern_type.05e) = ref_binding_pattern a [concrete]
// CHECK:STDOUT:       %a.var_patt: @G.%pattern_type.loc22 (%pattern_type.05e) = var_pattern %a.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a.var: ref @G.%as_type.loc22_11.2 (%as_type.e31) = var %a.var_patt
// CHECK:STDOUT:     %t.ref: @G.%T.binding.as_type (%T.binding.as_type) = name_ref t, %t
// CHECK:STDOUT:     %GetA.ref: %Transitive.assoc_type = name_ref GetA, @Transitive.%assoc3 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc22_17.1: @G.%.loc22_17 (%.c0a) = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element3 [symbolic = %impl.elem3.loc22_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc22_17: <bound method> = bound_method %t.ref, %impl.elem3.loc22_17.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_17.1: <specific function> = specific_impl_function %impl.elem3.loc22_17.1, @Transitive.WithSelf.GetA(constants.%T) [symbolic = %specific_impl_fn.loc22_17.2 (constants.%specific_impl_fn.c79)]
// CHECK:STDOUT:     %bound_method.loc22_23: <bound method> = bound_method %t.ref, %specific_impl_fn.loc22_17.1
// CHECK:STDOUT:     %.loc22_3.1: ref @G.%as_type.loc22_11.2 (%as_type.e31) = splice_block %a.var {}
// CHECK:STDOUT:     %Transitive.WithSelf.GetA.call: init @G.%as_type.loc22_11.2 (%as_type.e31) to %.loc22_3.1 = call %bound_method.loc22_23(%t.ref)
// CHECK:STDOUT:     assign %a.var, %Transitive.WithSelf.GetA.call
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %.loc22_11.3 [symbolic = %as_type.loc22_11.2 (constants.%as_type.e31)] {
// CHECK:STDOUT:       %T.ref.loc22: %Transitive.type = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc22_11.2: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %A.ref: %Transitive.assoc_type = name_ref A, @A.%assoc0 [concrete = constants.%assoc0.be3]
// CHECK:STDOUT:       %impl.elem0.loc22_11.1: %P.type = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.b09)]
// CHECK:STDOUT:       %as_type.loc22_11.1: type = facet_access_type %impl.elem0.loc22_11.1 [symbolic = %as_type.loc22_11.2 (constants.%as_type.e31)]
// CHECK:STDOUT:       %.loc22_11.3: type = converted %impl.elem0.loc22_11.1, %as_type.loc22_11.1 [symbolic = %as_type.loc22_11.2 (constants.%as_type.e31)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: ref @G.%as_type.loc22_11.2 (%as_type.e31) = ref_binding a, %a.var
// CHECK:STDOUT:     %observe.decl.loc24 = observe_decl @observe.loc24, (@observe.loc24.%T.ref.loc24_11, @observe.loc24.%T.as_type.loc24_12, @observe.loc24.%.loc24_12.1, @observe.loc24.%A.ref, @observe.loc24.%impl.elem0, @observe.loc24.%T.ref.loc24_18, @observe.loc24.%T.as_type.loc24_19, @observe.loc24.%.loc24_19.1, @observe.loc24.%B.ref, @observe.loc24.%impl.elem1, @observe.loc24.%as_type.loc24_19.1, @observe.loc24.%.loc24_19.2, @observe.loc24.%as_type.loc24_12, @observe.loc24.%.loc24_12.2, @observe.loc24.%eq, @observe.loc24.%Q.ref, @observe.loc24.%as_type.loc24_19.2, @observe.loc24.%.loc24_19.3, @observe.loc24.%impls)
// CHECK:STDOUT:     %a.ref.loc25: ref @G.%as_type.loc22_11.2 (%as_type.e31) = name_ref a, %a
// CHECK:STDOUT:     %Q.ref: type = name_ref Q, file.%Q.decl [concrete = constants.%Q.type]
// CHECK:STDOUT:     %InQ.ref: %Q.assoc_type = name_ref InQ, @Q.%assoc0 [concrete = constants.%assoc0.038]
// CHECK:STDOUT:     %.loc25_4.1: %Q_where.type = as_compatible constants.%as_type.e31 [symbolic = %as_type.loc25 (constants.%as_type.b6a)]
// CHECK:STDOUT:     %impl.elem0.loc25_4.1: @G.%.loc25_4.2 (%.d5e) = impl_witness_access %.loc25_4.1, element0 [symbolic = %impl.elem0.loc25_4.2 (constants.%impl.elem0.43b)]
// CHECK:STDOUT:     %bound_method.loc25_4: <bound method> = bound_method %a.ref.loc25, %impl.elem0.loc25_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_4.1: <specific function> = specific_impl_function %impl.elem0.loc25_4.1, @Q.WithSelf.InQ(constants.%Q.facet) [symbolic = %specific_impl_fn.loc25_4.2 (constants.%specific_impl_fn.909)]
// CHECK:STDOUT:     %bound_method.loc25_13: <bound method> = bound_method %a.ref.loc25, %specific_impl_fn.loc25_4.1
// CHECK:STDOUT:     %.loc25_3: @G.%as_type.loc22_11.2 (%as_type.e31) = acquire_value %a.ref.loc25
// CHECK:STDOUT:     %Q.WithSelf.InQ.call: init %empty_tuple.type = call %bound_method.loc25_13(%.loc25_3)
// CHECK:STDOUT:     %observe.decl.loc27 = observe_decl @observe.loc27, (@observe.loc27.%T.ref.loc27_11, @observe.loc27.%T.as_type.loc27_12, @observe.loc27.%.loc27_12.1, @observe.loc27.%A.ref, @observe.loc27.%impl.elem0, @observe.loc27.%T.ref.loc27_18, @observe.loc27.%T.as_type.loc27_19, @observe.loc27.%.loc27_19.1, @observe.loc27.%C.ref, @observe.loc27.%impl.elem2, @observe.loc27.%as_type.loc27_19.1, @observe.loc27.%.loc27_19.2, @observe.loc27.%as_type.loc27_12, @observe.loc27.%.loc27_12.2, @observe.loc27.%eq, @observe.loc27.%R.ref, @observe.loc27.%as_type.loc27_19.2, @observe.loc27.%.loc27_19.3, @observe.loc27.%impls)
// CHECK:STDOUT:     %TakesPQR.ref: %TakesPQR.type = name_ref TakesPQR, file.%TakesPQR.decl [concrete = constants.%TakesPQR]
// CHECK:STDOUT:     %a.ref.loc28: ref @G.%as_type.loc22_11.2 (%as_type.e31) = name_ref a, %a
// CHECK:STDOUT:     %.loc28_13.1: %Q_where.type = as_compatible constants.%as_type.e31 [symbolic = %as_type.loc25 (constants.%as_type.b6a)]
// CHECK:STDOUT:     %.loc28_13.2: %R_where.type = as_compatible constants.%as_type.e31 [symbolic = %as_type.loc28 (constants.%as_type.3e7)]
// CHECK:STDOUT:     %facet_value.loc28_13.1: %facet_type.7a2 = facet_value constants.%as_type.e31, (constants.%P.lookup_impl_witness, %.loc28_13.1, %.loc28_13.2) [symbolic = %facet_value.loc28_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc28_13.3: %facet_type.7a2 = converted constants.%as_type.e31, %facet_value.loc28_13.1 [symbolic = %facet_value.loc28_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc28_13.4: %Q_where.type = as_compatible constants.%as_type.e31 [symbolic = %as_type.loc25 (constants.%as_type.b6a)]
// CHECK:STDOUT:     %.loc28_13.5: %R_where.type = as_compatible constants.%as_type.e31 [symbolic = %as_type.loc28 (constants.%as_type.3e7)]
// CHECK:STDOUT:     %facet_value.loc28_13.2: %facet_type.7a2 = facet_value constants.%as_type.e31, (constants.%P.lookup_impl_witness, %.loc28_13.4, %.loc28_13.5) [symbolic = %facet_value.loc28_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc28_13.6: %facet_type.7a2 = converted constants.%as_type.e31, %facet_value.loc28_13.2 [symbolic = %facet_value.loc28_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %TakesPQR.specific_fn.loc28_3.1: <specific function> = specific_function %TakesPQR.ref, @TakesPQR(constants.%facet_value) [symbolic = %TakesPQR.specific_fn.loc28_3.2 (constants.%TakesPQR.specific_fn)]
// CHECK:STDOUT:     %.loc28_12: @G.%as_type.loc22_11.2 (%as_type.e31) = acquire_value %a.ref.loc28
// CHECK:STDOUT:     %TakesPQR.call: init %empty_tuple.type = call %TakesPQR.specific_fn.loc28_3.1(%.loc28_12)
// CHECK:STDOUT:     %impl.elem0.loc22_3.1: @G.%.loc22_3.2 (%.4d9) = impl_witness_access constants.%as_type.e31, element0 [symbolic = %impl.elem0.loc22_3.2 (constants.%impl.elem0.e40)]
// CHECK:STDOUT:     %bound_method.loc22_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc22_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_3.1: <specific function> = specific_impl_function %impl.elem0.loc22_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc22_3.2 (constants.%specific_impl_fn.bdf)]
// CHECK:STDOUT:     %bound_method.loc22_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc22_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc22_3.2(%a.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P(constants.%Self.d21) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc4_35 => constants.%Self.d21
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
// CHECK:STDOUT: specific @Q(constants.%Self.f64) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5_35 => constants.%Self.f64
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.377
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.2bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Self.f64) {
// CHECK:STDOUT:   %Self => constants.%Self.f64
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.1b2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R(constants.%Self.bea) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc6_35 => constants.%Self.bea
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
// CHECK:STDOUT: specific @Transitive(constants.%Self.e2a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc10 => constants.%Self.e2a
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.9bd
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.4e4
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.bde
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.e53
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.1ad
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.9f1
// CHECK:STDOUT:   %as_type.loc16_16 => constants.%as_type.79f
// CHECK:STDOUT:   %as_type.loc16_11 => constants.%as_type.0ea
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.8f7
// CHECK:STDOUT:   %as_type.loc16_21 => constants.%as_type.ecc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e5b
// CHECK:STDOUT:   %pattern_type.loc13_11 => constants.%pattern_type.b0d
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem0.loc13_28.1 => constants.%impl.elem0.9bd
// CHECK:STDOUT:   %as_type => constants.%as_type.0ea
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.7a7
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.bff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.TakesC(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e5b
// CHECK:STDOUT:   %pattern_type.loc14_13 => constants.%pattern_type.b0d
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem2.loc14_28.1 => constants.%impl.elem2.8f7
// CHECK:STDOUT:   %as_type => constants.%as_type.ecc
// CHECK:STDOUT:   %pattern_type.loc14_25 => constants.%pattern_type.364
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesPQR(constants.%U) {
// CHECK:STDOUT:   %U.loc19_13.1 => constants.%U
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.885
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc21_6.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc21 => constants.%pattern_type.993
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc10 => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.b09
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.32a
// CHECK:STDOUT:   %Transitive.WithSelf.GetA.type => constants.%Transitive.WithSelf.GetA.type.dc6
// CHECK:STDOUT:   %Transitive.WithSelf.GetA => constants.%Transitive.WithSelf.GetA.69d
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC.type => constants.%Transitive.WithSelf.TakesC.type.eda
// CHECK:STDOUT:   %Transitive.WithSelf.TakesC => constants.%Transitive.WithSelf.TakesC.67d
// CHECK:STDOUT:   %as_type.loc16_16 => constants.%as_type.846
// CHECK:STDOUT:   %as_type.loc16_11 => constants.%as_type.e31
// CHECK:STDOUT:   %impl.elem2 => constants.%impl.elem2.458
// CHECK:STDOUT:   %as_type.loc16_21 => constants.%as_type.c65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.WithSelf.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc13_11 => constants.%pattern_type.993
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8
// CHECK:STDOUT:   %impl.elem0.loc13_28.1 => constants.%impl.elem0.b09
// CHECK:STDOUT:   %as_type => constants.%as_type.e31
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.1b9
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.05e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q(constants.%Q.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.loc5_35 => constants.%Q.facet
// CHECK:STDOUT:   %Q.WithSelf.InQ.type => constants.%Q.WithSelf.InQ.type.177
// CHECK:STDOUT:   %Q.WithSelf.InQ => constants.%Q.WithSelf.InQ.24a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.WithSelf.InQ(constants.%Q.facet) {
// CHECK:STDOUT:   %Self => constants.%Q.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%as_type.e31
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.05e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TakesPQR(constants.%facet_value) {
// CHECK:STDOUT:   %U.loc19_13.1 => constants.%facet_value
// CHECK:STDOUT:   %U.binding.as_type => constants.%as_type.e31
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.05e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.b46
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
// CHECK:STDOUT:     %T.ref: %A.type = name_ref T, %T.loc9_14.1 [symbolic = %T.loc9_14.2 (constants.%T.5a3)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:     %.loc9_21: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     %.loc9_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T.5a3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.C.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.035 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %B.type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T.5ee)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:     %.loc10_21: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     %.loc10_18: type = splice_block %B.ref [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_14.1: %B.type = symbolic_binding T, 0 [symbolic = %T.loc10_14.2 (constants.%T.5ee)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @T.binding.as_type.as.D.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7f1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %C.type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T.94e)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:     %.loc11_21: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D.type]
// CHECK:STDOUT:     %.loc11_18: type = splice_block %C.ref [concrete = constants.%C.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_14.1: %C.type = symbolic_binding T, 0 [symbolic = %T.loc11_14.2 (constants.%T.94e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RequiresB.decl: %RequiresB.type = fn_decl @RequiresB [concrete = constants.%RequiresB] {
// CHECK:STDOUT:     %T.patt: %pattern_type.035 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresB.%pattern_type (%pattern_type.fae) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresB.%pattern_type (%pattern_type.fae) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_18: type = splice_block %B.ref [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_14.2: %B.type = symbolic_binding T, 0 [symbolic = %T.loc13_14.1 (constants.%T.5ee)]
// CHECK:STDOUT:     %x.param: @RequiresB.%T.binding.as_type (%T.binding.as_type.856) = value_param call_param0
// CHECK:STDOUT:     %.loc13_31.1: type = splice_block %.loc13_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)] {
// CHECK:STDOUT:       %T.ref: %B.type = name_ref T, %T.loc13_14.2 [symbolic = %T.loc13_14.1 (constants.%T.5ee)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:       %.loc13_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresB.%T.binding.as_type (%T.binding.as_type.856) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RequiresD.decl: %RequiresD.type = fn_decl @RequiresD [concrete = constants.%RequiresD] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f05 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresD.%pattern_type (%pattern_type.d60) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresD.%pattern_type (%pattern_type.d60) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_18: type = splice_block %D.ref [concrete = constants.%D.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_14.2: %D.type = symbolic_binding T, 0 [symbolic = %T.loc14_14.1 (constants.%T.233)]
// CHECK:STDOUT:     %x.param: @RequiresD.%T.binding.as_type (%T.binding.as_type.f5f) = value_param call_param0
// CHECK:STDOUT:     %.loc14_31.1: type = splice_block %.loc14_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.f5f)] {
// CHECK:STDOUT:       %T.ref: %D.type = name_ref T, %T.loc14_14.2 [symbolic = %T.loc14_14.1 (constants.%T.233)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.f5f)]
// CHECK:STDOUT:       %.loc14_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.f5f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresD.%T.binding.as_type (%T.binding.as_type.f5f) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresA.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresA.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc16_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = value_param call_param0
// CHECK:STDOUT:     %.loc16_24.1: type = splice_block %.loc16_24.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc16_14.2 [symbolic = %T.loc16_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:       %.loc16_24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   interface_with_self_decl @A [concrete]
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
// CHECK:STDOUT:   interface_with_self_decl @B [concrete]
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
// CHECK:STDOUT:   interface_with_self_decl @C [concrete]
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
// CHECK:STDOUT:   interface_with_self_decl @D [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.B.impl(%T.loc9_14.1: %A.type) {
// CHECK:STDOUT:   %T.loc9_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc9_14.2 (constants.%T.5a3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:   %B.impl_witness.loc9_28.2: <witness> = impl_witness %B.impl_witness_table, @T.binding.as_type.as.B.impl(%T.loc9_14.2) [symbolic = %B.impl_witness.loc9_28.2 (constants.%B.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc9_21 as %B.ref {
// CHECK:STDOUT:     %B.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.B.impl [concrete]
// CHECK:STDOUT:     %B.impl_witness.loc9_28.1: <witness> = impl_witness %B.impl_witness_table, @T.binding.as_type.as.B.impl(constants.%T.5a3) [symbolic = %B.impl_witness.loc9_28.2 (constants.%B.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %B.impl_witness.loc9_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.C.impl(%T.loc10_14.1: %B.type) {
// CHECK:STDOUT:   %T.loc10_14.2: %B.type = symbolic_binding T, 0 [symbolic = %T.loc10_14.2 (constants.%T.5ee)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc10_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
// CHECK:STDOUT:   %C.impl_witness.loc10_28.2: <witness> = impl_witness %C.impl_witness_table, @T.binding.as_type.as.C.impl(%T.loc10_14.2) [symbolic = %C.impl_witness.loc10_28.2 (constants.%C.impl_witness.c1a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc10_21 as %C.ref {
// CHECK:STDOUT:     %C.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.C.impl [concrete]
// CHECK:STDOUT:     %C.impl_witness.loc10_28.1: <witness> = impl_witness %C.impl_witness_table, @T.binding.as_type.as.C.impl(constants.%T.5ee) [symbolic = %C.impl_witness.loc10_28.2 (constants.%C.impl_witness.c1a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %C.impl_witness.loc10_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.binding.as_type.as.D.impl(%T.loc11_14.1: %C.type) {
// CHECK:STDOUT:   %T.loc11_14.2: %C.type = symbolic_binding T, 0 [symbolic = %T.loc11_14.2 (constants.%T.94e)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc11_14.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:   %D.impl_witness.loc11_28.2: <witness> = impl_witness %D.impl_witness_table, @T.binding.as_type.as.D.impl(%T.loc11_14.2) [symbolic = %D.impl_witness.loc11_28.2 (constants.%D.impl_witness.c16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc11_21 as %D.ref {
// CHECK:STDOUT:     %D.impl_witness_table = impl_witness_table (), @T.binding.as_type.as.D.impl [concrete]
// CHECK:STDOUT:     %D.impl_witness.loc11_28.1: <witness> = impl_witness %D.impl_witness_table, @T.binding.as_type.as.D.impl(constants.%T.94e) [symbolic = %D.impl_witness.loc11_28.2 (constants.%D.impl_witness.c16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %D.impl_witness.loc11_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresB(%T.loc13_14.2: %B.type) {
// CHECK:STDOUT:   %T.loc13_14.1: %B.type = symbolic_binding T, 0 [symbolic = %T.loc13_14.1 (constants.%T.5ee)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc13_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.856)]
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
// CHECK:STDOUT: generic fn @RequiresD(%T.loc14_14.2: %D.type) {
// CHECK:STDOUT:   %T.loc14_14.1: %D.type = symbolic_binding T, 0 [symbolic = %T.loc14_14.1 (constants.%T.233)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc14_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.f5f)]
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
// CHECK:STDOUT: generic fn @RequiresA(%T.loc16_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc16_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc16_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc16_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.057)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1a7)]
// CHECK:STDOUT:   %.loc17_14.3: require_specific_def_type = require_specific_def @T.binding.as_type.as.B.impl(%T.loc16_14.1) [symbolic = %.loc17_14.3 (constants.%.7ae)]
// CHECK:STDOUT:   %B.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc16_14.1, @B [symbolic = %B.lookup_impl_witness (constants.%B.lookup_impl_witness)]
// CHECK:STDOUT:   %B.facet.loc17_14.3: %B.type = facet_value %T.binding.as_type, (%B.lookup_impl_witness) [symbolic = %B.facet.loc17_14.3 (constants.%B.facet.83f)]
// CHECK:STDOUT:   %RequiresB.specific_fn.loc17_3.2: <specific function> = specific_function constants.%RequiresB, @RequiresB(%B.facet.loc17_14.3) [symbolic = %RequiresB.specific_fn.loc17_3.2 (constants.%RequiresB.specific_fn)]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc16_14.1, @C [symbolic = %C.lookup_impl_witness (constants.%C.lookup_impl_witness)]
// CHECK:STDOUT:   %C.facet: %C.type = facet_value %T.binding.as_type, (%C.lookup_impl_witness) [symbolic = %C.facet (constants.%C.facet.9cd)]
// CHECK:STDOUT:   %.loc21_14.3: require_specific_def_type = require_specific_def @T.binding.as_type.as.D.impl(%C.facet) [symbolic = %.loc21_14.3 (constants.%.c21)]
// CHECK:STDOUT:   %D.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc16_14.1, @D [symbolic = %D.lookup_impl_witness (constants.%D.lookup_impl_witness)]
// CHECK:STDOUT:   %D.facet.loc21_14.3: %D.type = facet_value %T.binding.as_type, (%D.lookup_impl_witness) [symbolic = %D.facet.loc21_14.3 (constants.%D.facet.514)]
// CHECK:STDOUT:   %RequiresD.specific_fn.loc21_3.2: <specific function> = specific_function constants.%RequiresD, @RequiresD(%D.facet.loc21_14.3) [symbolic = %RequiresD.specific_fn.loc21_3.2 (constants.%RequiresD.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RequiresB.ref: %RequiresB.type = name_ref RequiresB, file.%RequiresB.decl [concrete = constants.%RequiresB]
// CHECK:STDOUT:     %x.ref.loc17: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = name_ref x, %x
// CHECK:STDOUT:     %B.facet.loc17_14.1: %B.type = facet_value constants.%T.binding.as_type.0b3, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc17_14.3 (constants.%B.facet.83f)]
// CHECK:STDOUT:     %.loc17_14.1: %B.type = converted constants.%T.binding.as_type.0b3, %B.facet.loc17_14.1 [symbolic = %B.facet.loc17_14.3 (constants.%B.facet.83f)]
// CHECK:STDOUT:     %B.facet.loc17_14.2: %B.type = facet_value constants.%T.binding.as_type.0b3, (constants.%B.lookup_impl_witness) [symbolic = %B.facet.loc17_14.3 (constants.%B.facet.83f)]
// CHECK:STDOUT:     %.loc17_14.2: %B.type = converted constants.%T.binding.as_type.0b3, %B.facet.loc17_14.2 [symbolic = %B.facet.loc17_14.3 (constants.%B.facet.83f)]
// CHECK:STDOUT:     %RequiresB.specific_fn.loc17_3.1: <specific function> = specific_function %RequiresB.ref, @RequiresB(constants.%B.facet.83f) [symbolic = %RequiresB.specific_fn.loc17_3.2 (constants.%RequiresB.specific_fn)]
// CHECK:STDOUT:     %RequiresB.call: init %empty_tuple.type = call %RequiresB.specific_fn.loc17_3.1(%x.ref.loc17)
// CHECK:STDOUT:     %observe.decl.loc19 = observe_decl @observe.loc19, (@observe.loc19.%T.ref, @observe.loc19.%B.ref, @observe.loc19.%T.as_type, @observe.loc19.%.loc19, @observe.loc19.%impls)
// CHECK:STDOUT:     %observe.decl.loc20 = observe_decl @observe.loc20, (@observe.loc20.%T.ref, @observe.loc20.%C.ref, @observe.loc20.%T.as_type, @observe.loc20.%.loc20, @observe.loc20.%impls)
// CHECK:STDOUT:     %RequiresD.ref: %RequiresD.type = name_ref RequiresD, file.%RequiresD.decl [concrete = constants.%RequiresD]
// CHECK:STDOUT:     %x.ref.loc21: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = name_ref x, %x
// CHECK:STDOUT:     %D.facet.loc21_14.1: %D.type = facet_value constants.%T.binding.as_type.0b3, (constants.%D.lookup_impl_witness) [symbolic = %D.facet.loc21_14.3 (constants.%D.facet.514)]
// CHECK:STDOUT:     %.loc21_14.1: %D.type = converted constants.%T.binding.as_type.0b3, %D.facet.loc21_14.1 [symbolic = %D.facet.loc21_14.3 (constants.%D.facet.514)]
// CHECK:STDOUT:     %D.facet.loc21_14.2: %D.type = facet_value constants.%T.binding.as_type.0b3, (constants.%D.lookup_impl_witness) [symbolic = %D.facet.loc21_14.3 (constants.%D.facet.514)]
// CHECK:STDOUT:     %.loc21_14.2: %D.type = converted constants.%T.binding.as_type.0b3, %D.facet.loc21_14.2 [symbolic = %D.facet.loc21_14.3 (constants.%D.facet.514)]
// CHECK:STDOUT:     %RequiresD.specific_fn.loc21_3.1: <specific function> = specific_function %RequiresD.ref, @RequiresD(constants.%D.facet.514) [symbolic = %RequiresD.specific_fn.loc21_3.2 (constants.%RequiresD.specific_fn)]
// CHECK:STDOUT:     %RequiresD.call: init %empty_tuple.type = call %RequiresD.specific_fn.loc21_3.1(%x.ref.loc21)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self.b7c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%Self.b8b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.B.impl(constants.%T.5a3) {
// CHECK:STDOUT:   %T.loc9_14.2 => constants.%T.5a3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %B.impl_witness.loc9_28.2 => constants.%B.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%B.facet.234) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.C.impl(constants.%T.5ee) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T.5ee
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.856
// CHECK:STDOUT:   %C.impl_witness.loc10_28.2 => constants.%C.impl_witness.c1a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%C.facet.50a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.D.impl(constants.%T.94e) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T.94e
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.d92
// CHECK:STDOUT:   %D.impl_witness.loc11_28.2 => constants.%D.impl_witness.c16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%D.facet.d95) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresB(constants.%T.5ee) {
// CHECK:STDOUT:   %T.loc13_14.1 => constants.%T.5ee
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.856
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresD(constants.%T.233) {
// CHECK:STDOUT:   %T.loc14_14.1 => constants.%T.233
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.f5f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T.5a3) {
// CHECK:STDOUT:   %T.loc16_14.1 => constants.%T.5a3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresB(constants.%B.facet.83f) {
// CHECK:STDOUT:   %T.loc13_14.1 => constants.%B.facet.83f
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1a7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.C.impl(constants.%B.facet.83f) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%B.facet.83f
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %C.impl_witness.loc10_28.2 => constants.%C.impl_witness.257
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.binding.as_type.as.D.impl(constants.%C.facet.9cd) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%C.facet.9cd
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %D.impl_witness.loc11_28.2 => constants.%D.impl_witness.555
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresD(constants.%D.facet.514) {
// CHECK:STDOUT:   %T.loc14_14.1 => constants.%D.facet.514
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
// CHECK:STDOUT:   %pattern_type.2d9: type = pattern_type %U.binding.as_type.e57 [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF [concrete]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a2d: <witness> = require_complete_type %U.binding.as_type.e57 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %U.binding.as_type.89d: %A.type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %U.binding.as_type.e57, (%U.binding.as_type.89d) [symbolic]
// CHECK:STDOUT:   %RequiresA.specific_fn: <specific function> = specific_function %RequiresA, @RequiresA(%A.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %.loc7_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_31.1: type = splice_block %.loc7_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.2: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %.Self.ref: %B.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   interface_with_self_decl @A [concrete]
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
// CHECK:STDOUT:   interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc7_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc7_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
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
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %A.type, %U.loc8_13.2: %B_where.type) {
// CHECK:STDOUT:   %T.loc8_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc8_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc8_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe, (@observe.%T.ref, @observe.%U.ref, @observe.%U.as_type, @observe.%.loc9_16, @observe.%T.as_type, @observe.%.loc9_11, @observe.%eq)
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %x.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%U.binding.as_type.loc11 (%U.binding.as_type.e57) = value_param call_param0
// CHECK:STDOUT:       %.loc11_16.1: type = splice_block %.loc11_16.2 [symbolic = %U.binding.as_type.loc11 (constants.%U.binding.as_type.e57)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc8_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type.loc11 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:         %.loc11_16.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type.loc11 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%U.binding.as_type.loc11 (%U.binding.as_type.e57) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc8_6.2: %A.type, @F.%U.loc8_13.2: %B_where.type) {
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type.loc11: type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type.loc11 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type.loc11 [symbolic = %pattern_type (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type.loc11 [symbolic = %require_complete (constants.%require_complete.a2d)]
// CHECK:STDOUT:   %U.binding.as_type.loc12: %A.type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type.loc12 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:   %A.facet.loc12_16.3: %A.type = facet_value %U.binding.as_type.loc11, (%U.binding.as_type.loc12) [symbolic = %A.facet.loc12_16.3 (constants.%A.facet)]
// CHECK:STDOUT:   %RequiresA.specific_fn.loc12_5.2: <specific function> = specific_function constants.%RequiresA, @RequiresA(%A.facet.loc12_16.3) [symbolic = %RequiresA.specific_fn.loc12_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%U.binding.as_type.loc11 (%U.binding.as_type.e57)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @InnerF.%U.binding.as_type.loc11 (%U.binding.as_type.e57) = name_ref x, %x
// CHECK:STDOUT:     %.loc12_16.1: %A.type = as_compatible constants.%U.binding.as_type.e57 [symbolic = %U.binding.as_type.loc12 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:     %A.facet.loc12_16.1: %A.type = facet_value constants.%U.binding.as_type.e57, (%.loc12_16.1) [symbolic = %A.facet.loc12_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc12_16.2: %A.type = converted constants.%U.binding.as_type.e57, %A.facet.loc12_16.1 [symbolic = %A.facet.loc12_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc12_16.3: %A.type = as_compatible constants.%U.binding.as_type.e57 [symbolic = %U.binding.as_type.loc12 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:     %A.facet.loc12_16.2: %A.type = facet_value constants.%U.binding.as_type.e57, (%.loc12_16.3) [symbolic = %A.facet.loc12_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc12_16.4: %A.type = converted constants.%U.binding.as_type.e57, %A.facet.loc12_16.2 [symbolic = %A.facet.loc12_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %RequiresA.specific_fn.loc12_5.1: <specific function> = specific_function %RequiresA.ref, @RequiresA(constants.%A.facet) [symbolic = %RequiresA.specific_fn.loc12_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:     %RequiresA.call: init %empty_tuple.type = call %RequiresA.specific_fn.loc12_5.1(%x.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc8_13.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.binding.as_type.loc11 => constants.%U.binding.as_type.e57
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%A.facet) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%A.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%U.binding.as_type.e57
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.a2d
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %U.binding.as_type.89d: %A.type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %U.binding.as_type.e57, (%U.binding.as_type.89d) [symbolic]
// CHECK:STDOUT:   %RequiresA.specific_fn: <specific function> = specific_function %RequiresA, @RequiresA(%A.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %.loc7_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_31.1: type = splice_block %.loc7_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.2: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %.Self.ref: %B.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   interface_with_self_decl @A [concrete]
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
// CHECK:STDOUT:   interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc7_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc7_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
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
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %A.type, %U.loc8_13.2: %B_where.type) {
// CHECK:STDOUT:   %T.loc8_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc8_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %x.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%U.binding.as_type.loc10 (%U.binding.as_type.e57) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %.loc10_16.2 [symbolic = %U.binding.as_type.loc10 (constants.%U.binding.as_type.e57)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc8_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type.loc10 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:         %.loc10_16.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type.loc10 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%U.binding.as_type.loc10 (%U.binding.as_type.e57) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc8_6.2: %A.type, @F.%U.loc8_13.2: %B_where.type) {
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type.loc10: type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type.loc10 (constants.%U.binding.as_type.e57)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type.loc10 [symbolic = %pattern_type (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type.loc10 [symbolic = %require_complete (constants.%require_complete.a2d)]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %U.binding.as_type.loc13: %A.type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type.loc13 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:   %A.facet.loc13_16.3: %A.type = facet_value %U.binding.as_type.loc10, (%U.binding.as_type.loc13) [symbolic = %A.facet.loc13_16.3 (constants.%A.facet)]
// CHECK:STDOUT:   %RequiresA.specific_fn.loc13_5.2: <specific function> = specific_function constants.%RequiresA, @RequiresA(%A.facet.loc13_16.3) [symbolic = %RequiresA.specific_fn.loc13_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%U.binding.as_type.loc10 (%U.binding.as_type.e57)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe, (@observe.%T.ref, @observe.%U.ref, @observe.%U.as_type, @observe.%.loc11_18, @observe.%T.as_type, @observe.%.loc11_13, @observe.%eq)
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @InnerF.%U.binding.as_type.loc10 (%U.binding.as_type.e57) = name_ref x, %x
// CHECK:STDOUT:     %.loc13_16.1: %A.type = as_compatible constants.%U.binding.as_type.e57 [symbolic = %U.binding.as_type.loc13 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:     %A.facet.loc13_16.1: %A.type = facet_value constants.%U.binding.as_type.e57, (%.loc13_16.1) [symbolic = %A.facet.loc13_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc13_16.2: %A.type = converted constants.%U.binding.as_type.e57, %A.facet.loc13_16.1 [symbolic = %A.facet.loc13_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc13_16.3: %A.type = as_compatible constants.%U.binding.as_type.e57 [symbolic = %U.binding.as_type.loc13 (constants.%U.binding.as_type.89d)]
// CHECK:STDOUT:     %A.facet.loc13_16.2: %A.type = facet_value constants.%U.binding.as_type.e57, (%.loc13_16.3) [symbolic = %A.facet.loc13_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc13_16.4: %A.type = converted constants.%U.binding.as_type.e57, %A.facet.loc13_16.2 [symbolic = %A.facet.loc13_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %RequiresA.specific_fn.loc13_5.1: <specific function> = specific_function %RequiresA.ref, @RequiresA(constants.%A.facet) [symbolic = %RequiresA.specific_fn.loc13_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:     %RequiresA.call: init %empty_tuple.type = call %RequiresA.specific_fn.loc13_5.1(%x.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc8_13.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.binding.as_type.loc10 => constants.%U.binding.as_type.e57
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%A.facet) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%A.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%U.binding.as_type.e57
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.a2d
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
// CHECK:STDOUT:     %.loc7_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_31.1: type = splice_block %.loc7_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.2: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %.Self.ref: %B.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   interface_with_self_decl @A [concrete]
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
// CHECK:STDOUT:   interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc7_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc7_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
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
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %A.type, %U.loc8_13.2: %B_where.type) {
// CHECK:STDOUT:   %T.loc8_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc8_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc8_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %x.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %x.param: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %.loc10_16.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc8_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:         %.loc10_16.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %observe.decl = observe_decl @observe, (@observe.%T.ref, @observe.%U.ref, @observe.%U.as_type, @observe.%.loc21_16, @observe.%T.as_type, @observe.%.loc21_11, @observe.%eq)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc8_6.2: %A.type, @F.%U.loc8_13.2: %B_where.type) {
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
// CHECK:STDOUT: specific @A(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc8_13.1 => constants.%U
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
// CHECK:STDOUT:     %.loc7_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_31.1: type = splice_block %.loc7_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.2d9) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.2d9) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.3: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.2: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %.Self.ref: %B.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:     %x.param: @F.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_40.1: type = splice_block %.loc8_40.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc8_13.2 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc8_40.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%U.binding.as_type (%U.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   interface_with_self_decl @A [concrete]
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
// CHECK:STDOUT:   interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc7_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc7_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
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
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %A.type, %U.loc8_13.2: %B_where.type) {
// CHECK:STDOUT:   %T.loc8_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc8_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc8_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
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
// CHECK:STDOUT:       %.loc10_23.1: type = splice_block %.loc10_23.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc8_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:         %.loc10_23.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %x: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @F.%U.binding.as_type (%U.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InnerF(@F.%T.loc8_6.2: %A.type, @F.%U.loc8_13.2: %B_where.type) {
// CHECK:STDOUT:   %U: %B_where.type = symbolic_binding U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.a2d)]
// CHECK:STDOUT:   %T: %A.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InnerF.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe, (@observe.%T.ref, @observe.%U.ref, @observe.%U.as_type, @observe.%.loc11_18, @observe.%T.as_type, @observe.%.loc11_13, @observe.%eq)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc8_13.1 => constants.%U
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
// CHECK:STDOUT:     %.loc7_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_31.1: type = splice_block %.loc7_31.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_31.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
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
// CHECK:STDOUT:     %.loc8_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.4: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref.loc8_17: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:       %.Self.3: %B.type = symbolic_binding .Self [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %.Self.ref: %B.type = name_ref .Self, %.Self.3 [symbolic_self = constants.%.Self.d66]
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.3 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:     %.loc8_41: type = splice_block %B.ref.loc8_41 [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref.loc8_41: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc8_37.2: %B.type = symbolic_binding V, 2 [symbolic = %V.loc8_37.1 (constants.%V)]
// CHECK:STDOUT:     %x.param: @F.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_47.1: type = splice_block %.loc8_47.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc8_13.2 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc8_47.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%U.binding.as_type (%U.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   interface_with_self_decl @A [concrete]
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
// CHECK:STDOUT:   interface_with_self_decl @B [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RequiresA(%T.loc7_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc7_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
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
// CHECK:STDOUT: generic fn @F(%T.loc8_6.2: %A.type, %U.loc8_13.2: %B_where.type, %V.loc8_37.2: %B.type) {
// CHECK:STDOUT:   %T.loc8_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc8_13.1: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc8_37.1: %B.type = symbolic_binding V, 2 [symbolic = %V.loc8_37.1 (constants.%V)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc8_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.a2d)]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 2, %V.loc8_37.1 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe, (@observe.%T.ref, @observe.%V.ref, @observe.%V.as_type, @observe.%.loc9_16, @observe.%T.as_type, @observe.%.loc9_11, @observe.%eq)
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @F.%U.binding.as_type (%U.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self.d0b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U, constants.%V) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc8_13.1 => constants.%U
// CHECK:STDOUT:   %V.loc8_37.1 => constants.%V
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_scope.carbon
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
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc4_19.2: type = symbolic_binding U, 1 [symbolic = %U.loc4_19.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.2: type, %U.loc4_19.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc4_19.1: type = symbolic_binding U, 1 [symbolic = %U.loc4_19.1 (constants.%U)]
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
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T
// CHECK:STDOUT:   %U.loc4_19.1 => constants.%U
// CHECK:STDOUT: }
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
// CHECK:STDOUT:     %.loc7: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @A [concrete]
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
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: %A.type) {
// CHECK:STDOUT:   %T.loc7_6.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:

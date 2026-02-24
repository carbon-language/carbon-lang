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

fn RequiresA[T:! A](x: T) {}
fn F[T:! C](x: T) {
  observe T impls B;

  // CHECK:STDERR: fail_todo_impls.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `C` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_impls.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T:! A](x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
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

fn TakesPQR[U:! P & Q & R](u: U) {}

fn G[T:! Transitive](t: T) {
  var a: T.A = t.GetA();

  observe T.A == T.B impls Q;
  a.(Q.InQ)();

  observe T.A == T.C impls R;
  TakesPQR(a);
}

// --- nested_functions_in_outer.carbon

library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T:! A](x: T) {}
fn F[T:! A, U:! B where .Self == T](x: T, y: U) {
  observe T == U;

  fn InnerF(y: U) {
    RequiresA(y);
  }
}

// --- nested_functions_in_inner.carbon

library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T:! A](x: T) {}
fn F[T:! A, U:! B where .Self == T](x: T, y: U) {

  fn InnerF(y: U) {
    observe T == U;

    RequiresA(y);
  }
}

// --- fail_nested_functions_lexical_order.carbon

library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T:! A](x: T) {}
fn F[T:! A, U:! B where .Self == T](x: T, y: U) {

  fn InnerF(y: U) {
    // CHECK:STDERR: fail_nested_functions_lexical_order.carbon:[[@LINE+7]]:5: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     RequiresA(y);
    // CHECK:STDERR:     ^~~~~~~~~~~~
    // CHECK:STDERR: fail_nested_functions_lexical_order.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR: fn RequiresA[T:! A](x: T) {}
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    RequiresA(y);
  }

  observe T == U;
}

// --- fail_nested_functions_leak_upward.carbon

library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T:! A](x: T) {}
fn F[T:! A, U:! B where .Self == T](x: T, y: U) {

  fn InnerF(y: U) {
    observe T == U;
  }

  // CHECK:STDERR: fail_nested_functions_leak_upward.carbon:[[@LINE+7]]:3: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(y);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_nested_functions_leak_upward.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T:! A](x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(y);
}

// --- fail_not_observed.carbon

library "[[@TEST_NAME]]";

interface A {}
interface B {}

fn RequiresA[T:! A](x: T) {}
fn F[T:! A, U:! B where .Self == T, V:! B](x: T, y: U) {
  observe T == V;

  // CHECK:STDERR: fail_not_observed.carbon:[[@LINE+7]]:3: error: cannot convert type `U` that implements `B where...` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(y);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_not_observed.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T:! A](x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(y);
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

fn F[T:! A](x: T) {
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
// CHECK:STDOUT:   %P.InP.type: type = fn_type @P.InP [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %P.InP: %P.InP.type = struct_value () [concrete]
// CHECK:STDOUT:   %P.assoc_type: type = assoc_entity_type @P [concrete]
// CHECK:STDOUT:   %assoc0.671: %P.assoc_type = assoc_entity element0, @P.%P.InP.decl [concrete]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.f64: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.1b2: type = symbolic_binding_type Self, 0, %Self.f64 [symbolic]
// CHECK:STDOUT:   %pattern_type.992: type = pattern_type %Self.binding.as_type.1b2 [symbolic]
// CHECK:STDOUT:   %Q.InQ.type: type = fn_type @Q.InQ [concrete]
// CHECK:STDOUT:   %Q.InQ: %Q.InQ.type = struct_value () [concrete]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.135: %Q.assoc_type = assoc_entity element0, @Q.%Q.InQ.decl [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.bea: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.296: type = symbolic_binding_type Self, 0, %Self.bea [symbolic]
// CHECK:STDOUT:   %pattern_type.351: type = pattern_type %Self.binding.as_type.296 [symbolic]
// CHECK:STDOUT:   %R.InR.type: type = fn_type @R.InR [concrete]
// CHECK:STDOUT:   %R.InR: %R.InR.type = struct_value () [concrete]
// CHECK:STDOUT:   %R.assoc_type: type = assoc_entity_type @R [concrete]
// CHECK:STDOUT:   %assoc0.4f7: %R.assoc_type = assoc_entity element0, @R.%R.InR.decl [concrete]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.e2a: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.398: %Transitive.assoc_type = assoc_entity element0, @Transitive.%A [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.e5b: type = symbolic_binding_type Self, 0, %Self.e2a [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.804: <witness> = lookup_impl_witness %Self.e2a, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.9bd: %P.type = impl_witness_access %Transitive.lookup_impl_witness.804, element0 [symbolic]
// CHECK:STDOUT:   %Q_where.type: type = facet_type <@Q where TODO> [concrete]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.%B [concrete]
// CHECK:STDOUT:   %impl.elem1.4e4: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.804, element1 [symbolic]
// CHECK:STDOUT:   %R_where.type: type = facet_type <@R where TODO> [concrete]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.%C [concrete]
// CHECK:STDOUT:   %pattern_type.b0d: type = pattern_type %Self.binding.as_type.e5b [symbolic]
// CHECK:STDOUT:   %as_type.0ea: type = facet_access_type %impl.elem0.9bd [symbolic]
// CHECK:STDOUT:   %.e64: form = init_form %as_type.0ea, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.bff: type = pattern_type %as_type.0ea [symbolic]
// CHECK:STDOUT:   %Transitive.GetA.type: type = fn_type @Transitive.GetA [concrete]
// CHECK:STDOUT:   %Transitive.GetA: %Transitive.GetA.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, @Transitive.%Transitive.GetA.decl [concrete]
// CHECK:STDOUT:   %impl.elem2.8f7: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.804, element2 [symbolic]
// CHECK:STDOUT:   %as_type.ecc: type = facet_access_type %impl.elem2.8f7 [symbolic]
// CHECK:STDOUT:   %pattern_type.364: type = pattern_type %as_type.ecc [symbolic]
// CHECK:STDOUT:   %Transitive.TakesC.type: type = fn_type @Transitive.TakesC [concrete]
// CHECK:STDOUT:   %Transitive.TakesC: %Transitive.TakesC.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, @Transitive.%Transitive.TakesC.decl [concrete]
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
// CHECK:STDOUT:   %as_type.e31: type = facet_access_type %impl.elem0.b09 [symbolic]
// CHECK:STDOUT:   %require_complete.b46: <witness> = require_complete_type %as_type.e31 [symbolic]
// CHECK:STDOUT:   %pattern_type.05e: type = pattern_type %as_type.e31 [symbolic]
// CHECK:STDOUT:   %.f7d: type = fn_type_with_self_type %Transitive.GetA.type, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.f7d = impl_witness_access %Transitive.lookup_impl_witness.6d8, element3 [symbolic]
// CHECK:STDOUT:   %.d40: form = init_form %as_type.e31, call_param1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5b1: <specific function> = specific_impl_function %impl.elem3, @Transitive.GetA(%T) [symbolic]
// CHECK:STDOUT:   %impl.elem1.32a: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element1 [symbolic]
// CHECK:STDOUT:   %as_type.846: type = facet_access_type %impl.elem1.32a [symbolic]
// CHECK:STDOUT:   %require_complete.8ba: <witness> = require_complete_type %as_type.846 [symbolic]
// CHECK:STDOUT:   %pattern_type.a21: type = pattern_type %as_type.846 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ff3: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6d3: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.50f: %ImplicitAs.Convert.type.6d3 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d5a: type = facet_type <@ImplicitAs, @ImplicitAs(%as_type.846)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.11b: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%as_type.846) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.b4b: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%as_type.846) [symbolic]
// CHECK:STDOUT:   %assoc0.849: %ImplicitAs.assoc_type.b4b = assoc_entity element0, imports.%Core.import_ref.218 [symbolic]
// CHECK:STDOUT:   %require_complete.abb: <witness> = require_complete_type %ImplicitAs.type.d5a [symbolic]
// CHECK:STDOUT:   %assoc0.0bc: %ImplicitAs.assoc_type.ff3 = assoc_entity element0, imports.%Core.import_ref.fa2 [symbolic]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.752: type = assoc_entity_type @As, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %As.Convert.type.47d: type = fn_type @As.Convert, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %As.Convert.099: %As.Convert.type.47d = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0b4: %ImplicitAs.type.d5a = facet_value %as_type.e31, (%impl.elem1.4e4) [symbolic]
// CHECK:STDOUT:   %.1e5: type = fn_type_with_self_type %ImplicitAs.Convert.type.11b, %ImplicitAs.facet.0b4 [symbolic]
// CHECK:STDOUT:   %impl.elem0.694: %.1e5 = impl_witness_access %impl.elem1.4e4, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.dd7: <specific function> = specific_impl_function %impl.elem0.694, @ImplicitAs.Convert(%as_type.846, %ImplicitAs.facet.0b4) [symbolic]
// CHECK:STDOUT:   %Q.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.32a, @Q [symbolic]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.846, (%Q.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.736: type = fn_type_with_self_type %Q.InQ.type, %Q.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.3bb: %.736 = impl_witness_access %Q.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7f3: <specific function> = specific_impl_function %impl.elem0.3bb, @Q.InQ(%Q.facet) [symbolic]
// CHECK:STDOUT:   %impl.elem2.458: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element2 [symbolic]
// CHECK:STDOUT:   %as_type.c65: type = facet_access_type %impl.elem2.458 [symbolic]
// CHECK:STDOUT:   %R.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem2.458, @R [symbolic]
// CHECK:STDOUT:   %R.facet: %R.type = facet_value %as_type.c65, (%R.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.a55: type = fn_type_with_self_type %R.InR.type, %R.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.2a7: %.a55 = impl_witness_access %R.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7db: type = pattern_type %as_type.c65 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.88b: <specific function> = specific_impl_function %impl.elem0.2a7, @R.InR(%R.facet) [symbolic]
// CHECK:STDOUT:   %require_complete.60d: <witness> = require_complete_type %as_type.c65 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.ac2: type = facet_type <@ImplicitAs, @ImplicitAs(%as_type.c65)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.366: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%as_type.c65) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.eeb: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%as_type.c65) [symbolic]
// CHECK:STDOUT:   %assoc0.399: %ImplicitAs.assoc_type.eeb = assoc_entity element0, imports.%Core.import_ref.218 [symbolic]
// CHECK:STDOUT:   %require_complete.49e: <witness> = require_complete_type %ImplicitAs.type.ac2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.facet.304: %ImplicitAs.type.ac2 = facet_value %as_type.e31, (%impl.elem2.8f7) [symbolic]
// CHECK:STDOUT:   %.ee0: type = fn_type_with_self_type %ImplicitAs.Convert.type.366, %ImplicitAs.facet.304 [symbolic]
// CHECK:STDOUT:   %impl.elem0.757: %.ee0 = impl_witness_access %impl.elem2.8f7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.207: <specific function> = specific_impl_function %impl.elem0.757, @ImplicitAs.Convert(%as_type.c65, %ImplicitAs.facet.304) [symbolic]
// CHECK:STDOUT:   %As.type.ae5: type = facet_type <@As, @As(%as_type.c65)> [symbolic]
// CHECK:STDOUT:   %As.Convert.type.d13: type = fn_type @As.Convert, @As(%as_type.c65) [symbolic]
// CHECK:STDOUT:   %As.assoc_type.69a: type = assoc_entity_type @As, @As(%as_type.c65) [symbolic]
// CHECK:STDOUT:   %assoc0.685: %As.assoc_type.69a = assoc_entity element0, imports.%Core.import_ref.77d [symbolic]
// CHECK:STDOUT:   %require_complete.ea4: <witness> = require_complete_type %As.type.ae5 [symbolic]
// CHECK:STDOUT:   %assoc0.5c9: %As.assoc_type.752 = assoc_entity element0, imports.%Core.import_ref.1ae [symbolic]
// CHECK:STDOUT:   %As.facet: %As.type.ae5 = facet_value %as_type.e31, (%impl.elem2.8f7) [symbolic]
// CHECK:STDOUT:   %.d27: type = fn_type_with_self_type %As.Convert.type.d13, %As.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1b9: %.d27 = impl_witness_access %impl.elem2.8f7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0f7: <specific function> = specific_impl_function %impl.elem0.1b9, @As.Convert(%as_type.c65, %As.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.facet.7a0: %Destroy.type = facet_value %as_type.c65, (%impl.elem2.8f7) [symbolic]
// CHECK:STDOUT:   %.e6f: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.7a0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.ff7: %.e6f = impl_witness_access %impl.elem2.8f7, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d83: <specific function> = specific_impl_function %impl.elem0.ff7, @Destroy.Op(%Destroy.facet.7a0) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.69c: %Destroy.type = facet_value %as_type.846, (%impl.elem1.4e4) [symbolic]
// CHECK:STDOUT:   %.f9e: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.69c [symbolic]
// CHECK:STDOUT:   %impl.elem0.50f: %.f9e = impl_witness_access %impl.elem1.4e4, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e4c: <specific function> = specific_impl_function %impl.elem0.50f, @Destroy.Op(%Destroy.facet.69c) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.96d: %Destroy.type = facet_value %as_type.e31, (%impl.elem0.9bd) [symbolic]
// CHECK:STDOUT:   %.8f2: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.96d [symbolic]
// CHECK:STDOUT:   %impl.elem0.28e: %.8f2 = impl_witness_access %impl.elem0.9bd, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1b0: <specific function> = specific_impl_function %impl.elem0.28e, @Destroy.Op(%Destroy.facet.96d) [symbolic]
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
// CHECK:STDOUT:   %Core.import_ref.a80: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ff3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.0bc)]
// CHECK:STDOUT:   %Core.import_ref.218: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.6d3) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.50f)]
// CHECK:STDOUT:   %Core.import_ref.fa2 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.e0b: @As.%As.assoc_type (%As.assoc_type.752) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @As.%assoc0 (constants.%assoc0.5c9)]
// CHECK:STDOUT:   %Core.import_ref.77d: @As.%As.Convert.type (%As.Convert.type.47d) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @As.%As.Convert (constants.%As.Convert.099)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.1ae = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d21]
// CHECK:STDOUT:   %P.InP.decl: %P.InP.type = fn_decl @P.InP [concrete = constants.%P.InP] {
// CHECK:STDOUT:     %self.patt: @P.InP.%pattern_type (%pattern_type.2eb) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @P.InP.%pattern_type (%pattern_type.2eb) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @P.InP.%Self.binding.as_type (%Self.binding.as_type.6c8) = value_param call_param0
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)] {
// CHECK:STDOUT:       %Self.ref: %P.type = name_ref Self, @P.%Self [symbolic = %Self (constants.%Self.d21)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:       %.loc4_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @P.InP.%Self.binding.as_type (%Self.binding.as_type.6c8) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %P.assoc_type = assoc_entity element0, %P.InP.decl [concrete = constants.%assoc0.671]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InP = %assoc0
// CHECK:STDOUT:   witness = (%P.InP.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f64]
// CHECK:STDOUT:   %Q.InQ.decl: %Q.InQ.type = fn_decl @Q.InQ [concrete = constants.%Q.InQ] {
// CHECK:STDOUT:     %self.patt: @Q.InQ.%pattern_type (%pattern_type.992) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Q.InQ.%pattern_type (%pattern_type.992) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Q.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2) = value_param call_param0
// CHECK:STDOUT:     %.loc5_28.1: type = splice_block %.loc5_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)] {
// CHECK:STDOUT:       %Self.ref: %Q.type = name_ref Self, @Q.%Self [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:       %.loc5_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Q.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Q.assoc_type = assoc_entity element0, %Q.InQ.decl [concrete = constants.%assoc0.135]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InQ = %assoc0
// CHECK:STDOUT:   witness = (%Q.InQ.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = constants.%Self.bea]
// CHECK:STDOUT:   %R.InR.decl: %R.InR.type = fn_decl @R.InR [concrete = constants.%R.InR] {
// CHECK:STDOUT:     %self.patt: @R.InR.%pattern_type (%pattern_type.351) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @R.InR.%pattern_type (%pattern_type.351) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @R.InR.%Self.binding.as_type (%Self.binding.as_type.296) = value_param call_param0
// CHECK:STDOUT:     %.loc6_28.1: type = splice_block %.loc6_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)] {
// CHECK:STDOUT:       %Self.ref: %R.type = name_ref Self, @R.%Self [symbolic = %Self (constants.%Self.bea)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:       %.loc6_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @R.InR.%Self.binding.as_type (%Self.binding.as_type.296) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %R.assoc_type = assoc_entity element0, %R.InR.decl [concrete = constants.%assoc0.4f7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InR = %assoc0
// CHECK:STDOUT:   witness = (%R.InR.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e2a]
// CHECK:STDOUT:   %A: %P.type = assoc_const_decl @A [concrete] {
// CHECK:STDOUT:     %assoc0: %Transitive.assoc_type = assoc_entity element0, @Transitive.%A [concrete = constants.%assoc0.398]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: %Q_where.type = assoc_const_decl @B [concrete] {
// CHECK:STDOUT:     %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.%B [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: %R_where.type = assoc_const_decl @C [concrete] {
// CHECK:STDOUT:     %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.%C [concrete = constants.%assoc2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Transitive.GetA.decl: %Transitive.GetA.type = fn_decl @Transitive.GetA [concrete = constants.%Transitive.GetA] {
// CHECK:STDOUT:     %self.patt: @Transitive.GetA.%pattern_type.loc13_11 (%pattern_type.b0d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Transitive.GetA.%pattern_type.loc13_11 (%pattern_type.b0d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Transitive.GetA.%pattern_type.loc13_25 (%pattern_type.bff) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Transitive.GetA.%pattern_type.loc13_25 (%pattern_type.bff) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc13_28.2: %P.type = impl_witness_access constants.%Transitive.lookup_impl_witness.804, element0 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %A.ref: %P.type = name_ref A, %impl.elem0.loc13_28.2 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %A.as_type: type = facet_access_type %A.ref [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:     %.loc13_28.2: type = converted %A.ref, %A.as_type [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:     %.loc13_28.3: form = init_form %.loc13_28.2, call_param1 [symbolic = %.loc13_28.1 (constants.%.e64)]
// CHECK:STDOUT:     %self.param: @Transitive.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_param call_param0
// CHECK:STDOUT:     %.loc13_17.1: type = splice_block %.loc13_17.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:       %.loc13_17.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Transitive.GetA.%as_type (%as_type.0ea) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Transitive.GetA.%as_type (%as_type.0ea) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, %Transitive.GetA.decl [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Transitive.TakesC.decl: %Transitive.TakesC.type = fn_decl @Transitive.TakesC [concrete = constants.%Transitive.TakesC] {
// CHECK:STDOUT:     %self.patt: @Transitive.TakesC.%pattern_type.loc14_13 (%pattern_type.b0d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Transitive.TakesC.%pattern_type.loc14_13 (%pattern_type.b0d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %c.patt: @Transitive.TakesC.%pattern_type.loc14_25 (%pattern_type.364) = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @Transitive.TakesC.%pattern_type.loc14_25 (%pattern_type.364) = value_param_pattern %c.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Transitive.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:       %.loc14_19.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_binding self, %self.param
// CHECK:STDOUT:     %c.param: @Transitive.TakesC.%as_type (%as_type.ecc) = value_param call_param1
// CHECK:STDOUT:     %.loc14_28.1: type = splice_block %.loc14_28.2 [symbolic = %as_type (constants.%as_type.ecc)] {
// CHECK:STDOUT:       %impl.elem2.loc14_28.2: %R_where.type = impl_witness_access constants.%Transitive.lookup_impl_witness.804, element2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:       %C.ref: %R_where.type = name_ref C, %impl.elem2.loc14_28.2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:       %C.as_type: type = facet_access_type %C.ref [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:       %.loc14_28.2: type = converted %C.ref, %C.as_type [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @Transitive.TakesC.%as_type (%as_type.ecc) = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, %Transitive.TakesC.decl [concrete = constants.%assoc4]
// CHECK:STDOUT:   %Transitive.observe.decl = observe_decl @Transitive.observe, (@Transitive.observe.%impl.elem0, @Transitive.observe.%A.ref, @Transitive.observe.%impl.elem1, @Transitive.observe.%B.ref, @Transitive.observe.%B.as_type.loc16_16.1, @Transitive.observe.%.loc16_16.1, @Transitive.observe.%A.as_type, @Transitive.observe.%.loc16_11, @Transitive.observe.%eq.loc16_13, @Transitive.observe.%impl.elem2, @Transitive.observe.%C.ref, @Transitive.observe.%C.as_type, @Transitive.observe.%.loc16_21, @Transitive.observe.%B.as_type.loc16_16.2, @Transitive.observe.%.loc16_16.2, @Transitive.observe.%eq.loc16_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .P = <poisoned>
// CHECK:STDOUT:   .A = @A.%assoc0
// CHECK:STDOUT:   .Q = <poisoned>
// CHECK:STDOUT:   .B = @B.%assoc1
// CHECK:STDOUT:   .R = <poisoned>
// CHECK:STDOUT:   .C = @C.%assoc2
// CHECK:STDOUT:   .GetA = %assoc3
// CHECK:STDOUT:   .TakesC = %assoc4
// CHECK:STDOUT:   witness = (%A, %B, %C, %Transitive.GetA.decl, %Transitive.TakesC.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @A(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   assoc_const A:! %P.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @B(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem0: %P.type = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const B:! %Q_where.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @C(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem1: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1 (constants.%impl.elem1.4e4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const C:! %R_where.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.InP(@P.%Self: %P.type) {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d21)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @P.InP.%Self.binding.as_type (%Self.binding.as_type.6c8));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.InQ(@Q.%Self: %Q.type) {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.992)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Q.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.InR(@R.%Self: %R.type) {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.bea)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.351)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @R.InR.%Self.binding.as_type (%Self.binding.as_type.296));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.GetA(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:   %pattern_type.loc13_11: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc13_11 (constants.%pattern_type.b0d)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem0.loc13_28.1: %P.type = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc13_28.1 [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:   %.loc13_28.1: form = init_form %as_type, call_param1 [symbolic = %.loc13_28.1 (constants.%.e64)]
// CHECK:STDOUT:   %pattern_type.loc13_25: type = pattern_type %as_type [symbolic = %pattern_type.loc13_25 (constants.%pattern_type.bff)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b)) -> out %return.param: @Transitive.GetA.%as_type (%as_type.0ea);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.TakesC(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:   %pattern_type.loc14_13: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc14_13 (constants.%pattern_type.b0d)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem2.loc14_28.1: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2.loc14_28.1 [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:   %pattern_type.loc14_25: type = pattern_type %as_type [symbolic = %pattern_type.loc14_25 (constants.%pattern_type.364)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b), %c.param: @Transitive.TakesC.%as_type (%as_type.ecc));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc19_6.2: %Transitive.type) {
// CHECK:STDOUT:   %T.loc19_6.1: %Transitive.type = symbolic_binding T, 0 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc19_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc19: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc19 (constants.%pattern_type.993)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc19: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc19 (constants.%require_complete.1c0)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.loc20: <witness> = lookup_impl_witness %T.loc19_6.1, @Transitive [symbolic = %Transitive.lookup_impl_witness.loc20 (constants.%Transitive.lookup_impl_witness.6d8)]
// CHECK:STDOUT:   %impl.elem0.loc20_11.2: %P.type = impl_witness_access %Transitive.lookup_impl_witness.loc20, element0 [symbolic = %impl.elem0.loc20_11.2 (constants.%impl.elem0.b09)]
// CHECK:STDOUT:   %as_type.loc20_11.2: type = facet_access_type %impl.elem0.loc20_11.2 [symbolic = %as_type.loc20_11.2 (constants.%as_type.e31)]
// CHECK:STDOUT:   %require_complete.loc20: <witness> = require_complete_type %as_type.loc20_11.2 [symbolic = %require_complete.loc20 (constants.%require_complete.b46)]
// CHECK:STDOUT:   %pattern_type.loc20: type = pattern_type %as_type.loc20_11.2 [symbolic = %pattern_type.loc20 (constants.%pattern_type.05e)]
// CHECK:STDOUT:   %.loc20_17: type = fn_type_with_self_type constants.%Transitive.GetA.type, %T.loc19_6.1 [symbolic = %.loc20_17 (constants.%.f7d)]
// CHECK:STDOUT:   %impl.elem3.loc20_17.2: @F.%.loc20_17 (%.f7d) = impl_witness_access %Transitive.lookup_impl_witness.loc20, element3 [symbolic = %impl.elem3.loc20_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_17.2: <specific function> = specific_impl_function %impl.elem3.loc20_17.2, @Transitive.GetA(%T.loc19_6.1) [symbolic = %specific_impl_fn.loc20_17.2 (constants.%specific_impl_fn.5b1)]
// CHECK:STDOUT:   %impl.elem1.loc22_11.2: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.loc20, element1 [symbolic = %impl.elem1.loc22_11.2 (constants.%impl.elem1.32a)]
// CHECK:STDOUT:   %as_type.loc22_11.2: type = facet_access_type %impl.elem1.loc22_11.2 [symbolic = %as_type.loc22_11.2 (constants.%as_type.846)]
// CHECK:STDOUT:   %require_complete.loc22_11: <witness> = require_complete_type %as_type.loc22_11.2 [symbolic = %require_complete.loc22_11 (constants.%require_complete.8ba)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %as_type.loc22_11.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.a21)]
// CHECK:STDOUT:   %ImplicitAs.type.loc22_3.2: type = facet_type <@ImplicitAs, @ImplicitAs(%as_type.loc22_11.2)> [symbolic = %ImplicitAs.type.loc22_3.2 (constants.%ImplicitAs.type.d5a)]
// CHECK:STDOUT:   %require_complete.loc22_3: <witness> = require_complete_type %ImplicitAs.type.loc22_3.2 [symbolic = %require_complete.loc22_3 (constants.%require_complete.abb)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.loc22: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%as_type.loc22_11.2) [symbolic = %ImplicitAs.assoc_type.loc22 (constants.%ImplicitAs.assoc_type.b4b)]
// CHECK:STDOUT:   %assoc0.loc22: @F.%ImplicitAs.assoc_type.loc22 (%ImplicitAs.assoc_type.b4b) = assoc_entity element0, imports.%Core.import_ref.218 [symbolic = %assoc0.loc22 (constants.%assoc0.849)]
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.loc22: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness.loc22 (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem1.loc22_3: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.loc22, element1 [symbolic = %impl.elem1.loc22_3 (constants.%impl.elem1.4e4)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.loc22: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%as_type.loc22_11.2) [symbolic = %ImplicitAs.Convert.type.loc22 (constants.%ImplicitAs.Convert.type.11b)]
// CHECK:STDOUT:   %ImplicitAs.facet.loc22: @F.%ImplicitAs.type.loc22_3.2 (%ImplicitAs.type.d5a) = facet_value %as_type.loc20_11.2, (%impl.elem1.loc22_3) [symbolic = %ImplicitAs.facet.loc22 (constants.%ImplicitAs.facet.0b4)]
// CHECK:STDOUT:   %.loc22_3.6: type = fn_type_with_self_type %ImplicitAs.Convert.type.loc22, %ImplicitAs.facet.loc22 [symbolic = %.loc22_3.6 (constants.%.1e5)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.3: @F.%.loc22_3.6 (%.1e5) = impl_witness_access %impl.elem1.loc22_3, element0 [symbolic = %impl.elem0.loc22_3.3 (constants.%impl.elem0.694)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_3.3: <specific function> = specific_impl_function %impl.elem0.loc22_3.3, @ImplicitAs.Convert(%as_type.loc22_11.2, %ImplicitAs.facet.loc22) [symbolic = %specific_impl_fn.loc22_3.3 (constants.%specific_impl_fn.dd7)]
// CHECK:STDOUT:   %Q.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.loc22_11.2, @Q [symbolic = %Q.lookup_impl_witness (constants.%Q.lookup_impl_witness)]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.loc22_11.2, (%Q.lookup_impl_witness) [symbolic = %Q.facet (constants.%Q.facet)]
// CHECK:STDOUT:   %.loc23_4: type = fn_type_with_self_type constants.%Q.InQ.type, %Q.facet [symbolic = %.loc23_4 (constants.%.736)]
// CHECK:STDOUT:   %impl.elem0.loc23_4.2: @F.%.loc23_4 (%.736) = impl_witness_access %Q.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc23_4.2 (constants.%impl.elem0.3bb)]
// CHECK:STDOUT:   %specific_impl_fn.loc23_4.2: <specific function> = specific_impl_function %impl.elem0.loc23_4.2, @Q.InQ(%Q.facet) [symbolic = %specific_impl_fn.loc23_4.2 (constants.%specific_impl_fn.7f3)]
// CHECK:STDOUT:   %impl.elem2.loc25_7.2: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.loc20, element2 [symbolic = %impl.elem2.loc25_7.2 (constants.%impl.elem2.458)]
// CHECK:STDOUT:   %as_type.loc25_9.2: type = facet_access_type %impl.elem2.loc25_7.2 [symbolic = %as_type.loc25_9.2 (constants.%as_type.c65)]
// CHECK:STDOUT:   %R.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem2.loc25_7.2, @R [symbolic = %R.lookup_impl_witness (constants.%R.lookup_impl_witness)]
// CHECK:STDOUT:   %R.facet: %R.type = facet_value %as_type.loc25_9.2, (%R.lookup_impl_witness) [symbolic = %R.facet (constants.%R.facet)]
// CHECK:STDOUT:   %.loc25_9.2: type = fn_type_with_self_type constants.%R.InR.type, %R.facet [symbolic = %.loc25_9.2 (constants.%.a55)]
// CHECK:STDOUT:   %impl.elem0.loc25_9.2: @F.%.loc25_9.2 (%.a55) = impl_witness_access %R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_9.2 (constants.%impl.elem0.2a7)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_9.2: <specific function> = specific_impl_function %impl.elem0.loc25_9.2, @R.InR(%R.facet) [symbolic = %specific_impl_fn.loc25_9.2 (constants.%specific_impl_fn.88b)]
// CHECK:STDOUT:   %require_complete.loc25_3.1: <witness> = require_complete_type %as_type.loc25_9.2 [symbolic = %require_complete.loc25_3.1 (constants.%require_complete.60d)]
// CHECK:STDOUT:   %ImplicitAs.type.loc25_3.2: type = facet_type <@ImplicitAs, @ImplicitAs(%as_type.loc25_9.2)> [symbolic = %ImplicitAs.type.loc25_3.2 (constants.%ImplicitAs.type.ac2)]
// CHECK:STDOUT:   %require_complete.loc25_3.2: <witness> = require_complete_type %ImplicitAs.type.loc25_3.2 [symbolic = %require_complete.loc25_3.2 (constants.%require_complete.49e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.loc25: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%as_type.loc25_9.2) [symbolic = %ImplicitAs.assoc_type.loc25 (constants.%ImplicitAs.assoc_type.eeb)]
// CHECK:STDOUT:   %assoc0.loc25: @F.%ImplicitAs.assoc_type.loc25 (%ImplicitAs.assoc_type.eeb) = assoc_entity element0, imports.%Core.import_ref.218 [symbolic = %assoc0.loc25 (constants.%assoc0.399)]
// CHECK:STDOUT:   %impl.elem2.loc25_3: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.loc22, element2 [symbolic = %impl.elem2.loc25_3 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.loc25: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%as_type.loc25_9.2) [symbolic = %ImplicitAs.Convert.type.loc25 (constants.%ImplicitAs.Convert.type.366)]
// CHECK:STDOUT:   %ImplicitAs.facet.loc25: @F.%ImplicitAs.type.loc25_3.2 (%ImplicitAs.type.ac2) = facet_value %as_type.loc20_11.2, (%impl.elem2.loc25_3) [symbolic = %ImplicitAs.facet.loc25 (constants.%ImplicitAs.facet.304)]
// CHECK:STDOUT:   %.loc25_3.9: type = fn_type_with_self_type %ImplicitAs.Convert.type.loc25, %ImplicitAs.facet.loc25 [symbolic = %.loc25_3.9 (constants.%.ee0)]
// CHECK:STDOUT:   %impl.elem0.loc25_3.3: @F.%.loc25_3.9 (%.ee0) = impl_witness_access %impl.elem2.loc25_3, element0 [symbolic = %impl.elem0.loc25_3.3 (constants.%impl.elem0.757)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_3.3: <specific function> = specific_impl_function %impl.elem0.loc25_3.3, @ImplicitAs.Convert(%as_type.loc25_9.2, %ImplicitAs.facet.loc25) [symbolic = %specific_impl_fn.loc25_3.3 (constants.%specific_impl_fn.207)]
// CHECK:STDOUT:   %As.type.loc26_6.2: type = facet_type <@As, @As(%as_type.loc25_9.2)> [symbolic = %As.type.loc26_6.2 (constants.%As.type.ae5)]
// CHECK:STDOUT:   %require_complete.loc26: <witness> = require_complete_type %As.type.loc26_6.2 [symbolic = %require_complete.loc26 (constants.%require_complete.ea4)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%as_type.loc25_9.2) [symbolic = %As.assoc_type (constants.%As.assoc_type.69a)]
// CHECK:STDOUT:   %assoc0.loc26: @F.%As.assoc_type (%As.assoc_type.69a) = assoc_entity element0, imports.%Core.import_ref.77d [symbolic = %assoc0.loc26 (constants.%assoc0.685)]
// CHECK:STDOUT:   %As.Convert.type: type = fn_type @As.Convert, @As(%as_type.loc25_9.2) [symbolic = %As.Convert.type (constants.%As.Convert.type.d13)]
// CHECK:STDOUT:   %As.facet: @F.%As.type.loc26_6.2 (%As.type.ae5) = facet_value %as_type.loc20_11.2, (%impl.elem2.loc25_3) [symbolic = %As.facet (constants.%As.facet)]
// CHECK:STDOUT:   %.loc26_6.8: type = fn_type_with_self_type %As.Convert.type, %As.facet [symbolic = %.loc26_6.8 (constants.%.d27)]
// CHECK:STDOUT:   %impl.elem0.loc26_6.3: @F.%.loc26_6.8 (%.d27) = impl_witness_access %impl.elem2.loc25_3, element0 [symbolic = %impl.elem0.loc26_6.3 (constants.%impl.elem0.1b9)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_6.3: <specific function> = specific_impl_function %impl.elem0.loc26_6.3, @As.Convert(%as_type.loc25_9.2, %As.facet) [symbolic = %specific_impl_fn.loc26_6.3 (constants.%specific_impl_fn.0f7)]
// CHECK:STDOUT:   %Destroy.facet.loc26: %Destroy.type = facet_value %as_type.loc25_9.2, (%impl.elem2.loc25_3) [symbolic = %Destroy.facet.loc26 (constants.%Destroy.facet.7a0)]
// CHECK:STDOUT:   %.loc26_6.9: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc26 [symbolic = %.loc26_6.9 (constants.%.e6f)]
// CHECK:STDOUT:   %impl.elem0.loc26_6.4: @F.%.loc26_6.9 (%.e6f) = impl_witness_access %impl.elem2.loc25_3, element0 [symbolic = %impl.elem0.loc26_6.4 (constants.%impl.elem0.ff7)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_6.4: <specific function> = specific_impl_function %impl.elem0.loc26_6.4, @Destroy.Op(%Destroy.facet.loc26) [symbolic = %specific_impl_fn.loc26_6.4 (constants.%specific_impl_fn.d83)]
// CHECK:STDOUT:   %Destroy.facet.loc22: %Destroy.type = facet_value %as_type.loc22_11.2, (%impl.elem1.loc22_3) [symbolic = %Destroy.facet.loc22 (constants.%Destroy.facet.69c)]
// CHECK:STDOUT:   %.loc22_3.7: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc22 [symbolic = %.loc22_3.7 (constants.%.f9e)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.4: @F.%.loc22_3.7 (%.f9e) = impl_witness_access %impl.elem1.loc22_3, element0 [symbolic = %impl.elem0.loc22_3.4 (constants.%impl.elem0.50f)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_3.4: <specific function> = specific_impl_function %impl.elem0.loc22_3.4, @Destroy.Op(%Destroy.facet.loc22) [symbolic = %specific_impl_fn.loc22_3.4 (constants.%specific_impl_fn.e4c)]
// CHECK:STDOUT:   %impl.elem0.loc20_3.2: %P.type = impl_witness_access %Transitive.lookup_impl_witness.loc22, element0 [symbolic = %impl.elem0.loc20_3.2 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:   %Destroy.facet.loc20: %Destroy.type = facet_value %as_type.loc20_11.2, (%impl.elem0.loc20_3.2) [symbolic = %Destroy.facet.loc20 (constants.%Destroy.facet.96d)]
// CHECK:STDOUT:   %.loc20_3.3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc20 [symbolic = %.loc20_3.3 (constants.%.8f2)]
// CHECK:STDOUT:   %impl.elem0.loc20_3.3: @F.%.loc20_3.3 (%.8f2) = impl_witness_access %impl.elem0.loc20_3.2, element0 [symbolic = %impl.elem0.loc20_3.3 (constants.%impl.elem0.28e)]
// CHECK:STDOUT:   %specific_impl_fn.loc20_3.2: <specific function> = specific_impl_function %impl.elem0.loc20_3.3, @Destroy.Op(%Destroy.facet.loc20) [symbolic = %specific_impl_fn.loc20_3.2 (constants.%specific_impl_fn.1b0)]
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
// CHECK:STDOUT:     %impl.elem3.loc20_17.1: @F.%.loc20_17 (%.f7d) = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element3 [symbolic = %impl.elem3.loc20_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc20_17: <bound method> = bound_method %t.ref, %impl.elem3.loc20_17.1
// CHECK:STDOUT:     %specific_impl_fn.loc20_17.1: <specific function> = specific_impl_function %impl.elem3.loc20_17.1, @Transitive.GetA(constants.%T) [symbolic = %specific_impl_fn.loc20_17.2 (constants.%specific_impl_fn.5b1)]
// CHECK:STDOUT:     %bound_method.loc20_23: <bound method> = bound_method %t.ref, %specific_impl_fn.loc20_17.1
// CHECK:STDOUT:     %.loc20_3.1: ref @F.%as_type.loc20_11.2 (%as_type.e31) = splice_block %a.var {}
// CHECK:STDOUT:     %Transitive.GetA.call: init @F.%as_type.loc20_11.2 (%as_type.e31) to %.loc20_3.1 = call %bound_method.loc20_23(%t.ref)
// CHECK:STDOUT:     assign %a.var, %Transitive.GetA.call
// CHECK:STDOUT:     %.loc20_11.1: type = splice_block %.loc20_11.3 [symbolic = %as_type.loc20_11.2 (constants.%as_type.e31)] {
// CHECK:STDOUT:       %T.ref.loc20: %Transitive.type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc20: type = facet_access_type %T.ref.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc20_11.2: type = converted %T.ref.loc20, %T.as_type.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %A.ref: %Transitive.assoc_type = name_ref A, @A.%assoc0 [concrete = constants.%assoc0.398]
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
// CHECK:STDOUT:     %.loc22_3.1: @F.%ImplicitAs.assoc_type.loc22 (%ImplicitAs.assoc_type.b4b) = specific_constant imports.%Core.import_ref.a80, @ImplicitAs(constants.%as_type.846) [symbolic = %assoc0.loc22 (constants.%assoc0.849)]
// CHECK:STDOUT:     %Convert.ref.loc22: @F.%ImplicitAs.assoc_type.loc22 (%ImplicitAs.assoc_type.b4b) = name_ref Convert, %.loc22_3.1 [symbolic = %assoc0.loc22 (constants.%assoc0.849)]
// CHECK:STDOUT:     %.loc22_3.2: %Q_where.type = as_compatible constants.%impl.elem1.4e4 [symbolic = %impl.elem1.loc22_3 (constants.%impl.elem1.4e4)]
// CHECK:STDOUT:     %impl.elem0.loc22_3.1: @F.%.loc22_3.6 (%.1e5) = impl_witness_access %.loc22_3.2, element0 [symbolic = %impl.elem0.loc22_3.3 (constants.%impl.elem0.694)]
// CHECK:STDOUT:     %bound_method.loc22_3.1: <bound method> = bound_method %a.ref.loc22, %impl.elem0.loc22_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_3.1: <specific function> = specific_impl_function %impl.elem0.loc22_3.1, @ImplicitAs.Convert(constants.%as_type.846, constants.%ImplicitAs.facet.0b4) [symbolic = %specific_impl_fn.loc22_3.3 (constants.%specific_impl_fn.dd7)]
// CHECK:STDOUT:     %bound_method.loc22_3.2: <bound method> = bound_method %a.ref.loc22, %specific_impl_fn.loc22_3.1
// CHECK:STDOUT:     %.loc22_3.3: ref @F.%as_type.loc22_11.2 (%as_type.846) = splice_block %b.var {}
// CHECK:STDOUT:     %.loc22_16: @F.%as_type.loc20_11.2 (%as_type.e31) = acquire_value %a.ref.loc22
// CHECK:STDOUT:     %ImplicitAs.Convert.call.loc22: init @F.%as_type.loc22_11.2 (%as_type.846) to %.loc22_3.3 = call %bound_method.loc22_3.2(%.loc22_16)
// CHECK:STDOUT:     %.loc22_3.4: init @F.%as_type.loc22_11.2 (%as_type.846) = converted %a.ref.loc22, %ImplicitAs.Convert.call.loc22
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
// CHECK:STDOUT:     %InQ.ref: %Q.assoc_type = name_ref InQ, @Q.%assoc0 [concrete = constants.%assoc0.135]
// CHECK:STDOUT:     %impl.elem0.loc23_4.1: @F.%.loc23_4 (%.736) = impl_witness_access constants.%Q.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc23_4.2 (constants.%impl.elem0.3bb)]
// CHECK:STDOUT:     %bound_method.loc23_4: <bound method> = bound_method %b.ref, %impl.elem0.loc23_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc23_4.1: <specific function> = specific_impl_function %impl.elem0.loc23_4.1, @Q.InQ(constants.%Q.facet) [symbolic = %specific_impl_fn.loc23_4.2 (constants.%specific_impl_fn.7f3)]
// CHECK:STDOUT:     %bound_method.loc23_9: <bound method> = bound_method %b.ref, %specific_impl_fn.loc23_4.1
// CHECK:STDOUT:     %.loc23_3: @F.%as_type.loc22_11.2 (%as_type.846) = acquire_value %b.ref
// CHECK:STDOUT:     %Q.InQ.call: init %empty_tuple.type = call %bound_method.loc23_9(%.loc23_3)
// CHECK:STDOUT:     %a.ref.loc25: ref @F.%as_type.loc20_11.2 (%as_type.e31) = name_ref a, %a
// CHECK:STDOUT:     %T.ref.loc25: %Transitive.type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc25: type = facet_access_type %T.ref.loc25 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc25_7: type = converted %T.ref.loc25, %T.as_type.loc25 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %C.ref.loc25: %Transitive.assoc_type = name_ref C, @C.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc25_7.1: %R_where.type = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element2 [symbolic = %impl.elem2.loc25_7.2 (constants.%impl.elem2.458)]
// CHECK:STDOUT:     %as_type.loc25_9.1: type = facet_access_type %impl.elem2.loc25_7.1 [symbolic = %as_type.loc25_9.2 (constants.%as_type.c65)]
// CHECK:STDOUT:     %.loc25_9.1: type = converted %impl.elem2.loc25_7.1, %as_type.loc25_9.1 [symbolic = %as_type.loc25_9.2 (constants.%as_type.c65)]
// CHECK:STDOUT:     %InR.ref.loc25: %R.assoc_type = name_ref InR, @R.%assoc0 [concrete = constants.%assoc0.4f7]
// CHECK:STDOUT:     %impl.elem0.loc25_9.1: @F.%.loc25_9.2 (%.a55) = impl_witness_access constants.%R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_9.2 (constants.%impl.elem0.2a7)]
// CHECK:STDOUT:     %bound_method.loc25_4: <bound method> = bound_method %a.ref.loc25, %impl.elem0.loc25_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_9.1: <specific function> = specific_impl_function %impl.elem0.loc25_9.1, @R.InR(constants.%R.facet) [symbolic = %specific_impl_fn.loc25_9.2 (constants.%specific_impl_fn.88b)]
// CHECK:STDOUT:     %bound_method.loc25_15: <bound method> = bound_method %a.ref.loc25, %specific_impl_fn.loc25_9.1
// CHECK:STDOUT:     %ImplicitAs.type.loc25_3.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%as_type.c65)> [symbolic = %ImplicitAs.type.loc25_3.2 (constants.%ImplicitAs.type.ac2)]
// CHECK:STDOUT:     %.loc25_3.1: @F.%ImplicitAs.assoc_type.loc25 (%ImplicitAs.assoc_type.eeb) = specific_constant imports.%Core.import_ref.a80, @ImplicitAs(constants.%as_type.c65) [symbolic = %assoc0.loc25 (constants.%assoc0.399)]
// CHECK:STDOUT:     %Convert.ref.loc25: @F.%ImplicitAs.assoc_type.loc25 (%ImplicitAs.assoc_type.eeb) = name_ref Convert, %.loc25_3.1 [symbolic = %assoc0.loc25 (constants.%assoc0.399)]
// CHECK:STDOUT:     %.loc25_3.2: %R_where.type = as_compatible constants.%impl.elem2.8f7 [symbolic = %impl.elem2.loc25_3 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:     %impl.elem0.loc25_3.1: @F.%.loc25_3.9 (%.ee0) = impl_witness_access %.loc25_3.2, element0 [symbolic = %impl.elem0.loc25_3.3 (constants.%impl.elem0.757)]
// CHECK:STDOUT:     %bound_method.loc25_3.1: <bound method> = bound_method %a.ref.loc25, %impl.elem0.loc25_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_3.1: <specific function> = specific_impl_function %impl.elem0.loc25_3.1, @ImplicitAs.Convert(constants.%as_type.c65, constants.%ImplicitAs.facet.304) [symbolic = %specific_impl_fn.loc25_3.3 (constants.%specific_impl_fn.207)]
// CHECK:STDOUT:     %bound_method.loc25_3.2: <bound method> = bound_method %a.ref.loc25, %specific_impl_fn.loc25_3.1
// CHECK:STDOUT:     %.loc25_3.3: ref @F.%as_type.loc25_9.2 (%as_type.c65) = temporary_storage
// CHECK:STDOUT:     %.loc25_3.4: @F.%as_type.loc20_11.2 (%as_type.e31) = acquire_value %a.ref.loc25
// CHECK:STDOUT:     %ImplicitAs.Convert.call.loc25: init @F.%as_type.loc25_9.2 (%as_type.c65) to %.loc25_3.3 = call %bound_method.loc25_3.2(%.loc25_3.4)
// CHECK:STDOUT:     %.loc25_3.5: init @F.%as_type.loc25_9.2 (%as_type.c65) = converted %a.ref.loc25, %ImplicitAs.Convert.call.loc25
// CHECK:STDOUT:     %.loc25_3.6: ref @F.%as_type.loc25_9.2 (%as_type.c65) = temporary %.loc25_3.3, %.loc25_3.5
// CHECK:STDOUT:     %.loc25_3.7: @F.%as_type.loc25_9.2 (%as_type.c65) = acquire_value %.loc25_3.6
// CHECK:STDOUT:     %R.InR.call.loc25: init %empty_tuple.type = call %bound_method.loc25_15(%.loc25_3.7)
// CHECK:STDOUT:     %a.ref.loc26: ref @F.%as_type.loc20_11.2 (%as_type.e31) = name_ref a, %a
// CHECK:STDOUT:     %T.ref.loc26: %Transitive.type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc26: type = facet_access_type %T.ref.loc26 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc26_10.1: type = converted %T.ref.loc26, %T.as_type.loc26 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %C.ref.loc26: %Transitive.assoc_type = name_ref C, @C.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc26: %R_where.type = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element2 [symbolic = %impl.elem2.loc25_7.2 (constants.%impl.elem2.458)]
// CHECK:STDOUT:     %as_type.loc26: type = facet_access_type %impl.elem2.loc26 [symbolic = %as_type.loc25_9.2 (constants.%as_type.c65)]
// CHECK:STDOUT:     %.loc26_10.2: type = converted %impl.elem2.loc26, %as_type.loc26 [symbolic = %as_type.loc25_9.2 (constants.%as_type.c65)]
// CHECK:STDOUT:     %As.type.loc26_6.1: type = facet_type <@As, @As(constants.%as_type.c65)> [symbolic = %As.type.loc26_6.2 (constants.%As.type.ae5)]
// CHECK:STDOUT:     %.loc26_6.1: @F.%As.assoc_type (%As.assoc_type.69a) = specific_constant imports.%Core.import_ref.e0b, @As(constants.%as_type.c65) [symbolic = %assoc0.loc26 (constants.%assoc0.685)]
// CHECK:STDOUT:     %Convert.ref.loc26: @F.%As.assoc_type (%As.assoc_type.69a) = name_ref Convert, %.loc26_6.1 [symbolic = %assoc0.loc26 (constants.%assoc0.685)]
// CHECK:STDOUT:     %.loc26_6.2: %R_where.type = as_compatible constants.%impl.elem2.8f7 [symbolic = %impl.elem2.loc25_3 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:     %impl.elem0.loc26_6.1: @F.%.loc26_6.8 (%.d27) = impl_witness_access %.loc26_6.2, element0 [symbolic = %impl.elem0.loc26_6.3 (constants.%impl.elem0.1b9)]
// CHECK:STDOUT:     %bound_method.loc26_6.1: <bound method> = bound_method %a.ref.loc26, %impl.elem0.loc26_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc26_6.1: <specific function> = specific_impl_function %impl.elem0.loc26_6.1, @As.Convert(constants.%as_type.c65, constants.%As.facet) [symbolic = %specific_impl_fn.loc26_6.3 (constants.%specific_impl_fn.0f7)]
// CHECK:STDOUT:     %bound_method.loc26_6.2: <bound method> = bound_method %a.ref.loc26, %specific_impl_fn.loc26_6.1
// CHECK:STDOUT:     %.loc26_6.3: ref @F.%as_type.loc25_9.2 (%as_type.c65) = temporary_storage
// CHECK:STDOUT:     %.loc26_4: @F.%as_type.loc20_11.2 (%as_type.e31) = acquire_value %a.ref.loc26
// CHECK:STDOUT:     %As.Convert.call: init @F.%as_type.loc25_9.2 (%as_type.c65) to %.loc26_6.3 = call %bound_method.loc26_6.2(%.loc26_4)
// CHECK:STDOUT:     %.loc26_6.4: init @F.%as_type.loc25_9.2 (%as_type.c65) = converted %a.ref.loc26, %As.Convert.call
// CHECK:STDOUT:     %.loc26_6.5: ref @F.%as_type.loc25_9.2 (%as_type.c65) = temporary %.loc26_6.3, %.loc26_6.4
// CHECK:STDOUT:     %InR.ref.loc26: %R.assoc_type = name_ref InR, @R.%assoc0 [concrete = constants.%assoc0.4f7]
// CHECK:STDOUT:     %impl.elem0.loc26_13: @F.%.loc25_9.2 (%.a55) = impl_witness_access constants.%R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_9.2 (constants.%impl.elem0.2a7)]
// CHECK:STDOUT:     %bound_method.loc26_13: <bound method> = bound_method %.loc26_6.5, %impl.elem0.loc26_13
// CHECK:STDOUT:     %specific_impl_fn.loc26_13: <specific function> = specific_impl_function %impl.elem0.loc26_13, @R.InR(constants.%R.facet) [symbolic = %specific_impl_fn.loc25_9.2 (constants.%specific_impl_fn.88b)]
// CHECK:STDOUT:     %bound_method.loc26_18: <bound method> = bound_method %.loc26_6.5, %specific_impl_fn.loc26_13
// CHECK:STDOUT:     %.loc26_6.6: @F.%as_type.loc25_9.2 (%as_type.c65) = acquire_value %.loc26_6.5
// CHECK:STDOUT:     %R.InR.call.loc26: init %empty_tuple.type = call %bound_method.loc26_18(%.loc26_6.6)
// CHECK:STDOUT:     %.loc26_6.7: %R_where.type = as_compatible constants.%impl.elem2.8f7 [symbolic = %impl.elem2.loc25_3 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:     %impl.elem0.loc26_6.2: @F.%.loc26_6.9 (%.e6f) = impl_witness_access %.loc26_6.7, element0 [symbolic = %impl.elem0.loc26_6.4 (constants.%impl.elem0.ff7)]
// CHECK:STDOUT:     %bound_method.loc26_6.3: <bound method> = bound_method %.loc26_6.5, %impl.elem0.loc26_6.2
// CHECK:STDOUT:     %specific_impl_fn.loc26_6.2: <specific function> = specific_impl_function %impl.elem0.loc26_6.2, @Destroy.Op(constants.%Destroy.facet.7a0) [symbolic = %specific_impl_fn.loc26_6.4 (constants.%specific_impl_fn.d83)]
// CHECK:STDOUT:     %bound_method.loc26_6.4: <bound method> = bound_method %.loc26_6.5, %specific_impl_fn.loc26_6.2
// CHECK:STDOUT:     %Destroy.Op.call.loc26: init %empty_tuple.type = call %bound_method.loc26_6.4(%.loc26_6.5)
// CHECK:STDOUT:     %.loc25_3.8: %R_where.type = as_compatible constants.%impl.elem2.8f7 [symbolic = %impl.elem2.loc25_3 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:     %impl.elem0.loc25_3.2: @F.%.loc26_6.9 (%.e6f) = impl_witness_access %.loc25_3.8, element0 [symbolic = %impl.elem0.loc26_6.4 (constants.%impl.elem0.ff7)]
// CHECK:STDOUT:     %bound_method.loc25_3.3: <bound method> = bound_method %.loc25_3.6, %impl.elem0.loc25_3.2
// CHECK:STDOUT:     %specific_impl_fn.loc25_3.2: <specific function> = specific_impl_function %impl.elem0.loc25_3.2, @Destroy.Op(constants.%Destroy.facet.7a0) [symbolic = %specific_impl_fn.loc26_6.4 (constants.%specific_impl_fn.d83)]
// CHECK:STDOUT:     %bound_method.loc25_3.4: <bound method> = bound_method %.loc25_3.6, %specific_impl_fn.loc25_3.2
// CHECK:STDOUT:     %Destroy.Op.call.loc25: init %empty_tuple.type = call %bound_method.loc25_3.4(%.loc25_3.6)
// CHECK:STDOUT:     %.loc22_3.5: %Q_where.type = as_compatible constants.%impl.elem1.4e4 [symbolic = %impl.elem1.loc22_3 (constants.%impl.elem1.4e4)]
// CHECK:STDOUT:     %impl.elem0.loc22_3.2: @F.%.loc22_3.7 (%.f9e) = impl_witness_access %.loc22_3.5, element0 [symbolic = %impl.elem0.loc22_3.4 (constants.%impl.elem0.50f)]
// CHECK:STDOUT:     %bound_method.loc22_3.3: <bound method> = bound_method %b.var, %impl.elem0.loc22_3.2
// CHECK:STDOUT:     %specific_impl_fn.loc22_3.2: <specific function> = specific_impl_function %impl.elem0.loc22_3.2, @Destroy.Op(constants.%Destroy.facet.69c) [symbolic = %specific_impl_fn.loc22_3.4 (constants.%specific_impl_fn.e4c)]
// CHECK:STDOUT:     %bound_method.loc22_3.4: <bound method> = bound_method %b.var, %specific_impl_fn.loc22_3.2
// CHECK:STDOUT:     %Destroy.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22_3.4(%b.var)
// CHECK:STDOUT:     %.loc20_3.2: %P.type = as_compatible constants.%impl.elem0.9bd [symbolic = %impl.elem0.loc20_3.2 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %impl.elem0.loc20_3.1: @F.%.loc20_3.3 (%.8f2) = impl_witness_access %.loc20_3.2, element0 [symbolic = %impl.elem0.loc20_3.3 (constants.%impl.elem0.28e)]
// CHECK:STDOUT:     %bound_method.loc20_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc20_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc20_3.1: <specific function> = specific_impl_function %impl.elem0.loc20_3.1, @Destroy.Op(constants.%Destroy.facet.96d) [symbolic = %specific_impl_fn.loc20_3.2 (constants.%specific_impl_fn.1b0)]
// CHECK:STDOUT:     %bound_method.loc20_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc20_3.1
// CHECK:STDOUT:     %Destroy.Op.call.loc20: init %empty_tuple.type = call %bound_method.loc20_3.2(%a.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.InP(constants.%Self.d21) {
// CHECK:STDOUT:   %Self => constants.%Self.d21
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.6c8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.InQ(constants.%Self.f64) {
// CHECK:STDOUT:   %Self => constants.%Self.f64
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.1b2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.InR(constants.%Self.bea) {
// CHECK:STDOUT:   %Self => constants.%Self.bea
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.296
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.351
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self.e2a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.9bd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.4e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.GetA(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e5b
// CHECK:STDOUT:   %pattern_type.loc13_11 => constants.%pattern_type.b0d
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem0.loc13_28.1 => constants.%impl.elem0.9bd
// CHECK:STDOUT:   %as_type => constants.%as_type.0ea
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.e64
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.bff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.TakesC(constants.%Self.e2a) {
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
// CHECK:STDOUT: specific @A(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc13_11 => constants.%pattern_type.993
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8
// CHECK:STDOUT:   %impl.elem0.loc13_28.1 => constants.%impl.elem0.b09
// CHECK:STDOUT:   %as_type => constants.%as_type.e31
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.d40
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.05e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.b09
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.InQ(constants.%Q.facet) {
// CHECK:STDOUT:   %Self => constants.%Q.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%as_type.846
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.32a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.InR(constants.%R.facet) {
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
// CHECK:STDOUT:     %.loc8_24.1: type = splice_block %.loc8_24.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc8_14.2 [symbolic = %T.loc8_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:       %.loc8_24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
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
// CHECK:STDOUT:   %B.Self.binding.as_type.impls.A.type.require.decl = require_decl @B.Self.binding.as_type.impls.A.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %A.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @B.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.8ee)]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @B.Self.binding.as_type.impls.A.type.require {
// CHECK:STDOUT:     require @B.Self.binding.as_type.impls.A.type.require.%Self.as_type impls @B.Self.binding.as_type.impls.A.type.require.%A.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b7c]
// CHECK:STDOUT:   %C.Self.binding.as_type.impls.B.type.require.decl = require_decl @C.Self.binding.as_type.impls.B.type.require [concrete] {
// CHECK:STDOUT:     require %Self.as_type impls %B.ref
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type: type = facet_access_type @C.%Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.89a)]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT:   @C.Self.binding.as_type.impls.B.type.require {
// CHECK:STDOUT:     require @C.Self.binding.as_type.impls.B.type.require.%Self.as_type impls @C.Self.binding.as_type.impls.B.type.require.%B.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @B.Self.binding.as_type.impls.A.type.require(@B.%Self: %B.type) {
// CHECK:STDOUT:   %Self: %B.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d0b)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.8ee)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @C.Self.binding.as_type.impls.B.type.require(@C.%Self: %C.type) {
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
// CHECK:STDOUT: specific @B.Self.binding.as_type.impls.A.type.require(constants.%Self.d0b) {
// CHECK:STDOUT:   %Self => constants.%Self.d0b
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.8ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.Self.binding.as_type.impls.B.type.require(constants.%Self.b7c) {
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
// CHECK:STDOUT:   %P.InP.type: type = fn_type @P.InP [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %P.InP: %P.InP.type = struct_value () [concrete]
// CHECK:STDOUT:   %P.assoc_type: type = assoc_entity_type @P [concrete]
// CHECK:STDOUT:   %assoc0.671: %P.assoc_type = assoc_entity element0, @P.%P.InP.decl [concrete]
// CHECK:STDOUT:   %Q.type: type = facet_type <@Q> [concrete]
// CHECK:STDOUT:   %Self.f64: %Q.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.1b2: type = symbolic_binding_type Self, 0, %Self.f64 [symbolic]
// CHECK:STDOUT:   %pattern_type.992: type = pattern_type %Self.binding.as_type.1b2 [symbolic]
// CHECK:STDOUT:   %Q.InQ.type: type = fn_type @Q.InQ [concrete]
// CHECK:STDOUT:   %Q.InQ: %Q.InQ.type = struct_value () [concrete]
// CHECK:STDOUT:   %Q.assoc_type: type = assoc_entity_type @Q [concrete]
// CHECK:STDOUT:   %assoc0.135: %Q.assoc_type = assoc_entity element0, @Q.%Q.InQ.decl [concrete]
// CHECK:STDOUT:   %R.type: type = facet_type <@R> [concrete]
// CHECK:STDOUT:   %Self.bea: %R.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.296: type = symbolic_binding_type Self, 0, %Self.bea [symbolic]
// CHECK:STDOUT:   %pattern_type.351: type = pattern_type %Self.binding.as_type.296 [symbolic]
// CHECK:STDOUT:   %R.InR.type: type = fn_type @R.InR [concrete]
// CHECK:STDOUT:   %R.InR: %R.InR.type = struct_value () [concrete]
// CHECK:STDOUT:   %R.assoc_type: type = assoc_entity_type @R [concrete]
// CHECK:STDOUT:   %assoc0.4f7: %R.assoc_type = assoc_entity element0, @R.%R.InR.decl [concrete]
// CHECK:STDOUT:   %Transitive.type: type = facet_type <@Transitive> [concrete]
// CHECK:STDOUT:   %Self.e2a: %Transitive.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Transitive.assoc_type: type = assoc_entity_type @Transitive [concrete]
// CHECK:STDOUT:   %assoc0.398: %Transitive.assoc_type = assoc_entity element0, @Transitive.%A [concrete]
// CHECK:STDOUT:   %Self.binding.as_type.e5b: type = symbolic_binding_type Self, 0, %Self.e2a [symbolic]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.804: <witness> = lookup_impl_witness %Self.e2a, @Transitive [symbolic]
// CHECK:STDOUT:   %impl.elem0.9bd: %P.type = impl_witness_access %Transitive.lookup_impl_witness.804, element0 [symbolic]
// CHECK:STDOUT:   %Q_where.type: type = facet_type <@Q where TODO> [concrete]
// CHECK:STDOUT:   %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.%B [concrete]
// CHECK:STDOUT:   %impl.elem1.4e4: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.804, element1 [symbolic]
// CHECK:STDOUT:   %R_where.type: type = facet_type <@R where TODO> [concrete]
// CHECK:STDOUT:   %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.%C [concrete]
// CHECK:STDOUT:   %pattern_type.b0d: type = pattern_type %Self.binding.as_type.e5b [symbolic]
// CHECK:STDOUT:   %as_type.0ea: type = facet_access_type %impl.elem0.9bd [symbolic]
// CHECK:STDOUT:   %.e64: form = init_form %as_type.0ea, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.bff: type = pattern_type %as_type.0ea [symbolic]
// CHECK:STDOUT:   %Transitive.GetA.type: type = fn_type @Transitive.GetA [concrete]
// CHECK:STDOUT:   %Transitive.GetA: %Transitive.GetA.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, @Transitive.%Transitive.GetA.decl [concrete]
// CHECK:STDOUT:   %impl.elem2.8f7: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.804, element2 [symbolic]
// CHECK:STDOUT:   %as_type.ecc: type = facet_access_type %impl.elem2.8f7 [symbolic]
// CHECK:STDOUT:   %pattern_type.364: type = pattern_type %as_type.ecc [symbolic]
// CHECK:STDOUT:   %Transitive.TakesC.type: type = fn_type @Transitive.TakesC [concrete]
// CHECK:STDOUT:   %Transitive.TakesC: %Transitive.TakesC.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, @Transitive.%Transitive.TakesC.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.890: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
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
// CHECK:STDOUT:   %as_type.e31: type = facet_access_type %impl.elem0.b09 [symbolic]
// CHECK:STDOUT:   %require_complete.b46: <witness> = require_complete_type %as_type.e31 [symbolic]
// CHECK:STDOUT:   %pattern_type.05e: type = pattern_type %as_type.e31 [symbolic]
// CHECK:STDOUT:   %.f7d: type = fn_type_with_self_type %Transitive.GetA.type, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.f7d = impl_witness_access %Transitive.lookup_impl_witness.6d8, element3 [symbolic]
// CHECK:STDOUT:   %.d40: form = init_form %as_type.e31, call_param1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5b1: <specific function> = specific_impl_function %impl.elem3, @Transitive.GetA(%T) [symbolic]
// CHECK:STDOUT:   %impl.elem1.32a: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element1 [symbolic]
// CHECK:STDOUT:   %as_type.846: type = facet_access_type %impl.elem1.32a [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.6ed: %Q.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element1 [symbolic]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.e31, (%impl.elem1.6ed) [symbolic]
// CHECK:STDOUT:   %.fcd: type = fn_type_with_self_type %Q.InQ.type, %Q.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.38b: %.fcd = impl_witness_access %impl.elem1.6ed, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d03: <specific function> = specific_impl_function %impl.elem0.38b, @Q.InQ(%Q.facet) [symbolic]
// CHECK:STDOUT:   %impl.elem2.458: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element2 [symbolic]
// CHECK:STDOUT:   %as_type.c65: type = facet_access_type %impl.elem2.458 [symbolic]
// CHECK:STDOUT:   %P.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.b09, @P [symbolic]
// CHECK:STDOUT:   %impl.elem2.a7e: %R.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element2 [symbolic]
// CHECK:STDOUT:   %facet_value: %facet_type.7a2 = facet_value %as_type.e31, (%P.lookup_impl_witness, %impl.elem1.6ed, %impl.elem2.a7e) [symbolic]
// CHECK:STDOUT:   %TakesPQR.specific_fn: <specific function> = specific_function %TakesPQR, @TakesPQR(%facet_value) [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.e31, (%impl.elem0.9bd) [symbolic]
// CHECK:STDOUT:   %.8f2: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.28e: %.8f2 = impl_witness_access %impl.elem0.9bd, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1b0: <specific function> = specific_impl_function %impl.elem0.28e, @Destroy.Op(%Destroy.facet) [symbolic]
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
// CHECK:STDOUT:       %impl.elem0.loc19_19: %.890 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc19_19: <bound method> = bound_method %P.ref, %impl.elem0.loc19_19 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.aef]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc19_19: init type = call %bound_method.loc19_19(%P.ref, %Q.ref) [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %R.ref: type = name_ref R, file.%R.decl [concrete = constants.%R.type]
// CHECK:STDOUT:       %impl.elem0.loc19_23: %.890 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc19_23: <bound method> = bound_method %type.as.BitAndWith.impl.Op.call.loc19_19, %impl.elem0.loc19_23 [concrete = constants.%type.as.BitAndWith.impl.Op.bound.cf2]
// CHECK:STDOUT:       %.loc19_19.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc19_19 [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %.loc19_19.2: type = converted %type.as.BitAndWith.impl.Op.call.loc19_19, %.loc19_19.1 [concrete = constants.%facet_type.dec]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call.loc19_23: init type = call %bound_method.loc19_23(%.loc19_19.2, %R.ref) [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:       %.loc19_23.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc19_23 [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:       %.loc19_23.3: type = converted %type.as.BitAndWith.impl.Op.call.loc19_23, %.loc19_23.2 [concrete = constants.%facet_type.7a2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc19_13.2: %facet_type.7a2 = symbolic_binding U, 0 [symbolic = %U.loc19_13.1 (constants.%U)]
// CHECK:STDOUT:     %u.param: @TakesPQR.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc19_31.1: type = splice_block %.loc19_31.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %facet_type.7a2 = name_ref U, %U.loc19_13.2 [symbolic = %U.loc19_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc19_31.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
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
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d21]
// CHECK:STDOUT:   %P.InP.decl: %P.InP.type = fn_decl @P.InP [concrete = constants.%P.InP] {
// CHECK:STDOUT:     %self.patt: @P.InP.%pattern_type (%pattern_type.2eb) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @P.InP.%pattern_type (%pattern_type.2eb) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @P.InP.%Self.binding.as_type (%Self.binding.as_type.6c8) = value_param call_param0
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)] {
// CHECK:STDOUT:       %Self.ref: %P.type = name_ref Self, @P.%Self [symbolic = %Self (constants.%Self.d21)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:       %.loc4_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @P.InP.%Self.binding.as_type (%Self.binding.as_type.6c8) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %P.assoc_type = assoc_entity element0, %P.InP.decl [concrete = constants.%assoc0.671]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InP = %assoc0
// CHECK:STDOUT:   witness = (%P.InP.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Q {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f64]
// CHECK:STDOUT:   %Q.InQ.decl: %Q.InQ.type = fn_decl @Q.InQ [concrete = constants.%Q.InQ] {
// CHECK:STDOUT:     %self.patt: @Q.InQ.%pattern_type (%pattern_type.992) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Q.InQ.%pattern_type (%pattern_type.992) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Q.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2) = value_param call_param0
// CHECK:STDOUT:     %.loc5_28.1: type = splice_block %.loc5_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)] {
// CHECK:STDOUT:       %Self.ref: %Q.type = name_ref Self, @Q.%Self [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:       %.loc5_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Q.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Q.assoc_type = assoc_entity element0, %Q.InQ.decl [concrete = constants.%assoc0.135]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InQ = %assoc0
// CHECK:STDOUT:   witness = (%Q.InQ.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @R {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = constants.%Self.bea]
// CHECK:STDOUT:   %R.InR.decl: %R.InR.type = fn_decl @R.InR [concrete = constants.%R.InR] {
// CHECK:STDOUT:     %self.patt: @R.InR.%pattern_type (%pattern_type.351) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @R.InR.%pattern_type (%pattern_type.351) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @R.InR.%Self.binding.as_type (%Self.binding.as_type.296) = value_param call_param0
// CHECK:STDOUT:     %.loc6_28.1: type = splice_block %.loc6_28.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)] {
// CHECK:STDOUT:       %Self.ref: %R.type = name_ref Self, @R.%Self [symbolic = %Self (constants.%Self.bea)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:       %.loc6_28.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @R.InR.%Self.binding.as_type (%Self.binding.as_type.296) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %R.assoc_type = assoc_entity element0, %R.InR.decl [concrete = constants.%assoc0.4f7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .InR = %assoc0
// CHECK:STDOUT:   witness = (%R.InR.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Transitive {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e2a]
// CHECK:STDOUT:   %A: %P.type = assoc_const_decl @A [concrete] {
// CHECK:STDOUT:     %assoc0: %Transitive.assoc_type = assoc_entity element0, @Transitive.%A [concrete = constants.%assoc0.398]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: %Q_where.type = assoc_const_decl @B [concrete] {
// CHECK:STDOUT:     %assoc1: %Transitive.assoc_type = assoc_entity element1, @Transitive.%B [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: %R_where.type = assoc_const_decl @C [concrete] {
// CHECK:STDOUT:     %assoc2: %Transitive.assoc_type = assoc_entity element2, @Transitive.%C [concrete = constants.%assoc2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Transitive.GetA.decl: %Transitive.GetA.type = fn_decl @Transitive.GetA [concrete = constants.%Transitive.GetA] {
// CHECK:STDOUT:     %self.patt: @Transitive.GetA.%pattern_type.loc13_11 (%pattern_type.b0d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Transitive.GetA.%pattern_type.loc13_11 (%pattern_type.b0d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Transitive.GetA.%pattern_type.loc13_25 (%pattern_type.bff) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Transitive.GetA.%pattern_type.loc13_25 (%pattern_type.bff) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %impl.elem0.loc13_28.2: %P.type = impl_witness_access constants.%Transitive.lookup_impl_witness.804, element0 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %A.ref: %P.type = name_ref A, %impl.elem0.loc13_28.2 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %A.as_type: type = facet_access_type %A.ref [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:     %.loc13_28.2: type = converted %A.ref, %A.as_type [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:     %.loc13_28.3: form = init_form %.loc13_28.2, call_param1 [symbolic = %.loc13_28.1 (constants.%.e64)]
// CHECK:STDOUT:     %self.param: @Transitive.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_param call_param0
// CHECK:STDOUT:     %.loc13_17.1: type = splice_block %.loc13_17.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:       %.loc13_17.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Transitive.GetA.%as_type (%as_type.0ea) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Transitive.GetA.%as_type (%as_type.0ea) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc3: %Transitive.assoc_type = assoc_entity element3, %Transitive.GetA.decl [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Transitive.TakesC.decl: %Transitive.TakesC.type = fn_decl @Transitive.TakesC [concrete = constants.%Transitive.TakesC] {
// CHECK:STDOUT:     %self.patt: @Transitive.TakesC.%pattern_type.loc14_13 (%pattern_type.b0d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Transitive.TakesC.%pattern_type.loc14_13 (%pattern_type.b0d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %c.patt: @Transitive.TakesC.%pattern_type.loc14_25 (%pattern_type.364) = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: @Transitive.TakesC.%pattern_type.loc14_25 (%pattern_type.364) = value_param_pattern %c.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Transitive.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_param call_param0
// CHECK:STDOUT:     %.loc14_19.1: type = splice_block %.loc14_19.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)] {
// CHECK:STDOUT:       %Self.ref: %Transitive.type = name_ref Self, @Transitive.%Self [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:       %.loc14_19.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Transitive.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b) = value_binding self, %self.param
// CHECK:STDOUT:     %c.param: @Transitive.TakesC.%as_type (%as_type.ecc) = value_param call_param1
// CHECK:STDOUT:     %.loc14_28.1: type = splice_block %.loc14_28.2 [symbolic = %as_type (constants.%as_type.ecc)] {
// CHECK:STDOUT:       %impl.elem2.loc14_28.2: %R_where.type = impl_witness_access constants.%Transitive.lookup_impl_witness.804, element2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:       %C.ref: %R_where.type = name_ref C, %impl.elem2.loc14_28.2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:       %C.as_type: type = facet_access_type %C.ref [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:       %.loc14_28.2: type = converted %C.ref, %C.as_type [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @Transitive.TakesC.%as_type (%as_type.ecc) = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc4: %Transitive.assoc_type = assoc_entity element4, %Transitive.TakesC.decl [concrete = constants.%assoc4]
// CHECK:STDOUT:   %Transitive.observe.decl = observe_decl @Transitive.observe, (@Transitive.observe.%impl.elem0, @Transitive.observe.%A.ref, @Transitive.observe.%impl.elem1, @Transitive.observe.%B.ref, @Transitive.observe.%B.as_type.loc16_16.1, @Transitive.observe.%.loc16_16.1, @Transitive.observe.%A.as_type, @Transitive.observe.%.loc16_11, @Transitive.observe.%eq.loc16_13, @Transitive.observe.%impl.elem2, @Transitive.observe.%C.ref, @Transitive.observe.%C.as_type, @Transitive.observe.%.loc16_21, @Transitive.observe.%B.as_type.loc16_16.2, @Transitive.observe.%.loc16_16.2, @Transitive.observe.%eq.loc16_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .P = <poisoned>
// CHECK:STDOUT:   .A = @A.%assoc0
// CHECK:STDOUT:   .Q = <poisoned>
// CHECK:STDOUT:   .B = @B.%assoc1
// CHECK:STDOUT:   .R = <poisoned>
// CHECK:STDOUT:   .C = @C.%assoc2
// CHECK:STDOUT:   .GetA = %assoc3
// CHECK:STDOUT:   .TakesC = %assoc4
// CHECK:STDOUT:   witness = (%A, %B, %C, %Transitive.GetA.decl, %Transitive.TakesC.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @A(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   assoc_const A:! %P.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @B(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem0: %P.type = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const B:! %Q_where.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @C(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem1: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness, element1 [symbolic = %impl.elem1 (constants.%impl.elem1.4e4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const C:! %R_where.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @P.InP(@P.%Self: %P.type) {
// CHECK:STDOUT:   %Self: %P.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.d21)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.6c8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.2eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @P.InP.%Self.binding.as_type (%Self.binding.as_type.6c8));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Q.InQ(@Q.%Self: %Q.type) {
// CHECK:STDOUT:   %Self: %Q.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f64)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.1b2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.992)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Q.InQ.%Self.binding.as_type (%Self.binding.as_type.1b2));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @R.InR(@R.%Self: %R.type) {
// CHECK:STDOUT:   %Self: %R.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.bea)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.296)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.351)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @R.InR.%Self.binding.as_type (%Self.binding.as_type.296));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.GetA(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:   %pattern_type.loc13_11: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc13_11 (constants.%pattern_type.b0d)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem0.loc13_28.1: %P.type = impl_witness_access %Transitive.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_28.1 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem0.loc13_28.1 [symbolic = %as_type (constants.%as_type.0ea)]
// CHECK:STDOUT:   %.loc13_28.1: form = init_form %as_type, call_param1 [symbolic = %.loc13_28.1 (constants.%.e64)]
// CHECK:STDOUT:   %pattern_type.loc13_25: type = pattern_type %as_type [symbolic = %pattern_type.loc13_25 (constants.%pattern_type.bff)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.GetA.%Self.binding.as_type (%Self.binding.as_type.e5b)) -> out %return.param: @Transitive.GetA.%as_type (%as_type.0ea);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Transitive.TakesC(@Transitive.%Self: %Transitive.type) {
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.e5b)]
// CHECK:STDOUT:   %pattern_type.loc14_13: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc14_13 (constants.%pattern_type.b0d)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem2.loc14_28.1: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc14_28.1 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:   %as_type: type = facet_access_type %impl.elem2.loc14_28.1 [symbolic = %as_type (constants.%as_type.ecc)]
// CHECK:STDOUT:   %pattern_type.loc14_25: type = pattern_type %as_type [symbolic = %pattern_type.loc14_25 (constants.%pattern_type.364)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Transitive.TakesC.%Self.binding.as_type (%Self.binding.as_type.e5b), %c.param: @Transitive.TakesC.%as_type (%as_type.ecc));
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
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.loc22_11: <witness> = lookup_impl_witness %T.loc21_6.1, @Transitive [symbolic = %Transitive.lookup_impl_witness.loc22_11 (constants.%Transitive.lookup_impl_witness.6d8)]
// CHECK:STDOUT:   %impl.elem0.loc22_11.2: %P.type = impl_witness_access %Transitive.lookup_impl_witness.loc22_11, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.b09)]
// CHECK:STDOUT:   %as_type.loc22_11.2: type = facet_access_type %impl.elem0.loc22_11.2 [symbolic = %as_type.loc22_11.2 (constants.%as_type.e31)]
// CHECK:STDOUT:   %require_complete.loc22: <witness> = require_complete_type %as_type.loc22_11.2 [symbolic = %require_complete.loc22 (constants.%require_complete.b46)]
// CHECK:STDOUT:   %pattern_type.loc22: type = pattern_type %as_type.loc22_11.2 [symbolic = %pattern_type.loc22 (constants.%pattern_type.05e)]
// CHECK:STDOUT:   %.loc22_17: type = fn_type_with_self_type constants.%Transitive.GetA.type, %T.loc21_6.1 [symbolic = %.loc22_17 (constants.%.f7d)]
// CHECK:STDOUT:   %impl.elem3.loc22_17.2: @G.%.loc22_17 (%.f7d) = impl_witness_access %Transitive.lookup_impl_witness.loc22_11, element3 [symbolic = %impl.elem3.loc22_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_17.2: <specific function> = specific_impl_function %impl.elem3.loc22_17.2, @Transitive.GetA(%T.loc21_6.1) [symbolic = %specific_impl_fn.loc22_17.2 (constants.%specific_impl_fn.5b1)]
// CHECK:STDOUT:   %impl.elem1.loc24: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.loc22_11, element1 [symbolic = %impl.elem1.loc24 (constants.%impl.elem1.32a)]
// CHECK:STDOUT:   %as_type.loc24: type = facet_access_type %impl.elem1.loc24 [symbolic = %as_type.loc24 (constants.%as_type.846)]
// CHECK:STDOUT:   %impl.elem1.loc25: %Q.type = impl_witness_access %Transitive.lookup_impl_witness.loc22_11, element1 [symbolic = %impl.elem1.loc25 (constants.%impl.elem1.6ed)]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.loc22_11.2, (%impl.elem1.loc25) [symbolic = %Q.facet (constants.%Q.facet)]
// CHECK:STDOUT:   %.loc25_4.2: type = fn_type_with_self_type constants.%Q.InQ.type, %Q.facet [symbolic = %.loc25_4.2 (constants.%.fcd)]
// CHECK:STDOUT:   %impl.elem0.loc25_4.2: @G.%.loc25_4.2 (%.fcd) = impl_witness_access %impl.elem1.loc25, element0 [symbolic = %impl.elem0.loc25_4.2 (constants.%impl.elem0.38b)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_4.2: <specific function> = specific_impl_function %impl.elem0.loc25_4.2, @Q.InQ(%Q.facet) [symbolic = %specific_impl_fn.loc25_4.2 (constants.%specific_impl_fn.d03)]
// CHECK:STDOUT:   %impl.elem2.loc27: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.loc22_11, element2 [symbolic = %impl.elem2.loc27 (constants.%impl.elem2.458)]
// CHECK:STDOUT:   %as_type.loc27: type = facet_access_type %impl.elem2.loc27 [symbolic = %as_type.loc27 (constants.%as_type.c65)]
// CHECK:STDOUT:   %P.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.loc22_11.2, @P [symbolic = %P.lookup_impl_witness (constants.%P.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem2.loc28: %R.type = impl_witness_access %Transitive.lookup_impl_witness.loc22_11, element2 [symbolic = %impl.elem2.loc28 (constants.%impl.elem2.a7e)]
// CHECK:STDOUT:   %facet_value.loc28_13.3: %facet_type.7a2 = facet_value %as_type.loc22_11.2, (%P.lookup_impl_witness, %impl.elem1.loc25, %impl.elem2.loc28) [symbolic = %facet_value.loc28_13.3 (constants.%facet_value)]
// CHECK:STDOUT:   %TakesPQR.specific_fn.loc28_3.2: <specific function> = specific_function constants.%TakesPQR, @TakesPQR(%facet_value.loc28_13.3) [symbolic = %TakesPQR.specific_fn.loc28_3.2 (constants.%TakesPQR.specific_fn)]
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.loc22_3: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness.loc22_3 (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.2: %P.type = impl_witness_access %Transitive.lookup_impl_witness.loc22_3, element0 [symbolic = %impl.elem0.loc22_3.2 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %as_type.loc22_11.2, (%impl.elem0.loc22_3.2) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc22_3.3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc22_3.3 (constants.%.8f2)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.3: @G.%.loc22_3.3 (%.8f2) = impl_witness_access %impl.elem0.loc22_3.2, element0 [symbolic = %impl.elem0.loc22_3.3 (constants.%impl.elem0.28e)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_3.2: <specific function> = specific_impl_function %impl.elem0.loc22_3.3, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc22_3.2 (constants.%specific_impl_fn.1b0)]
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
// CHECK:STDOUT:     %impl.elem3.loc22_17.1: @G.%.loc22_17 (%.f7d) = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element3 [symbolic = %impl.elem3.loc22_17.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc22_17: <bound method> = bound_method %t.ref, %impl.elem3.loc22_17.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_17.1: <specific function> = specific_impl_function %impl.elem3.loc22_17.1, @Transitive.GetA(constants.%T) [symbolic = %specific_impl_fn.loc22_17.2 (constants.%specific_impl_fn.5b1)]
// CHECK:STDOUT:     %bound_method.loc22_23: <bound method> = bound_method %t.ref, %specific_impl_fn.loc22_17.1
// CHECK:STDOUT:     %.loc22_3.1: ref @G.%as_type.loc22_11.2 (%as_type.e31) = splice_block %a.var {}
// CHECK:STDOUT:     %Transitive.GetA.call: init @G.%as_type.loc22_11.2 (%as_type.e31) to %.loc22_3.1 = call %bound_method.loc22_23(%t.ref)
// CHECK:STDOUT:     assign %a.var, %Transitive.GetA.call
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %.loc22_11.3 [symbolic = %as_type.loc22_11.2 (constants.%as_type.e31)] {
// CHECK:STDOUT:       %T.ref.loc22: %Transitive.type = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc22_11.2: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %A.ref: %Transitive.assoc_type = name_ref A, @A.%assoc0 [concrete = constants.%assoc0.398]
// CHECK:STDOUT:       %impl.elem0.loc22_11.1: %P.type = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element0 [symbolic = %impl.elem0.loc22_11.2 (constants.%impl.elem0.b09)]
// CHECK:STDOUT:       %as_type.loc22_11.1: type = facet_access_type %impl.elem0.loc22_11.1 [symbolic = %as_type.loc22_11.2 (constants.%as_type.e31)]
// CHECK:STDOUT:       %.loc22_11.3: type = converted %impl.elem0.loc22_11.1, %as_type.loc22_11.1 [symbolic = %as_type.loc22_11.2 (constants.%as_type.e31)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: ref @G.%as_type.loc22_11.2 (%as_type.e31) = ref_binding a, %a.var
// CHECK:STDOUT:     %observe.decl.loc24 = observe_decl @observe.loc24, (@observe.loc24.%T.ref.loc24_11, @observe.loc24.%T.as_type.loc24_12, @observe.loc24.%.loc24_12.1, @observe.loc24.%A.ref, @observe.loc24.%impl.elem0, @observe.loc24.%T.ref.loc24_18, @observe.loc24.%T.as_type.loc24_19, @observe.loc24.%.loc24_19.1, @observe.loc24.%B.ref, @observe.loc24.%impl.elem1, @observe.loc24.%as_type.loc24_19.1, @observe.loc24.%.loc24_19.2, @observe.loc24.%as_type.loc24_12, @observe.loc24.%.loc24_12.2, @observe.loc24.%eq, @observe.loc24.%Q.ref, @observe.loc24.%as_type.loc24_19.2, @observe.loc24.%.loc24_19.3, @observe.loc24.%impls)
// CHECK:STDOUT:     %a.ref.loc25: ref @G.%as_type.loc22_11.2 (%as_type.e31) = name_ref a, %a
// CHECK:STDOUT:     %Q.ref: type = name_ref Q, file.%Q.decl [concrete = constants.%Q.type]
// CHECK:STDOUT:     %InQ.ref: %Q.assoc_type = name_ref InQ, @Q.%assoc0 [concrete = constants.%assoc0.135]
// CHECK:STDOUT:     %.loc25_4.1: %Q.type = as_compatible constants.%impl.elem1.32a [symbolic = %impl.elem1.loc25 (constants.%impl.elem1.6ed)]
// CHECK:STDOUT:     %impl.elem0.loc25_4.1: @G.%.loc25_4.2 (%.fcd) = impl_witness_access %.loc25_4.1, element0 [symbolic = %impl.elem0.loc25_4.2 (constants.%impl.elem0.38b)]
// CHECK:STDOUT:     %bound_method.loc25_4: <bound method> = bound_method %a.ref.loc25, %impl.elem0.loc25_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_4.1: <specific function> = specific_impl_function %impl.elem0.loc25_4.1, @Q.InQ(constants.%Q.facet) [symbolic = %specific_impl_fn.loc25_4.2 (constants.%specific_impl_fn.d03)]
// CHECK:STDOUT:     %bound_method.loc25_13: <bound method> = bound_method %a.ref.loc25, %specific_impl_fn.loc25_4.1
// CHECK:STDOUT:     %.loc25_3: @G.%as_type.loc22_11.2 (%as_type.e31) = acquire_value %a.ref.loc25
// CHECK:STDOUT:     %Q.InQ.call: init %empty_tuple.type = call %bound_method.loc25_13(%.loc25_3)
// CHECK:STDOUT:     %observe.decl.loc27 = observe_decl @observe.loc27, (@observe.loc27.%T.ref.loc27_11, @observe.loc27.%T.as_type.loc27_12, @observe.loc27.%.loc27_12.1, @observe.loc27.%A.ref, @observe.loc27.%impl.elem0, @observe.loc27.%T.ref.loc27_18, @observe.loc27.%T.as_type.loc27_19, @observe.loc27.%.loc27_19.1, @observe.loc27.%C.ref, @observe.loc27.%impl.elem2, @observe.loc27.%as_type.loc27_19.1, @observe.loc27.%.loc27_19.2, @observe.loc27.%as_type.loc27_12, @observe.loc27.%.loc27_12.2, @observe.loc27.%eq, @observe.loc27.%R.ref, @observe.loc27.%as_type.loc27_19.2, @observe.loc27.%.loc27_19.3, @observe.loc27.%impls)
// CHECK:STDOUT:     %TakesPQR.ref: %TakesPQR.type = name_ref TakesPQR, file.%TakesPQR.decl [concrete = constants.%TakesPQR]
// CHECK:STDOUT:     %a.ref.loc28: ref @G.%as_type.loc22_11.2 (%as_type.e31) = name_ref a, %a
// CHECK:STDOUT:     %.loc28_13.1: %Q.type = as_compatible constants.%impl.elem1.32a [symbolic = %impl.elem1.loc25 (constants.%impl.elem1.6ed)]
// CHECK:STDOUT:     %.loc28_13.2: %R.type = as_compatible constants.%impl.elem2.458 [symbolic = %impl.elem2.loc28 (constants.%impl.elem2.a7e)]
// CHECK:STDOUT:     %facet_value.loc28_13.1: %facet_type.7a2 = facet_value constants.%as_type.e31, (constants.%P.lookup_impl_witness, %.loc28_13.1, %.loc28_13.2) [symbolic = %facet_value.loc28_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc28_13.3: %facet_type.7a2 = converted constants.%as_type.e31, %facet_value.loc28_13.1 [symbolic = %facet_value.loc28_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc28_13.4: %Q.type = as_compatible constants.%impl.elem1.32a [symbolic = %impl.elem1.loc25 (constants.%impl.elem1.6ed)]
// CHECK:STDOUT:     %.loc28_13.5: %R.type = as_compatible constants.%impl.elem2.458 [symbolic = %impl.elem2.loc28 (constants.%impl.elem2.a7e)]
// CHECK:STDOUT:     %facet_value.loc28_13.2: %facet_type.7a2 = facet_value constants.%as_type.e31, (constants.%P.lookup_impl_witness, %.loc28_13.4, %.loc28_13.5) [symbolic = %facet_value.loc28_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc28_13.6: %facet_type.7a2 = converted constants.%as_type.e31, %facet_value.loc28_13.2 [symbolic = %facet_value.loc28_13.3 (constants.%facet_value)]
// CHECK:STDOUT:     %TakesPQR.specific_fn.loc28_3.1: <specific function> = specific_function %TakesPQR.ref, @TakesPQR(constants.%facet_value) [symbolic = %TakesPQR.specific_fn.loc28_3.2 (constants.%TakesPQR.specific_fn)]
// CHECK:STDOUT:     %.loc28_12: @G.%as_type.loc22_11.2 (%as_type.e31) = acquire_value %a.ref.loc28
// CHECK:STDOUT:     %TakesPQR.call: init %empty_tuple.type = call %TakesPQR.specific_fn.loc28_3.1(%.loc28_12)
// CHECK:STDOUT:     %.loc22_3.2: %P.type = as_compatible constants.%impl.elem0.9bd [symbolic = %impl.elem0.loc22_3.2 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %impl.elem0.loc22_3.1: @G.%.loc22_3.3 (%.8f2) = impl_witness_access %.loc22_3.2, element0 [symbolic = %impl.elem0.loc22_3.3 (constants.%impl.elem0.28e)]
// CHECK:STDOUT:     %bound_method.loc22_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc22_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_3.1: <specific function> = specific_impl_function %impl.elem0.loc22_3.1, @Destroy.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc22_3.2 (constants.%specific_impl_fn.1b0)]
// CHECK:STDOUT:     %bound_method.loc22_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc22_3.1
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %bound_method.loc22_3.2(%a.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @P.InP(constants.%Self.d21) {
// CHECK:STDOUT:   %Self => constants.%Self.d21
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.6c8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.InQ(constants.%Self.f64) {
// CHECK:STDOUT:   %Self => constants.%Self.f64
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.1b2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @R.InR(constants.%Self.bea) {
// CHECK:STDOUT:   %Self => constants.%Self.bea
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.296
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.351
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%Self.e2a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.9bd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.4e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.GetA(constants.%Self.e2a) {
// CHECK:STDOUT:   %Self => constants.%Self.e2a
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.e5b
// CHECK:STDOUT:   %pattern_type.loc13_11 => constants.%pattern_type.b0d
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem0.loc13_28.1 => constants.%impl.elem0.9bd
// CHECK:STDOUT:   %as_type => constants.%as_type.0ea
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.e64
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.bff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.TakesC(constants.%Self.e2a) {
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
// CHECK:STDOUT: specific @A(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Transitive.GetA(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc13_11 => constants.%pattern_type.993
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8
// CHECK:STDOUT:   %impl.elem0.loc13_28.1 => constants.%impl.elem0.b09
// CHECK:STDOUT:   %as_type => constants.%as_type.e31
// CHECK:STDOUT:   %.loc13_28.1 => constants.%.d40
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.05e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.b09
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Q.InQ(constants.%Q.facet) {
// CHECK:STDOUT:   %Self => constants.%Q.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%as_type.e31
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.05e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Transitive.lookup_impl_witness => constants.%Transitive.lookup_impl_witness.6d8
// CHECK:STDOUT:   %impl.elem1 => constants.%impl.elem1.32a
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
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.2d9: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a2d: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF [concrete]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %U.binding.as_type, (%T) [symbolic]
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
// CHECK:STDOUT:     %.loc7_24.1: type = splice_block %.loc7_24.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_37 (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_37 (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: @F.%pattern_type.loc8_43 (%pattern_type.2d9) = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @F.%pattern_type.loc8_43 (%pattern_type.2d9) = value_param_pattern %y.patt, call_param1 [concrete]
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
// CHECK:STDOUT:       %T.ref.loc8_34: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref.loc8_34
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:     %x.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_40.1: type = splice_block %.loc8_40.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_40: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref.loc8_40 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc8_40.2: type = converted %T.ref.loc8_40, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: @F.%U.binding.as_type (%U.binding.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc8_46.1: type = splice_block %.loc8_46.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc8_13.2 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc8_46.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @F.%U.binding.as_type (%U.binding.as_type) = value_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
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
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_37: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc8_37 (constants.%pattern_type.057)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc8_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_43: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc8_43 (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_38: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc8_38 (constants.%require_complete.1a7)]
// CHECK:STDOUT:   %require_complete.loc8_44: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete.loc8_44 (constants.%require_complete.a2d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.binding.as_type (%T.binding.as_type), %y.param: @F.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe, (@observe.%T.ref, @observe.%U.ref, @observe.%U.as_type, @observe.%.loc9_16, @observe.%T.as_type, @observe.%.loc9_11, @observe.%eq)
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %y.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern y [concrete]
// CHECK:STDOUT:       %y.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %y.param: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc11_16.1: type = splice_block %.loc11_16.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc8_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:         %.loc11_16.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %y: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_binding y, %y.param
// CHECK:STDOUT:     }
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
// CHECK:STDOUT:   %A.facet.loc12_16.3: %A.type = facet_value %U.binding.as_type, (%T) [symbolic = %A.facet.loc12_16.3 (constants.%A.facet)]
// CHECK:STDOUT:   %RequiresA.specific_fn.loc12_5.2: <specific function> = specific_function constants.%RequiresA, @RequiresA(%A.facet.loc12_16.3) [symbolic = %RequiresA.specific_fn.loc12_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @InnerF.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %y.ref: @InnerF.%U.binding.as_type (%U.binding.as_type) = name_ref y, %y
// CHECK:STDOUT:     %.loc12_16.1: %A.type = as_compatible constants.%T [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %A.facet.loc12_16.1: %A.type = facet_value constants.%U.binding.as_type, (%.loc12_16.1) [symbolic = %A.facet.loc12_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc12_16.2: %A.type = converted constants.%U.binding.as_type, %A.facet.loc12_16.1 [symbolic = %A.facet.loc12_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc12_16.3: %A.type = as_compatible constants.%T [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %A.facet.loc12_16.2: %A.type = facet_value constants.%U.binding.as_type, (%.loc12_16.3) [symbolic = %A.facet.loc12_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc12_16.4: %A.type = converted constants.%U.binding.as_type, %A.facet.loc12_16.2 [symbolic = %A.facet.loc12_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %RequiresA.specific_fn.loc12_5.1: <specific function> = specific_function %RequiresA.ref, @RequiresA(constants.%A.facet) [symbolic = %RequiresA.specific_fn.loc12_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:     %RequiresA.call: init %empty_tuple.type = call %RequiresA.specific_fn.loc12_5.1(%y.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_37 => constants.%pattern_type.057
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_43 => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%A.facet) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%A.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%U.binding.as_type
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
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.2d9: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a2d: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %InnerF.type: type = fn_type @InnerF [concrete]
// CHECK:STDOUT:   %InnerF: %InnerF.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %U.binding.as_type, (%T) [symbolic]
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
// CHECK:STDOUT:     %.loc7_24.1: type = splice_block %.loc7_24.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_37 (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_37 (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: @F.%pattern_type.loc8_43 (%pattern_type.2d9) = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @F.%pattern_type.loc8_43 (%pattern_type.2d9) = value_param_pattern %y.patt, call_param1 [concrete]
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
// CHECK:STDOUT:       %T.ref.loc8_34: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref.loc8_34
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:     %x.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_40.1: type = splice_block %.loc8_40.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_40: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref.loc8_40 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc8_40.2: type = converted %T.ref.loc8_40, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: @F.%U.binding.as_type (%U.binding.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc8_46.1: type = splice_block %.loc8_46.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc8_13.2 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc8_46.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @F.%U.binding.as_type (%U.binding.as_type) = value_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
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
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_37: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc8_37 (constants.%pattern_type.057)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc8_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_43: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc8_43 (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_38: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc8_38 (constants.%require_complete.1a7)]
// CHECK:STDOUT:   %require_complete.loc8_44: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete.loc8_44 (constants.%require_complete.a2d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.binding.as_type (%T.binding.as_type), %y.param: @F.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %y.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern y [concrete]
// CHECK:STDOUT:       %y.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %y.param: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %.loc10_16.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc8_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:         %.loc10_16.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %y: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_binding y, %y.param
// CHECK:STDOUT:     }
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
// CHECK:STDOUT:   %A.facet.loc13_16.3: %A.type = facet_value %U.binding.as_type, (%T) [symbolic = %A.facet.loc13_16.3 (constants.%A.facet)]
// CHECK:STDOUT:   %RequiresA.specific_fn.loc13_5.2: <specific function> = specific_function constants.%RequiresA, @RequiresA(%A.facet.loc13_16.3) [symbolic = %RequiresA.specific_fn.loc13_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @InnerF.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe, (@observe.%T.ref, @observe.%U.ref, @observe.%U.as_type, @observe.%.loc11_18, @observe.%T.as_type, @observe.%.loc11_13, @observe.%eq)
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %y.ref: @InnerF.%U.binding.as_type (%U.binding.as_type) = name_ref y, %y
// CHECK:STDOUT:     %.loc13_16.1: %A.type = as_compatible constants.%T [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %A.facet.loc13_16.1: %A.type = facet_value constants.%U.binding.as_type, (%.loc13_16.1) [symbolic = %A.facet.loc13_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc13_16.2: %A.type = converted constants.%U.binding.as_type, %A.facet.loc13_16.1 [symbolic = %A.facet.loc13_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc13_16.3: %A.type = as_compatible constants.%T [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %A.facet.loc13_16.2: %A.type = facet_value constants.%U.binding.as_type, (%.loc13_16.3) [symbolic = %A.facet.loc13_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %.loc13_16.4: %A.type = converted constants.%U.binding.as_type, %A.facet.loc13_16.2 [symbolic = %A.facet.loc13_16.3 (constants.%A.facet)]
// CHECK:STDOUT:     %RequiresA.specific_fn.loc13_5.1: <specific function> = specific_function %RequiresA.ref, @RequiresA(constants.%A.facet) [symbolic = %RequiresA.specific_fn.loc13_5.2 (constants.%RequiresA.specific_fn)]
// CHECK:STDOUT:     %RequiresA.call: init %empty_tuple.type = call %RequiresA.specific_fn.loc13_5.1(%y.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_37 => constants.%pattern_type.057
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_43 => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InnerF(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresA(constants.%A.facet) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%A.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%U.binding.as_type
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
// CHECK:STDOUT:     %.loc7_24.1: type = splice_block %.loc7_24.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_37 (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_37 (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: @F.%pattern_type.loc8_43 (%pattern_type.2d9) = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @F.%pattern_type.loc8_43 (%pattern_type.2d9) = value_param_pattern %y.patt, call_param1 [concrete]
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
// CHECK:STDOUT:       %T.ref.loc8_34: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref.loc8_34
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:     %x.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_40.1: type = splice_block %.loc8_40.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_40: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref.loc8_40 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc8_40.2: type = converted %T.ref.loc8_40, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: @F.%U.binding.as_type (%U.binding.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc8_46.1: type = splice_block %.loc8_46.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc8_13.2 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc8_46.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @F.%U.binding.as_type (%U.binding.as_type) = value_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
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
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_37: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc8_37 (constants.%pattern_type.057)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc8_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_43: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc8_43 (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_38: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc8_38 (constants.%require_complete.1a7)]
// CHECK:STDOUT:   %require_complete.loc8_44: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete.loc8_44 (constants.%require_complete.a2d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.binding.as_type (%T.binding.as_type), %y.param: @F.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %y.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern y [concrete]
// CHECK:STDOUT:       %y.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %y.param: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %.loc10_16.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc8_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:         %.loc10_16.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %y: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_binding y, %y.param
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
// CHECK:STDOUT:   fn(%y.param: @InnerF.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %y.ref: @InnerF.%U.binding.as_type (%U.binding.as_type) = name_ref y, %y
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_37 => constants.%pattern_type.057
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_43 => constants.%pattern_type.2d9
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
// CHECK:STDOUT:     %.loc7_24.1: type = splice_block %.loc7_24.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_37 (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_37 (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: @F.%pattern_type.loc8_43 (%pattern_type.2d9) = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @F.%pattern_type.loc8_43 (%pattern_type.2d9) = value_param_pattern %y.patt, call_param1 [concrete]
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
// CHECK:STDOUT:       %T.ref.loc8_34: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.2 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref.loc8_34
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:     %x.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_40.1: type = splice_block %.loc8_40.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_40: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref.loc8_40 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc8_40.2: type = converted %T.ref.loc8_40, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: @F.%U.binding.as_type (%U.binding.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc8_46.1: type = splice_block %.loc8_46.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc8_13.2 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc8_46.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @F.%U.binding.as_type (%U.binding.as_type) = value_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
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
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_37: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc8_37 (constants.%pattern_type.057)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc8_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_43: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc8_43 (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_38: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc8_38 (constants.%require_complete.1a7)]
// CHECK:STDOUT:   %require_complete.loc8_44: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete.loc8_44 (constants.%require_complete.a2d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.binding.as_type (%T.binding.as_type), %y.param: @F.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %InnerF.decl: %InnerF.type = fn_decl @InnerF [concrete = constants.%InnerF] {
// CHECK:STDOUT:       %y.patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_binding_pattern y [concrete]
// CHECK:STDOUT:       %y.param_patt: @InnerF.%pattern_type (%pattern_type.2d9) = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %y.param: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %.loc10_16.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:         %U.ref: %B_where.type = name_ref U, @F.%U.loc8_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:         %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:         %.loc10_16.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %y: @InnerF.%U.binding.as_type (%U.binding.as_type) = value_binding y, %y.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %y.ref: @F.%U.binding.as_type (%U.binding.as_type) = name_ref y, %y
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
// CHECK:STDOUT:   fn(%y.param: @InnerF.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe, (@observe.%T.ref, @observe.%U.ref, @observe.%U.as_type, @observe.%.loc11_18, @observe.%T.as_type, @observe.%.loc11_13, @observe.%eq)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_37 => constants.%pattern_type.057
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_43 => constants.%pattern_type.2d9
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
// CHECK:STDOUT:     %.loc7_24.1: type = splice_block %.loc7_24.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_14.2 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.14b = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.035 = symbolic_binding_pattern V, 2 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc8_44 (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc8_44 (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: @F.%pattern_type.loc8_50 (%pattern_type.2d9) = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @F.%pattern_type.loc8_50 (%pattern_type.2d9) = value_param_pattern %y.patt, call_param1 [concrete]
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
// CHECK:STDOUT:       %T.ref.loc8_34: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %.loc8_19.2: type = where_expr %.Self.3 [concrete = constants.%B_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%B.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref.loc8_34
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_13.2: %B_where.type = symbolic_binding U, 1 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:     %.loc8_41: type = splice_block %B.ref.loc8_41 [concrete = constants.%B.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %B.ref.loc8_41: type = name_ref B, file.%B.decl [concrete = constants.%B.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc8_37.2: %B.type = symbolic_binding V, 2 [symbolic = %V.loc8_37.1 (constants.%V)]
// CHECK:STDOUT:     %x.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_47.1: type = splice_block %.loc8_47.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc8_47: %A.type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref.loc8_47 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc8_47.2: type = converted %T.ref.loc8_47, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: @F.%U.binding.as_type (%U.binding.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc8_53.1: type = splice_block %.loc8_53.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %B_where.type = name_ref U, %U.loc8_13.2 [symbolic = %U.loc8_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc8_53.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @F.%U.binding.as_type (%U.binding.as_type) = value_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
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
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc8_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_44: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc8_44 (constants.%pattern_type.057)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc8_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc8_50: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc8_50 (constants.%pattern_type.2d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_45: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc8_45 (constants.%require_complete.1a7)]
// CHECK:STDOUT:   %require_complete.loc8_51: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete.loc8_51 (constants.%require_complete.a2d)]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 2, %V.loc8_37.1 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.binding.as_type (%T.binding.as_type), %y.param: @F.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %observe.decl = observe_decl @observe, (@observe.%T.ref, @observe.%V.ref, @observe.%V.as_type, @observe.%.loc9_16, @observe.%T.as_type, @observe.%.loc9_11, @observe.%eq)
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %y.ref: @F.%U.binding.as_type (%U.binding.as_type) = name_ref y, %y
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_44 => constants.%pattern_type.057
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc8_50 => constants.%pattern_type.2d9
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
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_10: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_6.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc7_16.1: type = splice_block %.loc7_16.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_16.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
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
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.057)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:

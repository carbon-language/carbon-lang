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

// --- fail_todo_equivalent_impls.carbon

library "[[@TEST_NAME]]";

interface A {}
interface B { require impls A; }
interface C { require impls B; }
interface D {}

fn RequiresA[T:! A](x: T) {}
fn RequiresC[T:! C](x: T) {}
fn F[T:! C, U:! D where .Self == T, V:! D](x: T, y: U) {
  observe T == U impls B;

  // CHECK:STDERR: fail_todo_equivalent_impls.carbon:[[@LINE+7]]:3: error: cannot convert type `T` that implements `C` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   RequiresA(x);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_equivalent_impls.carbon:[[@LINE-8]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn RequiresA[T:! A](x: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  RequiresA(x);

  RequiresC(y);
}

// --- fail_wrong_scope.carbon

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
// CHECK:STDOUT:   %As.type.8b0: type = facet_type <@As, @As(%as_type.846)> [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d5a = facet_value %as_type.e31, (%impl.elem1.4e4) [symbolic]
// CHECK:STDOUT:   %.1e5: type = fn_type_with_self_type %ImplicitAs.Convert.type.11b, %ImplicitAs.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.694: %.1e5 = impl_witness_access %impl.elem1.4e4, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.dd7: <specific function> = specific_impl_function %impl.elem0.694, @ImplicitAs.Convert(%as_type.846, %ImplicitAs.facet) [symbolic]
// CHECK:STDOUT:   %Q.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.32a, @Q [symbolic]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.846, (%Q.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.736: type = fn_type_with_self_type %Q.InQ.type, %Q.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.3bb: %.736 = impl_witness_access %Q.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7f3: <specific function> = specific_impl_function %impl.elem0.3bb, @Q.InQ(%Q.facet) [symbolic]
// CHECK:STDOUT:   %impl.elem2.458: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.6d8, element2 [symbolic]
// CHECK:STDOUT:   %as_type.c65: type = facet_access_type %impl.elem2.458 [symbolic]
// CHECK:STDOUT:   %require_complete.60d: <witness> = require_complete_type %as_type.c65 [symbolic]
// CHECK:STDOUT:   %As.type.ae5: type = facet_type <@As, @As(%as_type.c65)> [symbolic]
// CHECK:STDOUT:   %As.Convert.type.d13: type = fn_type @As.Convert, @As(%as_type.c65) [symbolic]
// CHECK:STDOUT:   %As.assoc_type.69a: type = assoc_entity_type @As, @As(%as_type.c65) [symbolic]
// CHECK:STDOUT:   %assoc0.685: %As.assoc_type.69a = assoc_entity element0, imports.%Core.import_ref.77d [symbolic]
// CHECK:STDOUT:   %require_complete.ea4: <witness> = require_complete_type %As.type.ae5 [symbolic]
// CHECK:STDOUT:   %assoc0.5c9: %As.assoc_type.752 = assoc_entity element0, imports.%Core.import_ref.1ae [symbolic]
// CHECK:STDOUT:   %As.facet: %As.type.ae5 = facet_value %as_type.e31, (%impl.elem2.8f7) [symbolic]
// CHECK:STDOUT:   %.d27: type = fn_type_with_self_type %As.Convert.type.d13, %As.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1b9: %.d27 = impl_witness_access %impl.elem2.8f7, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7db: type = pattern_type %as_type.c65 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0f7: <specific function> = specific_impl_function %impl.elem0.1b9, @As.Convert(%as_type.c65, %As.facet) [symbolic]
// CHECK:STDOUT:   %R.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem2.458, @R [symbolic]
// CHECK:STDOUT:   %R.facet: %R.type = facet_value %as_type.c65, (%R.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.a55: type = fn_type_with_self_type %R.InR.type, %R.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.2a7: %.a55 = impl_witness_access %R.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.88b: <specific function> = specific_impl_function %impl.elem0.2a7, @R.InR(%R.facet) [symbolic]
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
// CHECK:STDOUT:   observe_decl @<null name>, (<unexpected>.inst{{[0-9A-F]+}}.loc16_11, <unexpected>.inst{{[0-9A-F]+}}.loc16_11, <unexpected>.inst{{[0-9A-F]+}}.loc16_16, <unexpected>.inst{{[0-9A-F]+}}.loc16_16, <unexpected>.inst{{[0-9A-F]+}}.loc16_16, <unexpected>.inst{{[0-9A-F]+}}.loc16_16, <unexpected>.inst{{[0-9A-F]+}}.loc16_11, <unexpected>.inst{{[0-9A-F]+}}.loc16_11, <unexpected>.inst{{[0-9A-F]+}}.loc16_13, <unexpected>.inst{{[0-9A-F]+}}.loc16_21, <unexpected>.inst{{[0-9A-F]+}}.loc16_21, <unexpected>.inst{{[0-9A-F]+}}.loc16_21, <unexpected>.inst{{[0-9A-F]+}}.loc16_21, <unexpected>.inst{{[0-9A-F]+}}.loc16_16, <unexpected>.inst{{[0-9A-F]+}}.loc16_16, <unexpected>.inst{{[0-9A-F]+}}.loc16_18)
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
// CHECK:STDOUT:   %ImplicitAs.type.loc22_3.3: type = facet_type <@ImplicitAs, @ImplicitAs(%as_type.loc22_11.2)> [symbolic = %ImplicitAs.type.loc22_3.3 (constants.%ImplicitAs.type.d5a)]
// CHECK:STDOUT:   %require_complete.loc22_3: <witness> = require_complete_type %ImplicitAs.type.loc22_3.3 [symbolic = %require_complete.loc22_3 (constants.%require_complete.abb)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%as_type.loc22_11.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.b4b)]
// CHECK:STDOUT:   %assoc0.loc22: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.b4b) = assoc_entity element0, imports.%Core.import_ref.218 [symbolic = %assoc0.loc22 (constants.%assoc0.849)]
// CHECK:STDOUT:   %As.type.loc22_3.2: type = facet_type <@As, @As(%as_type.loc22_11.2)> [symbolic = %As.type.loc22_3.2 (constants.%As.type.8b0)]
// CHECK:STDOUT:   %Self: %Transitive.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e2a)]
// CHECK:STDOUT:   %Transitive.lookup_impl_witness.loc22: <witness> = lookup_impl_witness %Self, @Transitive [symbolic = %Transitive.lookup_impl_witness.loc22 (constants.%Transitive.lookup_impl_witness.804)]
// CHECK:STDOUT:   %impl.elem1.loc22_3: %Q_where.type = impl_witness_access %Transitive.lookup_impl_witness.loc22, element1 [symbolic = %impl.elem1.loc22_3 (constants.%impl.elem1.4e4)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%as_type.loc22_11.2) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.11b)]
// CHECK:STDOUT:   %ImplicitAs.facet: @F.%ImplicitAs.type.loc22_3.3 (%ImplicitAs.type.d5a) = facet_value %as_type.loc20_11.2, (%impl.elem1.loc22_3) [symbolic = %ImplicitAs.facet (constants.%ImplicitAs.facet)]
// CHECK:STDOUT:   %.loc22_3.4: type = fn_type_with_self_type %ImplicitAs.Convert.type, %ImplicitAs.facet [symbolic = %.loc22_3.4 (constants.%.1e5)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.3: @F.%.loc22_3.4 (%.1e5) = impl_witness_access %impl.elem1.loc22_3, element0 [symbolic = %impl.elem0.loc22_3.3 (constants.%impl.elem0.694)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_3.3: <specific function> = specific_impl_function %impl.elem0.loc22_3.3, @ImplicitAs.Convert(%as_type.loc22_11.2, %ImplicitAs.facet) [symbolic = %specific_impl_fn.loc22_3.3 (constants.%specific_impl_fn.dd7)]
// CHECK:STDOUT:   %Q.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1.loc22_11.2, @Q [symbolic = %Q.lookup_impl_witness (constants.%Q.lookup_impl_witness)]
// CHECK:STDOUT:   %Q.facet: %Q.type = facet_value %as_type.loc22_11.2, (%Q.lookup_impl_witness) [symbolic = %Q.facet (constants.%Q.facet)]
// CHECK:STDOUT:   %.loc23_4: type = fn_type_with_self_type constants.%Q.InQ.type, %Q.facet [symbolic = %.loc23_4 (constants.%.736)]
// CHECK:STDOUT:   %impl.elem0.loc23_4.2: @F.%.loc23_4 (%.736) = impl_witness_access %Q.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc23_4.2 (constants.%impl.elem0.3bb)]
// CHECK:STDOUT:   %specific_impl_fn.loc23_4.2: <specific function> = specific_impl_function %impl.elem0.loc23_4.2, @Q.InQ(%Q.facet) [symbolic = %specific_impl_fn.loc23_4.2 (constants.%specific_impl_fn.7f3)]
// CHECK:STDOUT:   %impl.elem2.loc25_10.2: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.loc20, element2 [symbolic = %impl.elem2.loc25_10.2 (constants.%impl.elem2.458)]
// CHECK:STDOUT:   %as_type.loc25_10.2: type = facet_access_type %impl.elem2.loc25_10.2 [symbolic = %as_type.loc25_10.2 (constants.%as_type.c65)]
// CHECK:STDOUT:   %require_complete.loc25_6.1: <witness> = require_complete_type %as_type.loc25_10.2 [symbolic = %require_complete.loc25_6.1 (constants.%require_complete.60d)]
// CHECK:STDOUT:   %As.type.loc25_6.3: type = facet_type <@As, @As(%as_type.loc25_10.2)> [symbolic = %As.type.loc25_6.3 (constants.%As.type.ae5)]
// CHECK:STDOUT:   %require_complete.loc25_6.2: <witness> = require_complete_type %As.type.loc25_6.3 [symbolic = %require_complete.loc25_6.2 (constants.%require_complete.ea4)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%as_type.loc25_10.2) [symbolic = %As.assoc_type (constants.%As.assoc_type.69a)]
// CHECK:STDOUT:   %assoc0.loc25: @F.%As.assoc_type (%As.assoc_type.69a) = assoc_entity element0, imports.%Core.import_ref.77d [symbolic = %assoc0.loc25 (constants.%assoc0.685)]
// CHECK:STDOUT:   %impl.elem2.loc25_6: %R_where.type = impl_witness_access %Transitive.lookup_impl_witness.loc22, element2 [symbolic = %impl.elem2.loc25_6 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:   %As.Convert.type: type = fn_type @As.Convert, @As(%as_type.loc25_10.2) [symbolic = %As.Convert.type (constants.%As.Convert.type.d13)]
// CHECK:STDOUT:   %As.facet: @F.%As.type.loc25_6.3 (%As.type.ae5) = facet_value %as_type.loc20_11.2, (%impl.elem2.loc25_6) [symbolic = %As.facet (constants.%As.facet)]
// CHECK:STDOUT:   %.loc25_6.6: type = fn_type_with_self_type %As.Convert.type, %As.facet [symbolic = %.loc25_6.6 (constants.%.d27)]
// CHECK:STDOUT:   %impl.elem0.loc25_6.3: @F.%.loc25_6.6 (%.d27) = impl_witness_access %impl.elem2.loc25_6, element0 [symbolic = %impl.elem0.loc25_6.3 (constants.%impl.elem0.1b9)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_6.3: <specific function> = specific_impl_function %impl.elem0.loc25_6.3, @As.Convert(%as_type.loc25_10.2, %As.facet) [symbolic = %specific_impl_fn.loc25_6.3 (constants.%specific_impl_fn.0f7)]
// CHECK:STDOUT:   %R.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem2.loc25_10.2, @R [symbolic = %R.lookup_impl_witness (constants.%R.lookup_impl_witness)]
// CHECK:STDOUT:   %R.facet: %R.type = facet_value %as_type.loc25_10.2, (%R.lookup_impl_witness) [symbolic = %R.facet (constants.%R.facet)]
// CHECK:STDOUT:   %.loc25_13: type = fn_type_with_self_type constants.%R.InR.type, %R.facet [symbolic = %.loc25_13 (constants.%.a55)]
// CHECK:STDOUT:   %impl.elem0.loc25_13.2: @F.%.loc25_13 (%.a55) = impl_witness_access %R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_13.2 (constants.%impl.elem0.2a7)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_13.2: <specific function> = specific_impl_function %impl.elem0.loc25_13.2, @R.InR(%R.facet) [symbolic = %specific_impl_fn.loc25_13.2 (constants.%specific_impl_fn.88b)]
// CHECK:STDOUT:   %Destroy.facet.loc25: %Destroy.type = facet_value %as_type.loc25_10.2, (%impl.elem2.loc25_6) [symbolic = %Destroy.facet.loc25 (constants.%Destroy.facet.7a0)]
// CHECK:STDOUT:   %.loc25_6.7: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc25 [symbolic = %.loc25_6.7 (constants.%.e6f)]
// CHECK:STDOUT:   %impl.elem0.loc25_6.4: @F.%.loc25_6.7 (%.e6f) = impl_witness_access %impl.elem2.loc25_6, element0 [symbolic = %impl.elem0.loc25_6.4 (constants.%impl.elem0.ff7)]
// CHECK:STDOUT:   %specific_impl_fn.loc25_6.4: <specific function> = specific_impl_function %impl.elem0.loc25_6.4, @Destroy.Op(%Destroy.facet.loc25) [symbolic = %specific_impl_fn.loc25_6.4 (constants.%specific_impl_fn.d83)]
// CHECK:STDOUT:   %Destroy.facet.loc22: %Destroy.type = facet_value %as_type.loc22_11.2, (%impl.elem1.loc22_3) [symbolic = %Destroy.facet.loc22 (constants.%Destroy.facet.69c)]
// CHECK:STDOUT:   %.loc22_3.5: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc22 [symbolic = %.loc22_3.5 (constants.%.f9e)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.4: @F.%.loc22_3.5 (%.f9e) = impl_witness_access %impl.elem1.loc22_3, element0 [symbolic = %impl.elem0.loc22_3.4 (constants.%impl.elem0.50f)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_3.4: <specific function> = specific_impl_function %impl.elem0.loc22_3.4, @Destroy.Op(%Destroy.facet.loc22) [symbolic = %specific_impl_fn.loc22_3.4 (constants.%specific_impl_fn.e4c)]
// CHECK:STDOUT:   %impl.elem0.loc20_3.2: %P.type = impl_witness_access %Transitive.lookup_impl_witness.loc22, element0 [symbolic = %impl.elem0.loc20_3.2 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:   %Destroy.facet.loc20: %Destroy.type = facet_value %as_type.loc20_11.2, (%impl.elem0.loc20_3.2) [symbolic = %Destroy.facet.loc20 (constants.%Destroy.facet.96d)]
// CHECK:STDOUT:   %.loc20_3.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet.loc20 [symbolic = %.loc20_3.2 (constants.%.8f2)]
// CHECK:STDOUT:   %impl.elem0.loc20_3.3: @F.%.loc20_3.2 (%.8f2) = impl_witness_access %impl.elem0.loc20_3.2, element0 [symbolic = %impl.elem0.loc20_3.3 (constants.%impl.elem0.28e)]
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
// CHECK:STDOUT:     %ImplicitAs.type.loc22_3.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%as_type.846)> [symbolic = %ImplicitAs.type.loc22_3.3 (constants.%ImplicitAs.type.d5a)]
// CHECK:STDOUT:     %.loc22_3.1: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.b4b) = specific_constant imports.%Core.import_ref.a80, @ImplicitAs(constants.%as_type.846) [symbolic = %assoc0.loc22 (constants.%assoc0.849)]
// CHECK:STDOUT:     %Convert.ref.loc22: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.b4b) = name_ref Convert, %.loc22_3.1 [symbolic = %assoc0.loc22 (constants.%assoc0.849)]
// CHECK:STDOUT:     %As.type.loc22_3.1: type = facet_type <@As, @As(constants.%as_type.846)> [symbolic = %As.type.loc22_3.2 (constants.%As.type.8b0)]
// CHECK:STDOUT:     %ImplicitAs.type.loc22_3.2: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%as_type.846)> [symbolic = %ImplicitAs.type.loc22_3.3 (constants.%ImplicitAs.type.d5a)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc22_3.1: %Q_where.type = impl_witness_assoc_constant constants.%impl.elem1.4e4 [symbolic = %impl.elem1.loc22_3 (constants.%impl.elem1.4e4)]
// CHECK:STDOUT:     %impl.elem0.loc22_3.1: @F.%.loc22_3.4 (%.1e5) = impl_witness_access %impl_witness_assoc_constant.loc22_3.1, element0 [symbolic = %impl.elem0.loc22_3.3 (constants.%impl.elem0.694)]
// CHECK:STDOUT:     %bound_method.loc22_3.1: <bound method> = bound_method %a.ref.loc22, %impl.elem0.loc22_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_3.1: <specific function> = specific_impl_function %impl.elem0.loc22_3.1, @ImplicitAs.Convert(constants.%as_type.846, constants.%ImplicitAs.facet) [symbolic = %specific_impl_fn.loc22_3.3 (constants.%specific_impl_fn.dd7)]
// CHECK:STDOUT:     %bound_method.loc22_3.2: <bound method> = bound_method %a.ref.loc22, %specific_impl_fn.loc22_3.1
// CHECK:STDOUT:     %.loc22_3.2: ref @F.%as_type.loc22_11.2 (%as_type.846) = splice_block %b.var {}
// CHECK:STDOUT:     %.loc22_16: @F.%as_type.loc20_11.2 (%as_type.e31) = acquire_value %a.ref.loc22
// CHECK:STDOUT:     %ImplicitAs.Convert.call: init @F.%as_type.loc22_11.2 (%as_type.846) to %.loc22_3.2 = call %bound_method.loc22_3.2(%.loc22_16)
// CHECK:STDOUT:     %.loc22_3.3: init @F.%as_type.loc22_11.2 (%as_type.846) = converted %a.ref.loc22, %ImplicitAs.Convert.call
// CHECK:STDOUT:     assign %b.var, %.loc22_3.3
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
// CHECK:STDOUT:     %.loc25_10.1: type = converted %T.ref.loc25, %T.as_type.loc25 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %C.ref: %Transitive.assoc_type = name_ref C, @C.%assoc2 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc25_10.1: %R_where.type = impl_witness_access constants.%Transitive.lookup_impl_witness.6d8, element2 [symbolic = %impl.elem2.loc25_10.2 (constants.%impl.elem2.458)]
// CHECK:STDOUT:     %as_type.loc25_10.1: type = facet_access_type %impl.elem2.loc25_10.1 [symbolic = %as_type.loc25_10.2 (constants.%as_type.c65)]
// CHECK:STDOUT:     %.loc25_10.2: type = converted %impl.elem2.loc25_10.1, %as_type.loc25_10.1 [symbolic = %as_type.loc25_10.2 (constants.%as_type.c65)]
// CHECK:STDOUT:     %As.type.loc25_6.1: type = facet_type <@As, @As(constants.%as_type.c65)> [symbolic = %As.type.loc25_6.3 (constants.%As.type.ae5)]
// CHECK:STDOUT:     %.loc25_6.1: @F.%As.assoc_type (%As.assoc_type.69a) = specific_constant imports.%Core.import_ref.e0b, @As(constants.%as_type.c65) [symbolic = %assoc0.loc25 (constants.%assoc0.685)]
// CHECK:STDOUT:     %Convert.ref.loc25: @F.%As.assoc_type (%As.assoc_type.69a) = name_ref Convert, %.loc25_6.1 [symbolic = %assoc0.loc25 (constants.%assoc0.685)]
// CHECK:STDOUT:     %As.type.loc25_6.2: type = facet_type <@As, @As(constants.%as_type.c65)> [symbolic = %As.type.loc25_6.3 (constants.%As.type.ae5)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc25_6.1: %R_where.type = impl_witness_assoc_constant constants.%impl.elem2.8f7 [symbolic = %impl.elem2.loc25_6 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:     %impl.elem0.loc25_6.1: @F.%.loc25_6.6 (%.d27) = impl_witness_access %impl_witness_assoc_constant.loc25_6.1, element0 [symbolic = %impl.elem0.loc25_6.3 (constants.%impl.elem0.1b9)]
// CHECK:STDOUT:     %bound_method.loc25_6.1: <bound method> = bound_method %a.ref.loc25, %impl.elem0.loc25_6.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_6.1: <specific function> = specific_impl_function %impl.elem0.loc25_6.1, @As.Convert(constants.%as_type.c65, constants.%As.facet) [symbolic = %specific_impl_fn.loc25_6.3 (constants.%specific_impl_fn.0f7)]
// CHECK:STDOUT:     %bound_method.loc25_6.2: <bound method> = bound_method %a.ref.loc25, %specific_impl_fn.loc25_6.1
// CHECK:STDOUT:     %.loc25_6.2: ref @F.%as_type.loc25_10.2 (%as_type.c65) = temporary_storage
// CHECK:STDOUT:     %.loc25_4: @F.%as_type.loc20_11.2 (%as_type.e31) = acquire_value %a.ref.loc25
// CHECK:STDOUT:     %As.Convert.call: init @F.%as_type.loc25_10.2 (%as_type.c65) to %.loc25_6.2 = call %bound_method.loc25_6.2(%.loc25_4)
// CHECK:STDOUT:     %.loc25_6.3: init @F.%as_type.loc25_10.2 (%as_type.c65) = converted %a.ref.loc25, %As.Convert.call
// CHECK:STDOUT:     %.loc25_6.4: ref @F.%as_type.loc25_10.2 (%as_type.c65) = temporary %.loc25_6.2, %.loc25_6.3
// CHECK:STDOUT:     %InR.ref: %R.assoc_type = name_ref InR, @R.%assoc0 [concrete = constants.%assoc0.4f7]
// CHECK:STDOUT:     %impl.elem0.loc25_13.1: @F.%.loc25_13 (%.a55) = impl_witness_access constants.%R.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc25_13.2 (constants.%impl.elem0.2a7)]
// CHECK:STDOUT:     %bound_method.loc25_13: <bound method> = bound_method %.loc25_6.4, %impl.elem0.loc25_13.1
// CHECK:STDOUT:     %specific_impl_fn.loc25_13.1: <specific function> = specific_impl_function %impl.elem0.loc25_13.1, @R.InR(constants.%R.facet) [symbolic = %specific_impl_fn.loc25_13.2 (constants.%specific_impl_fn.88b)]
// CHECK:STDOUT:     %bound_method.loc25_18: <bound method> = bound_method %.loc25_6.4, %specific_impl_fn.loc25_13.1
// CHECK:STDOUT:     %.loc25_6.5: @F.%as_type.loc25_10.2 (%as_type.c65) = acquire_value %.loc25_6.4
// CHECK:STDOUT:     %R.InR.call: init %empty_tuple.type = call %bound_method.loc25_18(%.loc25_6.5)
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc25_6.2: %R_where.type = impl_witness_assoc_constant constants.%impl.elem2.8f7 [symbolic = %impl.elem2.loc25_6 (constants.%impl.elem2.8f7)]
// CHECK:STDOUT:     %impl.elem0.loc25_6.2: @F.%.loc25_6.7 (%.e6f) = impl_witness_access %impl_witness_assoc_constant.loc25_6.2, element0 [symbolic = %impl.elem0.loc25_6.4 (constants.%impl.elem0.ff7)]
// CHECK:STDOUT:     %bound_method.loc25_6.3: <bound method> = bound_method %.loc25_6.4, %impl.elem0.loc25_6.2
// CHECK:STDOUT:     %specific_impl_fn.loc25_6.2: <specific function> = specific_impl_function %impl.elem0.loc25_6.2, @Destroy.Op(constants.%Destroy.facet.7a0) [symbolic = %specific_impl_fn.loc25_6.4 (constants.%specific_impl_fn.d83)]
// CHECK:STDOUT:     %bound_method.loc25_6.4: <bound method> = bound_method %.loc25_6.4, %specific_impl_fn.loc25_6.2
// CHECK:STDOUT:     %Destroy.Op.call.loc25: init %empty_tuple.type = call %bound_method.loc25_6.4(%.loc25_6.4)
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc22_3.2: %Q_where.type = impl_witness_assoc_constant constants.%impl.elem1.4e4 [symbolic = %impl.elem1.loc22_3 (constants.%impl.elem1.4e4)]
// CHECK:STDOUT:     %impl.elem0.loc22_3.2: @F.%.loc22_3.5 (%.f9e) = impl_witness_access %impl_witness_assoc_constant.loc22_3.2, element0 [symbolic = %impl.elem0.loc22_3.4 (constants.%impl.elem0.50f)]
// CHECK:STDOUT:     %bound_method.loc22_3.3: <bound method> = bound_method %b.var, %impl.elem0.loc22_3.2
// CHECK:STDOUT:     %specific_impl_fn.loc22_3.2: <specific function> = specific_impl_function %impl.elem0.loc22_3.2, @Destroy.Op(constants.%Destroy.facet.69c) [symbolic = %specific_impl_fn.loc22_3.4 (constants.%specific_impl_fn.e4c)]
// CHECK:STDOUT:     %bound_method.loc22_3.4: <bound method> = bound_method %b.var, %specific_impl_fn.loc22_3.2
// CHECK:STDOUT:     %Destroy.Op.call.loc22: init %empty_tuple.type = call %bound_method.loc22_3.4(%b.var)
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc20: %P.type = impl_witness_assoc_constant constants.%impl.elem0.9bd [symbolic = %impl.elem0.loc20_3.2 (constants.%impl.elem0.9bd)]
// CHECK:STDOUT:     %impl.elem0.loc20_3.1: @F.%.loc20_3.2 (%.8f2) = impl_witness_access %impl_witness_assoc_constant.loc20, element0 [symbolic = %impl.elem0.loc20_3.3 (constants.%impl.elem0.28e)]
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
// CHECK:STDOUT:     observe_decl @<null name>, (<unexpected>.inst{{[0-9A-F]+}}.loc10_11, <unexpected>.inst{{[0-9A-F]+}}.loc10_19, <unexpected>.inst{{[0-9A-F]+}}.loc10_11, <unexpected>.inst{{[0-9A-F]+}}.loc10_11, <unexpected>.inst{{[0-9A-F]+}}.loc10_13)
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
// CHECK:STDOUT: --- fail_todo_equivalent_impls.carbon
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
// CHECK:STDOUT:   %D.type: type = facet_type <@D> [concrete]
// CHECK:STDOUT:   %Self.b8b: %D.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.5a3: %A.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.faf: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.0b3: type = symbolic_binding_type T, 0, %T.5a3 [symbolic]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %T.binding.as_type.0b3 [symbolic]
// CHECK:STDOUT:   %RequiresA.type: type = fn_type @RequiresA [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %RequiresA: %RequiresA.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1a7: <witness> = require_complete_type %T.binding.as_type.0b3 [symbolic]
// CHECK:STDOUT:   %T.94e: %C.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7f1: type = pattern_type %C.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type.d92: type = symbolic_binding_type T, 0, %T.94e [symbolic]
// CHECK:STDOUT:   %pattern_type.901: type = pattern_type %T.binding.as_type.d92 [symbolic]
// CHECK:STDOUT:   %RequiresC.type: type = fn_type @RequiresC [concrete]
// CHECK:STDOUT:   %RequiresC: %RequiresC.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.791: <witness> = require_complete_type %T.binding.as_type.d92 [symbolic]
// CHECK:STDOUT:   %.Self.fc0: %D.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %D_where.type: type = facet_type <@D where TODO> [concrete]
// CHECK:STDOUT:   %U: %D_where.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.fab: type = pattern_type %D_where.type [concrete]
// CHECK:STDOUT:   %V: %D.type = symbolic_binding V, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.f05: type = pattern_type %D.type [concrete]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U [symbolic]
// CHECK:STDOUT:   %pattern_type.93c: type = pattern_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.831: <witness> = require_complete_type %U.binding.as_type [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %C.facet: %C.type = facet_value %U.binding.as_type, (%T.94e) [symbolic]
// CHECK:STDOUT:   %RequiresC.specific_fn: <specific function> = specific_function %RequiresC, @RequiresC(%C.facet) [symbolic]
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
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .RequiresA = %RequiresA.decl
// CHECK:STDOUT:     .RequiresC = %RequiresC.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %B.decl: type = interface_decl @B [concrete = constants.%B.type] {} {}
// CHECK:STDOUT:   %C.decl: type = interface_decl @C [concrete = constants.%C.type] {} {}
// CHECK:STDOUT:   %D.decl: type = interface_decl @D [concrete = constants.%D.type] {} {}
// CHECK:STDOUT:   %RequiresA.decl: %RequiresA.type = fn_decl @RequiresA [concrete = constants.%RequiresA] {
// CHECK:STDOUT:     %T.patt: %pattern_type.faf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresA.%pattern_type (%pattern_type.057) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresA.%pattern_type (%pattern_type.057) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_18: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_14.2: %A.type = symbolic_binding T, 0 [symbolic = %T.loc9_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:     %x.param: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = value_param call_param0
// CHECK:STDOUT:     %.loc9_24.1: type = splice_block %.loc9_24.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)] {
// CHECK:STDOUT:       %T.ref: %A.type = name_ref T, %T.loc9_14.2 [symbolic = %T.loc9_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:       %.loc9_24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresA.%T.binding.as_type (%T.binding.as_type.0b3) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RequiresC.decl: %RequiresC.type = fn_decl @RequiresC [concrete = constants.%RequiresC] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7f1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @RequiresC.%pattern_type (%pattern_type.901) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @RequiresC.%pattern_type (%pattern_type.901) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_18: type = splice_block %C.ref [concrete = constants.%C.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_14.2: %C.type = symbolic_binding T, 0 [symbolic = %T.loc10_14.1 (constants.%T.94e)]
// CHECK:STDOUT:     %x.param: @RequiresC.%T.binding.as_type (%T.binding.as_type.d92) = value_param call_param0
// CHECK:STDOUT:     %.loc10_24.1: type = splice_block %.loc10_24.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)] {
// CHECK:STDOUT:       %T.ref: %C.type = name_ref T, %T.loc10_14.2 [symbolic = %T.loc10_14.1 (constants.%T.94e)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:       %.loc10_24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @RequiresC.%T.binding.as_type (%T.binding.as_type.d92) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7f1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.fab = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.f05 = symbolic_binding_pattern V, 2 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc11_44 (%pattern_type.901) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc11_44 (%pattern_type.901) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: @F.%pattern_type.loc11_50 (%pattern_type.93c) = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @F.%pattern_type.loc11_50 (%pattern_type.93c) = value_param_pattern %y.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_10: type = splice_block %C.ref [concrete = constants.%C.type] {
// CHECK:STDOUT:       %.Self.4: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_6.2: %C.type = symbolic_binding T, 0 [symbolic = %T.loc11_6.1 (constants.%T.94e)]
// CHECK:STDOUT:     %.loc11_19.1: type = splice_block %.loc11_19.2 [concrete = constants.%D_where.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %D.ref.loc11_17: type = name_ref D, file.%D.decl [concrete = constants.%D.type]
// CHECK:STDOUT:       %.Self.3: %D.type = symbolic_binding .Self [symbolic_self = constants.%.Self.fc0]
// CHECK:STDOUT:       %.Self.ref: %D.type = name_ref .Self, %.Self.3 [symbolic_self = constants.%.Self.fc0]
// CHECK:STDOUT:       %T.ref.loc11_34: %C.type = name_ref T, %T.loc11_6.2 [symbolic = %T.loc11_6.1 (constants.%T.94e)]
// CHECK:STDOUT:       %.loc11_19.2: type = where_expr %.Self.3 [concrete = constants.%D_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%D.type
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %T.ref.loc11_34
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc11_13.2: %D_where.type = symbolic_binding U, 1 [symbolic = %U.loc11_13.1 (constants.%U)]
// CHECK:STDOUT:     %.loc11_41: type = splice_block %D.ref.loc11_41 [concrete = constants.%D.type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %D.ref.loc11_41: type = name_ref D, file.%D.decl [concrete = constants.%D.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc11_37.2: %D.type = symbolic_binding V, 2 [symbolic = %V.loc11_37.1 (constants.%V)]
// CHECK:STDOUT:     %x.param: @F.%T.binding.as_type (%T.binding.as_type.d92) = value_param call_param0
// CHECK:STDOUT:     %.loc11_47.1: type = splice_block %.loc11_47.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)] {
// CHECK:STDOUT:       %T.ref.loc11_47: %C.type = name_ref T, %T.loc11_6.2 [symbolic = %T.loc11_6.1 (constants.%T.94e)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref.loc11_47 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:       %.loc11_47.2: type = converted %T.ref.loc11_47, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.binding.as_type (%T.binding.as_type.d92) = value_binding x, %x.param
// CHECK:STDOUT:     %y.param: @F.%U.binding.as_type (%U.binding.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc11_53.1: type = splice_block %.loc11_53.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)] {
// CHECK:STDOUT:       %U.ref: %D_where.type = name_ref U, %U.loc11_13.2 [symbolic = %U.loc11_13.1 (constants.%U)]
// CHECK:STDOUT:       %U.as_type: type = facet_access_type %U.ref [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:       %.loc11_53.2: type = converted %U.ref, %U.as_type [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
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
// CHECK:STDOUT: interface @D {
// CHECK:STDOUT:   %Self: %D.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b8b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
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
// CHECK:STDOUT: generic fn @RequiresA(%T.loc9_14.2: %A.type) {
// CHECK:STDOUT:   %T.loc9_14.1: %A.type = symbolic_binding T, 0 [symbolic = %T.loc9_14.1 (constants.%T.5a3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.0b3)]
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
// CHECK:STDOUT: generic fn @RequiresC(%T.loc10_14.2: %C.type) {
// CHECK:STDOUT:   %T.loc10_14.1: %C.type = symbolic_binding T, 0 [symbolic = %T.loc10_14.1 (constants.%T.94e)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc10_14.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.901)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.791)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RequiresC.%T.binding.as_type (%T.binding.as_type.d92)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11_6.2: %C.type, %U.loc11_13.2: %D_where.type, %V.loc11_37.2: %D.type) {
// CHECK:STDOUT:   %T.loc11_6.1: %C.type = symbolic_binding T, 0 [symbolic = %T.loc11_6.1 (constants.%T.94e)]
// CHECK:STDOUT:   %U.loc11_13.1: %D_where.type = symbolic_binding U, 1 [symbolic = %U.loc11_13.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc11_37.1: %D.type = symbolic_binding V, 2 [symbolic = %V.loc11_37.1 (constants.%V)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc11_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type.d92)]
// CHECK:STDOUT:   %pattern_type.loc11_44: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc11_44 (constants.%pattern_type.901)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc11_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc11_50: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc11_50 (constants.%pattern_type.93c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_45: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc11_45 (constants.%require_complete.791)]
// CHECK:STDOUT:   %require_complete.loc11_51: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete.loc11_51 (constants.%require_complete.831)]
// CHECK:STDOUT:   %C.facet.loc23_14.3: %C.type = facet_value %U.binding.as_type, (%T.loc11_6.1) [symbolic = %C.facet.loc23_14.3 (constants.%C.facet)]
// CHECK:STDOUT:   %RequiresC.specific_fn.loc23_3.2: <specific function> = specific_function constants.%RequiresC, @RequiresC(%C.facet.loc23_14.3) [symbolic = %RequiresC.specific_fn.loc23_3.2 (constants.%RequiresC.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.binding.as_type (%T.binding.as_type.d92), %y.param: @F.%U.binding.as_type (%U.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     observe_decl @<null name>, (<unexpected>.inst{{[0-9A-F]+}}.loc12_11, <unexpected>.inst{{[0-9A-F]+}}.loc12_16, <unexpected>.inst{{[0-9A-F]+}}.loc12_16, <unexpected>.inst{{[0-9A-F]+}}.loc12_16, <unexpected>.inst{{[0-9A-F]+}}.loc12_11, <unexpected>.inst{{[0-9A-F]+}}.loc12_11, <unexpected>.inst{{[0-9A-F]+}}.loc12_13, <unexpected>.inst{{[0-9A-F]+}}.loc12_24, <unexpected>.inst{{[0-9A-F]+}}.loc12_16, <unexpected>.inst{{[0-9A-F]+}}.loc12_16, <unexpected>.inst{{[0-9A-F]+}}.loc12_18)
// CHECK:STDOUT:     %RequiresA.ref: %RequiresA.type = name_ref RequiresA, file.%RequiresA.decl [concrete = constants.%RequiresA]
// CHECK:STDOUT:     %x.ref: @F.%T.binding.as_type (%T.binding.as_type.d92) = name_ref x, %x
// CHECK:STDOUT:     %RequiresC.ref: %RequiresC.type = name_ref RequiresC, file.%RequiresC.decl [concrete = constants.%RequiresC]
// CHECK:STDOUT:     %y.ref: @F.%U.binding.as_type (%U.binding.as_type) = name_ref y, %y
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc23_14.1: %C.type = impl_witness_assoc_constant constants.%T.94e [symbolic = %T.loc11_6.1 (constants.%T.94e)]
// CHECK:STDOUT:     %C.facet.loc23_14.1: %C.type = facet_value constants.%U.binding.as_type, (%impl_witness_assoc_constant.loc23_14.1) [symbolic = %C.facet.loc23_14.3 (constants.%C.facet)]
// CHECK:STDOUT:     %.loc23_14.1: %C.type = converted constants.%U.binding.as_type, %C.facet.loc23_14.1 [symbolic = %C.facet.loc23_14.3 (constants.%C.facet)]
// CHECK:STDOUT:     %impl_witness_assoc_constant.loc23_14.2: %C.type = impl_witness_assoc_constant constants.%T.94e [symbolic = %T.loc11_6.1 (constants.%T.94e)]
// CHECK:STDOUT:     %C.facet.loc23_14.2: %C.type = facet_value constants.%U.binding.as_type, (%impl_witness_assoc_constant.loc23_14.2) [symbolic = %C.facet.loc23_14.3 (constants.%C.facet)]
// CHECK:STDOUT:     %.loc23_14.2: %C.type = converted constants.%U.binding.as_type, %C.facet.loc23_14.2 [symbolic = %C.facet.loc23_14.3 (constants.%C.facet)]
// CHECK:STDOUT:     %RequiresC.specific_fn.loc23_3.1: <specific function> = specific_function %RequiresC.ref, @RequiresC(constants.%C.facet) [symbolic = %RequiresC.specific_fn.loc23_3.2 (constants.%RequiresC.specific_fn)]
// CHECK:STDOUT:     %RequiresC.call: init %empty_tuple.type = call %RequiresC.specific_fn.loc23_3.1(%y.ref)
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
// CHECK:STDOUT:   %T.loc9_14.1 => constants.%T.5a3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.0b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.057
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresC(constants.%T.94e) {
// CHECK:STDOUT:   %T.loc10_14.1 => constants.%T.94e
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.d92
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.901
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.94e, constants.%U, constants.%V) {
// CHECK:STDOUT:   %T.loc11_6.1 => constants.%T.94e
// CHECK:STDOUT:   %U.loc11_13.1 => constants.%U
// CHECK:STDOUT:   %V.loc11_37.1 => constants.%V
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type.d92
// CHECK:STDOUT:   %pattern_type.loc11_44 => constants.%pattern_type.901
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc11_50 => constants.%pattern_type.93c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RequiresC(constants.%C.facet) {
// CHECK:STDOUT:   %T.loc10_14.1 => constants.%C.facet
// CHECK:STDOUT:   %T.binding.as_type => constants.%U.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.93c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.831
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
// CHECK:STDOUT:     %T.loc2_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc2_9.1 (constants.%T)]
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc2_19.2: type = symbolic_binding U, 1 [symbolic = %U.loc2_19.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc2_9.2: type, %U.loc2_19.2: type) {
// CHECK:STDOUT:   %T.loc2_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc2_9.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc2_19.1: type = symbolic_binding U, 1 [symbolic = %U.loc2_19.1 (constants.%U)]
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
// CHECK:STDOUT:   %T.loc2_9.1 => constants.%T
// CHECK:STDOUT:   %U.loc2_19.1 => constants.%U
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

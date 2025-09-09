// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/compound_member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/compound_member_access.carbon

// --- associated_constant.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
}

// Simple member access.
fn Simple1(T:! J, S:! T.U) {}

// This should be equivalent to `Simple1` above, but using compound member access.
fn Compound1(V:! J, W:! V.(J.U)) {}

// --- non_instance.carbon
library "[[@TEST_NAME]]";

interface K1 {
  fn Q1();
}

// Simple member access.
fn Simple2(T:! K1) {
  T.Q1();
}

// This should be equivalent to `Simple2` above, but using compound member access.
fn Compound2(V:! K1) {
  V.(K1.Q1)();
}

// --- fail_caller_instance_interface_not.carbon
library "[[@TEST_NAME]]";

interface K2 {
  fn Q2();
}

// Simple member access allows this.
fn Simple3(T:! K2, x: T) {
  x.Q2();
}

// Compound member access does not. It tries to convert `y` to `K2`, but only
// its type `V` can.
fn Compound3(V:! K2, y: V) {
  // CHECK:STDERR: fail_caller_instance_interface_not.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `V` into type implementing `K2` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   y.(K2.Q2)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_caller_instance_interface_not.carbon:[[@LINE+4]]:3: note: type `V` does not implement interface `Core.ImplicitAs(K2)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   y.(K2.Q2)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  y.(K2.Q2)();
}

// --- instance.carbon
library "[[@TEST_NAME]]";

interface L1 {
  fn R1[self: Self]();
  fn S1[addr self: Self*]();
}

// Simple member access.
fn Simple4(T:! L1, x: T) {
  x.R1();
  var xcopy: T = x;
  xcopy.S1();
}

// This should be equivalent to `Simple4` above, but using compound member access.
fn Compound4(V:! L1, y: V) {
  y.(L1.R1)();
  var ycopy: V = y;
  ycopy.(L1.S1)();
}

// --- fail_interface_instance_caller_not.carbon

library "[[@TEST_NAME]]";

interface L2 {
  fn R2[self: Self]();
  fn S2[addr self: Self*]();
}

// Simple member access. Fails since calling an instance method without an object.
fn Simple5(T:! L2) {
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE+7]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   T.R2();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE-9]]:3: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:   fn R2[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  T.R2();
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE+7]]:3: error: missing object argument in method call [MissingObjectInMethodCall]
  // CHECK:STDERR:   T.S2();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE-16]]:3: note: calling function declared here [InCallToFunction]
  // CHECK:STDERR:   fn S2[addr self: Self*]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  T.S2();
}

// TODO: Expected to fail in the same way as `Simple5`.
fn Compound5(V:! L2) {
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE+4]]:3: error: cannot access member of interface `L2` in type `L2` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   V.(L2.R2)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  V.(L2.R2)();
  // CHECK:STDERR: fail_interface_instance_caller_not.carbon:[[@LINE+4]]:3: error: cannot access member of interface `L2` in type `L2` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   V.(L2.S2)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  V.(L2.S2)();
}

// --- fail_combine_non_instance.carbon
library "[[@TEST_NAME]]";

interface A {
  fn G();
}

class C {}
impl C as A {
  fn G() {}
}

// Since `A.G` is a non-instance method, compound member access may only be
// used with a type, not an instance.
fn Fails() {
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `C` into type implementing `A` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   ({} as C).((A & A).G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+4]]:3: note: type `C` does not implement interface `Core.ImplicitAs(A)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   ({} as C).((A & A).G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as C).((A & A).G)();
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `C` into type implementing `A` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   (({} as C) as (C as (A & A))).((A & A).G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+4]]:3: note: type `C` does not implement interface `Core.ImplicitAs(A)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   (({} as C) as (C as (A & A))).((A & A).G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (({} as C) as (C as (A & A))).((A & A).G)();
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `C` into type implementing `A` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   (({} as C) as (C as (A & A))).(A.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+4]]:3: note: type `C` does not implement interface `Core.ImplicitAs(A)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   (({} as C) as (C as (A & A))).(A.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (({} as C) as (C as (A & A))).(A.G)();
}

// --- allowed_combine_non_instance.carbon
library "[[@TEST_NAME]]";

interface A {
  fn G();
}

class C {}
impl C as A {
  fn G() {}
}

fn Works() {
  C.((A & A).G)();
  (C as (A & A)).((A & A).G)();
  (C as (A & A)).(A.G)();
}

// CHECK:STDOUT: --- associated_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.fdf: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.0df834.1: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %J.facet.276c17.1: %J.type = facet_value %T.as_type, (%J.lookup_impl_witness.0df834.1) [symbolic]
// CHECK:STDOUT:   %impl.elem0.3a2921.1: type = impl_witness_access %J.lookup_impl_witness.0df834.1, element0 [symbolic]
// CHECK:STDOUT:   %S: %impl.elem0.3a2921.1 = bind_symbolic_name S, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.b7a83b.1: type = pattern_type %impl.elem0.3a2921.1 [symbolic]
// CHECK:STDOUT:   %Simple1.type: type = fn_type @Simple1 [concrete]
// CHECK:STDOUT:   %Simple1: %Simple1.type = struct_value () [concrete]
// CHECK:STDOUT:   %V: %J.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.0df834.2: <witness> = lookup_impl_witness %V, @J [symbolic]
// CHECK:STDOUT:   %J.facet.276c17.2: %J.type = facet_value %V.as_type, (%J.lookup_impl_witness.0df834.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.3a2921.2: type = impl_witness_access %J.lookup_impl_witness.0df834.2, element0 [symbolic]
// CHECK:STDOUT:   %W: %impl.elem0.3a2921.2 = bind_symbolic_name W, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.b7a83b.2: type = pattern_type %impl.elem0.3a2921.2 [symbolic]
// CHECK:STDOUT:   %Compound1.type: type = fn_type @Compound1 [concrete]
// CHECK:STDOUT:   %Compound1: %Compound1.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .Simple1 = %Simple1.decl
// CHECK:STDOUT:     .Compound1 = %Compound1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %Simple1.decl: %Simple1.type = fn_decl @Simple1 [concrete = constants.%Simple1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.fdf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: @Simple1.%pattern_type (%pattern_type.b7a83b.1) = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_16: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.2: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_12.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_12.1 (constants.%T)]
// CHECK:STDOUT:     %.loc8_24.1: type = splice_block %impl.elem0.loc8_24.2 [symbolic = %impl.elem0.loc8_24.1 (constants.%impl.elem0.3a2921.1)] {
// CHECK:STDOUT:       %.Self.1: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: %J.type = name_ref T, %T.loc8_12.2 [symbolic = %T.loc8_12.1 (constants.%T)]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %T.as_type.loc8_24.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_24.2: type = converted %T.ref, %T.as_type.loc8_24.2 [symbolic = %T.as_type.loc8_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %impl.elem0.loc8_24.2: type = impl_witness_access constants.%J.lookup_impl_witness.0df834.1, element0 [symbolic = %impl.elem0.loc8_24.1 (constants.%impl.elem0.3a2921.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc8_19.2: @Simple1.%impl.elem0.loc8_24.1 (%impl.elem0.3a2921.1) = bind_symbolic_name S, 1 [symbolic = %S.loc8_19.1 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound1.decl: %Compound1.type = fn_decl @Compound1 [concrete = constants.%Compound1] {
// CHECK:STDOUT:     %V.patt: %pattern_type.fdf = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %W.patt: @Compound1.%pattern_type (%pattern_type.b7a83b.2) = symbolic_binding_pattern W, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_18: type = splice_block %J.ref.loc11_18 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.2: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref.loc11_18: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc11_14.2: %J.type = bind_symbolic_name V, 0 [symbolic = %V.loc11_14.1 (constants.%V)]
// CHECK:STDOUT:     %.loc11_26: type = splice_block %impl.elem0.loc11_26.2 [symbolic = %impl.elem0.loc11_26.1 (constants.%impl.elem0.3a2921.2)] {
// CHECK:STDOUT:       %.Self.1: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %V.ref: %J.type = name_ref V, %V.loc11_14.2 [symbolic = %V.loc11_14.1 (constants.%V)]
// CHECK:STDOUT:       %J.ref.loc11_28: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc11_26.2: type = impl_witness_access constants.%J.lookup_impl_witness.0df834.2, element0 [symbolic = %impl.elem0.loc11_26.1 (constants.%impl.elem0.3a2921.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc11_21.2: @Compound1.%impl.elem0.loc11_26.1 (%impl.elem0.3a2921.2) = bind_symbolic_name W, 1 [symbolic = %W.loc11_21.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   witness = (%U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple1(%T.loc8_12.2: %J.type, %S.loc8_19.2: @Simple1.%impl.elem0.loc8_24.1 (%impl.elem0.3a2921.1)) {
// CHECK:STDOUT:   %T.loc8_12.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_12.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc8_24.1: type = facet_access_type %T.loc8_12.1 [symbolic = %T.as_type.loc8_24.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_12.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.0df834.1)]
// CHECK:STDOUT:   %impl.elem0.loc8_24.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_24.1 (constants.%impl.elem0.3a2921.1)]
// CHECK:STDOUT:   %S.loc8_19.1: @Simple1.%impl.elem0.loc8_24.1 (%impl.elem0.3a2921.1) = bind_symbolic_name S, 1 [symbolic = %S.loc8_19.1 (constants.%S)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc8_24.1 [symbolic = %pattern_type (constants.%pattern_type.b7a83b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound1(%V.loc11_14.2: %J.type, %W.loc11_21.2: @Compound1.%impl.elem0.loc11_26.1 (%impl.elem0.3a2921.2)) {
// CHECK:STDOUT:   %V.loc11_14.1: %J.type = bind_symbolic_name V, 0 [symbolic = %V.loc11_14.1 (constants.%V)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %V.loc11_14.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.0df834.2)]
// CHECK:STDOUT:   %impl.elem0.loc11_26.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_26.1 (constants.%impl.elem0.3a2921.2)]
// CHECK:STDOUT:   %W.loc11_21.1: @Compound1.%impl.elem0.loc11_26.1 (%impl.elem0.3a2921.2) = bind_symbolic_name W, 1 [symbolic = %W.loc11_21.1 (constants.%W)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc11_26.1 [symbolic = %pattern_type (constants.%pattern_type.b7a83b.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.276c17.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple1(constants.%T, constants.%S) {
// CHECK:STDOUT:   %T.loc8_12.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_24.1 => constants.%T.as_type
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.0df834.1
// CHECK:STDOUT:   %impl.elem0.loc8_24.1 => constants.%impl.elem0.3a2921.1
// CHECK:STDOUT:   %S.loc8_19.1 => constants.%S
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b7a83b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.276c17.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound1(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc11_14.1 => constants.%V
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.0df834.2
// CHECK:STDOUT:   %impl.elem0.loc11_26.1 => constants.%impl.elem0.3a2921.2
// CHECK:STDOUT:   %W.loc11_21.1 => constants.%W
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b7a83b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K1.type: type = facet_type <@K1> [concrete]
// CHECK:STDOUT:   %Self: %K1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K1.Q1.type: type = fn_type @K1.Q1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %K1.Q1: %K1.Q1.type = struct_value () [concrete]
// CHECK:STDOUT:   %K1.assoc_type: type = assoc_entity_type @K1 [concrete]
// CHECK:STDOUT:   %assoc0: %K1.assoc_type = assoc_entity element0, @K1.%K1.Q1.decl [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: %K1.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %K1.type [concrete]
// CHECK:STDOUT:   %Simple2.type: type = fn_type @Simple2 [concrete]
// CHECK:STDOUT:   %Simple2: %Simple2.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %K1.lookup_impl_witness.f3b0e3.1: <witness> = lookup_impl_witness %T, @K1 [symbolic]
// CHECK:STDOUT:   %K1.facet.e39cd4.1: %K1.type = facet_value %T.as_type, (%K1.lookup_impl_witness.f3b0e3.1) [symbolic]
// CHECK:STDOUT:   %.b4827e.1: type = fn_type_with_self_type %K1.Q1.type, %K1.facet.e39cd4.1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.964929.1: %.b4827e.1 = impl_witness_access %K1.lookup_impl_witness.f3b0e3.1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c6f659.1: <specific function> = specific_impl_function %impl.elem0.964929.1, @K1.Q1(%K1.facet.e39cd4.1) [symbolic]
// CHECK:STDOUT:   %V: %K1.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %Compound2.type: type = fn_type @Compound2 [concrete]
// CHECK:STDOUT:   %Compound2: %Compound2.type = struct_value () [concrete]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %K1.lookup_impl_witness.f3b0e3.2: <witness> = lookup_impl_witness %V, @K1 [symbolic]
// CHECK:STDOUT:   %K1.facet.e39cd4.2: %K1.type = facet_value %V.as_type, (%K1.lookup_impl_witness.f3b0e3.2) [symbolic]
// CHECK:STDOUT:   %.b4827e.2: type = fn_type_with_self_type %K1.Q1.type, %K1.facet.e39cd4.2 [symbolic]
// CHECK:STDOUT:   %impl.elem0.964929.2: %.b4827e.2 = impl_witness_access %K1.lookup_impl_witness.f3b0e3.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c6f659.2: <specific function> = specific_impl_function %impl.elem0.964929.2, @K1.Q1(%K1.facet.e39cd4.2) [symbolic]
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
// CHECK:STDOUT:     .K1 = %K1.decl
// CHECK:STDOUT:     .Simple2 = %Simple2.decl
// CHECK:STDOUT:     .Compound2 = %Compound2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %K1.decl: type = interface_decl @K1 [concrete = constants.%K1.type] {} {}
// CHECK:STDOUT:   %Simple2.decl: %Simple2.type = fn_decl @Simple2 [concrete = constants.%Simple2] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %K1.ref [concrete = constants.%K1.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K1.ref: type = name_ref K1, file.%K1.decl [concrete = constants.%K1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_12.2: %K1.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_12.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound2.decl: %Compound2.type = fn_decl @Compound2 [concrete = constants.%Compound2] {
// CHECK:STDOUT:     %V.patt: %pattern_type = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13: type = splice_block %K1.ref.loc13 [concrete = constants.%K1.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K1.ref.loc13: type = name_ref K1, file.%K1.decl [concrete = constants.%K1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc13_14.2: %K1.type = bind_symbolic_name V, 0 [symbolic = %V.loc13_14.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K1 {
// CHECK:STDOUT:   %Self: %K1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K1.Q1.decl: %K1.Q1.type = fn_decl @K1.Q1 [concrete = constants.%K1.Q1] {} {}
// CHECK:STDOUT:   %assoc0: %K1.assoc_type = assoc_entity element0, %K1.Q1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Q1 = %assoc0
// CHECK:STDOUT:   witness = (%K1.Q1.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K1.Q1(@K1.%Self: %K1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple2(%T.loc8_12.2: %K1.type) {
// CHECK:STDOUT:   %T.loc8_12.1: %K1.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_12.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc9_4.2: type = facet_access_type %T.loc8_12.1 [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %K1.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_12.1, @K1 [symbolic = %K1.lookup_impl_witness (constants.%K1.lookup_impl_witness.f3b0e3.1)]
// CHECK:STDOUT:   %K1.facet: %K1.type = facet_value %T.as_type.loc9_4.2, (%K1.lookup_impl_witness) [symbolic = %K1.facet (constants.%K1.facet.e39cd4.1)]
// CHECK:STDOUT:   %.loc9_4.2: type = fn_type_with_self_type constants.%K1.Q1.type, %K1.facet [symbolic = %.loc9_4.2 (constants.%.b4827e.1)]
// CHECK:STDOUT:   %impl.elem0.loc9_4.2: @Simple2.%.loc9_4.2 (%.b4827e.1) = impl_witness_access %K1.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0.964929.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_4.2: <specific function> = specific_impl_function %impl.elem0.loc9_4.2, @K1.Q1(%K1.facet) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn.c6f659.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %K1.type = name_ref T, %T.loc8_12.2 [symbolic = %T.loc8_12.1 (constants.%T)]
// CHECK:STDOUT:     %Q1.ref: %K1.assoc_type = name_ref Q1, @K1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc9_4.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_4.1: type = converted %T.ref, %T.as_type.loc9_4.1 [symbolic = %T.as_type.loc9_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc9_4.1: @Simple2.%.loc9_4.2 (%.b4827e.1) = impl_witness_access constants.%K1.lookup_impl_witness.f3b0e3.1, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0.964929.1)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_4.1: <specific function> = specific_impl_function %impl.elem0.loc9_4.1, @K1.Q1(constants.%K1.facet.e39cd4.1) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn.c6f659.1)]
// CHECK:STDOUT:     %.loc9_8: init %empty_tuple.type = call %specific_impl_fn.loc9_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound2(%V.loc13_14.2: %K1.type) {
// CHECK:STDOUT:   %V.loc13_14.1: %K1.type = bind_symbolic_name V, 0 [symbolic = %V.loc13_14.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %K1.lookup_impl_witness: <witness> = lookup_impl_witness %V.loc13_14.1, @K1 [symbolic = %K1.lookup_impl_witness (constants.%K1.lookup_impl_witness.f3b0e3.2)]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V.loc13_14.1 [symbolic = %V.as_type (constants.%V.as_type)]
// CHECK:STDOUT:   %K1.facet: %K1.type = facet_value %V.as_type, (%K1.lookup_impl_witness) [symbolic = %K1.facet (constants.%K1.facet.e39cd4.2)]
// CHECK:STDOUT:   %.loc14_4: type = fn_type_with_self_type constants.%K1.Q1.type, %K1.facet [symbolic = %.loc14_4 (constants.%.b4827e.2)]
// CHECK:STDOUT:   %impl.elem0.loc14_4.2: @Compound2.%.loc14_4 (%.b4827e.2) = impl_witness_access %K1.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0.964929.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_4.2: <specific function> = specific_impl_function %impl.elem0.loc14_4.2, @K1.Q1(%K1.facet) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.c6f659.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %V.ref: %K1.type = name_ref V, %V.loc13_14.2 [symbolic = %V.loc13_14.1 (constants.%V)]
// CHECK:STDOUT:     %K1.ref.loc14: type = name_ref K1, file.%K1.decl [concrete = constants.%K1.type]
// CHECK:STDOUT:     %Q1.ref: %K1.assoc_type = name_ref Q1, @K1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc14_4.1: @Compound2.%.loc14_4 (%.b4827e.2) = impl_witness_access constants.%K1.lookup_impl_witness.f3b0e3.2, element0 [symbolic = %impl.elem0.loc14_4.2 (constants.%impl.elem0.964929.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_4.1: <specific function> = specific_impl_function %impl.elem0.loc14_4.1, @K1.Q1(constants.%K1.facet.e39cd4.2) [symbolic = %specific_impl_fn.loc14_4.2 (constants.%specific_impl_fn.c6f659.2)]
// CHECK:STDOUT:     %.loc14_13: init %empty_tuple.type = call %specific_impl_fn.loc14_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K1.Q1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple2(constants.%T) {
// CHECK:STDOUT:   %T.loc8_12.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K1.Q1(constants.%K1.facet.e39cd4.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound2(constants.%V) {
// CHECK:STDOUT:   %V.loc13_14.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K1.Q1(constants.%K1.facet.e39cd4.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_caller_instance_interface_not.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K2.type: type = facet_type <@K2> [concrete]
// CHECK:STDOUT:   %Self.593: %K2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K2.Q2.type: type = fn_type @K2.Q2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %K2.Q2: %K2.Q2.type = struct_value () [concrete]
// CHECK:STDOUT:   %K2.assoc_type: type = assoc_entity_type @K2 [concrete]
// CHECK:STDOUT:   %assoc0.d67: %K2.assoc_type = assoc_entity element0, @K2.%K2.Q2.decl [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: %K2.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.738: type = pattern_type %K2.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.f43b58.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Simple3.type: type = fn_type @Simple3 [concrete]
// CHECK:STDOUT:   %Simple3: %Simple3.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.893460.1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %K2.lookup_impl_witness: <witness> = lookup_impl_witness %T, @K2 [symbolic]
// CHECK:STDOUT:   %K2.facet: %K2.type = facet_value %T.as_type, (%K2.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.641: type = fn_type_with_self_type %K2.Q2.type, %K2.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.641 = impl_witness_access %K2.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @K2.Q2(%K2.facet) [symbolic]
// CHECK:STDOUT:   %V: %K2.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %pattern_type.f43b58.2: type = pattern_type %V.as_type [symbolic]
// CHECK:STDOUT:   %Compound3.type: type = fn_type @Compound3 [concrete]
// CHECK:STDOUT:   %Compound3: %Compound3.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.893460.2: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .K2 = %K2.decl
// CHECK:STDOUT:     .Simple3 = %Simple3.decl
// CHECK:STDOUT:     .Compound3 = %Compound3.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %K2.decl: type = interface_decl @K2 [concrete = constants.%K2.type] {} {}
// CHECK:STDOUT:   %Simple3.decl: %Simple3.type = fn_decl @Simple3 [concrete = constants.%Simple3] {
// CHECK:STDOUT:     %T.patt: %pattern_type.738 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Simple3.%pattern_type (%pattern_type.f43b58.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Simple3.%pattern_type (%pattern_type.f43b58.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_16: type = splice_block %K2.ref [concrete = constants.%K2.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K2.ref: type = name_ref K2, file.%K2.decl [concrete = constants.%K2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_12.2: %K2.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_12.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Simple3.%T.as_type.loc8_23.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_23.1: type = splice_block %.loc8_23.2 [symbolic = %T.as_type.loc8_23.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %K2.type = name_ref T, %T.loc8_12.2 [symbolic = %T.loc8_12.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc8_23.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc8_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc8_23.2: type = converted %T.ref, %T.as_type.loc8_23.2 [symbolic = %T.as_type.loc8_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Simple3.%T.as_type.loc8_23.1 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound3.decl: %Compound3.type = fn_decl @Compound3 [concrete = constants.%Compound3] {
// CHECK:STDOUT:     %V.patt: %pattern_type.738 = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %y.patt: @Compound3.%pattern_type (%pattern_type.f43b58.2) = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @Compound3.%pattern_type (%pattern_type.f43b58.2) = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_18: type = splice_block %K2.ref.loc14 [concrete = constants.%K2.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K2.ref.loc14: type = name_ref K2, file.%K2.decl [concrete = constants.%K2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc14_14.2: %K2.type = bind_symbolic_name V, 0 [symbolic = %V.loc14_14.1 (constants.%V)]
// CHECK:STDOUT:     %y.param: @Compound3.%V.as_type.loc14_25.1 (%V.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc14_25.1: type = splice_block %.loc14_25.2 [symbolic = %V.as_type.loc14_25.1 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref: %K2.type = name_ref V, %V.loc14_14.2 [symbolic = %V.loc14_14.1 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc14_25.2: type = facet_access_type %V.ref [symbolic = %V.as_type.loc14_25.1 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc14_25.2: type = converted %V.ref, %V.as_type.loc14_25.2 [symbolic = %V.as_type.loc14_25.1 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @Compound3.%V.as_type.loc14_25.1 (%V.as_type) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K2 {
// CHECK:STDOUT:   %Self: %K2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.593]
// CHECK:STDOUT:   %K2.Q2.decl: %K2.Q2.type = fn_decl @K2.Q2 [concrete = constants.%K2.Q2] {} {}
// CHECK:STDOUT:   %assoc0: %K2.assoc_type = assoc_entity element0, %K2.Q2.decl [concrete = constants.%assoc0.d67]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Q2 = %assoc0
// CHECK:STDOUT:   witness = (%K2.Q2.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K2.Q2(@K2.%Self: %K2.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple3(%T.loc8_12.2: %K2.type) {
// CHECK:STDOUT:   %T.loc8_12.1: %K2.type = bind_symbolic_name T, 0 [symbolic = %T.loc8_12.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc8_23.1: type = facet_access_type %T.loc8_12.1 [symbolic = %T.as_type.loc8_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc8_23.1 [symbolic = %pattern_type (constants.%pattern_type.f43b58.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc8_23.1 [symbolic = %require_complete (constants.%require_complete.893460.1)]
// CHECK:STDOUT:   %K2.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc8_12.1, @K2 [symbolic = %K2.lookup_impl_witness (constants.%K2.lookup_impl_witness)]
// CHECK:STDOUT:   %K2.facet: %K2.type = facet_value %T.as_type.loc8_23.1, (%K2.lookup_impl_witness) [symbolic = %K2.facet (constants.%K2.facet)]
// CHECK:STDOUT:   %.loc9_4.2: type = fn_type_with_self_type constants.%K2.Q2.type, %K2.facet [symbolic = %.loc9_4.2 (constants.%.641)]
// CHECK:STDOUT:   %impl.elem0.loc9_4.2: @Simple3.%.loc9_4.2 (%.641) = impl_witness_access %K2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_4.2: <specific function> = specific_impl_function %impl.elem0.loc9_4.2, @K2.Q2(%K2.facet) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Simple3.%T.as_type.loc8_23.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Simple3.%T.as_type.loc8_23.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Q2.ref: %K2.assoc_type = name_ref Q2, @K2.%assoc0 [concrete = constants.%assoc0.d67]
// CHECK:STDOUT:     %T.as_type.loc9: type = facet_access_type constants.%T [symbolic = %T.as_type.loc8_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_4.1: type = converted constants.%T, %T.as_type.loc9 [symbolic = %T.as_type.loc8_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc9_4.1: @Simple3.%.loc9_4.2 (%.641) = impl_witness_access constants.%K2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_4.1: <specific function> = specific_impl_function %impl.elem0.loc9_4.1, @K2.Q2(constants.%K2.facet) [symbolic = %specific_impl_fn.loc9_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc9_8: init %empty_tuple.type = call %specific_impl_fn.loc9_4.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound3(%V.loc14_14.2: %K2.type) {
// CHECK:STDOUT:   %V.loc14_14.1: %K2.type = bind_symbolic_name V, 0 [symbolic = %V.loc14_14.1 (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc14_25.1: type = facet_access_type %V.loc14_14.1 [symbolic = %V.as_type.loc14_25.1 (constants.%V.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %V.as_type.loc14_25.1 [symbolic = %pattern_type (constants.%pattern_type.f43b58.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %V.as_type.loc14_25.1 [symbolic = %require_complete (constants.%require_complete.893460.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @Compound3.%V.as_type.loc14_25.1 (%V.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %y.ref: @Compound3.%V.as_type.loc14_25.1 (%V.as_type) = name_ref y, %y
// CHECK:STDOUT:     %K2.ref.loc22: type = name_ref K2, file.%K2.decl [concrete = constants.%K2.type]
// CHECK:STDOUT:     %Q2.ref: %K2.assoc_type = name_ref Q2, @K2.%assoc0 [concrete = constants.%assoc0.d67]
// CHECK:STDOUT:     %.loc22: %K2.type = converted %y.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K2.Q2(constants.%Self.593) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple3(constants.%T) {
// CHECK:STDOUT:   %T.loc8_12.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc8_23.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f43b58.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K2.Q2(constants.%K2.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound3(constants.%V) {
// CHECK:STDOUT:   %V.loc14_14.1 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc14_25.1 => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f43b58.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L1.type: type = facet_type <@L1> [concrete]
// CHECK:STDOUT:   %Self.112: %L1.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.351: type = facet_access_type %Self.112 [symbolic]
// CHECK:STDOUT:   %pattern_type.cb4262.1: type = pattern_type %Self.as_type.351 [symbolic]
// CHECK:STDOUT:   %L1.R1.type: type = fn_type @L1.R1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %L1.R1: %L1.R1.type = struct_value () [concrete]
// CHECK:STDOUT:   %L1.assoc_type: type = assoc_entity_type @L1 [concrete]
// CHECK:STDOUT:   %assoc0.8b4: %L1.assoc_type = assoc_entity element0, @L1.%L1.R1.decl [concrete]
// CHECK:STDOUT:   %ptr.87f2ee.1: type = ptr_type %Self.as_type.351 [symbolic]
// CHECK:STDOUT:   %pattern_type.c0652c.1: type = pattern_type %ptr.87f2ee.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %L1.S1.type: type = fn_type @L1.S1 [concrete]
// CHECK:STDOUT:   %L1.S1: %L1.S1.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %L1.assoc_type = assoc_entity element1, @L1.%L1.S1.decl [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.112: %L1.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.1cf: type = pattern_type %L1.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.112 [symbolic]
// CHECK:STDOUT:   %pattern_type.cb4262.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Simple4.type: type = fn_type @Simple4 [concrete]
// CHECK:STDOUT:   %Simple4: %Simple4.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a52deb.1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %L1.lookup_impl_witness.d9f5df.1: <witness> = lookup_impl_witness %T.112, @L1 [symbolic]
// CHECK:STDOUT:   %L1.facet.1cc812.1: %L1.type = facet_value %T.as_type, (%L1.lookup_impl_witness.d9f5df.1) [symbolic]
// CHECK:STDOUT:   %.cad400.1: type = fn_type_with_self_type %L1.R1.type, %L1.facet.1cc812.1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.c5e026.1: %.cad400.1 = impl_witness_access %L1.lookup_impl_witness.d9f5df.1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5f85c4.1: <specific function> = specific_impl_function %impl.elem0.c5e026.1, @L1.R1(%L1.facet.1cc812.1) [symbolic]
// CHECK:STDOUT:   %.9b5ba3.1: type = fn_type_with_self_type %L1.S1.type, %L1.facet.1cc812.1 [symbolic]
// CHECK:STDOUT:   %impl.elem1.d07d0e.1: %.9b5ba3.1 = impl_witness_access %L1.lookup_impl_witness.d9f5df.1, element1 [symbolic]
// CHECK:STDOUT:   %ptr.87f2ee.2: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.c0652c.2: type = pattern_type %ptr.87f2ee.2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2473a0.1: <specific function> = specific_impl_function %impl.elem1.d07d0e.1, @L1.S1(%L1.facet.1cc812.1) [symbolic]
// CHECK:STDOUT:   %require_complete.5158e1.1: <witness> = require_complete_type %ptr.87f2ee.2 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.c7977f.1: <witness> = lookup_impl_witness %T.112, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.f26842.1: %Destroy.type = facet_value %T.as_type, (%Destroy.lookup_impl_witness.c7977f.1) [symbolic]
// CHECK:STDOUT:   %.acaa49.1: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.f26842.1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.b71e6c.1: %.acaa49.1 = impl_witness_access %Destroy.lookup_impl_witness.c7977f.1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.21ccd1.1: <specific function> = specific_impl_function %impl.elem0.b71e6c.1, @Destroy.Op(%Destroy.facet.f26842.1) [symbolic]
// CHECK:STDOUT:   %V: %L1.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.as_type: type = facet_access_type %V [symbolic]
// CHECK:STDOUT:   %pattern_type.cb4262.3: type = pattern_type %V.as_type [symbolic]
// CHECK:STDOUT:   %Compound4.type: type = fn_type @Compound4 [concrete]
// CHECK:STDOUT:   %Compound4: %Compound4.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.a52deb.2: <witness> = require_complete_type %V.as_type [symbolic]
// CHECK:STDOUT:   %L1.lookup_impl_witness.d9f5df.2: <witness> = lookup_impl_witness %V, @L1 [symbolic]
// CHECK:STDOUT:   %L1.facet.1cc812.2: %L1.type = facet_value %V.as_type, (%L1.lookup_impl_witness.d9f5df.2) [symbolic]
// CHECK:STDOUT:   %.cad400.2: type = fn_type_with_self_type %L1.R1.type, %L1.facet.1cc812.2 [symbolic]
// CHECK:STDOUT:   %impl.elem0.c5e026.2: %.cad400.2 = impl_witness_access %L1.lookup_impl_witness.d9f5df.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5f85c4.2: <specific function> = specific_impl_function %impl.elem0.c5e026.2, @L1.R1(%L1.facet.1cc812.2) [symbolic]
// CHECK:STDOUT:   %.9b5ba3.2: type = fn_type_with_self_type %L1.S1.type, %L1.facet.1cc812.2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.d07d0e.2: %.9b5ba3.2 = impl_witness_access %L1.lookup_impl_witness.d9f5df.2, element1 [symbolic]
// CHECK:STDOUT:   %ptr.87f2ee.3: type = ptr_type %V.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.c0652c.3: type = pattern_type %ptr.87f2ee.3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2473a0.2: <specific function> = specific_impl_function %impl.elem1.d07d0e.2, @L1.S1(%L1.facet.1cc812.2) [symbolic]
// CHECK:STDOUT:   %require_complete.5158e1.2: <witness> = require_complete_type %ptr.87f2ee.3 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.c7977f.2: <witness> = lookup_impl_witness %V, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.f26842.2: %Destroy.type = facet_value %V.as_type, (%Destroy.lookup_impl_witness.c7977f.2) [symbolic]
// CHECK:STDOUT:   %.acaa49.2: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.f26842.2 [symbolic]
// CHECK:STDOUT:   %impl.elem0.b71e6c.2: %.acaa49.2 = impl_witness_access %Destroy.lookup_impl_witness.c7977f.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.21ccd1.2: <specific function> = specific_impl_function %impl.elem0.b71e6c.2, @Destroy.Op(%Destroy.facet.f26842.2) [symbolic]
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
// CHECK:STDOUT:     .L1 = %L1.decl
// CHECK:STDOUT:     .Simple4 = %Simple4.decl
// CHECK:STDOUT:     .Compound4 = %Compound4.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %L1.decl: type = interface_decl @L1 [concrete = constants.%L1.type] {} {}
// CHECK:STDOUT:   %Simple4.decl: %Simple4.type = fn_decl @Simple4 [concrete = constants.%Simple4] {
// CHECK:STDOUT:     %T.patt: %pattern_type.1cf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Simple4.%pattern_type (%pattern_type.cb4262.2) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Simple4.%pattern_type (%pattern_type.cb4262.2) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_16: type = splice_block %L1.ref [concrete = constants.%L1.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L1.ref: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_12.2: %L1.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_12.1 (constants.%T.112)]
// CHECK:STDOUT:     %x.param: @Simple4.%T.as_type.loc9_23.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_23.1: type = splice_block %.loc9_23.2 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9: %L1.type = name_ref T, %T.loc9_12.2 [symbolic = %T.loc9_12.1 (constants.%T.112)]
// CHECK:STDOUT:       %T.as_type.loc9_23.2: type = facet_access_type %T.ref.loc9 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_23.2: type = converted %T.ref.loc9, %T.as_type.loc9_23.2 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Simple4.%T.as_type.loc9_23.1 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound4.decl: %Compound4.type = fn_decl @Compound4 [concrete = constants.%Compound4] {
// CHECK:STDOUT:     %V.patt: %pattern_type.1cf = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %y.patt: @Compound4.%pattern_type (%pattern_type.cb4262.3) = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: @Compound4.%pattern_type (%pattern_type.cb4262.3) = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_18: type = splice_block %L1.ref.loc16 [concrete = constants.%L1.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L1.ref.loc16: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc16_14.2: %L1.type = bind_symbolic_name V, 0 [symbolic = %V.loc16_14.1 (constants.%V)]
// CHECK:STDOUT:     %y.param: @Compound4.%V.as_type.loc16_25.1 (%V.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc16_25.1: type = splice_block %.loc16_25.2 [symbolic = %V.as_type.loc16_25.1 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref.loc16: %L1.type = name_ref V, %V.loc16_14.2 [symbolic = %V.loc16_14.1 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc16_25.2: type = facet_access_type %V.ref.loc16 [symbolic = %V.as_type.loc16_25.1 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc16_25.2: type = converted %V.ref.loc16, %V.as_type.loc16_25.2 [symbolic = %V.as_type.loc16_25.1 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @Compound4.%V.as_type.loc16_25.1 (%V.as_type) = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L1 {
// CHECK:STDOUT:   %Self: %L1.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.112]
// CHECK:STDOUT:   %L1.R1.decl: %L1.R1.type = fn_decl @L1.R1 [concrete = constants.%L1.R1] {
// CHECK:STDOUT:     %self.patt: @L1.R1.%pattern_type (%pattern_type.cb4262.1) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @L1.R1.%pattern_type (%pattern_type.cb4262.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @L1.R1.%Self.as_type.loc4_15.1 (%Self.as_type.351) = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: type = splice_block %.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.351)] {
// CHECK:STDOUT:       %Self.ref: %L1.type = name_ref Self, @L1.%Self [symbolic = %Self (constants.%Self.112)]
// CHECK:STDOUT:       %Self.as_type.loc4_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.351)]
// CHECK:STDOUT:       %.loc4_15.2: type = converted %Self.ref, %Self.as_type.loc4_15.2 [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.351)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @L1.R1.%Self.as_type.loc4_15.1 (%Self.as_type.351) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %L1.assoc_type = assoc_entity element0, %L1.R1.decl [concrete = constants.%assoc0.8b4]
// CHECK:STDOUT:   %L1.S1.decl: %L1.S1.type = fn_decl @L1.S1 [concrete = constants.%L1.S1] {
// CHECK:STDOUT:     %self.patt: @L1.S1.%pattern_type (%pattern_type.c0652c.1) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @L1.S1.%pattern_type (%pattern_type.c0652c.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc5_9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @L1.S1.%ptr.loc5_24.1 (%ptr.87f2ee.1) = value_param call_param0
// CHECK:STDOUT:     %.loc5_24.1: type = splice_block %ptr.loc5_24.2 [symbolic = %ptr.loc5_24.1 (constants.%ptr.87f2ee.1)] {
// CHECK:STDOUT:       %Self.ref: %L1.type = name_ref Self, @L1.%Self [symbolic = %Self (constants.%Self.112)]
// CHECK:STDOUT:       %Self.as_type.loc5_24.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_24.1 (constants.%Self.as_type.351)]
// CHECK:STDOUT:       %.loc5_24.2: type = converted %Self.ref, %Self.as_type.loc5_24.2 [symbolic = %Self.as_type.loc5_24.1 (constants.%Self.as_type.351)]
// CHECK:STDOUT:       %ptr.loc5_24.2: type = ptr_type %.loc5_24.2 [symbolic = %ptr.loc5_24.1 (constants.%ptr.87f2ee.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @L1.S1.%ptr.loc5_24.1 (%ptr.87f2ee.1) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %L1.assoc_type = assoc_entity element1, %L1.S1.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .R1 = %assoc0
// CHECK:STDOUT:   .S1 = %assoc1
// CHECK:STDOUT:   witness = (%L1.R1.decl, %L1.S1.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @L1.R1(@L1.%Self: %L1.type) {
// CHECK:STDOUT:   %Self: %L1.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.112)]
// CHECK:STDOUT:   %Self.as_type.loc4_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_15.1 (constants.%Self.as_type.351)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc4_15.1 [symbolic = %pattern_type (constants.%pattern_type.cb4262.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @L1.R1.%Self.as_type.loc4_15.1 (%Self.as_type.351));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @L1.S1(@L1.%Self: %L1.type) {
// CHECK:STDOUT:   %Self: %L1.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.112)]
// CHECK:STDOUT:   %Self.as_type.loc5_24.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_24.1 (constants.%Self.as_type.351)]
// CHECK:STDOUT:   %ptr.loc5_24.1: type = ptr_type %Self.as_type.loc5_24.1 [symbolic = %ptr.loc5_24.1 (constants.%ptr.87f2ee.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc5_24.1 [symbolic = %pattern_type (constants.%pattern_type.c0652c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @L1.S1.%ptr.loc5_24.1 (%ptr.87f2ee.1));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple4(%T.loc9_12.2: %L1.type) {
// CHECK:STDOUT:   %T.loc9_12.1: %L1.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_12.1 (constants.%T.112)]
// CHECK:STDOUT:   %T.as_type.loc9_23.1: type = facet_access_type %T.loc9_12.1 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_23.1 [symbolic = %pattern_type (constants.%pattern_type.cb4262.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc9_23.1 [symbolic = %require_complete.loc9 (constants.%require_complete.a52deb.1)]
// CHECK:STDOUT:   %L1.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_12.1, @L1 [symbolic = %L1.lookup_impl_witness (constants.%L1.lookup_impl_witness.d9f5df.1)]
// CHECK:STDOUT:   %L1.facet: %L1.type = facet_value %T.as_type.loc9_23.1, (%L1.lookup_impl_witness) [symbolic = %L1.facet (constants.%L1.facet.1cc812.1)]
// CHECK:STDOUT:   %.loc10_4.2: type = fn_type_with_self_type constants.%L1.R1.type, %L1.facet [symbolic = %.loc10_4.2 (constants.%.cad400.1)]
// CHECK:STDOUT:   %impl.elem0.loc10_4.2: @Simple4.%.loc10_4.2 (%.cad400.1) = impl_witness_access %L1.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0.c5e026.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_4.2: <specific function> = specific_impl_function %impl.elem0.loc10_4.2, @L1.R1(%L1.facet) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn.5f85c4.1)]
// CHECK:STDOUT:   %.loc12_8.2: type = fn_type_with_self_type constants.%L1.S1.type, %L1.facet [symbolic = %.loc12_8.2 (constants.%.9b5ba3.1)]
// CHECK:STDOUT:   %impl.elem1.loc12_8.2: @Simple4.%.loc12_8.2 (%.9b5ba3.1) = impl_witness_access %L1.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc12_8.2 (constants.%impl.elem1.d07d0e.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc12_8.2: <specific function> = specific_impl_function %impl.elem1.loc12_8.2, @L1.S1(%L1.facet) [symbolic = %specific_impl_fn.loc12_8.2 (constants.%specific_impl_fn.2473a0.1)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.loc9_23.1 [symbolic = %ptr (constants.%ptr.87f2ee.2)]
// CHECK:STDOUT:   %require_complete.loc12: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc12 (constants.%require_complete.5158e1.1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_12.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.c7977f.1)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %T.as_type.loc9_23.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.f26842.1)]
// CHECK:STDOUT:   %.loc11_3.3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc11_3.3 (constants.%.acaa49.1)]
// CHECK:STDOUT:   %impl.elem0.loc11_3.2: @Simple4.%.loc11_3.3 (%.acaa49.1) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_3.2 (constants.%impl.elem0.b71e6c.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_3.2: <specific function> = specific_impl_function %impl.elem0.loc11_3.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc11_3.2 (constants.%specific_impl_fn.21ccd1.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Simple4.%T.as_type.loc9_23.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc10: @Simple4.%T.as_type.loc9_23.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %R1.ref: %L1.assoc_type = name_ref R1, @L1.%assoc0 [concrete = constants.%assoc0.8b4]
// CHECK:STDOUT:     %T.as_type.loc10: type = facet_access_type constants.%T.112 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_4.1: type = converted constants.%T.112, %T.as_type.loc10 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc10_4.1: @Simple4.%.loc10_4.2 (%.cad400.1) = impl_witness_access constants.%L1.lookup_impl_witness.d9f5df.1, element0 [symbolic = %impl.elem0.loc10_4.2 (constants.%impl.elem0.c5e026.1)]
// CHECK:STDOUT:     %bound_method.loc10_4: <bound method> = bound_method %x.ref.loc10, %impl.elem0.loc10_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_4.1: <specific function> = specific_impl_function %impl.elem0.loc10_4.1, @L1.R1(constants.%L1.facet.1cc812.1) [symbolic = %specific_impl_fn.loc10_4.2 (constants.%specific_impl_fn.5f85c4.1)]
// CHECK:STDOUT:     %bound_method.loc10_8: <bound method> = bound_method %x.ref.loc10, %specific_impl_fn.loc10_4.1
// CHECK:STDOUT:     %.loc10_8: init %empty_tuple.type = call %bound_method.loc10_8(%x.ref.loc10)
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %xcopy.patt: @Simple4.%pattern_type (%pattern_type.cb4262.2) = binding_pattern xcopy [concrete]
// CHECK:STDOUT:       %xcopy.var_patt: @Simple4.%pattern_type (%pattern_type.cb4262.2) = var_pattern %xcopy.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %xcopy.var: ref @Simple4.%T.as_type.loc9_23.1 (%T.as_type) = var %xcopy.var_patt
// CHECK:STDOUT:     %x.ref.loc11: @Simple4.%T.as_type.loc9_23.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     assign %xcopy.var, %x.ref.loc11
// CHECK:STDOUT:     %.loc11_14.1: type = splice_block %.loc11_14.2 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc11: %L1.type = name_ref T, %T.loc9_12.2 [symbolic = %T.loc9_12.1 (constants.%T.112)]
// CHECK:STDOUT:       %T.as_type.loc11_14: type = facet_access_type %T.ref.loc11 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc11_14.2: type = converted %T.ref.loc11, %T.as_type.loc11_14 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %xcopy: ref @Simple4.%T.as_type.loc9_23.1 (%T.as_type) = bind_name xcopy, %xcopy.var
// CHECK:STDOUT:     %xcopy.ref: ref @Simple4.%T.as_type.loc9_23.1 (%T.as_type) = name_ref xcopy, %xcopy
// CHECK:STDOUT:     %S1.ref: %L1.assoc_type = name_ref S1, @L1.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc12: type = facet_access_type constants.%T.112 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc12_8.1: type = converted constants.%T.112, %T.as_type.loc12 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem1.loc12_8.1: @Simple4.%.loc12_8.2 (%.9b5ba3.1) = impl_witness_access constants.%L1.lookup_impl_witness.d9f5df.1, element1 [symbolic = %impl.elem1.loc12_8.2 (constants.%impl.elem1.d07d0e.1)]
// CHECK:STDOUT:     %bound_method.loc12_8: <bound method> = bound_method %xcopy.ref, %impl.elem1.loc12_8.1
// CHECK:STDOUT:     %specific_impl_fn.loc12_8.1: <specific function> = specific_impl_function %impl.elem1.loc12_8.1, @L1.S1(constants.%L1.facet.1cc812.1) [symbolic = %specific_impl_fn.loc12_8.2 (constants.%specific_impl_fn.2473a0.1)]
// CHECK:STDOUT:     %bound_method.loc12_12: <bound method> = bound_method %xcopy.ref, %specific_impl_fn.loc12_8.1
// CHECK:STDOUT:     %addr.loc12: @Simple4.%ptr (%ptr.87f2ee.2) = addr_of %xcopy.ref
// CHECK:STDOUT:     %.loc12_12: init %empty_tuple.type = call %bound_method.loc12_12(%addr.loc12)
// CHECK:STDOUT:     %T.as_type.loc11_3: type = facet_access_type constants.%T.112 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc11_3.1: type = converted constants.%T.112, %T.as_type.loc11_3 [symbolic = %T.as_type.loc9_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc11_3.1: @Simple4.%.loc11_3.3 (%.acaa49.1) = impl_witness_access constants.%Destroy.lookup_impl_witness.c7977f.1, element0 [symbolic = %impl.elem0.loc11_3.2 (constants.%impl.elem0.b71e6c.1)]
// CHECK:STDOUT:     %bound_method.loc11_3.1: <bound method> = bound_method %xcopy.var, %impl.elem0.loc11_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_3.1: <specific function> = specific_impl_function %impl.elem0.loc11_3.1, @Destroy.Op(constants.%Destroy.facet.f26842.1) [symbolic = %specific_impl_fn.loc11_3.2 (constants.%specific_impl_fn.21ccd1.1)]
// CHECK:STDOUT:     %bound_method.loc11_3.2: <bound method> = bound_method %xcopy.var, %specific_impl_fn.loc11_3.1
// CHECK:STDOUT:     %addr.loc11: @Simple4.%ptr (%ptr.87f2ee.2) = addr_of %xcopy.var
// CHECK:STDOUT:     %.loc11_3.2: init %empty_tuple.type = call %bound_method.loc11_3.2(%addr.loc11)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound4(%V.loc16_14.2: %L1.type) {
// CHECK:STDOUT:   %V.loc16_14.1: %L1.type = bind_symbolic_name V, 0 [symbolic = %V.loc16_14.1 (constants.%V)]
// CHECK:STDOUT:   %V.as_type.loc16_25.1: type = facet_access_type %V.loc16_14.1 [symbolic = %V.as_type.loc16_25.1 (constants.%V.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %V.as_type.loc16_25.1 [symbolic = %pattern_type (constants.%pattern_type.cb4262.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %V.as_type.loc16_25.1 [symbolic = %require_complete.loc16 (constants.%require_complete.a52deb.2)]
// CHECK:STDOUT:   %L1.lookup_impl_witness: <witness> = lookup_impl_witness %V.loc16_14.1, @L1 [symbolic = %L1.lookup_impl_witness (constants.%L1.lookup_impl_witness.d9f5df.2)]
// CHECK:STDOUT:   %L1.facet: %L1.type = facet_value %V.as_type.loc16_25.1, (%L1.lookup_impl_witness) [symbolic = %L1.facet (constants.%L1.facet.1cc812.2)]
// CHECK:STDOUT:   %.loc17_4: type = fn_type_with_self_type constants.%L1.R1.type, %L1.facet [symbolic = %.loc17_4 (constants.%.cad400.2)]
// CHECK:STDOUT:   %impl.elem0.loc17_4.2: @Compound4.%.loc17_4 (%.cad400.2) = impl_witness_access %L1.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc17_4.2 (constants.%impl.elem0.c5e026.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_4.2: <specific function> = specific_impl_function %impl.elem0.loc17_4.2, @L1.R1(%L1.facet) [symbolic = %specific_impl_fn.loc17_4.2 (constants.%specific_impl_fn.5f85c4.2)]
// CHECK:STDOUT:   %.loc19_8: type = fn_type_with_self_type constants.%L1.S1.type, %L1.facet [symbolic = %.loc19_8 (constants.%.9b5ba3.2)]
// CHECK:STDOUT:   %impl.elem1.loc19_8.2: @Compound4.%.loc19_8 (%.9b5ba3.2) = impl_witness_access %L1.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc19_8.2 (constants.%impl.elem1.d07d0e.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc19_8.2: <specific function> = specific_impl_function %impl.elem1.loc19_8.2, @L1.S1(%L1.facet) [symbolic = %specific_impl_fn.loc19_8.2 (constants.%specific_impl_fn.2473a0.2)]
// CHECK:STDOUT:   %ptr: type = ptr_type %V.as_type.loc16_25.1 [symbolic = %ptr (constants.%ptr.87f2ee.3)]
// CHECK:STDOUT:   %require_complete.loc19: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc19 (constants.%require_complete.5158e1.2)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %V.loc16_14.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.c7977f.2)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %V.as_type.loc16_25.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.f26842.2)]
// CHECK:STDOUT:   %.loc18_3.3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc18_3.3 (constants.%.acaa49.2)]
// CHECK:STDOUT:   %impl.elem0.loc18_3.2: @Compound4.%.loc18_3.3 (%.acaa49.2) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_3.2 (constants.%impl.elem0.b71e6c.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_3.2: <specific function> = specific_impl_function %impl.elem0.loc18_3.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc18_3.2 (constants.%specific_impl_fn.21ccd1.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @Compound4.%V.as_type.loc16_25.1 (%V.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %y.ref.loc17: @Compound4.%V.as_type.loc16_25.1 (%V.as_type) = name_ref y, %y
// CHECK:STDOUT:     %L1.ref.loc17: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     %R1.ref: %L1.assoc_type = name_ref R1, @L1.%assoc0 [concrete = constants.%assoc0.8b4]
// CHECK:STDOUT:     %impl.elem0.loc17_4.1: @Compound4.%.loc17_4 (%.cad400.2) = impl_witness_access constants.%L1.lookup_impl_witness.d9f5df.2, element0 [symbolic = %impl.elem0.loc17_4.2 (constants.%impl.elem0.c5e026.2)]
// CHECK:STDOUT:     %bound_method.loc17_4: <bound method> = bound_method %y.ref.loc17, %impl.elem0.loc17_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_4.1: <specific function> = specific_impl_function %impl.elem0.loc17_4.1, @L1.R1(constants.%L1.facet.1cc812.2) [symbolic = %specific_impl_fn.loc17_4.2 (constants.%specific_impl_fn.5f85c4.2)]
// CHECK:STDOUT:     %bound_method.loc17_13: <bound method> = bound_method %y.ref.loc17, %specific_impl_fn.loc17_4.1
// CHECK:STDOUT:     %.loc17_13: init %empty_tuple.type = call %bound_method.loc17_13(%y.ref.loc17)
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %ycopy.patt: @Compound4.%pattern_type (%pattern_type.cb4262.3) = binding_pattern ycopy [concrete]
// CHECK:STDOUT:       %ycopy.var_patt: @Compound4.%pattern_type (%pattern_type.cb4262.3) = var_pattern %ycopy.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ycopy.var: ref @Compound4.%V.as_type.loc16_25.1 (%V.as_type) = var %ycopy.var_patt
// CHECK:STDOUT:     %y.ref.loc18: @Compound4.%V.as_type.loc16_25.1 (%V.as_type) = name_ref y, %y
// CHECK:STDOUT:     assign %ycopy.var, %y.ref.loc18
// CHECK:STDOUT:     %.loc18_14.1: type = splice_block %.loc18_14.2 [symbolic = %V.as_type.loc16_25.1 (constants.%V.as_type)] {
// CHECK:STDOUT:       %V.ref.loc18: %L1.type = name_ref V, %V.loc16_14.2 [symbolic = %V.loc16_14.1 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc18_14: type = facet_access_type %V.ref.loc18 [symbolic = %V.as_type.loc16_25.1 (constants.%V.as_type)]
// CHECK:STDOUT:       %.loc18_14.2: type = converted %V.ref.loc18, %V.as_type.loc18_14 [symbolic = %V.as_type.loc16_25.1 (constants.%V.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ycopy: ref @Compound4.%V.as_type.loc16_25.1 (%V.as_type) = bind_name ycopy, %ycopy.var
// CHECK:STDOUT:     %ycopy.ref: ref @Compound4.%V.as_type.loc16_25.1 (%V.as_type) = name_ref ycopy, %ycopy
// CHECK:STDOUT:     %L1.ref.loc19: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     %S1.ref: %L1.assoc_type = name_ref S1, @L1.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc19_8.1: @Compound4.%.loc19_8 (%.9b5ba3.2) = impl_witness_access constants.%L1.lookup_impl_witness.d9f5df.2, element1 [symbolic = %impl.elem1.loc19_8.2 (constants.%impl.elem1.d07d0e.2)]
// CHECK:STDOUT:     %bound_method.loc19_8: <bound method> = bound_method %ycopy.ref, %impl.elem1.loc19_8.1
// CHECK:STDOUT:     %specific_impl_fn.loc19_8.1: <specific function> = specific_impl_function %impl.elem1.loc19_8.1, @L1.S1(constants.%L1.facet.1cc812.2) [symbolic = %specific_impl_fn.loc19_8.2 (constants.%specific_impl_fn.2473a0.2)]
// CHECK:STDOUT:     %bound_method.loc19_17: <bound method> = bound_method %ycopy.ref, %specific_impl_fn.loc19_8.1
// CHECK:STDOUT:     %addr.loc19: @Compound4.%ptr (%ptr.87f2ee.3) = addr_of %ycopy.ref
// CHECK:STDOUT:     %.loc19_17: init %empty_tuple.type = call %bound_method.loc19_17(%addr.loc19)
// CHECK:STDOUT:     %V.as_type.loc18_3: type = facet_access_type constants.%V [symbolic = %V.as_type.loc16_25.1 (constants.%V.as_type)]
// CHECK:STDOUT:     %.loc18_3.1: type = converted constants.%V, %V.as_type.loc18_3 [symbolic = %V.as_type.loc16_25.1 (constants.%V.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc18_3.1: @Compound4.%.loc18_3.3 (%.acaa49.2) = impl_witness_access constants.%Destroy.lookup_impl_witness.c7977f.2, element0 [symbolic = %impl.elem0.loc18_3.2 (constants.%impl.elem0.b71e6c.2)]
// CHECK:STDOUT:     %bound_method.loc18_3.1: <bound method> = bound_method %ycopy.var, %impl.elem0.loc18_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_3.1: <specific function> = specific_impl_function %impl.elem0.loc18_3.1, @Destroy.Op(constants.%Destroy.facet.f26842.2) [symbolic = %specific_impl_fn.loc18_3.2 (constants.%specific_impl_fn.21ccd1.2)]
// CHECK:STDOUT:     %bound_method.loc18_3.2: <bound method> = bound_method %ycopy.var, %specific_impl_fn.loc18_3.1
// CHECK:STDOUT:     %addr.loc18: @Compound4.%ptr (%ptr.87f2ee.3) = addr_of %ycopy.var
// CHECK:STDOUT:     %.loc18_3.2: init %empty_tuple.type = call %bound_method.loc18_3.2(%addr.loc18)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.R1(constants.%Self.112) {
// CHECK:STDOUT:   %Self => constants.%Self.112
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%Self.as_type.351
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb4262.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.S1(constants.%Self.112) {
// CHECK:STDOUT:   %Self => constants.%Self.112
// CHECK:STDOUT:   %Self.as_type.loc5_24.1 => constants.%Self.as_type.351
// CHECK:STDOUT:   %ptr.loc5_24.1 => constants.%ptr.87f2ee.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c0652c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple4(constants.%T.112) {
// CHECK:STDOUT:   %T.loc9_12.1 => constants.%T.112
// CHECK:STDOUT:   %T.as_type.loc9_23.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb4262.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.R1(constants.%L1.facet.1cc812.1) {
// CHECK:STDOUT:   %Self => constants.%L1.facet.1cc812.1
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb4262.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.S1(constants.%L1.facet.1cc812.1) {
// CHECK:STDOUT:   %Self => constants.%L1.facet.1cc812.1
// CHECK:STDOUT:   %Self.as_type.loc5_24.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc5_24.1 => constants.%ptr.87f2ee.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c0652c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound4(constants.%V) {
// CHECK:STDOUT:   %V.loc16_14.1 => constants.%V
// CHECK:STDOUT:   %V.as_type.loc16_25.1 => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb4262.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.R1(constants.%L1.facet.1cc812.2) {
// CHECK:STDOUT:   %Self => constants.%L1.facet.1cc812.2
// CHECK:STDOUT:   %Self.as_type.loc4_15.1 => constants.%V.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb4262.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.S1(constants.%L1.facet.1cc812.2) {
// CHECK:STDOUT:   %Self => constants.%L1.facet.1cc812.2
// CHECK:STDOUT:   %Self.as_type.loc5_24.1 => constants.%V.as_type
// CHECK:STDOUT:   %ptr.loc5_24.1 => constants.%ptr.87f2ee.3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c0652c.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_interface_instance_caller_not.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L2.type: type = facet_type <@L2> [concrete]
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.238fea.1: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %L2.R2.type: type = fn_type @L2.R2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %L2.R2: %L2.R2.type = struct_value () [concrete]
// CHECK:STDOUT:   %L2.assoc_type: type = assoc_entity_type @L2 [concrete]
// CHECK:STDOUT:   %assoc0: %L2.assoc_type = assoc_entity element0, @L2.%L2.R2.decl [concrete]
// CHECK:STDOUT:   %ptr.fb1cff.1: type = ptr_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.40020d.1: type = pattern_type %ptr.fb1cff.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %L2.S2.type: type = fn_type @L2.S2 [concrete]
// CHECK:STDOUT:   %L2.S2: %L2.S2.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %L2.assoc_type = assoc_entity element1, @L2.%L2.S2.decl [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: %L2.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9e9: type = pattern_type %L2.type [concrete]
// CHECK:STDOUT:   %Simple5.type: type = fn_type @Simple5 [concrete]
// CHECK:STDOUT:   %Simple5: %Simple5.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %L2.lookup_impl_witness: <witness> = lookup_impl_witness %T, @L2 [symbolic]
// CHECK:STDOUT:   %L2.facet: %L2.type = facet_value %T.as_type, (%L2.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.cc4: type = fn_type_with_self_type %L2.R2.type, %L2.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.cc4 = impl_witness_access %L2.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.238fea.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b6b: <specific function> = specific_impl_function %impl.elem0, @L2.R2(%L2.facet) [symbolic]
// CHECK:STDOUT:   %.c8d: type = fn_type_with_self_type %L2.S2.type, %L2.facet [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.c8d = impl_witness_access %L2.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %ptr.fb1cff.2: type = ptr_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.40020d.2: type = pattern_type %ptr.fb1cff.2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.077: <specific function> = specific_impl_function %impl.elem1, @L2.S2(%L2.facet) [symbolic]
// CHECK:STDOUT:   %V: %L2.type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %Compound5.type: type = fn_type @Compound5 [concrete]
// CHECK:STDOUT:   %Compound5: %Compound5.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .L2 = %L2.decl
// CHECK:STDOUT:     .Simple5 = %Simple5.decl
// CHECK:STDOUT:     .Compound5 = %Compound5.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %L2.decl: type = interface_decl @L2 [concrete = constants.%L2.type] {} {}
// CHECK:STDOUT:   %Simple5.decl: %Simple5.type = fn_decl @Simple5 [concrete = constants.%Simple5] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9e9 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %L2.ref [concrete = constants.%L2.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L2.ref: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_12.2: %L2.type = bind_symbolic_name T, 0 [symbolic = %T.loc10_12.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound5.decl: %Compound5.type = fn_decl @Compound5 [concrete = constants.%Compound5] {
// CHECK:STDOUT:     %V.patt: %pattern_type.9e9 = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc30: type = splice_block %L2.ref.loc30 [concrete = constants.%L2.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L2.ref.loc30: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc30_14.2: %L2.type = bind_symbolic_name V, 0 [symbolic = %V.loc30_14.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L2 {
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %L2.R2.decl: %L2.R2.type = fn_decl @L2.R2 [concrete = constants.%L2.R2] {
// CHECK:STDOUT:     %self.patt: @L2.R2.%pattern_type (%pattern_type.238fea.1) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @L2.R2.%pattern_type (%pattern_type.238fea.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @L2.R2.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_15.1: type = splice_block %.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %L2.type = name_ref Self, @L2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_15.2: type = converted %Self.ref, %Self.as_type.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @L2.R2.%Self.as_type.loc5_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %L2.assoc_type = assoc_entity element0, %L2.R2.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %L2.S2.decl: %L2.S2.type = fn_decl @L2.S2 [concrete = constants.%L2.S2] {
// CHECK:STDOUT:     %self.patt: @L2.S2.%pattern_type (%pattern_type.40020d.1) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @L2.S2.%pattern_type (%pattern_type.40020d.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc6_9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @L2.S2.%ptr.loc6_24.1 (%ptr.fb1cff.1) = value_param call_param0
// CHECK:STDOUT:     %.loc6_24.1: type = splice_block %ptr.loc6_24.2 [symbolic = %ptr.loc6_24.1 (constants.%ptr.fb1cff.1)] {
// CHECK:STDOUT:       %Self.ref: %L2.type = name_ref Self, @L2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc6_24.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc6_24.2: type = converted %Self.ref, %Self.as_type.loc6_24.2 [symbolic = %Self.as_type.loc6_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %ptr.loc6_24.2: type = ptr_type %.loc6_24.2 [symbolic = %ptr.loc6_24.1 (constants.%ptr.fb1cff.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @L2.S2.%ptr.loc6_24.1 (%ptr.fb1cff.1) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %L2.assoc_type = assoc_entity element1, %L2.S2.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .R2 = %assoc0
// CHECK:STDOUT:   .S2 = %assoc1
// CHECK:STDOUT:   witness = (%L2.R2.decl, %L2.S2.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @L2.R2(@L2.%Self: %L2.type) {
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_15.1 [symbolic = %pattern_type (constants.%pattern_type.238fea.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @L2.R2.%Self.as_type.loc5_15.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @L2.S2(@L2.%Self: %L2.type) {
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc6_24.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_24.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %ptr.loc6_24.1: type = ptr_type %Self.as_type.loc6_24.1 [symbolic = %ptr.loc6_24.1 (constants.%ptr.fb1cff.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc6_24.1 [symbolic = %pattern_type (constants.%pattern_type.40020d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @L2.S2.%ptr.loc6_24.1 (%ptr.fb1cff.1));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple5(%T.loc10_12.2: %L2.type) {
// CHECK:STDOUT:   %T.loc10_12.1: %L2.type = bind_symbolic_name T, 0 [symbolic = %T.loc10_12.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc18_4.2: type = facet_access_type %T.loc10_12.1 [symbolic = %T.as_type.loc18_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %L2.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc10_12.1, @L2 [symbolic = %L2.lookup_impl_witness (constants.%L2.lookup_impl_witness)]
// CHECK:STDOUT:   %L2.facet: %L2.type = facet_value %T.as_type.loc18_4.2, (%L2.lookup_impl_witness) [symbolic = %L2.facet (constants.%L2.facet)]
// CHECK:STDOUT:   %.loc18_4.2: type = fn_type_with_self_type constants.%L2.R2.type, %L2.facet [symbolic = %.loc18_4.2 (constants.%.cc4)]
// CHECK:STDOUT:   %impl.elem0.loc18_4.2: @Simple5.%.loc18_4.2 (%.cc4) = impl_witness_access %L2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_4.2: <specific function> = specific_impl_function %impl.elem0.loc18_4.2, @L2.R2(%L2.facet) [symbolic = %specific_impl_fn.loc18_4.2 (constants.%specific_impl_fn.b6b)]
// CHECK:STDOUT:   %.loc26_4.2: type = fn_type_with_self_type constants.%L2.S2.type, %L2.facet [symbolic = %.loc26_4.2 (constants.%.c8d)]
// CHECK:STDOUT:   %impl.elem1.loc26_4.2: @Simple5.%.loc26_4.2 (%.c8d) = impl_witness_access %L2.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc26_4.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_4.2: <specific function> = specific_impl_function %impl.elem1.loc26_4.2, @L2.S2(%L2.facet) [symbolic = %specific_impl_fn.loc26_4.2 (constants.%specific_impl_fn.077)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc18: %L2.type = name_ref T, %T.loc10_12.2 [symbolic = %T.loc10_12.1 (constants.%T)]
// CHECK:STDOUT:     %R2.ref: %L2.assoc_type = name_ref R2, @L2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %T.as_type.loc18_4.1: type = facet_access_type %T.ref.loc18 [symbolic = %T.as_type.loc18_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc18_4.1: type = converted %T.ref.loc18, %T.as_type.loc18_4.1 [symbolic = %T.as_type.loc18_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc18_4.1: @Simple5.%.loc18_4.2 (%.cc4) = impl_witness_access constants.%L2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc18_4.1: <specific function> = specific_impl_function %impl.elem0.loc18_4.1, @L2.R2(constants.%L2.facet) [symbolic = %specific_impl_fn.loc18_4.2 (constants.%specific_impl_fn.b6b)]
// CHECK:STDOUT:     %.loc18_8: init %empty_tuple.type = call %specific_impl_fn.loc18_4.1(<error>) [concrete = <error>]
// CHECK:STDOUT:     %T.ref.loc26: %L2.type = name_ref T, %T.loc10_12.2 [symbolic = %T.loc10_12.1 (constants.%T)]
// CHECK:STDOUT:     %S2.ref: %L2.assoc_type = name_ref S2, @L2.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %T.as_type.loc26: type = facet_access_type %T.ref.loc26 [symbolic = %T.as_type.loc18_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc26_4.1: type = converted %T.ref.loc26, %T.as_type.loc26 [symbolic = %T.as_type.loc18_4.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem1.loc26_4.1: @Simple5.%.loc26_4.2 (%.c8d) = impl_witness_access constants.%L2.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc26_4.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_4.1: <specific function> = specific_impl_function %impl.elem1.loc26_4.1, @L2.S2(constants.%L2.facet) [symbolic = %specific_impl_fn.loc26_4.2 (constants.%specific_impl_fn.077)]
// CHECK:STDOUT:     %addr: <error> = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc26_8: init %empty_tuple.type = call %specific_impl_fn.loc26_4.1(<error>) [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound5(%V.loc30_14.2: %L2.type) {
// CHECK:STDOUT:   %V.loc30_14.1: %L2.type = bind_symbolic_name V, 0 [symbolic = %V.loc30_14.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %V.ref.loc35: %L2.type = name_ref V, %V.loc30_14.2 [symbolic = %V.loc30_14.1 (constants.%V)]
// CHECK:STDOUT:     %L2.ref.loc35: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     %R2.ref: %L2.assoc_type = name_ref R2, @L2.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %V.ref.loc40: %L2.type = name_ref V, %V.loc30_14.2 [symbolic = %V.loc30_14.1 (constants.%V)]
// CHECK:STDOUT:     %L2.ref.loc40: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     %S2.ref: %L2.assoc_type = name_ref S2, @L2.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L2.R2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.238fea.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L2.S2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc6_24.1 => constants.%Self.as_type
// CHECK:STDOUT:   %ptr.loc6_24.1 => constants.%ptr.fb1cff.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.40020d.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple5(constants.%T) {
// CHECK:STDOUT:   %T.loc10_12.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L2.R2(constants.%L2.facet) {
// CHECK:STDOUT:   %Self => constants.%L2.facet
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.238fea.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L2.S2(constants.%L2.facet) {
// CHECK:STDOUT:   %Self => constants.%L2.facet
// CHECK:STDOUT:   %Self.as_type.loc6_24.1 => constants.%T.as_type
// CHECK:STDOUT:   %ptr.loc6_24.1 => constants.%ptr.fb1cff.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.40020d.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound5(constants.%V) {
// CHECK:STDOUT:   %V.loc30_14.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_combine_non_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.907: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.G.type: type = fn_type @A.G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.G: %A.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.d52: %A.assoc_type = assoc_entity element0, @A.%A.G.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.9d0: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.A.impl.G.type: type = fn_type @C.as.A.impl.G [concrete]
// CHECK:STDOUT:   %C.as.A.impl.G: %C.as.A.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %C, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %Fails.type: type = fn_type @Fails [concrete]
// CHECK:STDOUT:   %Fails: %Fails.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.b22: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.a4f: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude, loc13_42, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.a4f), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Fails = %Fails.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@C.as.A.impl.%C.as.A.impl.G.decl), @C.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   %Fails.decl: %Fails.type = fn_decl @Fails [concrete = constants.%Fails] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.907]
// CHECK:STDOUT:   %A.G.decl: %A.G.type = fn_decl @A.G [concrete = constants.%A.G] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.G.decl [concrete = constants.%assoc0.d52]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%A.G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc7: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.A.impl: %C.ref as %A.ref {
// CHECK:STDOUT:   %C.as.A.impl.G.decl: %C.as.A.impl.G.type = fn_decl @C.as.A.impl.G [concrete = constants.%C.as.A.impl.G] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.A.impl.G.decl
// CHECK:STDOUT:   witness = file.%A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.9d0]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.G(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.A.impl.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Fails() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc22_5.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc22_5.3: init %C = class_init (), %.loc22_5.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc22_5.4: ref %C = temporary %.loc22_5.2, %.loc22_5.3
// CHECK:STDOUT:   %.loc22_7: ref %C = converted %.loc22_5.1, %.loc22_5.4
// CHECK:STDOUT:   %A.ref.loc22_15: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc22_19: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc22: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %A.ref.loc22_15, %impl.elem0.loc22 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc22: init type = call %bound_method.loc22(%A.ref.loc22_15, %A.ref.loc22_19) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc22: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.d52]
// CHECK:STDOUT:   %.loc22_12: %A.type = converted %.loc22_7, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc30_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc30_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc30_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc30_6.3: init %C = class_init (), %.loc30_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc30_6.4: ref %C = temporary %.loc30_6.2, %.loc30_6.3
// CHECK:STDOUT:   %.loc30_8: ref %C = converted %.loc30_6.1, %.loc30_6.4
// CHECK:STDOUT:   %C.ref.loc30_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc30_24: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc30_28: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc30_26: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc30_26: <bound method> = bound_method %A.ref.loc30_24, %impl.elem0.loc30_26 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc30_26: init type = call %bound_method.loc30_26(%A.ref.loc30_24, %A.ref.loc30_28) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc30_29.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc30_26 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc30_29.2: type = converted %type.as.BitAndWith.impl.Op.call.loc30_26, %.loc30_29.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc30: %A.type = facet_value constants.%C, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc30_20: %A.type = converted %C.ref.loc30_18, %A.facet.loc30 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %as_type.loc30: type = facet_access_type %.loc30_20 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc30_30: type = converted %.loc30_20, %as_type.loc30 [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc30_35: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc30_39: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc30_37: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc30_37: <bound method> = bound_method %A.ref.loc30_35, %impl.elem0.loc30_37 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc30_37: init type = call %bound_method.loc30_37(%A.ref.loc30_35, %A.ref.loc30_39) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc30: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.d52]
// CHECK:STDOUT:   %.loc30_32: %A.type = converted %.loc30_8, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc38_6.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %C.ref.loc38_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc38_6.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc38_6.3: init %C = class_init (), %.loc38_6.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc38_6.4: ref %C = temporary %.loc38_6.2, %.loc38_6.3
// CHECK:STDOUT:   %.loc38_8: ref %C = converted %.loc38_6.1, %.loc38_6.4
// CHECK:STDOUT:   %C.ref.loc38_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc38_24: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc38_28: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc38: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc38: <bound method> = bound_method %A.ref.loc38_24, %impl.elem0.loc38 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc38: init type = call %bound_method.loc38(%A.ref.loc38_24, %A.ref.loc38_28) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc38_29.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc38 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc38_29.2: type = converted %type.as.BitAndWith.impl.Op.call.loc38, %.loc38_29.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc38: %A.type = facet_value constants.%C, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc38_20: %A.type = converted %C.ref.loc38_18, %A.facet.loc38 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %as_type.loc38: type = facet_access_type %.loc38_20 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc38_30: type = converted %.loc38_20, %as_type.loc38 [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc38_34: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc38: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.d52]
// CHECK:STDOUT:   %.loc38_32: %A.type = converted %.loc38_8, <error> [concrete = <error>]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc38: <bound method> = bound_method %.loc38_6.4, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc38: %ptr.019 = addr_of %.loc38_6.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc38: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound.loc38(%addr.loc38)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc30: <bound method> = bound_method %.loc30_6.4, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc30: %ptr.019 = addr_of %.loc30_6.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc30: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound.loc30(%addr.loc30)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %.loc22_5.4, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc22: %ptr.019 = addr_of %.loc22_5.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound.loc22(%addr.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.G(constants.%Self.907) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.G(constants.%A.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- allowed_combine_non_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.907: %A.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %A.G.type: type = fn_type @A.G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.G: %A.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.d52: %A.assoc_type = assoc_entity element0, @A.%A.G.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness file.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.A.impl.G.type: type = fn_type @C.as.A.impl.G [concrete]
// CHECK:STDOUT:   %C.as.A.impl.G: %C.as.A.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %C, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %Works.type: type = fn_type @Works [concrete]
// CHECK:STDOUT:   %Works: %Works.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.b22: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %.6b1: type = fn_type_with_self_type %A.G.type, %A.facet [concrete]
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
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.a4f: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude, loc13_42, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.a4f), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Works = %Works.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = interface_decl @A [concrete = constants.%A.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.A.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (@C.as.A.impl.%C.as.A.impl.G.decl), @C.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:   %Works.decl: %Works.type = fn_decl @Works [concrete = constants.%Works] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.907]
// CHECK:STDOUT:   %A.G.decl: %A.G.type = fn_decl @A.G [concrete = constants.%A.G] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.G.decl [concrete = constants.%assoc0.d52]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = %assoc0
// CHECK:STDOUT:   witness = (%A.G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc7: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.A.impl: %C.ref as %A.ref {
// CHECK:STDOUT:   %C.as.A.impl.G.decl: %C.as.A.impl.G.type = fn_decl @C.as.A.impl.G [concrete = constants.%C.as.A.impl.G] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.A.impl.G.decl
// CHECK:STDOUT:   witness = file.%A.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.G(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.A.impl.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Works() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc13_7: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc13_11: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc13_9: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %A.ref.loc13_7, %impl.elem0.loc13_9 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc13: init type = call %bound_method.loc13(%A.ref.loc13_7, %A.ref.loc13_11) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc13: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.d52]
// CHECK:STDOUT:   %A.facet.loc13: %A.type = facet_value constants.%C, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc13: %A.type = converted %C.ref.loc13, %A.facet.loc13 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %impl.elem0.loc13_4: %.6b1 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%C.as.A.impl.G]
// CHECK:STDOUT:   %C.as.A.impl.G.call.loc13: init %empty_tuple.type = call %impl.elem0.loc13_4()
// CHECK:STDOUT:   %C.ref.loc14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc14_10: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc14_14: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc14_12: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc14_12: <bound method> = bound_method %A.ref.loc14_10, %impl.elem0.loc14_12 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc14_12: init type = call %bound_method.loc14_12(%A.ref.loc14_10, %A.ref.loc14_14) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc14_15.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc14_12 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc14_15.2: type = converted %type.as.BitAndWith.impl.Op.call.loc14_12, %.loc14_15.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc14: %A.type = facet_value constants.%C, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc14_6: %A.type = converted %C.ref.loc14, %A.facet.loc14 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %A.ref.loc14_20: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc14_24: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc14_22: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc14_22: <bound method> = bound_method %A.ref.loc14_20, %impl.elem0.loc14_22 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc14_22: init type = call %bound_method.loc14_22(%A.ref.loc14_20, %A.ref.loc14_24) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc14: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.d52]
// CHECK:STDOUT:   %impl.elem0.loc14_17: %.6b1 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%C.as.A.impl.G]
// CHECK:STDOUT:   %C.as.A.impl.G.call.loc14: init %empty_tuple.type = call %impl.elem0.loc14_17()
// CHECK:STDOUT:   %C.ref.loc15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc15_10: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc15_14: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc15_12: %.b22 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc15: <bound method> = bound_method %A.ref.loc15_10, %impl.elem0.loc15_12 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc15: init type = call %bound_method.loc15(%A.ref.loc15_10, %A.ref.loc15_14) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc15_15.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc15 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc15_15.2: type = converted %type.as.BitAndWith.impl.Op.call.loc15, %.loc15_15.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc15: %A.type = facet_value constants.%C, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc15_6: %A.type = converted %C.ref.loc15, %A.facet.loc15 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %A.ref.loc15_19: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc15: %A.assoc_type = name_ref G, @A.%assoc0 [concrete = constants.%assoc0.d52]
// CHECK:STDOUT:   %impl.elem0.loc15_17: %.6b1 = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%C.as.A.impl.G]
// CHECK:STDOUT:   %C.as.A.impl.G.call.loc15: init %empty_tuple.type = call %impl.elem0.loc15_17()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.G(constants.%Self.907) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.G(constants.%A.facet) {}
// CHECK:STDOUT:

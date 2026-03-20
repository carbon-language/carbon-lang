// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
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
fn Simple1(T:! J, unused S:! T.U) {}

// This should be equivalent to `Simple1` above, but using compound member access.
fn Compound1(V:! J, unused W:! V.(J.U)) {}

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
  // CHECK:STDERR: fail_caller_instance_interface_not.carbon:[[@LINE+4]]:3: note: type `V` does not implement interface `Core.ImplicitAs(K2)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   y.(K2.Q2)();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  y.(K2.Q2)();
}

// --- instance.carbon
library "[[@TEST_NAME]]";

interface L1 {
  fn R1[self: Self]();
  fn S1[ref self: Self]();
}

// Simple member access.
fn Simple4(T:! L1, x: T, p: T*) {
  x.R1();
  p->S1();
}

// This should be equivalent to `Simple4` above, but using compound member access.
fn Compound4(V:! L1, y: V, p: V*) {
  y.(L1.R1)();
  p->(L1.S1)();
}

// --- fail_interface_instance_caller_not.carbon

library "[[@TEST_NAME]]";

interface L2 {
  fn R2[self: Self]();
  fn S2[ref self: Self]();
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
  // CHECK:STDERR:   fn S2[ref self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+4]]:3: note: type `C` does not implement interface `Core.ImplicitAs(A)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   ({} as C).((A & A).G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as C).((A & A).G)();
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `C` into type implementing `A` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   (({} as C) as (C as (A & A))).((A & A).G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+4]]:3: note: type `C` does not implement interface `Core.ImplicitAs(A)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   (({} as C) as (C as (A & A))).((A & A).G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (({} as C) as (C as (A & A))).((A & A).G)();
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+7]]:3: error: cannot implicitly convert non-type value of type `C` into type implementing `A` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   (({} as C) as (C as (A & A))).(A.G)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_combine_non_instance.carbon:[[@LINE+4]]:3: note: type `C` does not implement interface `Core.ImplicitAs(A)` [MissingImplInMemberAccessInContext]
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
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.WithSelf.%U [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %J.type [concrete]
// CHECK:STDOUT:   %T: %J.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.264a02.1: <witness> = lookup_impl_witness %T, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.5607ef.1: type = impl_witness_access %J.lookup_impl_witness.264a02.1, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ad6d12.1: type = pattern_type %impl.elem0.5607ef.1 [symbolic]
// CHECK:STDOUT:   %S: %impl.elem0.5607ef.1 = symbolic_binding S, 1 [symbolic]
// CHECK:STDOUT:   %Simple1.type: type = fn_type @Simple1 [concrete]
// CHECK:STDOUT:   %Simple1: %Simple1.type = struct_value () [concrete]
// CHECK:STDOUT:   %V: %J.type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %J.lookup_impl_witness.264a02.2: <witness> = lookup_impl_witness %V, @J [symbolic]
// CHECK:STDOUT:   %impl.elem0.5607ef.2: type = impl_witness_access %J.lookup_impl_witness.264a02.2, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ad6d12.2: type = pattern_type %impl.elem0.5607ef.2 [symbolic]
// CHECK:STDOUT:   %W: %impl.elem0.5607ef.2 = symbolic_binding W, 1 [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.f76 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %S.patt: @Simple1.%pattern_type (%pattern_type.ad6d12.1) = symbolic_binding_pattern S, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %J.ref [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16.2: %J.type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:     %.loc24.1: type = splice_block %impl.elem0.loc24.2 [symbolic = %impl.elem0.loc24.1 (constants.%impl.elem0.5607ef.1)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: %J.type = name_ref T, %T.loc16.2 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc24.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc24.2: type = impl_witness_access constants.%J.lookup_impl_witness.264a02.1, element0 [symbolic = %impl.elem0.loc24.1 (constants.%impl.elem0.5607ef.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %S.loc21.2: @Simple1.%impl.elem0.loc24.1 (%impl.elem0.5607ef.1) = symbolic_binding S, 1 [symbolic = %S.loc21.1 (constants.%S)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound1.decl: %Compound1.type = fn_decl @Compound1 [concrete = constants.%Compound1] {
// CHECK:STDOUT:     %V.patt: %pattern_type.f76 = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %W.patt: @Compound1.%pattern_type (%pattern_type.ad6d12.2) = symbolic_binding_pattern W, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc34: type = splice_block %J.ref.loc34 [concrete = constants.%J.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref.loc34: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc32.2: %J.type = symbolic_binding V, 0 [symbolic = %V.loc32.1 (constants.%V)]
// CHECK:STDOUT:     %.loc40: type = splice_block %impl.elem0.loc40.2 [symbolic = %impl.elem0.loc40.1 (constants.%impl.elem0.5607ef.2)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %V.ref: %J.type = name_ref V, %V.loc32.2 [symbolic = %V.loc32.1 (constants.%V)]
// CHECK:STDOUT:       %J.ref.loc42: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc40.2: type = impl_witness_access constants.%J.lookup_impl_witness.264a02.2, element0 [symbolic = %impl.elem0.loc40.1 (constants.%impl.elem0.5607ef.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %W.loc37.2: @Compound1.%impl.elem0.loc40.1 (%impl.elem0.5607ef.2) = symbolic_binding W, 1 [symbolic = %W.loc37.1 (constants.%W)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.WithSelf.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   witness = (@J.WithSelf.%U)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple1(%T.loc16.2: %J.type, %S.loc21.2: @Simple1.%impl.elem0.loc24.1 (%impl.elem0.5607ef.1)) {
// CHECK:STDOUT:   %T.loc16.1: %J.type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc16.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc16.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.264a02.1)]
// CHECK:STDOUT:   %impl.elem0.loc24.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc24.1 (constants.%impl.elem0.5607ef.1)]
// CHECK:STDOUT:   %S.loc21.1: @Simple1.%impl.elem0.loc24.1 (%impl.elem0.5607ef.1) = symbolic_binding S, 1 [symbolic = %S.loc21.1 (constants.%S)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc24.1 [symbolic = %pattern_type (constants.%pattern_type.ad6d12.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound1(%V.loc32.2: %J.type, %W.loc37.2: @Compound1.%impl.elem0.loc40.1 (%impl.elem0.5607ef.2)) {
// CHECK:STDOUT:   %V.loc32.1: %J.type = symbolic_binding V, 0 [symbolic = %V.loc32.1 (constants.%V)]
// CHECK:STDOUT:   %J.lookup_impl_witness: <witness> = lookup_impl_witness %V.loc32.1, @J [symbolic = %J.lookup_impl_witness (constants.%J.lookup_impl_witness.264a02.2)]
// CHECK:STDOUT:   %impl.elem0.loc40.1: type = impl_witness_access %J.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc40.1 (constants.%impl.elem0.5607ef.2)]
// CHECK:STDOUT:   %W.loc37.1: @Compound1.%impl.elem0.loc40.1 (%impl.elem0.5607ef.2) = symbolic_binding W, 1 [symbolic = %W.loc37.1 (constants.%W)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc40.1 [symbolic = %pattern_type (constants.%pattern_type.ad6d12.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple1(constants.%T, constants.%S) {
// CHECK:STDOUT:   %T.loc16.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.264a02.1
// CHECK:STDOUT:   %impl.elem0.loc24.1 => constants.%impl.elem0.5607ef.1
// CHECK:STDOUT:   %S.loc21.1 => constants.%S
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ad6d12.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%V) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound1(constants.%V, constants.%W) {
// CHECK:STDOUT:   %V.loc32.1 => constants.%V
// CHECK:STDOUT:   %J.lookup_impl_witness => constants.%J.lookup_impl_witness.264a02.2
// CHECK:STDOUT:   %impl.elem0.loc40.1 => constants.%impl.elem0.5607ef.2
// CHECK:STDOUT:   %W.loc37.1 => constants.%W
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ad6d12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K1.type: type = facet_type <@K1> [concrete]
// CHECK:STDOUT:   %Self: %K1.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %K1.WithSelf.Q1.type.cab: type = fn_type @K1.WithSelf.Q1, @K1.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %K1.WithSelf.Q1.d38: %K1.WithSelf.Q1.type.cab = struct_value () [symbolic]
// CHECK:STDOUT:   %K1.assoc_type: type = assoc_entity_type @K1 [concrete]
// CHECK:STDOUT:   %assoc0: %K1.assoc_type = assoc_entity element0, @K1.WithSelf.%K1.WithSelf.Q1.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %K1.type [concrete]
// CHECK:STDOUT:   %T: %K1.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Simple2.type: type = fn_type @Simple2 [concrete]
// CHECK:STDOUT:   %Simple2: %Simple2.type = struct_value () [concrete]
// CHECK:STDOUT:   %K1.WithSelf.Q1.type.0b941d.1: type = fn_type @K1.WithSelf.Q1, @K1.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %K1.WithSelf.Q1.2b1e8b.1: %K1.WithSelf.Q1.type.0b941d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %K1.lookup_impl_witness.962f78.1: <witness> = lookup_impl_witness %T, @K1 [symbolic]
// CHECK:STDOUT:   %.838067.1: type = fn_type_with_self_type %K1.WithSelf.Q1.type.0b941d.1, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0.6aa105.1: %.838067.1 = impl_witness_access %K1.lookup_impl_witness.962f78.1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c7d99a.1: <specific function> = specific_impl_function %impl.elem0.6aa105.1, @K1.WithSelf.Q1(%T) [symbolic]
// CHECK:STDOUT:   %V: %K1.type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %Compound2.type: type = fn_type @Compound2 [concrete]
// CHECK:STDOUT:   %Compound2: %Compound2.type = struct_value () [concrete]
// CHECK:STDOUT:   %K1.lookup_impl_witness.962f78.2: <witness> = lookup_impl_witness %V, @K1 [symbolic]
// CHECK:STDOUT:   %K1.WithSelf.Q1.type.0b941d.2: type = fn_type @K1.WithSelf.Q1, @K1.WithSelf(%V) [symbolic]
// CHECK:STDOUT:   %K1.WithSelf.Q1.2b1e8b.2: %K1.WithSelf.Q1.type.0b941d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.838067.2: type = fn_type_with_self_type %K1.WithSelf.Q1.type.0b941d.2, %V [symbolic]
// CHECK:STDOUT:   %impl.elem0.6aa105.2: %.838067.2 = impl_witness_access %K1.lookup_impl_witness.962f78.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c7d99a.2: <specific function> = specific_impl_function %impl.elem0.6aa105.2, @K1.WithSelf.Q1(%V) [symbolic]
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
// CHECK:STDOUT:     %.loc18: type = splice_block %K1.ref [concrete = constants.%K1.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K1.ref: type = name_ref K1, file.%K1.decl [concrete = constants.%K1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16.2: %K1.type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound2.decl: %Compound2.type = fn_decl @Compound2 [concrete = constants.%Compound2] {
// CHECK:STDOUT:     %V.patt: %pattern_type = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc33: type = splice_block %K1.ref.loc33 [concrete = constants.%K1.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K1.ref.loc33: type = name_ref K1, file.%K1.decl [concrete = constants.%K1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc31.2: %K1.type = symbolic_binding V, 0 [symbolic = %V.loc31.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K1 {
// CHECK:STDOUT:   %Self: %K1.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %K1.WithSelf.decl = interface_with_self_decl @K1 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %K1.WithSelf.Q1.decl: @K1.WithSelf.%K1.WithSelf.Q1.type (%K1.WithSelf.Q1.type.cab) = fn_decl @K1.WithSelf.Q1 [symbolic = @K1.WithSelf.%K1.WithSelf.Q1 (constants.%K1.WithSelf.Q1.d38)] {} {}
// CHECK:STDOUT:   %assoc0: %K1.assoc_type = assoc_entity element0, %K1.WithSelf.Q1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Q1 = @K1.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@K1.WithSelf.%K1.WithSelf.Q1.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K1.WithSelf.Q1(@K1.%Self: %K1.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple2(%T.loc16.2: %K1.type) {
// CHECK:STDOUT:   %T.loc16.1: %K1.type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc16.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %K1.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc16.1, @K1 [symbolic = %K1.lookup_impl_witness (constants.%K1.lookup_impl_witness.962f78.1)]
// CHECK:STDOUT:   %K1.WithSelf.Q1.type: type = fn_type @K1.WithSelf.Q1, @K1.WithSelf(%T.loc16.1) [symbolic = %K1.WithSelf.Q1.type (constants.%K1.WithSelf.Q1.type.0b941d.1)]
// CHECK:STDOUT:   %.loc22.2: type = fn_type_with_self_type %K1.WithSelf.Q1.type, %T.loc16.1 [symbolic = %.loc22.2 (constants.%.838067.1)]
// CHECK:STDOUT:   %impl.elem0.loc22.2: @Simple2.%.loc22.2 (%.838067.1) = impl_witness_access %K1.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22.2 (constants.%impl.elem0.6aa105.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc22.2: <specific function> = specific_impl_function %impl.elem0.loc22.2, @K1.WithSelf.Q1(%T.loc16.1) [symbolic = %specific_impl_fn.loc22.2 (constants.%specific_impl_fn.c7d99a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: %K1.type = name_ref T, %T.loc16.2 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc22.1: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Q1.ref: %K1.assoc_type = name_ref Q1, @K1.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc22.1: @Simple2.%.loc22.2 (%.838067.1) = impl_witness_access constants.%K1.lookup_impl_witness.962f78.1, element0 [symbolic = %impl.elem0.loc22.2 (constants.%impl.elem0.6aa105.1)]
// CHECK:STDOUT:     %specific_impl_fn.loc22.1: <specific function> = specific_impl_function %impl.elem0.loc22.1, @K1.WithSelf.Q1(constants.%T) [symbolic = %specific_impl_fn.loc22.2 (constants.%specific_impl_fn.c7d99a.1)]
// CHECK:STDOUT:     %K1.WithSelf.Q1.call: init %empty_tuple.type = call %specific_impl_fn.loc22.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound2(%V.loc31.2: %K1.type) {
// CHECK:STDOUT:   %V.loc31.1: %K1.type = symbolic_binding V, 0 [symbolic = %V.loc31.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %K1.lookup_impl_witness: <witness> = lookup_impl_witness %V.loc31.1, @K1 [symbolic = %K1.lookup_impl_witness (constants.%K1.lookup_impl_witness.962f78.2)]
// CHECK:STDOUT:   %K1.WithSelf.Q1.type: type = fn_type @K1.WithSelf.Q1, @K1.WithSelf(%V.loc31.1) [symbolic = %K1.WithSelf.Q1.type (constants.%K1.WithSelf.Q1.type.0b941d.2)]
// CHECK:STDOUT:   %.loc37: type = fn_type_with_self_type %K1.WithSelf.Q1.type, %V.loc31.1 [symbolic = %.loc37 (constants.%.838067.2)]
// CHECK:STDOUT:   %impl.elem0.loc37.2: @Compound2.%.loc37 (%.838067.2) = impl_witness_access %K1.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc37.2 (constants.%impl.elem0.6aa105.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc37.2: <specific function> = specific_impl_function %impl.elem0.loc37.2, @K1.WithSelf.Q1(%V.loc31.1) [symbolic = %specific_impl_fn.loc37.2 (constants.%specific_impl_fn.c7d99a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %V.ref: %K1.type = name_ref V, %V.loc31.2 [symbolic = %V.loc31.1 (constants.%V)]
// CHECK:STDOUT:     %K1.ref.loc39: type = name_ref K1, file.%K1.decl [concrete = constants.%K1.type]
// CHECK:STDOUT:     %Q1.ref: %K1.assoc_type = name_ref Q1, @K1.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc37.1: @Compound2.%.loc37 (%.838067.2) = impl_witness_access constants.%K1.lookup_impl_witness.962f78.2, element0 [symbolic = %impl.elem0.loc37.2 (constants.%impl.elem0.6aa105.2)]
// CHECK:STDOUT:     %specific_impl_fn.loc37.1: <specific function> = specific_impl_function %impl.elem0.loc37.1, @K1.WithSelf.Q1(constants.%V) [symbolic = %specific_impl_fn.loc37.2 (constants.%specific_impl_fn.c7d99a.2)]
// CHECK:STDOUT:     %K1.WithSelf.Q1.call: init %empty_tuple.type = call %specific_impl_fn.loc37.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K1.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %K1.WithSelf.Q1.type => constants.%K1.WithSelf.Q1.type.cab
// CHECK:STDOUT:   %K1.WithSelf.Q1 => constants.%K1.WithSelf.Q1.d38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K1.WithSelf.Q1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple2(constants.%T) {
// CHECK:STDOUT:   %T.loc16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K1.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %K1.WithSelf.Q1.type => constants.%K1.WithSelf.Q1.type.0b941d.1
// CHECK:STDOUT:   %K1.WithSelf.Q1 => constants.%K1.WithSelf.Q1.2b1e8b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K1.WithSelf.Q1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound2(constants.%V) {
// CHECK:STDOUT:   %V.loc31.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K1.WithSelf(constants.%V) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%V
// CHECK:STDOUT:   %K1.WithSelf.Q1.type => constants.%K1.WithSelf.Q1.type.0b941d.2
// CHECK:STDOUT:   %K1.WithSelf.Q1 => constants.%K1.WithSelf.Q1.2b1e8b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K1.WithSelf.Q1(constants.%V) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_caller_instance_interface_not.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K2.type: type = facet_type <@K2> [concrete]
// CHECK:STDOUT:   %Self.404: %K2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %K2.WithSelf.Q2.type.0f8: type = fn_type @K2.WithSelf.Q2, @K2.WithSelf(%Self.404) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %K2.WithSelf.Q2.c3d: %K2.WithSelf.Q2.type.0f8 = struct_value () [symbolic]
// CHECK:STDOUT:   %K2.assoc_type: type = assoc_entity_type @K2 [concrete]
// CHECK:STDOUT:   %assoc0.5ff: %K2.assoc_type = assoc_entity element0, @K2.WithSelf.%K2.WithSelf.Q2.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.8cb: type = pattern_type %K2.type [concrete]
// CHECK:STDOUT:   %T: %K2.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.6a7df6.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Simple3.type: type = fn_type @Simple3 [concrete]
// CHECK:STDOUT:   %Simple3: %Simple3.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.57854f.1: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %K2.WithSelf.Q2.type.3af: type = fn_type @K2.WithSelf.Q2, @K2.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %K2.WithSelf.Q2.c9a: %K2.WithSelf.Q2.type.3af = struct_value () [symbolic]
// CHECK:STDOUT:   %K2.lookup_impl_witness: <witness> = lookup_impl_witness %T, @K2 [symbolic]
// CHECK:STDOUT:   %.02c: type = fn_type_with_self_type %K2.WithSelf.Q2.type.3af, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.02c = impl_witness_access %K2.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @K2.WithSelf.Q2(%T) [symbolic]
// CHECK:STDOUT:   %V: %K2.type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 0, %V [symbolic]
// CHECK:STDOUT:   %pattern_type.6a7df6.2: type = pattern_type %V.binding.as_type [symbolic]
// CHECK:STDOUT:   %Compound3.type: type = fn_type @Compound3 [concrete]
// CHECK:STDOUT:   %Compound3: %Compound3.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.57854f.2: <witness> = require_complete_type %V.binding.as_type [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.8cb = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @Simple3.%pattern_type (%pattern_type.6a7df6.1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @Simple3.%pattern_type (%pattern_type.6a7df6.1) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %K2.ref [concrete = constants.%K2.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K2.ref: type = name_ref K2, file.%K2.decl [concrete = constants.%K2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16.2: %K2.type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Simple3.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc22.1: type = splice_block %.loc22.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref: %K2.type = name_ref T, %T.loc16.2 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc22.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Simple3.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound3.decl: %Compound3.type = fn_decl @Compound3 [concrete = constants.%Compound3] {
// CHECK:STDOUT:     %V.patt: %pattern_type.8cb = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %y.param_patt: @Compound3.%pattern_type (%pattern_type.6a7df6.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: @Compound3.%pattern_type (%pattern_type.6a7df6.2) = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc37: type = splice_block %K2.ref.loc37 [concrete = constants.%K2.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %K2.ref.loc37: type = name_ref K2, file.%K2.decl [concrete = constants.%K2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc35.2: %K2.type = symbolic_binding V, 0 [symbolic = %V.loc35.1 (constants.%V)]
// CHECK:STDOUT:     %y.param: @Compound3.%V.binding.as_type (%V.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc41.1: type = splice_block %.loc41.2 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)] {
// CHECK:STDOUT:       %V.ref: %K2.type = name_ref V, %V.loc35.2 [symbolic = %V.loc35.1 (constants.%V)]
// CHECK:STDOUT:       %V.as_type: type = facet_access_type %V.ref [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:       %.loc41.2: type = converted %V.ref, %V.as_type [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @Compound3.%V.binding.as_type (%V.binding.as_type) = value_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K2 {
// CHECK:STDOUT:   %Self: %K2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.404]
// CHECK:STDOUT:   %K2.WithSelf.decl = interface_with_self_decl @K2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %K2.WithSelf.Q2.decl: @K2.WithSelf.%K2.WithSelf.Q2.type (%K2.WithSelf.Q2.type.0f8) = fn_decl @K2.WithSelf.Q2 [symbolic = @K2.WithSelf.%K2.WithSelf.Q2 (constants.%K2.WithSelf.Q2.c3d)] {} {}
// CHECK:STDOUT:   %assoc0: %K2.assoc_type = assoc_entity element0, %K2.WithSelf.Q2.decl [concrete = constants.%assoc0.5ff]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Q2 = @K2.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@K2.WithSelf.%K2.WithSelf.Q2.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @K2.WithSelf.Q2(@K2.%Self: %K2.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple3(%T.loc16.2: %K2.type) {
// CHECK:STDOUT:   %T.loc16.1: %K2.type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc16.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.6a7df6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.57854f.1)]
// CHECK:STDOUT:   %K2.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc16.1, @K2 [symbolic = %K2.lookup_impl_witness (constants.%K2.lookup_impl_witness)]
// CHECK:STDOUT:   %K2.WithSelf.Q2.type: type = fn_type @K2.WithSelf.Q2, @K2.WithSelf(%T.loc16.1) [symbolic = %K2.WithSelf.Q2.type (constants.%K2.WithSelf.Q2.type.3af)]
// CHECK:STDOUT:   %.loc26: type = fn_type_with_self_type %K2.WithSelf.Q2.type, %T.loc16.1 [symbolic = %.loc26 (constants.%.02c)]
// CHECK:STDOUT:   %impl.elem0.loc26.2: @Simple3.%.loc26 (%.02c) = impl_witness_access %K2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc26.2: <specific function> = specific_impl_function %impl.elem0.loc26.2, @K2.WithSelf.Q2(%T.loc16.1) [symbolic = %specific_impl_fn.loc26.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Simple3.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Simple3.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Q2.ref: %K2.assoc_type = name_ref Q2, @K2.WithSelf.%assoc0 [concrete = constants.%assoc0.5ff]
// CHECK:STDOUT:     %impl.elem0.loc26.1: @Simple3.%.loc26 (%.02c) = impl_witness_access constants.%K2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc26.1: <specific function> = specific_impl_function %impl.elem0.loc26.1, @K2.WithSelf.Q2(constants.%T) [symbolic = %specific_impl_fn.loc26.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %K2.WithSelf.Q2.call: init %empty_tuple.type = call %specific_impl_fn.loc26.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound3(%V.loc35.2: %K2.type) {
// CHECK:STDOUT:   %V.loc35.1: %K2.type = symbolic_binding V, 0 [symbolic = %V.loc35.1 (constants.%V)]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 0, %V.loc35.1 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %V.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.6a7df6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %V.binding.as_type [symbolic = %require_complete (constants.%require_complete.57854f.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @Compound3.%V.binding.as_type (%V.binding.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %y.ref: @Compound3.%V.binding.as_type (%V.binding.as_type) = name_ref y, %y
// CHECK:STDOUT:     %K2.ref.loc47: type = name_ref K2, file.%K2.decl [concrete = constants.%K2.type]
// CHECK:STDOUT:     %Q2.ref: %K2.assoc_type = name_ref Q2, @K2.WithSelf.%assoc0 [concrete = constants.%assoc0.5ff]
// CHECK:STDOUT:     %.loc45: %K2.type = converted %y.ref, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K2.WithSelf(constants.%Self.404) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.404
// CHECK:STDOUT:   %K2.WithSelf.Q2.type => constants.%K2.WithSelf.Q2.type.0f8
// CHECK:STDOUT:   %K2.WithSelf.Q2 => constants.%K2.WithSelf.Q2.c3d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K2.WithSelf.Q2(constants.%Self.404) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple3(constants.%T) {
// CHECK:STDOUT:   %T.loc16.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6a7df6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K2.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %K2.WithSelf.Q2.type => constants.%K2.WithSelf.Q2.type.3af
// CHECK:STDOUT:   %K2.WithSelf.Q2 => constants.%K2.WithSelf.Q2.c9a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K2.WithSelf.Q2(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound3(constants.%V) {
// CHECK:STDOUT:   %V.loc35.1 => constants.%V
// CHECK:STDOUT:   %V.binding.as_type => constants.%V.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6a7df6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L1.type: type = facet_type <@L1> [concrete]
// CHECK:STDOUT:   %Self: %L1.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.9c1: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %L1.WithSelf.R1.type.504: type = fn_type @L1.WithSelf.R1, @L1.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %L1.WithSelf.R1.aa4: %L1.WithSelf.R1.type.504 = struct_value () [symbolic]
// CHECK:STDOUT:   %L1.assoc_type: type = assoc_entity_type @L1 [concrete]
// CHECK:STDOUT:   %assoc0: %L1.assoc_type = assoc_entity element0, @L1.WithSelf.%L1.WithSelf.R1.decl [concrete]
// CHECK:STDOUT:   %L1.WithSelf.S1.type.e03: type = fn_type @L1.WithSelf.S1, @L1.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %L1.WithSelf.S1.480: %L1.WithSelf.S1.type.e03 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %L1.assoc_type = assoc_entity element1, @L1.WithSelf.%L1.WithSelf.S1.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a5c: type = pattern_type %L1.type [concrete]
// CHECK:STDOUT:   %T: %L1.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.352a62.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %ptr.6c1f8e.1: type = ptr_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.740d56.1: type = pattern_type %ptr.6c1f8e.1 [symbolic]
// CHECK:STDOUT:   %Simple4.type: type = fn_type @Simple4 [concrete]
// CHECK:STDOUT:   %Simple4: %Simple4.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.592d00.1: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.94a5f1.1: <witness> = require_complete_type %ptr.6c1f8e.1 [symbolic]
// CHECK:STDOUT:   %L1.WithSelf.R1.type.0d7469.1: type = fn_type @L1.WithSelf.R1, @L1.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %L1.WithSelf.R1.f22128.1: %L1.WithSelf.R1.type.0d7469.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %L1.WithSelf.S1.type.79e6b4.1: type = fn_type @L1.WithSelf.S1, @L1.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %L1.WithSelf.S1.cd5111.1: %L1.WithSelf.S1.type.79e6b4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %L1.lookup_impl_witness.297fbf.1: <witness> = lookup_impl_witness %T, @L1 [symbolic]
// CHECK:STDOUT:   %.69b631.1: type = fn_type_with_self_type %L1.WithSelf.R1.type.0d7469.1, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0.60b493.1: %.69b631.1 = impl_witness_access %L1.lookup_impl_witness.297fbf.1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f023e9.1: <specific function> = specific_impl_function %impl.elem0.60b493.1, @L1.WithSelf.R1(%T) [symbolic]
// CHECK:STDOUT:   %.917578.1: type = fn_type_with_self_type %L1.WithSelf.S1.type.79e6b4.1, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1.e3fe4c.1: %.917578.1 = impl_witness_access %L1.lookup_impl_witness.297fbf.1, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2cc093.1: <specific function> = specific_impl_function %impl.elem1.e3fe4c.1, @L1.WithSelf.S1(%T) [symbolic]
// CHECK:STDOUT:   %V: %L1.type = symbolic_binding V, 0 [symbolic]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 0, %V [symbolic]
// CHECK:STDOUT:   %pattern_type.352a62.2: type = pattern_type %V.binding.as_type [symbolic]
// CHECK:STDOUT:   %ptr.6c1f8e.2: type = ptr_type %V.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.740d56.2: type = pattern_type %ptr.6c1f8e.2 [symbolic]
// CHECK:STDOUT:   %Compound4.type: type = fn_type @Compound4 [concrete]
// CHECK:STDOUT:   %Compound4: %Compound4.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.592d00.2: <witness> = require_complete_type %V.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.94a5f1.2: <witness> = require_complete_type %ptr.6c1f8e.2 [symbolic]
// CHECK:STDOUT:   %L1.lookup_impl_witness.297fbf.2: <witness> = lookup_impl_witness %V, @L1 [symbolic]
// CHECK:STDOUT:   %L1.WithSelf.R1.type.0d7469.2: type = fn_type @L1.WithSelf.R1, @L1.WithSelf(%V) [symbolic]
// CHECK:STDOUT:   %L1.WithSelf.R1.f22128.2: %L1.WithSelf.R1.type.0d7469.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %L1.WithSelf.S1.type.79e6b4.2: type = fn_type @L1.WithSelf.S1, @L1.WithSelf(%V) [symbolic]
// CHECK:STDOUT:   %L1.WithSelf.S1.cd5111.2: %L1.WithSelf.S1.type.79e6b4.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.69b631.2: type = fn_type_with_self_type %L1.WithSelf.R1.type.0d7469.2, %V [symbolic]
// CHECK:STDOUT:   %impl.elem0.60b493.2: %.69b631.2 = impl_witness_access %L1.lookup_impl_witness.297fbf.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f023e9.2: <specific function> = specific_impl_function %impl.elem0.60b493.2, @L1.WithSelf.R1(%V) [symbolic]
// CHECK:STDOUT:   %.917578.2: type = fn_type_with_self_type %L1.WithSelf.S1.type.79e6b4.2, %V [symbolic]
// CHECK:STDOUT:   %impl.elem1.e3fe4c.2: %.917578.2 = impl_witness_access %L1.lookup_impl_witness.297fbf.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2cc093.2: <specific function> = specific_impl_function %impl.elem1.e3fe4c.2, @L1.WithSelf.S1(%V) [symbolic]
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
// CHECK:STDOUT:     .L1 = %L1.decl
// CHECK:STDOUT:     .Simple4 = %Simple4.decl
// CHECK:STDOUT:     .Compound4 = %Compound4.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %L1.decl: type = interface_decl @L1 [concrete = constants.%L1.type] {} {}
// CHECK:STDOUT:   %Simple4.decl: %Simple4.type = fn_decl @Simple4 [concrete = constants.%Simple4] {
// CHECK:STDOUT:     %T.patt: %pattern_type.a5c = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @Simple4.%pattern_type.loc37 (%pattern_type.352a62.1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @Simple4.%pattern_type.loc37 (%pattern_type.352a62.1) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %p.param_patt: @Simple4.%pattern_type.loc41 (%pattern_type.740d56.1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: @Simple4.%pattern_type.loc41 (%pattern_type.740d56.1) = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc34: type = splice_block %L1.ref [concrete = constants.%L1.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L1.ref: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc32.2: %L1.type = symbolic_binding T, 0 [symbolic = %T.loc32.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Simple4.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc38.1: type = splice_block %.loc38.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc38: %L1.type = name_ref T, %T.loc32.2 [symbolic = %T.loc32.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc38: type = facet_access_type %T.ref.loc38 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc38.2: type = converted %T.ref.loc38, %T.as_type.loc38 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Simple4.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %p.param: @Simple4.%ptr.loc43.1 (%ptr.6c1f8e.1) = value_param call_param1
// CHECK:STDOUT:     %.loc43.1: type = splice_block %ptr.loc43.2 [symbolic = %ptr.loc43.1 (constants.%ptr.6c1f8e.1)] {
// CHECK:STDOUT:       %T.ref.loc42: %L1.type = name_ref T, %T.loc32.2 [symbolic = %T.loc32.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc43: type = facet_access_type %T.ref.loc42 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc43.2: type = converted %T.ref.loc42, %T.as_type.loc43 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %ptr.loc43.2: type = ptr_type %.loc43.2 [symbolic = %ptr.loc43.1 (constants.%ptr.6c1f8e.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @Simple4.%ptr.loc43.1 (%ptr.6c1f8e.1) = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound4.decl: %Compound4.type = fn_decl @Compound4 [concrete = constants.%Compound4] {
// CHECK:STDOUT:     %V.patt: %pattern_type.a5c = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:     %y.param_patt: @Compound4.%pattern_type.loc67 (%pattern_type.352a62.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %y.patt: @Compound4.%pattern_type.loc67 (%pattern_type.352a62.2) = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:     %p.param_patt: @Compound4.%pattern_type.loc71 (%pattern_type.740d56.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: @Compound4.%pattern_type.loc71 (%pattern_type.740d56.2) = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc64: type = splice_block %L1.ref.loc64 [concrete = constants.%L1.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L1.ref.loc64: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc62.2: %L1.type = symbolic_binding V, 0 [symbolic = %V.loc62.1 (constants.%V)]
// CHECK:STDOUT:     %y.param: @Compound4.%V.binding.as_type (%V.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc68.1: type = splice_block %.loc68.2 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)] {
// CHECK:STDOUT:       %V.ref.loc68: %L1.type = name_ref V, %V.loc62.2 [symbolic = %V.loc62.1 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc68: type = facet_access_type %V.ref.loc68 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:       %.loc68.2: type = converted %V.ref.loc68, %V.as_type.loc68 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: @Compound4.%V.binding.as_type (%V.binding.as_type) = value_binding y, %y.param
// CHECK:STDOUT:     %p.param: @Compound4.%ptr.loc73.1 (%ptr.6c1f8e.2) = value_param call_param1
// CHECK:STDOUT:     %.loc73.1: type = splice_block %ptr.loc73.2 [symbolic = %ptr.loc73.1 (constants.%ptr.6c1f8e.2)] {
// CHECK:STDOUT:       %V.ref.loc72: %L1.type = name_ref V, %V.loc62.2 [symbolic = %V.loc62.1 (constants.%V)]
// CHECK:STDOUT:       %V.as_type.loc73: type = facet_access_type %V.ref.loc72 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:       %.loc73.2: type = converted %V.ref.loc72, %V.as_type.loc73 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:       %ptr.loc73.2: type = ptr_type %.loc73.2 [symbolic = %ptr.loc73.1 (constants.%ptr.6c1f8e.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @Compound4.%ptr.loc73.1 (%ptr.6c1f8e.2) = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L1 {
// CHECK:STDOUT:   %Self: %L1.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %L1.WithSelf.decl = interface_with_self_decl @L1 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %L1.WithSelf.R1.decl: @L1.WithSelf.%L1.WithSelf.R1.type (%L1.WithSelf.R1.type.504) = fn_decl @L1.WithSelf.R1 [symbolic = @L1.WithSelf.%L1.WithSelf.R1 (constants.%L1.WithSelf.R1.aa4)] {
// CHECK:STDOUT:     %self.param_patt: @L1.WithSelf.R1.%pattern_type (%pattern_type.9c1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @L1.WithSelf.R1.%pattern_type (%pattern_type.9c1) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @L1.WithSelf.R1.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc12.1: type = splice_block %.loc12.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %L1.type = name_ref Self, @L1.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc12.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @L1.WithSelf.R1.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %L1.assoc_type = assoc_entity element0, %L1.WithSelf.R1.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %L1.WithSelf.S1.decl: @L1.WithSelf.%L1.WithSelf.S1.type (%L1.WithSelf.S1.type.e03) = fn_decl @L1.WithSelf.S1 [symbolic = @L1.WithSelf.%L1.WithSelf.S1 (constants.%L1.WithSelf.S1.480)] {
// CHECK:STDOUT:     %self.param_patt: @L1.WithSelf.S1.%pattern_type (%pattern_type.9c1) = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @L1.WithSelf.S1.%pattern_type (%pattern_type.9c1) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref @L1.WithSelf.S1.%Self.binding.as_type (%Self.binding.as_type) = ref_param call_param0
// CHECK:STDOUT:     %.loc23.1: type = splice_block %.loc23.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %L1.type = name_ref Self, @L1.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc23.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: ref @L1.WithSelf.S1.%Self.binding.as_type (%Self.binding.as_type) = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %L1.assoc_type = assoc_entity element1, %L1.WithSelf.S1.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .R1 = @L1.WithSelf.%assoc0
// CHECK:STDOUT:   .S1 = @L1.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@L1.WithSelf.%L1.WithSelf.R1.decl, @L1.WithSelf.%L1.WithSelf.S1.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @L1.WithSelf.R1(@L1.%Self: %L1.type) {
// CHECK:STDOUT:   %Self: %L1.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9c1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @L1.WithSelf.R1.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @L1.WithSelf.S1(@L1.%Self: %L1.type) {
// CHECK:STDOUT:   %Self: %L1.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9c1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @L1.WithSelf.S1.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple4(%T.loc32.2: %L1.type) {
// CHECK:STDOUT:   %T.loc32.1: %L1.type = symbolic_binding T, 0 [symbolic = %T.loc32.1 (constants.%T)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc32.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc37: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc37 (constants.%pattern_type.352a62.1)]
// CHECK:STDOUT:   %ptr.loc43.1: type = ptr_type %T.binding.as_type [symbolic = %ptr.loc43.1 (constants.%ptr.6c1f8e.1)]
// CHECK:STDOUT:   %pattern_type.loc41: type = pattern_type %ptr.loc43.1 [symbolic = %pattern_type.loc41 (constants.%pattern_type.740d56.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc37: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc37 (constants.%require_complete.592d00.1)]
// CHECK:STDOUT:   %require_complete.loc41: <witness> = require_complete_type %ptr.loc43.1 [symbolic = %require_complete.loc41 (constants.%require_complete.94a5f1.1)]
// CHECK:STDOUT:   %L1.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc32.1, @L1 [symbolic = %L1.lookup_impl_witness (constants.%L1.lookup_impl_witness.297fbf.1)]
// CHECK:STDOUT:   %L1.WithSelf.R1.type: type = fn_type @L1.WithSelf.R1, @L1.WithSelf(%T.loc32.1) [symbolic = %L1.WithSelf.R1.type (constants.%L1.WithSelf.R1.type.0d7469.1)]
// CHECK:STDOUT:   %.loc47: type = fn_type_with_self_type %L1.WithSelf.R1.type, %T.loc32.1 [symbolic = %.loc47 (constants.%.69b631.1)]
// CHECK:STDOUT:   %impl.elem0.loc47.2: @Simple4.%.loc47 (%.69b631.1) = impl_witness_access %L1.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc47.2 (constants.%impl.elem0.60b493.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc47.2: <specific function> = specific_impl_function %impl.elem0.loc47.2, @L1.WithSelf.R1(%T.loc32.1) [symbolic = %specific_impl_fn.loc47.2 (constants.%specific_impl_fn.f023e9.1)]
// CHECK:STDOUT:   %L1.WithSelf.S1.type: type = fn_type @L1.WithSelf.S1, @L1.WithSelf(%T.loc32.1) [symbolic = %L1.WithSelf.S1.type (constants.%L1.WithSelf.S1.type.79e6b4.1)]
// CHECK:STDOUT:   %.loc53.2: type = fn_type_with_self_type %L1.WithSelf.S1.type, %T.loc32.1 [symbolic = %.loc53.2 (constants.%.917578.1)]
// CHECK:STDOUT:   %impl.elem1.loc53.2: @Simple4.%.loc53.2 (%.917578.1) = impl_witness_access %L1.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc53.2 (constants.%impl.elem1.e3fe4c.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc53.2: <specific function> = specific_impl_function %impl.elem1.loc53.2, @L1.WithSelf.S1(%T.loc32.1) [symbolic = %specific_impl_fn.loc53.2 (constants.%specific_impl_fn.2cc093.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Simple4.%T.binding.as_type (%T.binding.as_type), %p.param: @Simple4.%ptr.loc43.1 (%ptr.6c1f8e.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Simple4.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %R1.ref: %L1.assoc_type = name_ref R1, @L1.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc47.1: @Simple4.%.loc47 (%.69b631.1) = impl_witness_access constants.%L1.lookup_impl_witness.297fbf.1, element0 [symbolic = %impl.elem0.loc47.2 (constants.%impl.elem0.60b493.1)]
// CHECK:STDOUT:     %bound_method.loc47: <bound method> = bound_method %x.ref, %impl.elem0.loc47.1
// CHECK:STDOUT:     %specific_impl_fn.loc47.1: <specific function> = specific_impl_function %impl.elem0.loc47.1, @L1.WithSelf.R1(constants.%T) [symbolic = %specific_impl_fn.loc47.2 (constants.%specific_impl_fn.f023e9.1)]
// CHECK:STDOUT:     %bound_method.loc50: <bound method> = bound_method %x.ref, %specific_impl_fn.loc47.1
// CHECK:STDOUT:     %L1.WithSelf.R1.call: init %empty_tuple.type = call %bound_method.loc50(%x.ref)
// CHECK:STDOUT:     %p.ref: @Simple4.%ptr.loc43.1 (%ptr.6c1f8e.1) = name_ref p, %p
// CHECK:STDOUT:     %.loc53.1: ref @Simple4.%T.binding.as_type (%T.binding.as_type) = deref %p.ref
// CHECK:STDOUT:     %S1.ref: %L1.assoc_type = name_ref S1, @L1.WithSelf.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc53.1: @Simple4.%.loc53.2 (%.917578.1) = impl_witness_access constants.%L1.lookup_impl_witness.297fbf.1, element1 [symbolic = %impl.elem1.loc53.2 (constants.%impl.elem1.e3fe4c.1)]
// CHECK:STDOUT:     %bound_method.loc53: <bound method> = bound_method %.loc53.1, %impl.elem1.loc53.1
// CHECK:STDOUT:     %specific_impl_fn.loc53.1: <specific function> = specific_impl_function %impl.elem1.loc53.1, @L1.WithSelf.S1(constants.%T) [symbolic = %specific_impl_fn.loc53.2 (constants.%specific_impl_fn.2cc093.1)]
// CHECK:STDOUT:     %bound_method.loc56: <bound method> = bound_method %.loc53.1, %specific_impl_fn.loc53.1
// CHECK:STDOUT:     %L1.WithSelf.S1.call: init %empty_tuple.type = call %bound_method.loc56(%.loc53.1)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound4(%V.loc62.2: %L1.type) {
// CHECK:STDOUT:   %V.loc62.1: %L1.type = symbolic_binding V, 0 [symbolic = %V.loc62.1 (constants.%V)]
// CHECK:STDOUT:   %V.binding.as_type: type = symbolic_binding_type V, 0, %V.loc62.1 [symbolic = %V.binding.as_type (constants.%V.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc67: type = pattern_type %V.binding.as_type [symbolic = %pattern_type.loc67 (constants.%pattern_type.352a62.2)]
// CHECK:STDOUT:   %ptr.loc73.1: type = ptr_type %V.binding.as_type [symbolic = %ptr.loc73.1 (constants.%ptr.6c1f8e.2)]
// CHECK:STDOUT:   %pattern_type.loc71: type = pattern_type %ptr.loc73.1 [symbolic = %pattern_type.loc71 (constants.%pattern_type.740d56.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc67: <witness> = require_complete_type %V.binding.as_type [symbolic = %require_complete.loc67 (constants.%require_complete.592d00.2)]
// CHECK:STDOUT:   %require_complete.loc71: <witness> = require_complete_type %ptr.loc73.1 [symbolic = %require_complete.loc71 (constants.%require_complete.94a5f1.2)]
// CHECK:STDOUT:   %L1.lookup_impl_witness: <witness> = lookup_impl_witness %V.loc62.1, @L1 [symbolic = %L1.lookup_impl_witness (constants.%L1.lookup_impl_witness.297fbf.2)]
// CHECK:STDOUT:   %L1.WithSelf.R1.type: type = fn_type @L1.WithSelf.R1, @L1.WithSelf(%V.loc62.1) [symbolic = %L1.WithSelf.R1.type (constants.%L1.WithSelf.R1.type.0d7469.2)]
// CHECK:STDOUT:   %.loc77: type = fn_type_with_self_type %L1.WithSelf.R1.type, %V.loc62.1 [symbolic = %.loc77 (constants.%.69b631.2)]
// CHECK:STDOUT:   %impl.elem0.loc77.2: @Compound4.%.loc77 (%.69b631.2) = impl_witness_access %L1.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc77.2 (constants.%impl.elem0.60b493.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc77.2: <specific function> = specific_impl_function %impl.elem0.loc77.2, @L1.WithSelf.R1(%V.loc62.1) [symbolic = %specific_impl_fn.loc77.2 (constants.%specific_impl_fn.f023e9.2)]
// CHECK:STDOUT:   %L1.WithSelf.S1.type: type = fn_type @L1.WithSelf.S1, @L1.WithSelf(%V.loc62.1) [symbolic = %L1.WithSelf.S1.type (constants.%L1.WithSelf.S1.type.79e6b4.2)]
// CHECK:STDOUT:   %.loc87.2: type = fn_type_with_self_type %L1.WithSelf.S1.type, %V.loc62.1 [symbolic = %.loc87.2 (constants.%.917578.2)]
// CHECK:STDOUT:   %impl.elem1.loc87.2: @Compound4.%.loc87.2 (%.917578.2) = impl_witness_access %L1.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc87.2 (constants.%impl.elem1.e3fe4c.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc87.2: <specific function> = specific_impl_function %impl.elem1.loc87.2, @L1.WithSelf.S1(%V.loc62.1) [symbolic = %specific_impl_fn.loc87.2 (constants.%specific_impl_fn.2cc093.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @Compound4.%V.binding.as_type (%V.binding.as_type), %p.param: @Compound4.%ptr.loc73.1 (%ptr.6c1f8e.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %y.ref: @Compound4.%V.binding.as_type (%V.binding.as_type) = name_ref y, %y
// CHECK:STDOUT:     %L1.ref.loc79: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     %R1.ref: %L1.assoc_type = name_ref R1, @L1.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc77.1: @Compound4.%.loc77 (%.69b631.2) = impl_witness_access constants.%L1.lookup_impl_witness.297fbf.2, element0 [symbolic = %impl.elem0.loc77.2 (constants.%impl.elem0.60b493.2)]
// CHECK:STDOUT:     %bound_method.loc77: <bound method> = bound_method %y.ref, %impl.elem0.loc77.1
// CHECK:STDOUT:     %specific_impl_fn.loc77.1: <specific function> = specific_impl_function %impl.elem0.loc77.1, @L1.WithSelf.R1(constants.%V) [symbolic = %specific_impl_fn.loc77.2 (constants.%specific_impl_fn.f023e9.2)]
// CHECK:STDOUT:     %bound_method.loc84: <bound method> = bound_method %y.ref, %specific_impl_fn.loc77.1
// CHECK:STDOUT:     %L1.WithSelf.R1.call: init %empty_tuple.type = call %bound_method.loc84(%y.ref)
// CHECK:STDOUT:     %p.ref: @Compound4.%ptr.loc73.1 (%ptr.6c1f8e.2) = name_ref p, %p
// CHECK:STDOUT:     %L1.ref.loc89: type = name_ref L1, file.%L1.decl [concrete = constants.%L1.type]
// CHECK:STDOUT:     %S1.ref: %L1.assoc_type = name_ref S1, @L1.WithSelf.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %.loc87.1: ref @Compound4.%V.binding.as_type (%V.binding.as_type) = deref %p.ref
// CHECK:STDOUT:     %impl.elem1.loc87.1: @Compound4.%.loc87.2 (%.917578.2) = impl_witness_access constants.%L1.lookup_impl_witness.297fbf.2, element1 [symbolic = %impl.elem1.loc87.2 (constants.%impl.elem1.e3fe4c.2)]
// CHECK:STDOUT:     %bound_method.loc87: <bound method> = bound_method %.loc87.1, %impl.elem1.loc87.1
// CHECK:STDOUT:     %specific_impl_fn.loc87.1: <specific function> = specific_impl_function %impl.elem1.loc87.1, @L1.WithSelf.S1(constants.%V) [symbolic = %specific_impl_fn.loc87.2 (constants.%specific_impl_fn.2cc093.2)]
// CHECK:STDOUT:     %bound_method.loc94: <bound method> = bound_method %.loc87.1, %specific_impl_fn.loc87.1
// CHECK:STDOUT:     %L1.WithSelf.S1.call: init %empty_tuple.type = call %bound_method.loc94(%.loc87.1)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %L1.WithSelf.R1.type => constants.%L1.WithSelf.R1.type.504
// CHECK:STDOUT:   %L1.WithSelf.R1 => constants.%L1.WithSelf.R1.aa4
// CHECK:STDOUT:   %L1.WithSelf.S1.type => constants.%L1.WithSelf.S1.type.e03
// CHECK:STDOUT:   %L1.WithSelf.S1 => constants.%L1.WithSelf.S1.480
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.WithSelf.R1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9c1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.WithSelf.S1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9c1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple4(constants.%T) {
// CHECK:STDOUT:   %T.loc32.1 => constants.%T
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc37 => constants.%pattern_type.352a62.1
// CHECK:STDOUT:   %ptr.loc43.1 => constants.%ptr.6c1f8e.1
// CHECK:STDOUT:   %pattern_type.loc41 => constants.%pattern_type.740d56.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %L1.WithSelf.R1.type => constants.%L1.WithSelf.R1.type.0d7469.1
// CHECK:STDOUT:   %L1.WithSelf.R1 => constants.%L1.WithSelf.R1.f22128.1
// CHECK:STDOUT:   %L1.WithSelf.S1.type => constants.%L1.WithSelf.S1.type.79e6b4.1
// CHECK:STDOUT:   %L1.WithSelf.S1 => constants.%L1.WithSelf.S1.cd5111.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.WithSelf.R1(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.352a62.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.WithSelf.S1(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.352a62.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound4(constants.%V) {
// CHECK:STDOUT:   %V.loc62.1 => constants.%V
// CHECK:STDOUT:   %V.binding.as_type => constants.%V.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc67 => constants.%pattern_type.352a62.2
// CHECK:STDOUT:   %ptr.loc73.1 => constants.%ptr.6c1f8e.2
// CHECK:STDOUT:   %pattern_type.loc71 => constants.%pattern_type.740d56.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.WithSelf(constants.%V) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%V
// CHECK:STDOUT:   %L1.WithSelf.R1.type => constants.%L1.WithSelf.R1.type.0d7469.2
// CHECK:STDOUT:   %L1.WithSelf.R1 => constants.%L1.WithSelf.R1.f22128.2
// CHECK:STDOUT:   %L1.WithSelf.S1.type => constants.%L1.WithSelf.S1.type.79e6b4.2
// CHECK:STDOUT:   %L1.WithSelf.S1 => constants.%L1.WithSelf.S1.cd5111.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.WithSelf.R1(constants.%V) {
// CHECK:STDOUT:   %Self => constants.%V
// CHECK:STDOUT:   %Self.binding.as_type => constants.%V.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.352a62.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L1.WithSelf.S1(constants.%V) {
// CHECK:STDOUT:   %Self => constants.%V
// CHECK:STDOUT:   %Self.binding.as_type => constants.%V.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.352a62.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_interface_instance_caller_not.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L2.type: type = facet_type <@L2> [concrete]
// CHECK:STDOUT:   %Self: %L2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.73b: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %L2.WithSelf.R2.type.e38: type = fn_type @L2.WithSelf.R2, @L2.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %L2.WithSelf.R2.f23: %L2.WithSelf.R2.type.e38 = struct_value () [symbolic]
// CHECK:STDOUT:   %L2.assoc_type: type = assoc_entity_type @L2 [concrete]
// CHECK:STDOUT:   %assoc0: %L2.assoc_type = assoc_entity element0, @L2.WithSelf.%L2.WithSelf.R2.decl [concrete]
// CHECK:STDOUT:   %L2.WithSelf.S2.type.7f8: type = fn_type @L2.WithSelf.S2, @L2.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %L2.WithSelf.S2.827: %L2.WithSelf.S2.type.7f8 = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc1: %L2.assoc_type = assoc_entity element1, @L2.WithSelf.%L2.WithSelf.S2.decl [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.5a0: type = pattern_type %L2.type [concrete]
// CHECK:STDOUT:   %T: %L2.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Simple5.type: type = fn_type @Simple5 [concrete]
// CHECK:STDOUT:   %Simple5: %Simple5.type = struct_value () [concrete]
// CHECK:STDOUT:   %L2.WithSelf.R2.type.ba3: type = fn_type @L2.WithSelf.R2, @L2.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %L2.WithSelf.R2.c61: %L2.WithSelf.R2.type.ba3 = struct_value () [symbolic]
// CHECK:STDOUT:   %L2.WithSelf.S2.type.68b: type = fn_type @L2.WithSelf.S2, @L2.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %L2.WithSelf.S2.470: %L2.WithSelf.S2.type.68b = struct_value () [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %L2.lookup_impl_witness: <witness> = lookup_impl_witness %T, @L2 [symbolic]
// CHECK:STDOUT:   %.f8e: type = fn_type_with_self_type %L2.WithSelf.R2.type.ba3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.f8e = impl_witness_access %L2.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %pattern_type.0fc: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b6c: <specific function> = specific_impl_function %impl.elem0, @L2.WithSelf.R2(%T) [symbolic]
// CHECK:STDOUT:   %.b1c: type = fn_type_with_self_type %L2.WithSelf.S2.type.68b, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.b1c = impl_witness_access %L2.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b67: <specific function> = specific_impl_function %impl.elem1, @L2.WithSelf.S2(%T) [symbolic]
// CHECK:STDOUT:   %V: %L2.type = symbolic_binding V, 0 [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.5a0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc34: type = splice_block %L2.ref [concrete = constants.%L2.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L2.ref: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc32.2: %L2.type = symbolic_binding T, 0 [symbolic = %T.loc32.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compound5.decl: %Compound5.type = fn_decl @Compound5 [concrete = constants.%Compound5] {
// CHECK:STDOUT:     %V.patt: %pattern_type.5a0 = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc55: type = splice_block %L2.ref.loc55 [concrete = constants.%L2.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %L2.ref.loc55: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc53.2: %L2.type = symbolic_binding V, 0 [symbolic = %V.loc53.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L2 {
// CHECK:STDOUT:   %Self: %L2.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %L2.WithSelf.decl = interface_with_self_decl @L2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %L2.WithSelf.R2.decl: @L2.WithSelf.%L2.WithSelf.R2.type (%L2.WithSelf.R2.type.e38) = fn_decl @L2.WithSelf.R2 [symbolic = @L2.WithSelf.%L2.WithSelf.R2 (constants.%L2.WithSelf.R2.f23)] {
// CHECK:STDOUT:     %self.param_patt: @L2.WithSelf.R2.%pattern_type (%pattern_type.73b) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @L2.WithSelf.R2.%pattern_type (%pattern_type.73b) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @L2.WithSelf.R2.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc12.1: type = splice_block %.loc12.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %L2.type = name_ref Self, @L2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc12.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @L2.WithSelf.R2.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %L2.assoc_type = assoc_entity element0, %L2.WithSelf.R2.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %L2.WithSelf.S2.decl: @L2.WithSelf.%L2.WithSelf.S2.type (%L2.WithSelf.S2.type.7f8) = fn_decl @L2.WithSelf.S2 [symbolic = @L2.WithSelf.%L2.WithSelf.S2 (constants.%L2.WithSelf.S2.827)] {
// CHECK:STDOUT:     %self.param_patt: @L2.WithSelf.S2.%pattern_type (%pattern_type.73b) = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @L2.WithSelf.S2.%pattern_type (%pattern_type.73b) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref @L2.WithSelf.S2.%Self.binding.as_type (%Self.binding.as_type) = ref_param call_param0
// CHECK:STDOUT:     %.loc23.1: type = splice_block %.loc23.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref: %L2.type = name_ref Self, @L2.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc23.2: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: ref @L2.WithSelf.S2.%Self.binding.as_type (%Self.binding.as_type) = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %L2.assoc_type = assoc_entity element1, %L2.WithSelf.S2.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .R2 = @L2.WithSelf.%assoc0
// CHECK:STDOUT:   .S2 = @L2.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@L2.WithSelf.%L2.WithSelf.R2.decl, @L2.WithSelf.%L2.WithSelf.S2.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @L2.WithSelf.R2(@L2.%Self: %L2.type) {
// CHECK:STDOUT:   %Self: %L2.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.73b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @L2.WithSelf.R2.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @L2.WithSelf.S2(@L2.%Self: %L2.type) {
// CHECK:STDOUT:   %Self: %L2.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.73b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: ref @L2.WithSelf.S2.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple5(%T.loc32.2: %L2.type) {
// CHECK:STDOUT:   %T.loc32.1: %L2.type = symbolic_binding T, 0 [symbolic = %T.loc32.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc32.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %L2.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc32.1, @L2 [symbolic = %L2.lookup_impl_witness (constants.%L2.lookup_impl_witness)]
// CHECK:STDOUT:   %L2.WithSelf.R2.type: type = fn_type @L2.WithSelf.R2, @L2.WithSelf(%T.loc32.1) [symbolic = %L2.WithSelf.R2.type (constants.%L2.WithSelf.R2.type.ba3)]
// CHECK:STDOUT:   %.loc38.2: type = fn_type_with_self_type %L2.WithSelf.R2.type, %T.loc32.1 [symbolic = %.loc38.2 (constants.%.f8e)]
// CHECK:STDOUT:   %impl.elem0.loc38.2: @Simple5.%.loc38.2 (%.f8e) = impl_witness_access %L2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc38.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc38.2: <specific function> = specific_impl_function %impl.elem0.loc38.2, @L2.WithSelf.R2(%T.loc32.1) [symbolic = %specific_impl_fn.loc38.2 (constants.%specific_impl_fn.b6c)]
// CHECK:STDOUT:   %L2.WithSelf.S2.type: type = fn_type @L2.WithSelf.S2, @L2.WithSelf(%T.loc32.1) [symbolic = %L2.WithSelf.S2.type (constants.%L2.WithSelf.S2.type.68b)]
// CHECK:STDOUT:   %.loc44.2: type = fn_type_with_self_type %L2.WithSelf.S2.type, %T.loc32.1 [symbolic = %.loc44.2 (constants.%.b1c)]
// CHECK:STDOUT:   %impl.elem1.loc44.2: @Simple5.%.loc44.2 (%.b1c) = impl_witness_access %L2.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc44.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc44.2: <specific function> = specific_impl_function %impl.elem1.loc44.2, @L2.WithSelf.S2(%T.loc32.1) [symbolic = %specific_impl_fn.loc44.2 (constants.%specific_impl_fn.b67)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc37: %L2.type = name_ref T, %T.loc32.2 [symbolic = %T.loc32.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc38: type = facet_access_type %T.ref.loc37 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc38.1: type = converted %T.ref.loc37, %T.as_type.loc38 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %R2.ref: %L2.assoc_type = name_ref R2, @L2.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc38.1: @Simple5.%.loc38.2 (%.f8e) = impl_witness_access constants.%L2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc38.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc38.1: <specific function> = specific_impl_function %impl.elem0.loc38.1, @L2.WithSelf.R2(constants.%T) [symbolic = %specific_impl_fn.loc38.2 (constants.%specific_impl_fn.b6c)]
// CHECK:STDOUT:     %L2.WithSelf.R2.call: init %empty_tuple.type = call %specific_impl_fn.loc38.1(<error>)
// CHECK:STDOUT:     %T.ref.loc43: %L2.type = name_ref T, %T.loc32.2 [symbolic = %T.loc32.1 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc44: type = facet_access_type %T.ref.loc43 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc44.1: type = converted %T.ref.loc43, %T.as_type.loc44 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %S2.ref: %L2.assoc_type = name_ref S2, @L2.WithSelf.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc44.1: @Simple5.%.loc44.2 (%.b1c) = impl_witness_access constants.%L2.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc44.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %specific_impl_fn.loc44.1: <specific function> = specific_impl_function %impl.elem1.loc44.1, @L2.WithSelf.S2(constants.%T) [symbolic = %specific_impl_fn.loc44.2 (constants.%specific_impl_fn.b67)]
// CHECK:STDOUT:     %L2.WithSelf.S2.call: init %empty_tuple.type = call %specific_impl_fn.loc44.1(<error>)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Compound5(%V.loc53.2: %L2.type) {
// CHECK:STDOUT:   %V.loc53.1: %L2.type = symbolic_binding V, 0 [symbolic = %V.loc53.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %V.ref.loc58: %L2.type = name_ref V, %V.loc53.2 [symbolic = %V.loc53.1 (constants.%V)]
// CHECK:STDOUT:     %L2.ref.loc61: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     %R2.ref: %L2.assoc_type = name_ref R2, @L2.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %V.ref.loc68: %L2.type = name_ref V, %V.loc53.2 [symbolic = %V.loc53.1 (constants.%V)]
// CHECK:STDOUT:     %L2.ref.loc71: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     %S2.ref: %L2.assoc_type = name_ref S2, @L2.WithSelf.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L2.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %L2.WithSelf.R2.type => constants.%L2.WithSelf.R2.type.e38
// CHECK:STDOUT:   %L2.WithSelf.R2 => constants.%L2.WithSelf.R2.f23
// CHECK:STDOUT:   %L2.WithSelf.S2.type => constants.%L2.WithSelf.S2.type.7f8
// CHECK:STDOUT:   %L2.WithSelf.S2 => constants.%L2.WithSelf.S2.827
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L2.WithSelf.R2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.73b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L2.WithSelf.S2(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.73b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple5(constants.%T) {
// CHECK:STDOUT:   %T.loc32.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L2.WithSelf(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %L2.WithSelf.R2.type => constants.%L2.WithSelf.R2.type.ba3
// CHECK:STDOUT:   %L2.WithSelf.R2 => constants.%L2.WithSelf.R2.c61
// CHECK:STDOUT:   %L2.WithSelf.S2.type => constants.%L2.WithSelf.S2.type.68b
// CHECK:STDOUT:   %L2.WithSelf.S2 => constants.%L2.WithSelf.S2.470
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L2.WithSelf.R2(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @L2.WithSelf.S2(constants.%T) {
// CHECK:STDOUT:   %Self => constants.%T
// CHECK:STDOUT:   %Self.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Compound5(constants.%V) {
// CHECK:STDOUT:   %V.loc53.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_combine_non_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.G.type.e1d: type = fn_type @A.WithSelf.G, @A.WithSelf(%Self.c51) [symbolic]
// CHECK:STDOUT:   %A.WithSelf.G.552: %A.WithSelf.G.type.e1d = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.eb1: %A.assoc_type = assoc_entity element0, @A.WithSelf.%A.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @C.as.A.impl.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.A.impl.G.type: type = fn_type @C.as.A.impl.G [concrete]
// CHECK:STDOUT:   %C.as.A.impl.G: %C.as.A.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %C, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %A.WithSelf.G.type.f93: type = fn_type @A.WithSelf.G, @A.WithSelf(%A.facet) [concrete]
// CHECK:STDOUT:   %A.WithSelf.G.9b2: %A.WithSelf.G.type.f93 = struct_value () [concrete]
// CHECK:STDOUT:   %Fails.type: type = fn_type @Fails [concrete]
// CHECK:STDOUT:   %Fails: %Fails.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4bd: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4bd, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.8d3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc130, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.8d3), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:   %Fails.decl: %Fails.type = fn_decl @Fails [concrete = constants.%Fails] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A.WithSelf.G.decl: @A.WithSelf.%A.WithSelf.G.type (%A.WithSelf.G.type.e1d) = fn_decl @A.WithSelf.G [symbolic = @A.WithSelf.%A.WithSelf.G (constants.%A.WithSelf.G.552)] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.WithSelf.G.decl [concrete = constants.%assoc0.eb1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = @A.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@A.WithSelf.%A.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.A.impl: %C.ref as %A.ref {
// CHECK:STDOUT:   %C.as.A.impl.G.decl: %C.as.A.impl.G.type = fn_decl @C.as.A.impl.G [concrete = constants.%C.as.A.impl.G] {} {}
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (%C.as.A.impl.G.decl), @C.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.A.impl.G.decl
// CHECK:STDOUT:   witness = %A.impl_witness
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
// CHECK:STDOUT: generic fn @A.WithSelf.G(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.A.impl.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Fails() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc36.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref.loc38: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc36.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc36.3: init %C to %.loc36.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc37: init %C = converted %.loc36.1, %.loc36.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %A.ref.loc43: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc45: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc44: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc44: <bound method> = bound_method %A.ref.loc43, %impl.elem0.loc44 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc44: init type = call %bound_method.loc44(%A.ref.loc43, %A.ref.loc45) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc47: %A.assoc_type = name_ref G, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.eb1]
// CHECK:STDOUT:   %.loc40: %A.type = converted %.loc37, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc56.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref.loc58: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc56.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc56.3: init %C to %.loc56.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc57: init %C = converted %.loc56.1, %.loc56.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref.loc62: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc65: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc67: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc66: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc66: <bound method> = bound_method %A.ref.loc65, %impl.elem0.loc66 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc66: init type = call %bound_method.loc66(%A.ref.loc65, %A.ref.loc67) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc68.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc66 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc68.2: type = converted %type.as.BitAndWith.impl.Op.call.loc66, %.loc68.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc63: %A.type = facet_value %C.ref.loc62, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc63: %A.type = converted %C.ref.loc62, %A.facet.loc63 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %as_type.loc69: type = facet_access_type %.loc63 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc69: type = converted %.loc63, %as_type.loc69 [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc74: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc76: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc75: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc75: <bound method> = bound_method %A.ref.loc74, %impl.elem0.loc75 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc75: init type = call %bound_method.loc75(%A.ref.loc74, %A.ref.loc76) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc78: %A.assoc_type = name_ref G, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.eb1]
// CHECK:STDOUT:   %.loc71: %A.type = converted %.loc57, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc87.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref.loc89: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc87.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc87.3: init %C to %.loc87.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc88: init %C = converted %.loc87.1, %.loc87.3 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref.loc93: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc96: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc98: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc97: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc97: <bound method> = bound_method %A.ref.loc96, %impl.elem0.loc97 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc97: init type = call %bound_method.loc97(%A.ref.loc96, %A.ref.loc98) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc99.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc97 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc99.2: type = converted %type.as.BitAndWith.impl.Op.call.loc97, %.loc99.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc94: %A.type = facet_value %C.ref.loc93, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc94: %A.type = converted %C.ref.loc93, %A.facet.loc94 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %as_type.loc100: type = facet_access_type %.loc94 [concrete = constants.%C]
// CHECK:STDOUT:   %.loc100: type = converted %.loc94, %as_type.loc100 [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc104: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc105: %A.assoc_type = name_ref G, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.eb1]
// CHECK:STDOUT:   %.loc102: %A.type = converted %.loc88, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.c51
// CHECK:STDOUT:   %A.WithSelf.G.type => constants.%A.WithSelf.G.type.e1d
// CHECK:STDOUT:   %A.WithSelf.G => constants.%A.WithSelf.G.552
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.G(constants.%Self.c51) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%A.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%A.facet
// CHECK:STDOUT:   %A.WithSelf.G.type => constants.%A.WithSelf.G.type.f93
// CHECK:STDOUT:   %A.WithSelf.G => constants.%A.WithSelf.G.9b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.G(constants.%A.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- allowed_combine_non_instance.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %Self.c51: %A.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %A.WithSelf.G.type.e1d: type = fn_type @A.WithSelf.G, @A.WithSelf(%Self.c51) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.WithSelf.G.552: %A.WithSelf.G.type.e1d = struct_value () [symbolic]
// CHECK:STDOUT:   %A.assoc_type: type = assoc_entity_type @A [concrete]
// CHECK:STDOUT:   %assoc0.eb1: %A.assoc_type = assoc_entity element0, @A.WithSelf.%A.WithSelf.G.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness @C.as.A.impl.%A.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.A.impl.G.type: type = fn_type @C.as.A.impl.G [concrete]
// CHECK:STDOUT:   %C.as.A.impl.G: %C.as.A.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.facet: %A.type = facet_value %C, (%A.impl_witness) [concrete]
// CHECK:STDOUT:   %A.WithSelf.G.type.f93: type = fn_type @A.WithSelf.G, @A.WithSelf(%A.facet) [concrete]
// CHECK:STDOUT:   %A.WithSelf.G.9b2: %A.WithSelf.G.type.f93 = struct_value () [concrete]
// CHECK:STDOUT:   %Works.type: type = fn_type @Works [concrete]
// CHECK:STDOUT:   %Works: %Works.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4bd: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4bd, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %A.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %.05a: type = fn_type_with_self_type %A.WithSelf.G.type.f93, %A.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.8d3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc130, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.8d3), @type.as.BitAndWith.impl [concrete]
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
// CHECK:STDOUT:   %Works.decl: %Works.type = fn_decl @Works [concrete = constants.%Works] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @A {
// CHECK:STDOUT:   %Self: %A.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c51]
// CHECK:STDOUT:   %A.WithSelf.decl = interface_with_self_decl @A [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %A.WithSelf.G.decl: @A.WithSelf.%A.WithSelf.G.type (%A.WithSelf.G.type.e1d) = fn_decl @A.WithSelf.G [symbolic = @A.WithSelf.%A.WithSelf.G (constants.%A.WithSelf.G.552)] {} {}
// CHECK:STDOUT:   %assoc0: %A.assoc_type = assoc_entity element0, %A.WithSelf.G.decl [concrete = constants.%assoc0.eb1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .G = @A.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@A.WithSelf.%A.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.A.impl: %C.ref as %A.ref {
// CHECK:STDOUT:   %C.as.A.impl.G.decl: %C.as.A.impl.G.type = fn_decl @C.as.A.impl.G [concrete = constants.%C.as.A.impl.G] {} {}
// CHECK:STDOUT:   %A.impl_witness_table = impl_witness_table (%C.as.A.impl.G.decl), @C.as.A.impl [concrete]
// CHECK:STDOUT:   %A.impl_witness: <witness> = impl_witness %A.impl_witness_table [concrete = constants.%A.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %C.as.A.impl.G.decl
// CHECK:STDOUT:   witness = %A.impl_witness
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
// CHECK:STDOUT: generic fn @A.WithSelf.G(@A.%Self: %A.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.A.impl.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Works() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc34: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc38: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc40: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc39: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc39: <bound method> = bound_method %A.ref.loc38, %impl.elem0.loc39 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc39: init type = call %bound_method.loc39(%A.ref.loc38, %A.ref.loc40) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc42: %A.assoc_type = name_ref G, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.eb1]
// CHECK:STDOUT:   %A.facet.loc35: %A.type = facet_value %C.ref.loc34, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc35: %A.type = converted %C.ref.loc34, %A.facet.loc35 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %impl.elem0.loc35: %.05a = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%C.as.A.impl.G]
// CHECK:STDOUT:   %C.as.A.impl.G.call.loc46: init %empty_tuple.type = call %impl.elem0.loc35()
// CHECK:STDOUT:   %C.ref.loc49: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc52: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc54: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc53: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc53: <bound method> = bound_method %A.ref.loc52, %impl.elem0.loc53 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc53: init type = call %bound_method.loc53(%A.ref.loc52, %A.ref.loc54) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc55.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc53 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc55.2: type = converted %type.as.BitAndWith.impl.Op.call.loc53, %.loc55.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc50: %A.type = facet_value %C.ref.loc49, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc50: %A.type = converted %C.ref.loc49, %A.facet.loc50 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %A.ref.loc60: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc62: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc61: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc61: <bound method> = bound_method %A.ref.loc60, %impl.elem0.loc61 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc61: init type = call %bound_method.loc61(%A.ref.loc60, %A.ref.loc62) [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc64: %A.assoc_type = name_ref G, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.eb1]
// CHECK:STDOUT:   %impl.elem0.loc57: %.05a = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%C.as.A.impl.G]
// CHECK:STDOUT:   %C.as.A.impl.G.call.loc68: init %empty_tuple.type = call %impl.elem0.loc57()
// CHECK:STDOUT:   %C.ref.loc71: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A.ref.loc74: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.ref.loc76: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %impl.elem0.loc75: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc75: <bound method> = bound_method %A.ref.loc74, %impl.elem0.loc75 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.call.loc75: init type = call %bound_method.loc75(%A.ref.loc74, %A.ref.loc76) [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc77.1: type = value_of_initializer %type.as.BitAndWith.impl.Op.call.loc75 [concrete = constants.%A.type]
// CHECK:STDOUT:   %.loc77.2: type = converted %type.as.BitAndWith.impl.Op.call.loc75, %.loc77.1 [concrete = constants.%A.type]
// CHECK:STDOUT:   %A.facet.loc72: %A.type = facet_value %C.ref.loc71, (constants.%A.impl_witness) [concrete = constants.%A.facet]
// CHECK:STDOUT:   %.loc72: %A.type = converted %C.ref.loc71, %A.facet.loc72 [concrete = constants.%A.facet]
// CHECK:STDOUT:   %A.ref.loc81: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:   %G.ref.loc82: %A.assoc_type = name_ref G, @A.WithSelf.%assoc0 [concrete = constants.%assoc0.eb1]
// CHECK:STDOUT:   %impl.elem0.loc79: %.05a = impl_witness_access constants.%A.impl_witness, element0 [concrete = constants.%C.as.A.impl.G]
// CHECK:STDOUT:   %C.as.A.impl.G.call.loc86: init %empty_tuple.type = call %impl.elem0.loc79()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%Self.c51) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.c51
// CHECK:STDOUT:   %A.WithSelf.G.type => constants.%A.WithSelf.G.type.e1d
// CHECK:STDOUT:   %A.WithSelf.G => constants.%A.WithSelf.G.552
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.G(constants.%Self.c51) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf(constants.%A.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%A.facet
// CHECK:STDOUT:   %A.WithSelf.G.type => constants.%A.WithSelf.G.type.f93
// CHECK:STDOUT:   %A.WithSelf.G => constants.%A.WithSelf.G.9b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.WithSelf.G(constants.%A.facet) {}
// CHECK:STDOUT:

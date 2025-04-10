// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/generic_redeclaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/generic_redeclaration.carbon

// --- same_self_and_interface.carbon

library "[[@TEST_NAME]]";

interface Interface {}

interface I {}
interface J {}
interface K {}
interface L {}

impl forall [T:! I] T as Interface;
impl forall [T:! J] T as Interface;
impl forall [T:! K] T as Interface;
impl forall [T:! L] T as Interface;

// These are different impls, so they are not redefinitions, even though the
// self type and constraint type are the same.
impl forall [T:! I] T as Interface {}
impl forall [T:! J] T as Interface {}
impl forall [T:! K] T as Interface {}
impl forall [T:! L] T as Interface {}

// --- fail_same_self_and_interface_redefined.carbon

library "[[@TEST_NAME]]";

interface I {}
interface J {}

impl forall [T:! I] T as J {}
// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE+7]]:1: error: redefinition of `impl T as J` [ImplRedefinition]
// CHECK:STDERR: impl forall [T:! I] T as J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE-4]]:1: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR: impl forall [T:! I] T as J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! I] T as J {}

// --- same_type_different_spelling.carbon

library "[[@TEST_NAME]]";

class C;
interface I {}

// We accept this because these two types are spelled differently, even though
// they evaluate to the same type constant. Any use of this impl will be
// ambiguous unless resolved by a `match_first` or similar.
//
// TODO: Produce a warning or maybe an error when this happens in a non-generic
// impl.
impl C as I {}
impl (C, C).0 as I {}

// --- fail_redefinition_generic_regions.carbon

interface I {}

impl forall [T:! type] T as I {
  fn A() {}
}

// CHECK:STDERR: fail_redefinition_generic_regions.carbon:[[@LINE+7]]:1: error: redefinition of `impl T as I` [ImplRedefinition]
// CHECK:STDERR: impl forall [T:! type] T as I {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition_generic_regions.carbon:[[@LINE-7]]:1: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR: impl forall [T:! type] T as I {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as I {
  // Although not referenced, B has the same generic region index of A, and
  // makes C a different index. We used to merge, and this was a crash.
  fn B() {}
  fn C() -> () { return (); }
  fn D() -> () {
    return C();
  }
}

// CHECK:STDOUT: --- same_self_and_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self.719: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self.09f: %K.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %Self.1d2: %L.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T.826: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.3ad: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.b70: type = facet_access_type %T.826 [symbolic]
// CHECK:STDOUT:   %impl_witness.d14: <witness> = impl_witness (), @impl.c3d(%T.826) [symbolic]
// CHECK:STDOUT:   %T.ccd: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.371: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.3df: type = facet_access_type %T.ccd [symbolic]
// CHECK:STDOUT:   %impl_witness.d94: <witness> = impl_witness (), @impl.793(%T.ccd) [symbolic]
// CHECK:STDOUT:   %T.09f: %K.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.036: %K.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.037: type = facet_access_type %T.09f [symbolic]
// CHECK:STDOUT:   %impl_witness.8aa: <witness> = impl_witness (), @impl.c93(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.1d2: %L.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.29d: %L.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.0ed: type = facet_access_type %T.1d2 [symbolic]
// CHECK:STDOUT:   %impl_witness.da5: <witness> = impl_witness (), @impl.9e6(%T.1d2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [concrete = constants.%K.type] {} {}
// CHECK:STDOUT:   %L.decl: type = interface_decl @L [concrete = constants.%L.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.c3d [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.3ad)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc11: %I.type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T.826)]
// CHECK:STDOUT:     %T.as_type.loc11_21.1: type = facet_access_type %T.ref.loc11 [symbolic = %T.as_type.loc11_21.2 (constants.%T.as_type.b70)]
// CHECK:STDOUT:     %.loc11: type = converted %T.ref.loc11, %T.as_type.loc11_21.1 [symbolic = %T.as_type.loc11_21.2 (constants.%T.as_type.b70)]
// CHECK:STDOUT:     %Interface.ref.loc11: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %I.ref.loc11: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc11_14.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T.826)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc11: <witness> = impl_witness (), @impl.c3d(constants.%T.826) [symbolic = @impl.c3d.%impl_witness (constants.%impl_witness.d14)]
// CHECK:STDOUT:   impl_decl @impl.793 [concrete] {
// CHECK:STDOUT:     %T.patt.loc12_14.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.371)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12: %J.type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.ccd)]
// CHECK:STDOUT:     %T.as_type.loc12_21.1: type = facet_access_type %T.ref.loc12 [symbolic = %T.as_type.loc12_21.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     %.loc12: type = converted %T.ref.loc12, %T.as_type.loc12_21.1 [symbolic = %T.as_type.loc12_21.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     %Interface.ref.loc12: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %J.ref.loc12: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc12_14.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.ccd)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc12: <witness> = impl_witness (), @impl.793(constants.%T.ccd) [symbolic = @impl.793.%impl_witness (constants.%impl_witness.d94)]
// CHECK:STDOUT:   impl_decl @impl.c93 [concrete] {
// CHECK:STDOUT:     %T.patt.loc13_14.1: %K.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_14.2 (constants.%T.patt.036)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13: %K.type = name_ref T, %T.loc13_14.1 [symbolic = %T.loc13_14.2 (constants.%T.09f)]
// CHECK:STDOUT:     %T.as_type.loc13_21.1: type = facet_access_type %T.ref.loc13 [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type.037)]
// CHECK:STDOUT:     %.loc13: type = converted %T.ref.loc13, %T.as_type.loc13_21.1 [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type.037)]
// CHECK:STDOUT:     %Interface.ref.loc13: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %K.ref.loc13: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     %T.loc13_14.1: %K.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_14.2 (constants.%T.09f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness (), @impl.c93(constants.%T.09f) [symbolic = @impl.c93.%impl_witness (constants.%impl_witness.8aa)]
// CHECK:STDOUT:   impl_decl @impl.9e6 [concrete] {
// CHECK:STDOUT:     %T.patt.loc14_14.1: %L.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt.29d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc14: %L.type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T.1d2)]
// CHECK:STDOUT:     %T.as_type.loc14_21.1: type = facet_access_type %T.ref.loc14 [symbolic = %T.as_type.loc14_21.2 (constants.%T.as_type.0ed)]
// CHECK:STDOUT:     %.loc14: type = converted %T.ref.loc14, %T.as_type.loc14_21.1 [symbolic = %T.as_type.loc14_21.2 (constants.%T.as_type.0ed)]
// CHECK:STDOUT:     %Interface.ref.loc14: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %L.ref.loc14: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     %T.loc14_14.1: %L.type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T.1d2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness (), @impl.9e6(constants.%T.1d2) [symbolic = @impl.9e6.%impl_witness (constants.%impl_witness.da5)]
// CHECK:STDOUT:   impl_decl @impl.c3d [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.3ad)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc18: %I.type = name_ref T, %T.loc18 [symbolic = %T.loc11_14.2 (constants.%T.826)]
// CHECK:STDOUT:     %T.as_type.loc18: type = facet_access_type %T.ref.loc18 [symbolic = %T.as_type.loc11_21.2 (constants.%T.as_type.b70)]
// CHECK:STDOUT:     %.loc18: type = converted %T.ref.loc18, %T.as_type.loc18 [symbolic = %T.as_type.loc11_21.2 (constants.%T.as_type.b70)]
// CHECK:STDOUT:     %Interface.ref.loc18: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %I.ref.loc18: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc18: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T.826)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.793 [concrete] {
// CHECK:STDOUT:     %T.patt.loc12_14.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.371)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc19: %J.type = name_ref T, %T.loc19 [symbolic = %T.loc12_14.2 (constants.%T.ccd)]
// CHECK:STDOUT:     %T.as_type.loc19: type = facet_access_type %T.ref.loc19 [symbolic = %T.as_type.loc12_21.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     %.loc19: type = converted %T.ref.loc19, %T.as_type.loc19 [symbolic = %T.as_type.loc12_21.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     %Interface.ref.loc19: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %J.ref.loc19: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %T.loc19: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.ccd)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.c93 [concrete] {
// CHECK:STDOUT:     %T.patt.loc13_14.1: %K.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_14.2 (constants.%T.patt.036)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc20: %K.type = name_ref T, %T.loc20 [symbolic = %T.loc13_14.2 (constants.%T.09f)]
// CHECK:STDOUT:     %T.as_type.loc20: type = facet_access_type %T.ref.loc20 [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type.037)]
// CHECK:STDOUT:     %.loc20: type = converted %T.ref.loc20, %T.as_type.loc20 [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type.037)]
// CHECK:STDOUT:     %Interface.ref.loc20: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %K.ref.loc20: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     %T.loc20: %K.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_14.2 (constants.%T.09f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.9e6 [concrete] {
// CHECK:STDOUT:     %T.patt.loc14_14.1: %L.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt.29d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc21: %L.type = name_ref T, %T.loc21 [symbolic = %T.loc14_14.2 (constants.%T.1d2)]
// CHECK:STDOUT:     %T.as_type.loc21: type = facet_access_type %T.ref.loc21 [symbolic = %T.as_type.loc14_21.2 (constants.%T.as_type.0ed)]
// CHECK:STDOUT:     %.loc21: type = converted %T.ref.loc21, %T.as_type.loc21 [symbolic = %T.as_type.loc14_21.2 (constants.%T.as_type.0ed)]
// CHECK:STDOUT:     %Interface.ref.loc21: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %L.ref.loc21: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     %T.loc21: %L.type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T.1d2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.719]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.09f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %L.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1d2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.c3d(%T.loc11_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc11_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T.826)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.3ad)]
// CHECK:STDOUT:   %T.as_type.loc11_21.2: type = facet_access_type %T.loc11_14.2 [symbolic = %T.as_type.loc11_21.2 (constants.%T.as_type.b70)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.c3d(%T.loc11_14.2) [symbolic = %impl_witness (constants.%impl_witness.d14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc11 as %Interface.ref.loc11 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.793(%T.loc12_14.1: %J.type) {
// CHECK:STDOUT:   %T.loc12_14.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.ccd)]
// CHECK:STDOUT:   %T.patt.loc12_14.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.371)]
// CHECK:STDOUT:   %T.as_type.loc12_21.2: type = facet_access_type %T.loc12_14.2 [symbolic = %T.as_type.loc12_21.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.793(%T.loc12_14.2) [symbolic = %impl_witness (constants.%impl_witness.d94)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc12 as %Interface.ref.loc12 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.c93(%T.loc13_14.1: %K.type) {
// CHECK:STDOUT:   %T.loc13_14.2: %K.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_14.2 (constants.%T.09f)]
// CHECK:STDOUT:   %T.patt.loc13_14.2: %K.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_14.2 (constants.%T.patt.036)]
// CHECK:STDOUT:   %T.as_type.loc13_21.2: type = facet_access_type %T.loc13_14.2 [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type.037)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.c93(%T.loc13_14.2) [symbolic = %impl_witness (constants.%impl_witness.8aa)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc13 as %Interface.ref.loc13 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9e6(%T.loc14_14.1: %L.type) {
// CHECK:STDOUT:   %T.loc14_14.2: %L.type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T.1d2)]
// CHECK:STDOUT:   %T.patt.loc14_14.2: %L.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt.29d)]
// CHECK:STDOUT:   %T.as_type.loc14_21.2: type = facet_access_type %T.loc14_14.2 [symbolic = %T.as_type.loc14_21.2 (constants.%T.as_type.0ed)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.9e6(%T.loc14_14.2) [symbolic = %impl_witness (constants.%impl_witness.da5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14 as %Interface.ref.loc14 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.c3d(constants.%T.826) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T.826
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T.patt.3ad
// CHECK:STDOUT:   %T.as_type.loc11_21.2 => constants.%T.as_type.b70
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.d14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.c3d(%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.793(constants.%T.ccd) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.ccd
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%T.patt.371
// CHECK:STDOUT:   %T.as_type.loc12_21.2 => constants.%T.as_type.3df
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.d94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.793(%T.loc12_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.c93(constants.%T.09f) {
// CHECK:STDOUT:   %T.loc13_14.2 => constants.%T.09f
// CHECK:STDOUT:   %T.patt.loc13_14.2 => constants.%T.patt.036
// CHECK:STDOUT:   %T.as_type.loc13_21.2 => constants.%T.as_type.037
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.8aa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.c93(%T.loc13_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9e6(constants.%T.1d2) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T.1d2
// CHECK:STDOUT:   %T.patt.loc14_14.2 => constants.%T.patt.29d
// CHECK:STDOUT:   %T.as_type.loc14_21.2 => constants.%T.as_type.0ed
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.da5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9e6(%T.loc14_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_same_self_and_interface_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %impl_witness.1896b7.1: <witness> = impl_witness (), @impl.bfd509.1(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.1896b7.2: <witness> = impl_witness (), @impl.bfd509.2(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.bfd509.1 [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc7_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc7: type = converted %T.ref, %T.as_type.loc7_21.1 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc7_14.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc7: <witness> = impl_witness (), @impl.bfd509.1(constants.%T) [symbolic = @impl.bfd509.1.%impl_witness (constants.%impl_witness.1896b7.1)]
// CHECK:STDOUT:   impl_decl @impl.bfd509.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc15_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc15_14.1 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc15_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc15_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc15: type = converted %T.ref, %T.as_type.loc15_21.1 [symbolic = %T.as_type.loc15_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc15_14.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc15: <witness> = impl_witness (), @impl.bfd509.2(constants.%T) [symbolic = @impl.bfd509.2.%impl_witness (constants.%impl_witness.1896b7.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.bfd509.1(%T.loc7_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc7_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_14.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc7_21.2: type = facet_access_type %T.loc7_14.2 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.bfd509.1(%T.loc7_14.2) [symbolic = %impl_witness (constants.%impl_witness.1896b7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc7 as %J.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.bfd509.2(%T.loc15_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc15_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc15_14.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc15_21.2: type = facet_access_type %T.loc15_14.2 [symbolic = %T.as_type.loc15_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.bfd509.2(%T.loc15_14.2) [symbolic = %impl_witness (constants.%impl_witness.1896b7.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc15 as %J.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.bfd509.1(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc7_21.2 => constants.%T.as_type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1896b7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.bfd509.1(%T.loc7_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.bfd509.2(constants.%T) {
// CHECK:STDOUT:   %T.loc15_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc15_21.2 => constants.%T.as_type
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1896b7.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.bfd509.2(%T.loc15_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_type_different_spelling.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.7704ae.1 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc13: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl.7704ae.2 [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc14_7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %C.ref.loc14_10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc14_11.1: %tuple.type = tuple_literal (%C.ref.loc14_7, %C.ref.loc14_10)
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %tuple: %tuple.type = tuple_value (%C.ref.loc14_7, %C.ref.loc14_10) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc14_11.2: %tuple.type = converted %.loc14_11.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:     %tuple.elem0: type = tuple_access %.loc14_11.2, element0 [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness () [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7704ae.1: %C.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7704ae.2: %tuple.elem0 as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redefinition_generic_regions.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness.1f09a1.1: <witness> = impl_witness (), @impl.2caff2.1(%T) [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A, @impl.2caff2.1(%T) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.1f09a1.2: <witness> = impl_witness (), @impl.2caff2.2(%T) [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B, @impl.2caff2.2(%T) [symbolic]
// CHECK:STDOUT:   %B: %B.type = struct_value () [symbolic]
// CHECK:STDOUT:   %C.type: type = fn_type @C, @impl.2caff2.2(%T) [symbolic]
// CHECK:STDOUT:   %C: %C.type = struct_value () [symbolic]
// CHECK:STDOUT:   %D.type: type = fn_type @D, @impl.2caff2.2(%T) [symbolic]
// CHECK:STDOUT:   %D: %D.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.specific_fn: <specific function> = specific_function %C, @C(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.2caff2.1 [concrete] {
// CHECK:STDOUT:     %T.patt.loc4_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_14.1 [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc4_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc4: <witness> = impl_witness (), @impl.2caff2.1(constants.%T) [symbolic = @impl.2caff2.1.%impl_witness (constants.%impl_witness.1f09a1.1)]
// CHECK:STDOUT:   impl_decl @impl.2caff2.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc15_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_14.1 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %T.loc15_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc15: <witness> = impl_witness (), @impl.2caff2.2(constants.%T) [symbolic = @impl.2caff2.2.%impl_witness (constants.%impl_witness.1f09a1.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2caff2.1(%T.loc4_14.1: type) {
// CHECK:STDOUT:   %T.loc4_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2caff2.1(%T.loc4_14.2) [symbolic = %impl_witness (constants.%impl_witness.1f09a1.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type: type = fn_type @A, @impl.2caff2.1(%T.loc4_14.2) [symbolic = %A.type (constants.%A.type)]
// CHECK:STDOUT:   %A: @impl.2caff2.1.%A.type (%A.type) = struct_value () [symbolic = %A (constants.%A)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %I.ref {
// CHECK:STDOUT:     %A.decl: @impl.2caff2.1.%A.type (%A.type) = fn_decl @A [symbolic = @impl.2caff2.1.%A (constants.%A)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     witness = file.%impl_witness.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2caff2.2(%T.loc15_14.1: type) {
// CHECK:STDOUT:   %T.loc15_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc15_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2caff2.2(%T.loc15_14.2) [symbolic = %impl_witness (constants.%impl_witness.1f09a1.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type: type = fn_type @B, @impl.2caff2.2(%T.loc15_14.2) [symbolic = %B.type (constants.%B.type)]
// CHECK:STDOUT:   %B: @impl.2caff2.2.%B.type (%B.type) = struct_value () [symbolic = %B (constants.%B)]
// CHECK:STDOUT:   %C.type: type = fn_type @C, @impl.2caff2.2(%T.loc15_14.2) [symbolic = %C.type (constants.%C.type)]
// CHECK:STDOUT:   %C: @impl.2caff2.2.%C.type (%C.type) = struct_value () [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %D.type: type = fn_type @D, @impl.2caff2.2(%T.loc15_14.2) [symbolic = %D.type (constants.%D.type)]
// CHECK:STDOUT:   %D: @impl.2caff2.2.%D.type (%D.type) = struct_value () [symbolic = %D (constants.%D)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %I.ref {
// CHECK:STDOUT:     %B.decl: @impl.2caff2.2.%B.type (%B.type) = fn_decl @B [symbolic = @impl.2caff2.2.%B (constants.%B)] {} {}
// CHECK:STDOUT:     %C.decl: @impl.2caff2.2.%C.type (%C.type) = fn_decl @C [symbolic = @impl.2caff2.2.%C (constants.%C)] {
// CHECK:STDOUT:       %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc19_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc19_14.2: type = converted %.loc19_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:       %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %D.decl: @impl.2caff2.2.%D.type (%D.type) = fn_decl @D [symbolic = @impl.2caff2.2.%D (constants.%D)] {
// CHECK:STDOUT:       %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc20_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc20_14.2: type = converted %.loc20_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:       %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     witness = file.%impl_witness.loc15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(@impl.2caff2.1.%T.loc4_14.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B(@impl.2caff2.2.%T.loc15_14.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C(@impl.2caff2.2.%T.loc15_14.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_26: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc19_27: %empty_tuple.type = converted %.loc19_26, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc19_27
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D(@impl.2caff2.2.%T.loc15_14.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C.type: type = fn_type @C, @impl.2caff2.2(%T) [symbolic = %C.type (constants.%C.type)]
// CHECK:STDOUT:   %C: @D.%C.type (%C.type) = struct_value () [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %C.specific_fn.loc21_12.2: <specific function> = specific_function %C, @C(%T) [symbolic = %C.specific_fn.loc21_12.2 (constants.%C.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc21_12: @D.%C.type (%C.type) = specific_constant @impl.2caff2.2.%C.decl, @impl.2caff2.2(constants.%T) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:     %C.ref: @D.%C.type (%C.type) = name_ref C, %.loc21_12 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:     %C.specific_fn.loc21_12.1: <specific function> = specific_function %C.ref, @C(constants.%T) [symbolic = %C.specific_fn.loc21_12.2 (constants.%C.specific_fn)]
// CHECK:STDOUT:     %C.call: init %empty_tuple.type = call %C.specific_fn.loc21_12.1()
// CHECK:STDOUT:     %.loc21_14.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:     %.loc21_14.2: ref %empty_tuple.type = temporary %.loc21_14.1, %C.call
// CHECK:STDOUT:     %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc21_15: %empty_tuple.type = converted %C.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc21_15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2caff2.1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1f09a1.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.type => constants.%A.type
// CHECK:STDOUT:   %A => constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2caff2.1(%T.loc4_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2caff2.2(constants.%T) {
// CHECK:STDOUT:   %T.loc15_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_14.2 => constants.%T.patt
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1f09a1.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type => constants.%B.type
// CHECK:STDOUT:   %B => constants.%B
// CHECK:STDOUT:   %C.type => constants.%C.type
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %D.type => constants.%D.type
// CHECK:STDOUT:   %D => constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2caff2.2(%T.loc15_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2caff2.2(@D.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@D.%T) {}
// CHECK:STDOUT:

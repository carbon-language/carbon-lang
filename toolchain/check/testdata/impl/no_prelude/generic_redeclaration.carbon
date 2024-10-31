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

// These are two different impls, so this is not a redefinition, even though the
// self type and constraint type are the same.
// CHECK:STDERR: same_self_and_interface.carbon:[[@LINE+4]]:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl forall [T:! I] T as Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! I] T as Interface {}
// CHECK:STDERR: same_self_and_interface.carbon:[[@LINE+4]]:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl forall [T:! J] T as Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! J] T as Interface {}
// CHECK:STDERR: same_self_and_interface.carbon:[[@LINE+4]]:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl forall [T:! K] T as Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! K] T as Interface {}
// CHECK:STDERR: same_self_and_interface.carbon:[[@LINE+4]]:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl forall [T:! L] T as Interface {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! L] T as Interface {}

// --- fail_same_self_and_interface_redefined.carbon

library "[[@TEST_NAME]]";

interface I {}
interface J {}

// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE+4]]:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl forall [T:! I] T as J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
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
// CHECK:STDERR: same_type_different_spelling.carbon:[[@LINE+4]]:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl C as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as I {}
// CHECK:STDERR: same_type_different_spelling.carbon:[[@LINE+3]]:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl (C, C).0 as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
impl (C, C).0 as I {}

// CHECK:STDOUT: --- same_self_and_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self.1: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self.2: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = interface_type @J [template]
// CHECK:STDOUT:   %Self.3: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = interface_type @K [template]
// CHECK:STDOUT:   %Self.4: %K.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L.type: type = interface_type @L [template]
// CHECK:STDOUT:   %Self.5: %L.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T.1: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.2: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.2: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.3: %K.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.3: %K.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.4: %L.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.4: %L.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.5: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.6: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.7: %K.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.8: %L.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%J.type] {} {}
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [template = constants.%K.type] {} {}
// CHECK:STDOUT:   %L.decl: type = interface_decl @L [template = constants.%L.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: %I.type = value_param_pattern %T.patt.loc11_14.1, runtime_param<invalid> [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T.1)]
// CHECK:STDOUT:     %.loc11_21.1: type = facet_type_access %T.ref [symbolic = %T.loc11_14.2 (constants.%T.1)]
// CHECK:STDOUT:     %.loc11_21.2: type = converted %T.ref, %.loc11_21.1 [symbolic = %T.loc11_14.2 (constants.%T.1)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %I.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_14.1: %I.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_14.2 (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {
// CHECK:STDOUT:     %T.patt.loc12_14.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: %J.type = value_param_pattern %T.patt.loc12_14.1, runtime_param<invalid> [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T.2)]
// CHECK:STDOUT:     %.loc12_21.1: type = facet_type_access %T.ref [symbolic = %T.loc12_14.2 (constants.%T.2)]
// CHECK:STDOUT:     %.loc12_21.2: type = converted %T.ref, %.loc12_21.1 [symbolic = %T.loc12_14.2 (constants.%T.2)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %J.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc12_14.1: %J.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_14.2 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %T.patt.loc13_14.1: %K.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_14.2 (constants.%T.patt.3)]
// CHECK:STDOUT:     %T.param_patt: %K.type = value_param_pattern %T.patt.loc13_14.1, runtime_param<invalid> [symbolic = %T.patt.loc13_14.2 (constants.%T.patt.3)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [template = constants.%K.type]
// CHECK:STDOUT:     %T.ref: %K.type = name_ref T, %T.loc13_14.1 [symbolic = %T.loc13_14.2 (constants.%T.3)]
// CHECK:STDOUT:     %.loc13_21.1: type = facet_type_access %T.ref [symbolic = %T.loc13_14.2 (constants.%T.3)]
// CHECK:STDOUT:     %.loc13_21.2: type = converted %T.ref, %.loc13_21.1 [symbolic = %T.loc13_14.2 (constants.%T.3)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %K.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc13_14.1: %K.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc13_14.2 (constants.%T.3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {
// CHECK:STDOUT:     %T.patt.loc14_14.1: %L.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt.4)]
// CHECK:STDOUT:     %T.param_patt: %L.type = value_param_pattern %T.patt.loc14_14.1, runtime_param<invalid> [symbolic = %T.patt.loc14_14.2 (constants.%T.patt.4)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %L.ref: type = name_ref L, file.%L.decl [template = constants.%L.type]
// CHECK:STDOUT:     %T.ref: %L.type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T.4)]
// CHECK:STDOUT:     %.loc14_21.1: type = facet_type_access %T.ref [symbolic = %T.loc14_14.2 (constants.%T.4)]
// CHECK:STDOUT:     %.loc14_21.2: type = converted %T.ref, %.loc14_21.1 [symbolic = %T.loc14_14.2 (constants.%T.4)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %L.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc14_14.1: %L.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc14_14.2 (constants.%T.4)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.5 [template] {
// CHECK:STDOUT:     %T.patt.loc22_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_14.2 (constants.%T.patt.5)]
// CHECK:STDOUT:     %T.param_patt: %I.type = value_param_pattern %T.patt.loc22_14.1, runtime_param<invalid> [symbolic = %T.patt.loc22_14.2 (constants.%T.patt.5)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc22_14.1 [symbolic = %T.loc22_14.2 (constants.%T.1)]
// CHECK:STDOUT:     %.loc22_21.1: type = facet_type_access %T.ref [symbolic = %T.loc22_14.2 (constants.%T.1)]
// CHECK:STDOUT:     %.loc22_21.2: type = converted %T.ref, %.loc22_21.1 [symbolic = %T.loc22_14.2 (constants.%T.1)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %I.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc22_14.1: %I.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc22_14.2 (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.6 [template] {
// CHECK:STDOUT:     %T.patt.loc27_14.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_14.2 (constants.%T.patt.6)]
// CHECK:STDOUT:     %T.param_patt: %J.type = value_param_pattern %T.patt.loc27_14.1, runtime_param<invalid> [symbolic = %T.patt.loc27_14.2 (constants.%T.patt.6)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc27_14.1 [symbolic = %T.loc27_14.2 (constants.%T.2)]
// CHECK:STDOUT:     %.loc27_21.1: type = facet_type_access %T.ref [symbolic = %T.loc27_14.2 (constants.%T.2)]
// CHECK:STDOUT:     %.loc27_21.2: type = converted %T.ref, %.loc27_21.1 [symbolic = %T.loc27_14.2 (constants.%T.2)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %J.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc27_14.1: %J.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc27_14.2 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.7 [template] {
// CHECK:STDOUT:     %T.patt.loc32_14.1: %K.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc32_14.2 (constants.%T.patt.7)]
// CHECK:STDOUT:     %T.param_patt: %K.type = value_param_pattern %T.patt.loc32_14.1, runtime_param<invalid> [symbolic = %T.patt.loc32_14.2 (constants.%T.patt.7)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [template = constants.%K.type]
// CHECK:STDOUT:     %T.ref: %K.type = name_ref T, %T.loc32_14.1 [symbolic = %T.loc32_14.2 (constants.%T.3)]
// CHECK:STDOUT:     %.loc32_21.1: type = facet_type_access %T.ref [symbolic = %T.loc32_14.2 (constants.%T.3)]
// CHECK:STDOUT:     %.loc32_21.2: type = converted %T.ref, %.loc32_21.1 [symbolic = %T.loc32_14.2 (constants.%T.3)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %K.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc32_14.1: %K.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc32_14.2 (constants.%T.3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.8 [template] {
// CHECK:STDOUT:     %T.patt.loc37_14.1: %L.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc37_14.2 (constants.%T.patt.8)]
// CHECK:STDOUT:     %T.param_patt: %L.type = value_param_pattern %T.patt.loc37_14.1, runtime_param<invalid> [symbolic = %T.patt.loc37_14.2 (constants.%T.patt.8)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %L.ref: type = name_ref L, file.%L.decl [template = constants.%L.type]
// CHECK:STDOUT:     %T.ref: %L.type = name_ref T, %T.loc37_14.1 [symbolic = %T.loc37_14.2 (constants.%T.4)]
// CHECK:STDOUT:     %.loc37_21.1: type = facet_type_access %T.ref [symbolic = %T.loc37_14.2 (constants.%T.4)]
// CHECK:STDOUT:     %.loc37_21.2: type = converted %T.ref, %.loc37_21.1 [symbolic = %T.loc37_14.2 (constants.%T.4)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %L.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc37_14.1: %L.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc37_14.2 (constants.%T.4)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %L.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(%T.loc11_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc11_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc11_21.2 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(%T.loc12_14.1: %J.type) {
// CHECK:STDOUT:   %T.loc12_14.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.2)]
// CHECK:STDOUT:   %T.patt.loc12_14.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc12_21.2 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%T.loc13_14.1: %K.type) {
// CHECK:STDOUT:   %T.loc13_14.2: %K.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_14.2 (constants.%T.3)]
// CHECK:STDOUT:   %T.patt.loc13_14.2: %K.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_14.2 (constants.%T.patt.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc13_21.2 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(%T.loc14_14.1: %L.type) {
// CHECK:STDOUT:   %T.loc14_14.2: %L.type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T.4)]
// CHECK:STDOUT:   %T.patt.loc14_14.2: %L.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_21.2 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(%T.loc22_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc22_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc22_14.2 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.loc22_14.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_14.2 (constants.%T.patt.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc22_21.2 as %Interface.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(%T.loc27_14.1: %J.type) {
// CHECK:STDOUT:   %T.loc27_14.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc27_14.2 (constants.%T.2)]
// CHECK:STDOUT:   %T.patt.loc27_14.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc27_14.2 (constants.%T.patt.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc27_21.2 as %Interface.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(%T.loc32_14.1: %K.type) {
// CHECK:STDOUT:   %T.loc32_14.2: %K.type = bind_symbolic_name T, 0 [symbolic = %T.loc32_14.2 (constants.%T.3)]
// CHECK:STDOUT:   %T.patt.loc32_14.2: %K.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc32_14.2 (constants.%T.patt.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc32_21.2 as %Interface.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(%T.loc37_14.1: %L.type) {
// CHECK:STDOUT:   %T.loc37_14.2: %L.type = bind_symbolic_name T, 0 [symbolic = %T.loc37_14.2 (constants.%T.4)]
// CHECK:STDOUT:   %T.patt.loc37_14.2: %L.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc37_14.2 (constants.%T.patt.8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc37_21.2 as %Interface.ref {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T.1) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T.2) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.2
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%T.3) {
// CHECK:STDOUT:   %T.loc13_14.2 => constants.%T.3
// CHECK:STDOUT:   %T.patt.loc13_14.2 => constants.%T.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%T.4) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T.4
// CHECK:STDOUT:   %T.patt.loc14_14.2 => constants.%T.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%T.1) {
// CHECK:STDOUT:   %T.loc22_14.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc22_14.2 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%T.2) {
// CHECK:STDOUT:   %T.loc27_14.2 => constants.%T.2
// CHECK:STDOUT:   %T.patt.loc27_14.2 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%T.3) {
// CHECK:STDOUT:   %T.loc32_14.2 => constants.%T.3
// CHECK:STDOUT:   %T.patt.loc32_14.2 => constants.%T.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%T.4) {
// CHECK:STDOUT:   %T.loc37_14.2 => constants.%T.4
// CHECK:STDOUT:   %T.patt.loc37_14.2 => constants.%T.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_same_self_and_interface_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = interface_type @J [template]
// CHECK:STDOUT:   %Self.2: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.2: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: %I.type = value_param_pattern %T.patt.loc11_14.1, runtime_param<invalid> [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref.loc11: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.ref.loc11: %I.type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:     %.loc11_21.1: type = facet_type_access %T.ref.loc11 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:     %.loc11_21.2: type = converted %T.ref.loc11, %.loc11_21.1 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.ref.loc11: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %T.param.loc11: %I.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_14.1: %I.type = bind_symbolic_name T, 0, %T.param.loc11 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: %I.type = value_param_pattern %T.patt.loc11_14.1, runtime_param<invalid> [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref.loc19: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.ref.loc19: %I.type = name_ref T, %T.loc19 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc19_21.1: type = facet_type_access %T.ref.loc19 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc19_21.2: type = converted %T.ref.loc19, %.loc19_21.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %J.ref.loc19: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %T.param.loc19: %I.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc19: %I.type = bind_symbolic_name T, 0, %T.param.loc19 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc11_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc11_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc11_21.2 as %J.ref.loc11 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_type_different_spelling.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value (%C, %C) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %C.ref.loc21_7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc21_10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc21_11.1: %.1 = tuple_literal (%C.ref.loc21_7, %C.ref.loc21_10)
// CHECK:STDOUT:     %.loc21_13: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:     %tuple: %.1 = tuple_value (%C.ref.loc21_7, %C.ref.loc21_10) [template = constants.%tuple]
// CHECK:STDOUT:     %.loc21_11.2: %.1 = converted %.loc21_11.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:     %.loc21_12: type = tuple_access %.loc21_11.2, element0 [template = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: impl @impl.1: %C.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %.loc21_12 as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:

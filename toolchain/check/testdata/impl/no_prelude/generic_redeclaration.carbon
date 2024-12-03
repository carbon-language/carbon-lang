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
impl forall [T:! I] T as Interface {}
impl forall [T:! J] T as Interface {}
impl forall [T:! K] T as Interface {}
impl forall [T:! L] T as Interface {}

// --- fail_same_self_and_interface_redefined.carbon

library "[[@TEST_NAME]]";

interface I {}
interface J {}

impl forall [T:! I] T as J {}
// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE+6]]:1: error: redefinition of `impl T as J` [ImplRedefinition]
// CHECK:STDERR: impl forall [T:! I] T as J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE-4]]:1: note: previous definition was here [ImplPreviousDefinition]
// CHECK:STDERR: impl forall [T:! I] T as J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

// CHECK:STDOUT: --- same_self_and_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [template]
// CHECK:STDOUT:   %Self.1: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.2: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [template]
// CHECK:STDOUT:   %Self.3: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [template]
// CHECK:STDOUT:   %Self.4: %K.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [template]
// CHECK:STDOUT:   %Self.5: %L.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T.1: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.1: type = facet_access_type %T.1 [symbolic]
// CHECK:STDOUT:   %T.2: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.2: %J.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.2: type = facet_access_type %T.2 [symbolic]
// CHECK:STDOUT:   %T.3: %K.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.3: %K.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.3: type = facet_access_type %T.3 [symbolic]
// CHECK:STDOUT:   %T.4: %L.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.4: %L.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.4: type = facet_access_type %T.4 [symbolic]
// CHECK:STDOUT:   %interface: <witness> = interface_witness () [template]
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
// CHECK:STDOUT:     %T.as_type.loc11_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc11_21.2 (constants.%T.as_type.1)]
// CHECK:STDOUT:     %.loc11: type = converted %T.ref, %T.as_type.loc11_21.1 [symbolic = %T.as_type.loc11_21.2 (constants.%T.as_type.1)]
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
// CHECK:STDOUT:     %T.as_type.loc12_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc12_21.2 (constants.%T.as_type.2)]
// CHECK:STDOUT:     %.loc12: type = converted %T.ref, %T.as_type.loc12_21.1 [symbolic = %T.as_type.loc12_21.2 (constants.%T.as_type.2)]
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
// CHECK:STDOUT:     %T.as_type.loc13_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type.3)]
// CHECK:STDOUT:     %.loc13: type = converted %T.ref, %T.as_type.loc13_21.1 [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type.3)]
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
// CHECK:STDOUT:     %T.as_type.loc14_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc14_21.2 (constants.%T.as_type.4)]
// CHECK:STDOUT:     %.loc14: type = converted %T.ref, %T.as_type.loc14_21.1 [symbolic = %T.as_type.loc14_21.2 (constants.%T.as_type.4)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %L.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc14_14.1: %L.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc14_14.2 (constants.%T.4)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.5 [template] {
// CHECK:STDOUT:     %T.patt.loc18_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: %I.type = value_param_pattern %T.patt.loc18_14.1, runtime_param<invalid> [symbolic = %T.patt.loc18_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc18_14.1 [symbolic = %T.loc18_14.2 (constants.%T.1)]
// CHECK:STDOUT:     %T.as_type.loc18_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc18_21.2 (constants.%T.as_type.1)]
// CHECK:STDOUT:     %.loc18: type = converted %T.ref, %T.as_type.loc18_21.1 [symbolic = %T.as_type.loc18_21.2 (constants.%T.as_type.1)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %I.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc18_14.1: %I.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc18_14.2 (constants.%T.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.6 [template] {
// CHECK:STDOUT:     %T.patt.loc19_14.1: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_14.2 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: %J.type = value_param_pattern %T.patt.loc19_14.1, runtime_param<invalid> [symbolic = %T.patt.loc19_14.2 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc19_14.1 [symbolic = %T.loc19_14.2 (constants.%T.2)]
// CHECK:STDOUT:     %T.as_type.loc19_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc19_21.2 (constants.%T.as_type.2)]
// CHECK:STDOUT:     %.loc19: type = converted %T.ref, %T.as_type.loc19_21.1 [symbolic = %T.as_type.loc19_21.2 (constants.%T.as_type.2)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %J.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc19_14.1: %J.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc19_14.2 (constants.%T.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.7 [template] {
// CHECK:STDOUT:     %T.patt.loc20_14.1: %K.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_14.2 (constants.%T.patt.3)]
// CHECK:STDOUT:     %T.param_patt: %K.type = value_param_pattern %T.patt.loc20_14.1, runtime_param<invalid> [symbolic = %T.patt.loc20_14.2 (constants.%T.patt.3)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [template = constants.%K.type]
// CHECK:STDOUT:     %T.ref: %K.type = name_ref T, %T.loc20_14.1 [symbolic = %T.loc20_14.2 (constants.%T.3)]
// CHECK:STDOUT:     %T.as_type.loc20_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc20_21.2 (constants.%T.as_type.3)]
// CHECK:STDOUT:     %.loc20: type = converted %T.ref, %T.as_type.loc20_21.1 [symbolic = %T.as_type.loc20_21.2 (constants.%T.as_type.3)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %K.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc20_14.1: %K.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc20_14.2 (constants.%T.3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.8 [template] {
// CHECK:STDOUT:     %T.patt.loc21_14.1: %L.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt.4)]
// CHECK:STDOUT:     %T.param_patt: %L.type = value_param_pattern %T.patt.loc21_14.1, runtime_param<invalid> [symbolic = %T.patt.loc21_14.2 (constants.%T.patt.4)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %L.ref: type = name_ref L, file.%L.decl [template = constants.%L.type]
// CHECK:STDOUT:     %T.ref: %L.type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T.4)]
// CHECK:STDOUT:     %T.as_type.loc21_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc21_21.2 (constants.%T.as_type.4)]
// CHECK:STDOUT:     %.loc21: type = converted %T.ref, %T.as_type.loc21_21.1 [symbolic = %T.as_type.loc21_21.2 (constants.%T.as_type.4)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.param: %L.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc21_14.1: %L.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc21_14.2 (constants.%T.4)]
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
// CHECK:STDOUT:   %T.as_type.loc11_21.2: type = facet_access_type %T.loc11_14.2 [symbolic = %T.as_type.loc11_21.2 (constants.%T.as_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc11 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(%T.loc12_14.1: %J.type) {
// CHECK:STDOUT:   %T.loc12_14.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T.2)]
// CHECK:STDOUT:   %T.patt.loc12_14.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt.2)]
// CHECK:STDOUT:   %T.as_type.loc12_21.2: type = facet_access_type %T.loc12_14.2 [symbolic = %T.as_type.loc12_21.2 (constants.%T.as_type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc12 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%T.loc13_14.1: %K.type) {
// CHECK:STDOUT:   %T.loc13_14.2: %K.type = bind_symbolic_name T, 0 [symbolic = %T.loc13_14.2 (constants.%T.3)]
// CHECK:STDOUT:   %T.patt.loc13_14.2: %K.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_14.2 (constants.%T.patt.3)]
// CHECK:STDOUT:   %T.as_type.loc13_21.2: type = facet_access_type %T.loc13_14.2 [symbolic = %T.as_type.loc13_21.2 (constants.%T.as_type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc13 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(%T.loc14_14.1: %L.type) {
// CHECK:STDOUT:   %T.loc14_14.2: %L.type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T.4)]
// CHECK:STDOUT:   %T.patt.loc14_14.2: %L.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt.4)]
// CHECK:STDOUT:   %T.as_type.loc14_21.2: type = facet_access_type %T.loc14_14.2 [symbolic = %T.as_type.loc14_21.2 (constants.%T.as_type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(%T.loc18_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc18_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc18_14.2 (constants.%T.1)]
// CHECK:STDOUT:   %T.patt.loc18_14.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_14.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   %T.as_type.loc18_21.2: type = facet_access_type %T.loc18_14.2 [symbolic = %T.as_type.loc18_21.2 (constants.%T.as_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc18 as %Interface.ref {
// CHECK:STDOUT:     %interface: <witness> = interface_witness () [template = constants.%interface]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %interface
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(%T.loc19_14.1: %J.type) {
// CHECK:STDOUT:   %T.loc19_14.2: %J.type = bind_symbolic_name T, 0 [symbolic = %T.loc19_14.2 (constants.%T.2)]
// CHECK:STDOUT:   %T.patt.loc19_14.2: %J.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_14.2 (constants.%T.patt.2)]
// CHECK:STDOUT:   %T.as_type.loc19_21.2: type = facet_access_type %T.loc19_14.2 [symbolic = %T.as_type.loc19_21.2 (constants.%T.as_type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc19 as %Interface.ref {
// CHECK:STDOUT:     %interface: <witness> = interface_witness () [template = constants.%interface]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %interface
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(%T.loc20_14.1: %K.type) {
// CHECK:STDOUT:   %T.loc20_14.2: %K.type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T.3)]
// CHECK:STDOUT:   %T.patt.loc20_14.2: %K.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_14.2 (constants.%T.patt.3)]
// CHECK:STDOUT:   %T.as_type.loc20_21.2: type = facet_access_type %T.loc20_14.2 [symbolic = %T.as_type.loc20_21.2 (constants.%T.as_type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc20 as %Interface.ref {
// CHECK:STDOUT:     %interface: <witness> = interface_witness () [template = constants.%interface]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %interface
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(%T.loc21_14.1: %L.type) {
// CHECK:STDOUT:   %T.loc21_14.2: %L.type = bind_symbolic_name T, 0 [symbolic = %T.loc21_14.2 (constants.%T.4)]
// CHECK:STDOUT:   %T.patt.loc21_14.2: %L.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_14.2 (constants.%T.patt.4)]
// CHECK:STDOUT:   %T.as_type.loc21_21.2: type = facet_access_type %T.loc21_14.2 [symbolic = %T.as_type.loc21_21.2 (constants.%T.as_type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc21 as %Interface.ref {
// CHECK:STDOUT:     %interface: <witness> = interface_witness () [template = constants.%interface]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %interface
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T.1) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T.1
// CHECK:STDOUT:   %T.as_type.loc11_21.2 => constants.%T.as_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T.2) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T.2
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%T.2
// CHECK:STDOUT:   %T.as_type.loc12_21.2 => constants.%T.as_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%T.3) {
// CHECK:STDOUT:   %T.loc13_14.2 => constants.%T.3
// CHECK:STDOUT:   %T.patt.loc13_14.2 => constants.%T.3
// CHECK:STDOUT:   %T.as_type.loc13_21.2 => constants.%T.as_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%T.4) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T.4
// CHECK:STDOUT:   %T.patt.loc14_14.2 => constants.%T.4
// CHECK:STDOUT:   %T.as_type.loc14_21.2 => constants.%T.as_type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%T.1) {
// CHECK:STDOUT:   %T.loc18_14.2 => constants.%T.1
// CHECK:STDOUT:   %T.patt.loc18_14.2 => constants.%T.1
// CHECK:STDOUT:   %T.as_type.loc18_21.2 => constants.%T.as_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%T.2) {
// CHECK:STDOUT:   %T.loc19_14.2 => constants.%T.2
// CHECK:STDOUT:   %T.patt.loc19_14.2 => constants.%T.2
// CHECK:STDOUT:   %T.as_type.loc19_21.2 => constants.%T.as_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%T.3) {
// CHECK:STDOUT:   %T.loc20_14.2 => constants.%T.3
// CHECK:STDOUT:   %T.patt.loc20_14.2 => constants.%T.3
// CHECK:STDOUT:   %T.as_type.loc20_21.2 => constants.%T.as_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%T.4) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T.4
// CHECK:STDOUT:   %T.patt.loc21_14.2 => constants.%T.4
// CHECK:STDOUT:   %T.as_type.loc21_21.2 => constants.%T.as_type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_same_self_and_interface_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [template]
// CHECK:STDOUT:   %Self.2: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %I.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc7_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %I.type = value_param_pattern %T.patt.loc7_14.1, runtime_param<invalid> [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref.loc7: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.ref.loc7: %I.type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.as_type.loc7_21.1: type = facet_access_type %T.ref.loc7 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc7: type = converted %T.ref.loc7, %T.as_type.loc7_21.1 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %J.ref.loc7: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %T.param.loc7: %I.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_14.1: %I.type = bind_symbolic_name T, 0, %T.param.loc7 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt.loc7_14.1: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: %I.type = value_param_pattern %T.patt.loc7_14.1, runtime_param<invalid> [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref.loc14: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.ref.loc14: %I.type = name_ref T, %T.loc14 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.as_type.loc14: type = facet_access_type %T.ref.loc14 [symbolic = constants.%T.as_type]
// CHECK:STDOUT:     %.loc14: type = converted %T.ref.loc14, %T.as_type.loc14 [symbolic = constants.%T.as_type]
// CHECK:STDOUT:     %J.ref.loc14: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %T.param.loc14: %I.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc14: %I.type = bind_symbolic_name T, 0, %T.param.loc14 [symbolic = constants.%T]
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
// CHECK:STDOUT: generic impl @impl(%T.loc7_14.1: %I.type) {
// CHECK:STDOUT:   %T.loc7_14.2: %I.type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_14.2: %I.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc7_21.2: type = facet_access_type %T.loc7_14.2 [symbolic = %T.as_type.loc7_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc7 as %J.ref.loc7 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <unexpected>.inst+26.loc7_28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_14.2 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc7_21.2 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_type_different_spelling.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %interface: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%C, %C) [template]
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
// CHECK:STDOUT:     %C.ref.loc14_7: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %C.ref.loc14_10: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc14_11.1: %tuple.type = tuple_literal (%C.ref.loc14_7, %C.ref.loc14_10)
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:     %tuple: %tuple.type = tuple_value (%C.ref.loc14_7, %C.ref.loc14_10) [template = constants.%tuple]
// CHECK:STDOUT:     %.loc14_11.2: %tuple.type = converted %.loc14_11.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:     %tuple.elem0: type = tuple_access %.loc14_11.2, element0 [template = constants.%C]
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
// CHECK:STDOUT:   %interface: <witness> = interface_witness () [template = constants.%interface]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %tuple.elem0 as %I.ref {
// CHECK:STDOUT:   %interface: <witness> = interface_witness () [template = constants.%interface]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:

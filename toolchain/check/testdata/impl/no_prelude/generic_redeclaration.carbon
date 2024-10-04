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
// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE+6]]:1: error: redefinition of `impl T as J`
// CHECK:STDERR: impl forall [T:! I] T as J {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_same_self_and_interface_redefined.carbon:[[@LINE-4]]:1: note: previous definition was here
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
// CHECK:STDOUT:   %T.2: %J.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.3: %K.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.4: %L.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: <witness> = interface_witness () [template]
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
// CHECK:STDOUT:     %T.patt: %I.type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.param: %I.type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11: %I.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc11 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:     %.loc11_21.1: type = facet_type_access %T.ref [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:     %.loc11_21.2: type = converted %T.ref, %.loc11_21.1 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {
// CHECK:STDOUT:     %T.patt: %J.type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %T.param: %J.type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc12: %J.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc12 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     %.loc12_21.1: type = facet_type_access %T.ref [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     %.loc12_21.2: type = converted %T.ref, %.loc12_21.1 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %T.patt: %K.type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [template = constants.%K.type]
// CHECK:STDOUT:     %T.param: %K.type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc13: %K.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:     %T.ref: %K.type = name_ref T, %T.loc13 [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:     %.loc13_21.1: type = facet_type_access %T.ref [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:     %.loc13_21.2: type = converted %T.ref, %.loc13_21.1 [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {
// CHECK:STDOUT:     %T.patt: %L.type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %L.ref: type = name_ref L, file.%L.decl [template = constants.%L.type]
// CHECK:STDOUT:     %T.param: %L.type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc14: %L.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T.4)]
// CHECK:STDOUT:     %T.ref: %L.type = name_ref T, %T.loc14 [symbolic = %T.1 (constants.%T.4)]
// CHECK:STDOUT:     %.loc14_21.1: type = facet_type_access %T.ref [symbolic = %T.1 (constants.%T.4)]
// CHECK:STDOUT:     %.loc14_21.2: type = converted %T.ref, %.loc14_21.1 [symbolic = %T.1 (constants.%T.4)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.5 [template] {
// CHECK:STDOUT:     %T.patt: %I.type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.param: %I.type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc18: %I.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:     %T.ref: %I.type = name_ref T, %T.loc18 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:     %.loc18_21.1: type = facet_type_access %T.ref [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:     %.loc18_21.2: type = converted %T.ref, %.loc18_21.1 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.6 [template] {
// CHECK:STDOUT:     %T.patt: %J.type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %T.param: %J.type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc19: %J.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     %T.ref: %J.type = name_ref T, %T.loc19 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     %.loc19_21.1: type = facet_type_access %T.ref [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     %.loc19_21.2: type = converted %T.ref, %.loc19_21.1 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.7 [template] {
// CHECK:STDOUT:     %T.patt: %K.type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [template = constants.%K.type]
// CHECK:STDOUT:     %T.param: %K.type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc20: %K.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:     %T.ref: %K.type = name_ref T, %T.loc20 [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:     %.loc20_21.1: type = facet_type_access %T.ref [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:     %.loc20_21.2: type = converted %T.ref, %.loc20_21.1 [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.8 [template] {
// CHECK:STDOUT:     %T.patt: %L.type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %L.ref: type = name_ref L, file.%L.decl [template = constants.%L.type]
// CHECK:STDOUT:     %T.param: %L.type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc21: %L.type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T.4)]
// CHECK:STDOUT:     %T.ref: %L.type = name_ref T, %T.loc21 [symbolic = %T.1 (constants.%T.4)]
// CHECK:STDOUT:     %.loc21_21.1: type = facet_type_access %T.ref [symbolic = %T.1 (constants.%T.4)]
// CHECK:STDOUT:     %.loc21_21.2: type = converted %T.ref, %.loc21_21.1 [symbolic = %T.1 (constants.%T.4)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
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
// CHECK:STDOUT: generic impl @impl.1(%T.loc11: %I.type) {
// CHECK:STDOUT:   %T.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc11_21.2 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(%T.loc12: %J.type) {
// CHECK:STDOUT:   %T.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc12_21.2 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%T.loc13: %K.type) {
// CHECK:STDOUT:   %T.1: %K.type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc13_21.2 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(%T.loc14: %L.type) {
// CHECK:STDOUT:   %T.1: %L.type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc14_21.2 as %Interface.ref;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(%T.loc18: %I.type) {
// CHECK:STDOUT:   %T.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc18_21.2 as %Interface.ref {
// CHECK:STDOUT:     %.loc18_36: <witness> = interface_witness () [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %.loc18_36
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(%T.loc19: %J.type) {
// CHECK:STDOUT:   %T.1: %J.type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc19_21.2 as %Interface.ref {
// CHECK:STDOUT:     %.loc19_36: <witness> = interface_witness () [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %.loc19_36
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.7(%T.loc20: %K.type) {
// CHECK:STDOUT:   %T.1: %K.type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc20_21.2 as %Interface.ref {
// CHECK:STDOUT:     %.loc20_36: <witness> = interface_witness () [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %.loc20_36
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(%T.loc21: %L.type) {
// CHECK:STDOUT:   %T.1: %L.type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc21_21.2 as %Interface.ref {
// CHECK:STDOUT:     %.loc21_36: <witness> = interface_witness () [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %.loc21_36
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T.2) {
// CHECK:STDOUT:   %T.1 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%T.3) {
// CHECK:STDOUT:   %T.1 => constants.%T.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%T.4) {
// CHECK:STDOUT:   %T.1 => constants.%T.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%T.2) {
// CHECK:STDOUT:   %T.1 => constants.%T.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.7(constants.%T.3) {
// CHECK:STDOUT:   %T.1 => constants.%T.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%T.4) {
// CHECK:STDOUT:   %T.1 => constants.%T.4
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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: <witness> = interface_witness () [template]
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
// CHECK:STDOUT:     %T.patt: %I.type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref.loc7: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.param.loc7: %I.type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7: %I.type = bind_symbolic_name T, 0, %T.param.loc7 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc7: %I.type = name_ref T, %T.loc7 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_21.1: type = facet_type_access %T.ref.loc7 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_21.2: type = converted %T.ref.loc7, %.loc7_21.1 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %J.ref.loc7: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %T.patt: %I.type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref.loc14: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %T.param.loc14: %I.type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc14: %I.type = bind_symbolic_name T, 0, %T.param.loc14 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc14: %I.type = name_ref T, %T.loc14 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc14_21.1: type = facet_type_access %T.ref.loc14 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc14_21.2: type = converted %T.ref.loc14, %.loc14_21.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %J.ref.loc14: type = name_ref J, file.%J.decl [template = constants.%J.type]
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
// CHECK:STDOUT: generic impl @impl(%T.loc7: %I.type) {
// CHECK:STDOUT:   %T.1: %I.type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc7_21.2 as %J.ref.loc7 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <unexpected>.inst+21.loc7_28
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_type_different_spelling.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: <witness> = interface_witness () [template]
// CHECK:STDOUT:   %.3: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%C, %C) [template]
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
// CHECK:STDOUT:     %.loc14_11.1: %.3 = tuple_literal (%C.ref.loc14_7, %C.ref.loc14_10)
// CHECK:STDOUT:     %.loc14_13: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:     %tuple: %.3 = tuple_value (%C.ref.loc14_7, %C.ref.loc14_10) [template = constants.%tuple]
// CHECK:STDOUT:     %.loc14_11.2: %.3 = converted %.loc14_11.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:     %.loc14_12: type = tuple_access %.loc14_11.2, element0 [template = constants.%C]
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
// CHECK:STDOUT:   %.loc13: <witness> = interface_witness () [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %.loc14_12 as %I.ref {
// CHECK:STDOUT:   %.loc14_20: <witness> = interface_witness () [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc14_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:

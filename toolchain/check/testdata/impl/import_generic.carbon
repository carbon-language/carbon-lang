// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/import_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/import_generic.carbon

// --- basic_import_generic_interface.carbon
library "[[@TEST_NAME]]";

interface I(T: type) {}
interface J(T: type) {
  extend require impls I(T);
}

// --- basic_import_generic_interface.impl.carbon
impl library "[[@TEST_NAME]]";

class C {}

impl C as I({}) {}

impl C as J({}) {}

// --- basic_import_generic_constraint.carbon
library "[[@TEST_NAME]]";

interface I(T: type) {}
constraint J(T: type) {
  extend require impls I(T);
}

// --- basic_import_generic_constraint.impl.carbon
impl library "[[@TEST_NAME]]";

class C {}

impl C as J({}) {}

// --- import_generic.carbon

library "[[@TEST_NAME]]";

class C {}
interface I(T: type) {}

// Has both declaration and definition.
impl forall [T: type] C as I(T);
impl forall [T: type] C as I(T) {}

// Only has definition.
impl forall [T: type] C as I(T*) {}

constraint N(T: type) {
  extend require impls I(T);
}

// --- fail_import_generic.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+2]]:1-32: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] C as I(T);

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+2]]:1-33: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] C as I(T) {}

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+2]]:1-33: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] C as I(T*);

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+2]]:1-34: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] C as I(T*) {}

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+2]]:1-32: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] C as N(T);
// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+2]]:1-33: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] C as N(T*);

// --- import_generic_with_different_specific.carbon

library "[[@TEST_NAME]]";

class C {}
interface I(T: type) {}

impl forall [T: type] C as I(T) {}

constraint N(T: type) {
  extend require impls I(T);
}

// --- import_generic_with_different_specific.impl.carbon
impl library "[[@TEST_NAME]]";

// These have different specifics than any impl decl in the api file, so they
// are allowed.
impl forall [T: type] C as I(T*) {}
impl forall [T: type] C as N(T**) {}

// --- fail_import_generic_decl.carbon

library "[[@TEST_NAME]]";

class D {}
interface J(T: type) {}

constraint N(T: type) {
  extend require impls J(T);
}

// CHECK:STDERR: fail_import_generic_decl.carbon:[[@LINE+2]]:1-32: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR:
impl forall [T: type] D as J(T);

// CHECK:STDERR: fail_import_generic_decl.carbon:[[@LINE+2]]:1-33: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR:
impl forall [T: type] D as J(T*);

// --- fail_import_generic_decl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+2]]:1-32: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] D as J(T);

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+2]]:1-33: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] D as J(T) {}

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+2]]:1-33: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] D as J(T*);

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+2]]:1-34: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] D as J(T*) {}

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+2]]:1-33: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] D as N(T) {}

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+2]]:1-34: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:
impl forall [T: type] D as N(T*) {}

// CHECK:STDOUT: --- basic_import_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.3fe: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.191: %I.type.3fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %J.type.23c: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.23c = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.5c8: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.ad7: %J.type.5c8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.ad7 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc3_14.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_16.1: type = splice_block %.loc3_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type.23c = interface_decl @J [concrete = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_16.1: type = splice_block %.loc4_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc3_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc3_14.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc3_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc3_14.1)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Self.loc3_22.2: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc3_22.2 (constants.%Self.191)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_22.1: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc3_22.2 (constants.%Self.191)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc4_14.1)> [symbolic = %J.type (constants.%J.type.5c8)]
// CHECK:STDOUT:   %Self.loc4_22.2: @J.%J.type (%J.type.5c8) = symbolic_binding Self, 1 [symbolic = %Self.loc4_22.2 (constants.%Self.ad7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_22.1: @J.%J.type (%J.type.5c8) = symbolic_binding Self, 1 [symbolic = %Self.loc4_22.2 (constants.%Self.ad7)]
// CHECK:STDOUT:     %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %J.WithSelf.Self.as_type.impls.I.type.require.decl = require_decl @J.WithSelf.Self.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc5_18.1 impls %I.type.loc5_27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc5_18.1: type = facet_access_type @J.%Self.loc4_22.1 [symbolic = %Self.as_type.loc5_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @J.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc5_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc5: type = specific_constant @J.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc5_27.1, @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.ad7) [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_22.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     extend @J.WithSelf.%.loc5
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require @J.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type.loc5_18.1 impls @J.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc5_27.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.WithSelf.Self.as_type.impls.I.type.require(@J.%T.loc4_14.2: type, @J.%Self.loc4_22.1: @J.%J.type (%J.type.5c8)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.5c8)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.as_type.impls.I.type.require.%J.type (%J.type.5c8) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.ad7)]
// CHECK:STDOUT:   %Self.as_type.loc5_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type.loc5_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.3fe)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc3_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %Self.loc3_22.2 => constants.%Self.191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.191) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.ad7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.ad7) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.5c8
// CHECK:STDOUT:   %Self => constants.%Self.ad7
// CHECK:STDOUT:   %Self.as_type.loc5_18.2 => constants.%Self.as_type
// CHECK:STDOUT:   %I.type.loc5_27.2 => constants.%I.type.3fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_interface.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.3fe: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.a67: %I.type.3fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.81d: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %.b5d: <witness> = impl_self_witness %C, @I, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.47c: %I.type.81d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.81d = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %J.type.23c: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.23c = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.5c8: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.9f1: %J.type.5c8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.9f1 [symbolic]
// CHECK:STDOUT:   %J.type.80f: type = facet_type <@J, @J(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %.cc7: <witness> = impl_self_witness %C, @J, @J(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.043: %J.type.80f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.7da: <witness> = complete_type_witness %I.type.81d [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %J.facet: %J.type.80f = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.335 = import_ref Main//basic_import_generic_interface, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.J: %J.type.23c = import_ref Main//basic_import_generic_interface, J, loaded [concrete = constants.%J.generic]
// CHECK:STDOUT:   %Main.import_ref.a43 = import_ref Main//basic_import_generic_interface, loc3_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//basic_import_generic_interface, loc3_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.6ec = import_ref Main//basic_import_generic_interface, loc5_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.91b: type = import_ref Main//basic_import_generic_interface, loc5_18, loaded [symbolic = @J.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %Main.import_ref.e01: type = import_ref Main//basic_import_generic_interface, loc5_27, loaded [symbolic = @J.WithSelf.Self.as_type.impls.I.type.require.%I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//basic_import_generic_interface, loc4_14, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.a19ec6.2: @J.%J.type (%J.type.5c8) = import_ref Main//basic_import_generic_interface, loc4_22, loaded [symbolic = @J.%Self (constants.%Self.9f1)]
// CHECK:STDOUT:   %Main.import_ref.ec8 = import_ref Main//basic_import_generic_interface, loc4_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.5: type = import_ref Main//basic_import_generic_interface, loc4_14, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc1_46.1 = import <none>
// CHECK:STDOUT:   %default.import.loc1_46.2 = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %.loc5_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_15: type = converted %.loc5_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%empty_struct_type)> [concrete = constants.%I.type.81d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5: <witness> = impl_self_witness @C.as.I.impl.%C.ref, @I, @I(constants.%empty_struct_type) [concrete = constants.%.b5d]
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: %J.type.23c = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %.loc7_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_15: type = converted %.loc7_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.type: type = facet_type <@J, @J(constants.%empty_struct_type)> [concrete = constants.%J.type.80f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.J.impl.%C.ref, @J, @J(constants.%empty_struct_type) [concrete = constants.%.cc7]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.2: type) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.a67)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.a43
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(imports.%Main.import_ref.b3bc94.5: type) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.5c8)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.5c8) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.9f1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ec8
// CHECK:STDOUT:     extend imports.%Main.import_ref.6ec
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.91b impls imports.%Main.import_ref.e01
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.WithSelf.Self.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.4: type, imports.%Main.import_ref.a19ec6.2: @J.%J.type (%J.type.5c8)) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.5c8)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.as_type.impls.I.type.require.%J.type (%J.type.5c8) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.9f1)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %I.type {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.type
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref as %J.type {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.type
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %Self => constants.%Self.a67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.a67) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.81d
// CHECK:STDOUT:   %Self => constants.%Self.47c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%Self.a67) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.9f1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.9f1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.5c8
// CHECK:STDOUT:   %Self => constants.%Self.9f1
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.80f
// CHECK:STDOUT:   %Self => constants.%Self.043
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%Self.9f1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.81d
// CHECK:STDOUT:   %require_complete => constants.%complete_type.7da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.81d
// CHECK:STDOUT:   %require_complete => constants.%complete_type.7da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %J.type => constants.%J.type.80f
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.81d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.81d
// CHECK:STDOUT:   %require_complete => constants.%complete_type.7da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.3fe: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.191: %I.type.3fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %J.type.28e: type = generic_named_constaint_type @J [concrete]
// CHECK:STDOUT:   %empty_struct: %J.type.28e = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.d68: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.22b: %J.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.22b [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc3_14.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_16.1: type = splice_block %.loc3_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc3_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type.28e = constraint_decl @J [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc3_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc3_14.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc3_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc3_14.1)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Self.loc3_22.2: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc3_22.2 (constants.%Self.191)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_22.1: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc3_22.2 (constants.%Self.191)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @J(%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc4_15.1)> [symbolic = %J.type (constants.%J.type.d68)]
// CHECK:STDOUT:   %Self.loc4_23.2: @J.%J.type (%J.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.22b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc4_23.1: @J.%J.type (%J.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.22b)]
// CHECK:STDOUT:     %J.WithSelf.decl = constraint_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %J.WithSelf.Self.as_type.impls.I.type.require.decl = require_decl @J.WithSelf.Self.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc5_18.1 impls %I.type.loc5_27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc5_18.1: type = facet_access_type @J.%Self.loc4_23.1 [symbolic = %Self.as_type.loc5_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @J.%T.loc4_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc5_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc5: type = specific_constant @J.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc5_27.1, @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.22b) [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_23.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     extend @J.WithSelf.%.loc5
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require @J.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type.loc5_18.1 impls @J.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc5_27.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.WithSelf.Self.as_type.impls.I.type.require(@J.%T.loc4_15.2: type, @J.%Self.loc4_23.1: @J.%J.type (%J.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.d68)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.as_type.impls.I.type.require.%J.type (%J.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.22b)]
// CHECK:STDOUT:   %Self.as_type.loc5_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type.loc5_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.3fe)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc3_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %Self.loc3_22.2 => constants.%Self.191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.191) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.22b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.22b) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.22b
// CHECK:STDOUT:   %Self.as_type.loc5_18.2 => constants.%Self.as_type
// CHECK:STDOUT:   %I.type.loc5_27.2 => constants.%I.type.3fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_constraint.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.type.28e: type = generic_named_constaint_type @J [concrete]
// CHECK:STDOUT:   %empty_struct.f37: %J.type.28e = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.d682d6.1: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.e1f642.1: %J.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.type.3fe: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.a67: %I.type.3fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e1f642.1 [symbolic]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.d682d6.2: type = facet_type <@J, @J(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %I.type.81d: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.47c: %I.type.81d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.7da: <witness> = complete_type_witness %I.type.81d [concrete]
// CHECK:STDOUT:   %.b5d: <witness> = impl_self_witness %C, @I, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.e1f642.2: %J.type.d682d6.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.81d = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//basic_import_generic_constraint, I, unloaded
// CHECK:STDOUT:   %Main.J: %J.type.28e = import_ref Main//basic_import_generic_constraint, J, loaded [concrete = constants.%empty_struct.f37]
// CHECK:STDOUT:   %Main.import_ref.afe = import_ref Main//basic_import_generic_constraint, loc5_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.a43 = import_ref Main//basic_import_generic_constraint, loc3_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//basic_import_generic_constraint, loc3_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.849: type = import_ref Main//basic_import_generic_constraint, loc5_18, loaded [symbolic = @J.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %Main.import_ref.e01: type = import_ref Main//basic_import_generic_constraint, loc5_27, loaded [symbolic = @J.WithSelf.Self.as_type.impls.I.type.require.%I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//basic_import_generic_constraint, loc4_15, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.006c33.2: @J.%J.type (%J.type.d682d6.1) = import_ref Main//basic_import_generic_constraint, loc4_23, loaded [symbolic = @J.%Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Main.import_ref.694 = import_ref Main//basic_import_generic_constraint, loc4_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.5: type = import_ref Main//basic_import_generic_constraint, loc4_15, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc1_47.1 = import <none>
// CHECK:STDOUT:   %default.import.loc1_47.2 = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: %J.type.28e = name_ref J, imports.%Main.J [concrete = constants.%empty_struct.f37]
// CHECK:STDOUT:     %.loc5_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:     %.loc5_15: type = converted %.loc5_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.type: type = facet_type <@J, @J(constants.%empty_struct_type)> [concrete = constants.%J.type.d682d6.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5: <witness> = impl_self_witness @C.as.I.impl.%C.ref, @I, @I(constants.%empty_struct_type) [concrete = constants.%.b5d]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.3: type) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.a67)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.a43
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @J(imports.%Main.import_ref.b3bc94.5: type) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.694
// CHECK:STDOUT:     extend imports.%Main.import_ref.afe
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.849 impls imports.%Main.import_ref.e01
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.WithSelf.Self.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.4: type, imports.%Main.import_ref.006c33.2: @J.%J.type (%J.type.d682d6.1)) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.as_type.impls.I.type.require.%J.type (%J.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %J.type {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %J.type
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %Self => constants.%Self.a67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.a67) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.e1f642.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.e1f642.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.1
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.81d
// CHECK:STDOUT:   %require_complete => constants.%complete_type.7da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.81d
// CHECK:STDOUT:   %Self => constants.%Self.47c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%Self.a67) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %J.type => constants.%J.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.81d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%Self.e1f642.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.81d
// CHECK:STDOUT:   %require_complete => constants.%complete_type.7da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.3fe: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.191: %I.type.3fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %.ece: <witness> = impl_self_witness %C, @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.51e: <witness> = impl_witness @C.as.I.impl.da9.%I.impl_witness_table, @C.as.I.impl.da9(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.45e: <witness> = require_complete_type %I.type.3fe [symbolic]
// CHECK:STDOUT:   %I.facet.77c: %I.type.3fe = facet_value %C, (%I.impl_witness.51e) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.a76: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %.5ab: <witness> = impl_self_witness %C, @I, @I(%ptr) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.7a8: <witness> = impl_witness @C.as.I.impl.982.%I.impl_witness_table, @C.as.I.impl.982(%T) [symbolic]
// CHECK:STDOUT:   %Self.0ef: %I.type.a76 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.078: <witness> = require_complete_type %I.type.a76 [symbolic]
// CHECK:STDOUT:   %I.facet.0b6: %I.type.a76 = facet_value %C, (%I.impl_witness.7a8) [symbolic]
// CHECK:STDOUT:   %N.type.b76: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b76 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d68: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.22b: %N.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.22b [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc5_14.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_16.1: type = splice_block %.loc5_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.da9 [concrete] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc8: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_31.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_31.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:     %.loc8_17.1: type = splice_block %.loc8_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc8: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @C.as.I.impl.da9.%C.ref.loc8, @I, @I(constants.%T) [symbolic = @C.as.I.impl.da9.%.loc8_32 (constants.%.ece)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.da9 [concrete] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc9: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc9 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc9: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_31.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc9: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc9_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = impl_self_witness @C.as.I.impl.da9.%C.ref.loc9, @I, @I(constants.%T) [symbolic = @C.as.I.impl.da9.%.loc8_32 (constants.%.ece)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.982 [concrete] {
// CHECK:STDOUT:     %T.patt.loc12_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_15.1 [symbolic = %T.loc12_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_31.1: type = ptr_type %T.ref [symbolic = %ptr.loc12_31.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc12_32.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc12_32.2 (constants.%I.type.a76)]
// CHECK:STDOUT:     %.loc12_17.1: type = splice_block %.loc12_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc12_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = impl_self_witness @C.as.I.impl.982.%C.ref, @I, @I(constants.%ptr) [symbolic = @C.as.I.impl.982.%.loc12_34 (constants.%.5ab)]
// CHECK:STDOUT:   %N.decl: %N.type.b76 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt.loc14_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_17.1: type = splice_block %.loc14_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc14_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_14.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc5_14.1)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Self.loc5_22.2: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc5_22.2 (constants.%Self.191)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_22.1: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc5_22.2 (constants.%Self.191)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc14_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc14_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc14_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc14_15.1)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self.loc14_23.2: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc14_23.2 (constants.%Self.22b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc14_23.1: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc14_23.2 (constants.%Self.22b)]
// CHECK:STDOUT:     %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.as_type.impls.I.type.require.decl = require_decl @N.WithSelf.Self.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc15_18.1 impls %I.type.loc15_27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc15_18.1: type = facet_access_type @N.%Self.loc14_23.1 [symbolic = %Self.as_type.loc15_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc14_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc15_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc15_27.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc15: type = specific_constant @N.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc15_27.1, @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.22b) [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc14_23.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     extend @N.WithSelf.%.loc15
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require @N.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type.loc15_18.1 impls @N.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc15_27.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.I.type.require(@N.%T.loc14_15.2: type, @N.%Self.loc14_23.1: @N.%N.type (%N.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.I.type.require.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.22b)]
// CHECK:STDOUT:   %Self.as_type.loc15_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc15_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type.loc15_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc15_27.2 (constants.%I.type.3fe)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.da9(%T.loc8_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc8_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc8_31.2: type = facet_type <@I, @I(%T.loc8_15.2)> [symbolic = %I.type.loc8_31.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:   %.loc8_32: <witness> = impl_self_witness constants.%C, @I, @I(%T.loc8_15.2) [symbolic = %.loc8_32 (constants.%.ece)]
// CHECK:STDOUT:   %I.impl_witness.loc8_32.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.da9(%T.loc8_15.2) [symbolic = %I.impl_witness.loc8_32.2 (constants.%I.impl_witness.51e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc8_31.2 [symbolic = %require_complete (constants.%require_complete.45e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref.loc8 as %I.type.loc8_31.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.da9 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc8_32.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.da9(constants.%T) [symbolic = %I.impl_witness.loc8_32.2 (constants.%I.impl_witness.51e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc8_31.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc8_32.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.982(%T.loc12_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc12_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc12_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc12_31.2: type = ptr_type %T.loc12_15.2 [symbolic = %ptr.loc12_31.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc12_32.2: type = facet_type <@I, @I(%ptr.loc12_31.2)> [symbolic = %I.type.loc12_32.2 (constants.%I.type.a76)]
// CHECK:STDOUT:   %.loc12_34: <witness> = impl_self_witness constants.%C, @I, @I(%ptr.loc12_31.2) [symbolic = %.loc12_34 (constants.%.5ab)]
// CHECK:STDOUT:   %I.impl_witness.loc12_34.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.982(%T.loc12_15.2) [symbolic = %I.impl_witness.loc12_34.2 (constants.%I.impl_witness.7a8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc12_32.2 [symbolic = %require_complete (constants.%require_complete.078)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc12_32.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.982 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc12_34.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.982(constants.%T) [symbolic = %I.impl_witness.loc12_34.2 (constants.%I.impl_witness.7a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc12_32.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc12_34.1
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %Self.loc5_22.2 => constants.%Self.191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.191) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.da9(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc8_31.2 => constants.%I.type.3fe
// CHECK:STDOUT:   %.loc8_32 => constants.%.ece
// CHECK:STDOUT:   %I.impl_witness.loc8_32.2 => constants.%I.impl_witness.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%I.facet.77c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.a76
// CHECK:STDOUT:   %Self.loc5_22.2 => constants.%Self.0ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.982(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc12_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc12_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc12_31.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc12_32.2 => constants.%I.type.a76
// CHECK:STDOUT:   %.loc12_34 => constants.%.5ab
// CHECK:STDOUT:   %I.impl_witness.loc12_34.2 => constants.%I.impl_witness.7a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr, constants.%Self.191) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr, constants.%I.facet.0b6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc14_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc14_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.22b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.22b) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.22b
// CHECK:STDOUT:   %Self.as_type.loc15_18.2 => constants.%Self.as_type
// CHECK:STDOUT:   %I.type.loc15_27.2 => constants.%I.type.3fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.3fe: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.a67: %I.type.3fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.ece: <witness> = impl_self_witness %C, @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.66f: <witness> = impl_witness imports.%I.impl_witness_table.5a9, @C.as.I.impl.119(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.45e: <witness> = require_complete_type %I.type.3fe [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.5ab: <witness> = impl_self_witness %C, @I, @I(%ptr) [symbolic]
// CHECK:STDOUT:   %I.type.a76: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %Self.74c: %I.type.a76 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.500: <witness> = impl_witness imports.%I.impl_witness_table.6bd, @C.as.I.impl.238(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.078: <witness> = require_complete_type %I.type.a76 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.type.b76: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b76 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e1f [symbolic]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%ptr)> [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//import_generic, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.I: %I.type.335 = import_ref Main//import_generic, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.N: %N.type.b76 = import_ref Main//import_generic, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.a43 = import_ref Main//import_generic, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//import_generic, loc5_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.3f8 = import_ref Main//import_generic, loc8_32, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//import_generic, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %I.impl_witness_table.5a9 = impl_witness_table (), @C.as.I.impl.119 [concrete]
// CHECK:STDOUT:   %Main.import_ref.335f1c.1: type = import_ref Main//import_generic, loc8_23, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.e015f6.1: type = import_ref Main//import_generic, loc8_31, loaded [symbolic = @C.as.I.impl.119.%I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Main.import_ref.e87 = import_ref Main//import_generic, loc8_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//import_generic, loc8_15, loaded [symbolic = @C.as.I.impl.119.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.766 = import_ref Main//import_generic, loc12_34, unloaded
// CHECK:STDOUT:   %I.impl_witness_table.6bd = impl_witness_table (), @C.as.I.impl.238 [concrete]
// CHECK:STDOUT:   %Main.import_ref.335f1c.2: type = import_ref Main//import_generic, loc12_23, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.a08: type = import_ref Main//import_generic, loc12_32, loaded [symbolic = @C.as.I.impl.238.%I.type (constants.%I.type.a76)]
// CHECK:STDOUT:   %Main.import_ref.891 = import_ref Main//import_generic, loc12_32, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//import_generic, loc12_15, loaded [symbolic = @C.as.I.impl.238.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e24 = import_ref Main//import_generic, loc15_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.849: type = import_ref Main//import_generic, loc15_18, loaded [symbolic = @N.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %Main.import_ref.e015f6.2: type = import_ref Main//import_generic, loc15_27, loaded [symbolic = @N.WithSelf.Self.as_type.impls.I.type.require.%I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.6: type = import_ref Main//import_generic, loc14_15, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.006c33.2: @N.%N.type (%N.type.d682d6.1) = import_ref Main//import_generic, loc14_23, loaded [symbolic = @N.%Self (constants.%Self.e1f)]
// CHECK:STDOUT:   %Main.import_ref.694 = import_ref Main//import_generic, loc14_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.7: type = import_ref Main//import_generic, loc14_15, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .N = imports.%Main.N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_30.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_30.2 = import <none>
// CHECK:STDOUT:   impl_decl @C.as.I.impl.da9cee.1 [concrete] {
// CHECK:STDOUT:     %T.patt.loc6_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_15.1 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc6_31.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc6_31.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:     %.loc6_17.1: type = splice_block %.loc6_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = impl_self_witness @C.as.I.impl.da9cee.1.%C.ref, @I, @I(constants.%T) [symbolic = @C.as.I.impl.da9cee.1.%.loc6_32 (constants.%.ece)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.da9cee.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc10_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_15.1 [symbolic = %T.loc10_15.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc10_31.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc10_31.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:     %.loc10_17.1: type = splice_block %.loc10_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc10_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @C.as.I.impl.da9cee.2.%C.ref, @I, @I(constants.%T) [symbolic = @C.as.I.impl.da9cee.2.%.loc10_33 (constants.%.ece)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.982b7a.1 [concrete] {
// CHECK:STDOUT:     %T.patt.loc14_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_15.1 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc14_31.1: type = ptr_type %T.ref [symbolic = %ptr.loc14_31.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc14_32.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.a76)]
// CHECK:STDOUT:     %.loc14_17.1: type = splice_block %.loc14_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc14_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @C.as.I.impl.982b7a.1.%C.ref, @I, @I(constants.%ptr) [symbolic = @C.as.I.impl.982b7a.1.%.loc14_33 (constants.%.5ab)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.982b7a.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc18_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc18_15.1 [symbolic = %T.loc18_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc18_31.1: type = ptr_type %T.ref [symbolic = %ptr.loc18_31.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc18_32.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc18_32.2 (constants.%I.type.a76)]
// CHECK:STDOUT:     %.loc18_17.1: type = splice_block %.loc18_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc18_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc18_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = impl_self_witness @C.as.I.impl.982b7a.2.%C.ref, @I, @I(constants.%ptr) [symbolic = @C.as.I.impl.982b7a.2.%.loc18_34 (constants.%.5ab)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.042f52.1 [concrete] {
// CHECK:STDOUT:     %T.patt.loc22_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.b76 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22_15.1 [symbolic = %T.loc22_15.2 (constants.%T)]
// CHECK:STDOUT:     %N.type.loc22_31.1: type = facet_type <@N, @N(constants.%T)> [symbolic = %N.type.loc22_31.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:     %.loc22_17.1: type = splice_block %.loc22_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc22_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc22_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22: <witness> = impl_self_witness @C.as.I.impl.042f52.1.%C.ref, @I, @I(constants.%T) [symbolic = @C.as.I.impl.042f52.1.%.loc22_32 (constants.%.ece)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.042f52.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc25_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.b76 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc25_15.1 [symbolic = %T.loc25_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc25_31.1: type = ptr_type %T.ref [symbolic = %ptr.loc25_31.2 (constants.%ptr)]
// CHECK:STDOUT:     %N.type.loc25_32.1: type = facet_type <@N, @N(constants.%ptr)> [symbolic = %N.type.loc25_32.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:     %.loc25_17.1: type = splice_block %.loc25_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc25_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc25_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc25_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25: <witness> = impl_self_witness @C.as.I.impl.042f52.2.%C.ref, @I, @I(constants.%ptr) [symbolic = @C.as.I.impl.042f52.2.%.loc25_33 (constants.%.5ab)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.2: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.a67)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.a43
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(imports.%Main.import_ref.b3bc94.7: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.694
// CHECK:STDOUT:     extend imports.%Main.import_ref.e24
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.849 impls imports.%Main.import_ref.e015f6.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.6: type, imports.%Main.import_ref.006c33.2: @N.%N.type (%N.type.d682d6.1)) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.I.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.119(imports.%Main.import_ref.b3bc94.3: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %.1: <witness> = impl_self_witness constants.%C, @I, @I(%T) [symbolic = %.1 (constants.%.ece)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table.5a9, @C.as.I.impl.119(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.66f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.45e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.335f1c.1 as imports.%Main.import_ref.e015f6.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend imports.%Main.import_ref.e87
// CHECK:STDOUT:     witness = imports.%Main.import_ref.3f8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.238(imports.%Main.import_ref.b3bc94.4: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%ptr)> [symbolic = %I.type (constants.%I.type.a76)]
// CHECK:STDOUT:   %.1: <witness> = impl_self_witness constants.%C, @I, @I(%ptr) [symbolic = %.1 (constants.%.5ab)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table.6bd, @C.as.I.impl.238(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.500)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.078)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.335f1c.2 as imports.%Main.import_ref.a08 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend imports.%Main.import_ref.891
// CHECK:STDOUT:     witness = imports.%Main.import_ref.766
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.da9cee.1(%T.loc6_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc6_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc6_31.2: type = facet_type <@I, @I(%T.loc6_15.2)> [symbolic = %I.type.loc6_31.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:   %.loc6_32: <witness> = impl_self_witness constants.%C, @I, @I(%T.loc6_15.2) [symbolic = %.loc6_32 (constants.%.ece)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc6_31.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.da9cee.2(%T.loc10_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc10_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc10_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc10_31.2: type = facet_type <@I, @I(%T.loc10_15.2)> [symbolic = %I.type.loc10_31.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:   %.loc10_33: <witness> = impl_self_witness constants.%C, @I, @I(%T.loc10_15.2) [symbolic = %.loc10_33 (constants.%.ece)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc10_31.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc10_31.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.982b7a.1(%T.loc14_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc14_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc14_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc14_31.2: type = ptr_type %T.loc14_15.2 [symbolic = %ptr.loc14_31.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc14_32.2: type = facet_type <@I, @I(%ptr.loc14_31.2)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.a76)]
// CHECK:STDOUT:   %.loc14_33: <witness> = impl_self_witness constants.%C, @I, @I(%ptr.loc14_31.2) [symbolic = %.loc14_33 (constants.%.5ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc14_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.982b7a.2(%T.loc18_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc18_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc18_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc18_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc18_31.2: type = ptr_type %T.loc18_15.2 [symbolic = %ptr.loc18_31.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc18_32.2: type = facet_type <@I, @I(%ptr.loc18_31.2)> [symbolic = %I.type.loc18_32.2 (constants.%I.type.a76)]
// CHECK:STDOUT:   %.loc18_34: <witness> = impl_self_witness constants.%C, @I, @I(%ptr.loc18_31.2) [symbolic = %.loc18_34 (constants.%.5ab)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc18_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc18_32.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.042f52.1(%T.loc22_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc22_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc22_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc22_15.2 (constants.%T)]
// CHECK:STDOUT:   %N.type.loc22_31.2: type = facet_type <@N, @N(%T.loc22_15.2)> [symbolic = %N.type.loc22_31.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %.loc22_32: <witness> = impl_self_witness constants.%C, @I, @I(%T.loc22_15.2) [symbolic = %.loc22_32 (constants.%.ece)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc22_31.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.042f52.2(%T.loc25_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc25_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc25_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc25_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc25_31.2: type = ptr_type %T.loc25_15.2 [symbolic = %ptr.loc25_31.2 (constants.%ptr)]
// CHECK:STDOUT:   %N.type.loc25_32.2: type = facet_type <@N, @N(%ptr.loc25_31.2)> [symbolic = %N.type.loc25_32.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:   %.loc25_33: <witness> = impl_self_witness constants.%C, @I, @I(%ptr.loc25_31.2) [symbolic = %.loc25_33 (constants.%.5ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc25_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "import_generic.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %Self => constants.%Self.a67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.a67) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.119(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %.1 => constants.%.ece
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.66f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.a76
// CHECK:STDOUT:   %Self => constants.%Self.74c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.238(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.a76
// CHECK:STDOUT:   %.1 => constants.%.5ab
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.500
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.da9cee.1(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_15.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc6_31.2 => constants.%I.type.3fe
// CHECK:STDOUT:   %.loc6_32 => constants.%.ece
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.da9cee.2(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc10_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc10_15.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc10_31.2 => constants.%I.type.3fe
// CHECK:STDOUT:   %.loc10_33 => constants.%.ece
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.982b7a.1(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc14_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc14_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc14_31.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc14_32.2 => constants.%I.type.a76
// CHECK:STDOUT:   %.loc14_33 => constants.%.5ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.982b7a.2(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc18_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc18_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc18_31.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc18_32.2 => constants.%I.type.a76
// CHECK:STDOUT:   %.loc18_34 => constants.%.5ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.e1f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.e1f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.e1f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %require_complete => constants.%require_complete.45e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.042f52.1(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc22_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc22_15.2 => constants.%T
// CHECK:STDOUT:   %N.type.loc22_31.2 => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %.loc22_32 => constants.%.ece
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.a76
// CHECK:STDOUT:   %require_complete => constants.%require_complete.078
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%ptr, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.a76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.042f52.2(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc25_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc25_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc25_31.2 => constants.%ptr
// CHECK:STDOUT:   %N.type.loc25_32.2 => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %.loc25_33 => constants.%.5ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_generic_with_different_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.3fe: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.191: %I.type.3fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %.ece: <witness> = impl_self_witness %C, @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.3fe [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.3fe = facet_value %C, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %N.type.b76: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b76 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d68: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.22b: %N.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.22b [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc5_14.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_16.1: type = splice_block %.loc5_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc7_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_15.1 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc7_31.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc7_31.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:     %.loc7_17.1: type = splice_block %.loc7_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C.as.I.impl.%C.ref, @I, @I(constants.%T) [symbolic = @C.as.I.impl.%.loc7_33 (constants.%.ece)]
// CHECK:STDOUT:   %N.decl: %N.type.b76 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt.loc9_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc9_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_14.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc5_14.1)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Self.loc5_22.2: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc5_22.2 (constants.%Self.191)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_22.1: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc5_22.2 (constants.%Self.191)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc9_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc9_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc9_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc9_15.1)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self.loc9_23.2: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc9_23.2 (constants.%Self.22b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc9_23.1: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc9_23.2 (constants.%Self.22b)]
// CHECK:STDOUT:     %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.as_type.impls.I.type.require.decl = require_decl @N.WithSelf.Self.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc10_18.1 impls %I.type.loc10_27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc10_18.1: type = facet_access_type @N.%Self.loc9_23.1 [symbolic = %Self.as_type.loc10_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc9_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc10_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc10_27.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10: type = specific_constant @N.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc10_27.1, @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.22b) [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc9_23.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     extend @N.WithSelf.%.loc10
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require @N.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type.loc10_18.1 impls @N.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc10_27.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.I.type.require(@N.%T.loc9_15.2: type, @N.%Self.loc9_23.1: @N.%N.type (%N.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.I.type.require.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.22b)]
// CHECK:STDOUT:   %Self.as_type.loc10_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type.loc10_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc10_27.2 (constants.%I.type.3fe)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc7_31.2: type = facet_type <@I, @I(%T.loc7_15.2)> [symbolic = %I.type.loc7_31.2 (constants.%I.type.3fe)]
// CHECK:STDOUT:   %.loc7_33: <witness> = impl_self_witness constants.%C, @I, @I(%T.loc7_15.2) [symbolic = %.loc7_33 (constants.%.ece)]
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%T.loc7_15.2) [symbolic = %I.impl_witness.loc7_33.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_31.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_31.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_33.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T) [symbolic = %I.impl_witness.loc7_33.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc7_31.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_33.1
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %Self.loc5_22.2 => constants.%Self.191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.191) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_15.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc7_31.2 => constants.%I.type.3fe
// CHECK:STDOUT:   %.loc7_33 => constants.%.ece
// CHECK:STDOUT:   %I.impl_witness.loc7_33.2 => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc9_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc9_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.22b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.22b) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.22b
// CHECK:STDOUT:   %Self.as_type.loc10_18.2 => constants.%Self.as_type
// CHECK:STDOUT:   %I.type.loc10_27.2 => constants.%I.type.3fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_generic_with_different_specific.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.3fe: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.a67: %I.type.3fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.ece: <witness> = impl_self_witness %C, @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.66f: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl.119(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.45e: <witness> = require_complete_type %I.type.3fe [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.a76: type = facet_type <@I, @I(%ptr.e8f)> [symbolic]
// CHECK:STDOUT:   %.5ab: <witness> = impl_self_witness %C, @I, @I(%ptr.e8f) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.7a8: <witness> = impl_witness @C.as.I.impl.982.%I.impl_witness_table, @C.as.I.impl.982(%T) [symbolic]
// CHECK:STDOUT:   %Self.74c: %I.type.a76 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.078: <witness> = require_complete_type %I.type.a76 [symbolic]
// CHECK:STDOUT:   %I.facet.0b6: %I.type.a76 = facet_value %C, (%I.impl_witness.7a8) [symbolic]
// CHECK:STDOUT:   %N.type.b76: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b76 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f642.1: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e1f642.1 [symbolic]
// CHECK:STDOUT:   %ptr.125: type = ptr_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%ptr.125)> [symbolic]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %I.type.0a0: type = facet_type <@I, @I(%ptr.125)> [symbolic]
// CHECK:STDOUT:   %require_complete.6fb: <witness> = require_complete_type %I.type.0a0 [symbolic]
// CHECK:STDOUT:   %.d9e: <witness> = impl_self_witness %C, @I, @I(%ptr.125) [symbolic]
// CHECK:STDOUT:   %I.impl_witness.6e9: <witness> = impl_witness @C.as.I.impl.042.%I.impl_witness_table, @C.as.I.impl.042(%T) [symbolic]
// CHECK:STDOUT:   %Self.e1f642.2: %N.type.d682d6.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.533: <witness> = require_complete_type %N.type.d682d6.2 [symbolic]
// CHECK:STDOUT:   %I.facet.e71: %I.type.0a0 = facet_value %C, (%I.impl_witness.6e9) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//import_generic_with_different_specific, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.I: %I.type.335 = import_ref Main//import_generic_with_different_specific, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.N: %N.type.b76 = import_ref Main//import_generic_with_different_specific, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.a43 = import_ref Main//import_generic_with_different_specific, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//import_generic_with_different_specific, loc5_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.3f8 = import_ref Main//import_generic_with_different_specific, loc7_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic_with_different_specific, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//import_generic_with_different_specific, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.119 [concrete]
// CHECK:STDOUT:   %Main.import_ref.335: type = import_ref Main//import_generic_with_different_specific, loc7_23, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.e015f6.1: type = import_ref Main//import_generic_with_different_specific, loc7_31, loaded [symbolic = @C.as.I.impl.119.%I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Main.import_ref.e87 = import_ref Main//import_generic_with_different_specific, loc7_31, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//import_generic_with_different_specific, loc7_15, loaded [symbolic = @C.as.I.impl.119.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e24 = import_ref Main//import_generic_with_different_specific, loc10_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.849: type = import_ref Main//import_generic_with_different_specific, loc10_18, loaded [symbolic = @N.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %Main.import_ref.e015f6.2: type = import_ref Main//import_generic_with_different_specific, loc10_27, loaded [symbolic = @N.WithSelf.Self.as_type.impls.I.type.require.%I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.5: type = import_ref Main//import_generic_with_different_specific, loc9_15, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.006c33.2: @N.%N.type (%N.type.d682d6.1) = import_ref Main//import_generic_with_different_specific, loc9_23, loaded [symbolic = @N.%Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Main.import_ref.694 = import_ref Main//import_generic_with_different_specific, loc9_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.6: type = import_ref Main//import_generic_with_different_specific, loc9_15, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .N = imports.%Main.N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc1_54.1 = import <none>
// CHECK:STDOUT:   %default.import.loc1_54.2 = import <none>
// CHECK:STDOUT:   impl_decl @C.as.I.impl.982 [concrete] {
// CHECK:STDOUT:     %T.patt.loc5_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_15.1 [symbolic = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc5_31.1: type = ptr_type %T.ref [symbolic = %ptr.loc5_31.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %I.type.loc5_32.1: type = facet_type <@I, @I(constants.%ptr.e8f)> [symbolic = %I.type.loc5_32.2 (constants.%I.type.a76)]
// CHECK:STDOUT:     %.loc5_17.1: type = splice_block %.loc5_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5: <witness> = impl_self_witness @C.as.I.impl.982.%C.ref, @I, @I(constants.%ptr.e8f) [symbolic = @C.as.I.impl.982.%.loc5_34 (constants.%.5ab)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.042 [concrete] {
// CHECK:STDOUT:     %T.patt.loc6_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.b76 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_15.1 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc6_31.1: type = ptr_type %T.ref [symbolic = %ptr.loc6_31.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %ptr.loc6_32.1: type = ptr_type %ptr.loc6_31.1 [symbolic = %ptr.loc6_32.2 (constants.%ptr.125)]
// CHECK:STDOUT:     %N.type.loc6_33.1: type = facet_type <@N, @N(constants.%ptr.125)> [symbolic = %N.type.loc6_33.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:     %.loc6_17.1: type = splice_block %.loc6_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = impl_self_witness @C.as.I.impl.042.%C.ref, @I, @I(constants.%ptr.125) [symbolic = @C.as.I.impl.042.%.loc6_35 (constants.%.d9e)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.2: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.a67)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.a43
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(imports.%Main.import_ref.b3bc94.6: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.694
// CHECK:STDOUT:     extend imports.%Main.import_ref.e24
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.849 impls imports.%Main.import_ref.e015f6.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.5: type, imports.%Main.import_ref.006c33.2: @N.%N.type (%N.type.d682d6.1)) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.I.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.119(imports.%Main.import_ref.b3bc94.3: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %.1: <witness> = impl_self_witness constants.%C, @I, @I(%T) [symbolic = %.1 (constants.%.ece)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl.119(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.66f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.45e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.335 as imports.%Main.import_ref.e015f6.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend imports.%Main.import_ref.e87
// CHECK:STDOUT:     witness = imports.%Main.import_ref.3f8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.982(%T.loc5_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc5_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc5_31.2: type = ptr_type %T.loc5_15.2 [symbolic = %ptr.loc5_31.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %I.type.loc5_32.2: type = facet_type <@I, @I(%ptr.loc5_31.2)> [symbolic = %I.type.loc5_32.2 (constants.%I.type.a76)]
// CHECK:STDOUT:   %.loc5_34: <witness> = impl_self_witness constants.%C, @I, @I(%ptr.loc5_31.2) [symbolic = %.loc5_34 (constants.%.5ab)]
// CHECK:STDOUT:   %I.impl_witness.loc5_34.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.982(%T.loc5_15.2) [symbolic = %I.impl_witness.loc5_34.2 (constants.%I.impl_witness.7a8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc5_32.2 [symbolic = %require_complete (constants.%require_complete.078)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc5_32.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.982 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc5_34.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.982(constants.%T) [symbolic = %I.impl_witness.loc5_34.2 (constants.%I.impl_witness.7a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc5_32.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc5_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.042(%T.loc6_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc6_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc6_31.2: type = ptr_type %T.loc6_15.2 [symbolic = %ptr.loc6_31.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %ptr.loc6_32.2: type = ptr_type %ptr.loc6_31.2 [symbolic = %ptr.loc6_32.2 (constants.%ptr.125)]
// CHECK:STDOUT:   %N.type.loc6_33.2: type = facet_type <@N, @N(%ptr.loc6_32.2)> [symbolic = %N.type.loc6_33.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:   %.loc6_35: <witness> = impl_self_witness constants.%C, @I, @I(%ptr.loc6_32.2) [symbolic = %.loc6_35 (constants.%.d9e)]
// CHECK:STDOUT:   %I.impl_witness.loc6_35.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.042(%T.loc6_15.2) [symbolic = %I.impl_witness.loc6_35.2 (constants.%I.impl_witness.6e9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %N.type.loc6_33.2 [symbolic = %require_complete (constants.%require_complete.533)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc6_33.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.042 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc6_35.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.042(constants.%T) [symbolic = %I.impl_witness.loc6_35.2 (constants.%I.impl_witness.6e9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %N.type.loc6_33.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc6_35.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %Self => constants.%Self.a67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.a67) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.119(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT:   %.1 => constants.%.ece
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.66f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr.e8f) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%ptr.e8f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.a76
// CHECK:STDOUT:   %Self => constants.%Self.74c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.982(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc5_31.2 => constants.%ptr.e8f
// CHECK:STDOUT:   %I.type.loc5_32.2 => constants.%I.type.a76
// CHECK:STDOUT:   %.loc5_34 => constants.%.5ab
// CHECK:STDOUT:   %I.impl_witness.loc5_34.2 => constants.%I.impl_witness.7a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr.e8f, constants.%Self.a67) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr.e8f, constants.%I.facet.0b6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.e1f642.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.e1f642.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.1
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.3fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr.125) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr.125, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT:   %I.type => constants.%I.type.0a0
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6fb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr.125) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%ptr.125, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.0a0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.042(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc6_31.2 => constants.%ptr.e8f
// CHECK:STDOUT:   %ptr.loc6_32.2 => constants.%ptr.125
// CHECK:STDOUT:   %N.type.loc6_33.2 => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %.loc6_35 => constants.%.d9e
// CHECK:STDOUT:   %I.impl_witness.loc6_35.2 => constants.%I.impl_witness.6e9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr.125, constants.%Self.e1f642.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT:   %I.type => constants.%I.type.0a0
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6fb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr.125, constants.%I.facet.e71) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.23c: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.23c = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.5c8: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.ad7: %J.type.5c8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %N.type.b76: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b76 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d68: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.22b: %N.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.22b [symbolic]
// CHECK:STDOUT:   %.1a5: <witness> = impl_self_witness %D, @J, @J(%T) [symbolic]
// CHECK:STDOUT:   %J.impl_witness.669: <witness> = impl_witness @D.as.J.impl.5a5.%J.impl_witness_table, @D.as.J.impl.5a5(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.c71: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %.7a3: <witness> = impl_self_witness %D, @J, @J(%ptr) [symbolic]
// CHECK:STDOUT:   %J.impl_witness.ee0: <witness> = impl_witness @D.as.J.impl.101.%J.impl_witness_table, @D.as.J.impl.101(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %J.decl: %J.type.23c = interface_decl @J [concrete = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt.loc5_14.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_16.1: type = splice_block %.loc5_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: %N.type.b76 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt.loc7_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_17.1: type = splice_block %.loc7_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.5a5 [concrete] {
// CHECK:STDOUT:     %T.patt.loc13_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.23c = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc13_15.1 [symbolic = %T.loc13_15.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc13_31.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc13_31.2 (constants.%J.type.5c8)]
// CHECK:STDOUT:     %.loc13_17.1: type = splice_block %.loc13_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc13_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc13_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @D.as.J.impl.5a5.%D.ref, @J, @J(constants.%T) [symbolic = @D.as.J.impl.5a5.%.loc13_32 (constants.%.1a5)]
// CHECK:STDOUT:   impl_decl @D.as.J.impl.101 [concrete] {
// CHECK:STDOUT:     %T.patt.loc17_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.23c = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc17_15.1 [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc17_31.1: type = ptr_type %T.ref [symbolic = %ptr.loc17_31.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc17_32.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc17_32.2 (constants.%J.type.c71)]
// CHECK:STDOUT:     %.loc17_17.1: type = splice_block %.loc17_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc17_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = impl_self_witness @D.as.J.impl.101.%D.ref, @J, @J(constants.%ptr) [symbolic = @D.as.J.impl.101.%.loc17_33 (constants.%.7a3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_14.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc5_14.1)> [symbolic = %J.type (constants.%J.type.5c8)]
// CHECK:STDOUT:   %Self.loc5_22.2: @J.%J.type (%J.type.5c8) = symbolic_binding Self, 1 [symbolic = %Self.loc5_22.2 (constants.%Self.ad7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_22.1: @J.%J.type (%J.type.5c8) = symbolic_binding Self, 1 [symbolic = %Self.loc5_22.2 (constants.%Self.ad7)]
// CHECK:STDOUT:     %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc7_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc7_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc7_15.1)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self.loc7_23.2: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc7_23.2 (constants.%Self.22b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc7_23.1: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc7_23.2 (constants.%Self.22b)]
// CHECK:STDOUT:     %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.as_type.impls.J.type.require.decl = require_decl @N.WithSelf.Self.as_type.impls.J.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc8_18.1 impls %J.type.loc8_27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc8_18.1: type = facet_access_type @N.%Self.loc7_23.1 [symbolic = %Self.as_type.loc8_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %J.ref: %J.type.23c = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc7_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %J.type.loc8_27.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc8_27.2 (constants.%J.type.5c8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc8: type = specific_constant @N.WithSelf.Self.as_type.impls.J.type.require.%J.type.loc8_27.1, @N.WithSelf.Self.as_type.impls.J.type.require(constants.%T, constants.%Self.22b) [symbolic = %J.type (constants.%J.type.5c8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc7_23.1
// CHECK:STDOUT:     .J = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .J = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     extend @N.WithSelf.%.loc8
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.J.type.require {
// CHECK:STDOUT:       require @N.WithSelf.Self.as_type.impls.J.type.require.%Self.as_type.loc8_18.1 impls @N.WithSelf.Self.as_type.impls.J.type.require.%J.type.loc8_27.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.J.type.require(@N.%T.loc7_15.2: type, @N.%Self.loc7_23.1: @N.%N.type (%N.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.J.type.require.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.22b)]
// CHECK:STDOUT:   %Self.as_type.loc8_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %J.type.loc8_27.2: type = facet_type <@J, @J(%T)> [symbolic = %J.type.loc8_27.2 (constants.%J.type.5c8)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.5a5(%T.loc13_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc13_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc13_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc13_15.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc13_31.2: type = facet_type <@J, @J(%T.loc13_15.2)> [symbolic = %J.type.loc13_31.2 (constants.%J.type.5c8)]
// CHECK:STDOUT:   %.loc13_32: <witness> = impl_self_witness constants.%D, @J, @J(%T.loc13_15.2) [symbolic = %.loc13_32 (constants.%.1a5)]
// CHECK:STDOUT:   %J.impl_witness.loc13_32.2: <witness> = impl_witness %J.impl_witness_table, @D.as.J.impl.5a5(%T.loc13_15.2) [symbolic = %J.impl_witness.loc13_32.2 (constants.%J.impl_witness.669)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc13_31.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.101(%T.loc17_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc17_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc17_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc17_31.2: type = ptr_type %T.loc17_15.2 [symbolic = %ptr.loc17_31.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc17_32.2: type = facet_type <@J, @J(%ptr.loc17_31.2)> [symbolic = %J.type.loc17_32.2 (constants.%J.type.c71)]
// CHECK:STDOUT:   %.loc17_33: <witness> = impl_self_witness constants.%D, @J, @J(%ptr.loc17_31.2) [symbolic = %.loc17_33 (constants.%.7a3)]
// CHECK:STDOUT:   %J.impl_witness.loc17_33.2: <witness> = impl_witness %J.impl_witness_table, @D.as.J.impl.101(%T.loc17_15.2) [symbolic = %J.impl_witness.loc17_33.2 (constants.%J.impl_witness.ee0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc17_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.5c8
// CHECK:STDOUT:   %Self.loc5_22.2 => constants.%Self.ad7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.ad7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.22b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.J.type.require(constants.%T, constants.%Self.22b) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.22b
// CHECK:STDOUT:   %Self.as_type.loc8_18.2 => constants.%Self.as_type
// CHECK:STDOUT:   %J.type.loc8_27.2 => constants.%J.type.5c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.5a5(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc13_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc13_15.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc13_31.2 => constants.%J.type.5c8
// CHECK:STDOUT:   %.loc13_32 => constants.%.1a5
// CHECK:STDOUT:   %J.impl_witness.loc13_32.2 => constants.%J.impl_witness.669
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.101(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc17_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc17_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc17_31.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc17_32.2 => constants.%J.type.c71
// CHECK:STDOUT:   %.loc17_33 => constants.%.7a3
// CHECK:STDOUT:   %J.impl_witness.loc17_33.2 => constants.%J.impl_witness.ee0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.23c: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.23c = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.5c8: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.9f1: %J.type.5c8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.1a5: <witness> = impl_self_witness %D, @J, @J(%T) [symbolic]
// CHECK:STDOUT:   %J.impl_witness.cb2: <witness> = impl_witness imports.%J.impl_witness_table.494, @D.as.J.impl.bdb(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.7a3: <witness> = impl_self_witness %D, @J, @J(%ptr) [symbolic]
// CHECK:STDOUT:   %J.type.c71: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %J.impl_witness.90a: <witness> = impl_witness imports.%J.impl_witness_table.dd7, @D.as.J.impl.fa6(%T) [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.type.b76: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b76 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.6ba: <witness> = require_complete_type %J.type.5c8 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e1f [symbolic]
// CHECK:STDOUT:   %D.type.facet: %type = facet_value %D, () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.769: <witness> = require_complete_type %J.type.c71 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//import_generic_decl, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.J: %J.type.23c = import_ref Main//import_generic_decl, J, loaded [concrete = constants.%J.generic]
// CHECK:STDOUT:   %Main.N: %N.type.b76 = import_ref Main//import_generic_decl, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.ec8 = import_ref Main//import_generic_decl, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//import_generic_decl, loc5_14, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic_decl, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.54a = import_ref Main//import_generic_decl, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %J.impl_witness_table.494 = impl_witness_table (), @D.as.J.impl.bdb [concrete]
// CHECK:STDOUT:   %Main.import_ref.2e2cf0.1: type = import_ref Main//import_generic_decl, loc13_23, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.1936e1.1: type = import_ref Main//import_generic_decl, loc13_31, loaded [symbolic = @D.as.J.impl.bdb.%J.type (constants.%J.type.5c8)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//import_generic_decl, loc13_15, loaded [symbolic = @D.as.J.impl.bdb.%T (constants.%T)]
// CHECK:STDOUT:   %J.impl_witness_table.dd7 = impl_witness_table (), @D.as.J.impl.fa6 [concrete]
// CHECK:STDOUT:   %Main.import_ref.2e2cf0.2: type = import_ref Main//import_generic_decl, loc17_23, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.5a0: type = import_ref Main//import_generic_decl, loc17_32, loaded [symbolic = @D.as.J.impl.fa6.%J.type (constants.%J.type.c71)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//import_generic_decl, loc17_15, loaded [symbolic = @D.as.J.impl.fa6.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.b19 = import_ref Main//import_generic_decl, loc8_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.849: type = import_ref Main//import_generic_decl, loc8_18, loaded [symbolic = @N.WithSelf.Self.as_type.impls.J.type.require.%Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %Main.import_ref.1936e1.2: type = import_ref Main//import_generic_decl, loc8_27, loaded [symbolic = @N.WithSelf.Self.as_type.impls.J.type.require.%J.type (constants.%J.type.5c8)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.6: type = import_ref Main//import_generic_decl, loc7_15, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.006c33.2: @N.%N.type (%N.type.d682d6.1) = import_ref Main//import_generic_decl, loc7_23, loaded [symbolic = @N.%Self (constants.%Self.e1f)]
// CHECK:STDOUT:   %Main.import_ref.694 = import_ref Main//import_generic_decl, loc7_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.7: type = import_ref Main//import_generic_decl, loc7_15, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:     .N = imports.%Main.N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_35.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_35.2 = import <none>
// CHECK:STDOUT:   impl_decl @D.as.J.impl.5a5643.1 [concrete] {
// CHECK:STDOUT:     %T.patt.loc6_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.23c = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_15.1 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc6_31.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc6_31.2 (constants.%J.type.5c8)]
// CHECK:STDOUT:     %.loc6_17.1: type = splice_block %.loc6_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = impl_self_witness @D.as.J.impl.5a5643.1.%D.ref, @J, @J(constants.%T) [symbolic = @D.as.J.impl.5a5643.1.%.loc6_32 (constants.%.1a5)]
// CHECK:STDOUT:   impl_decl @D.as.J.impl.5a5643.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc10_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.23c = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc10_15.1 [symbolic = %T.loc10_15.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc10_31.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc10_31.2 (constants.%J.type.5c8)]
// CHECK:STDOUT:     %.loc10_17.1: type = splice_block %.loc10_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc10_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc10_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @D.as.J.impl.5a5643.2.%D.ref, @J, @J(constants.%T) [symbolic = @D.as.J.impl.5a5643.2.%.loc10_33 (constants.%.1a5)]
// CHECK:STDOUT:   impl_decl @D.as.J.impl.1010f3.1 [concrete] {
// CHECK:STDOUT:     %T.patt.loc14_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.23c = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_15.1 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc14_31.1: type = ptr_type %T.ref [symbolic = %ptr.loc14_31.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc14_32.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.c71)]
// CHECK:STDOUT:     %.loc14_17.1: type = splice_block %.loc14_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc14_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @D.as.J.impl.1010f3.1.%D.ref, @J, @J(constants.%ptr) [symbolic = @D.as.J.impl.1010f3.1.%.loc14_33 (constants.%.7a3)]
// CHECK:STDOUT:   impl_decl @D.as.J.impl.1010f3.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc18_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.23c = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc18_15.1 [symbolic = %T.loc18_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc18_31.1: type = ptr_type %T.ref [symbolic = %ptr.loc18_31.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc18_32.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc18_32.2 (constants.%J.type.c71)]
// CHECK:STDOUT:     %.loc18_17.1: type = splice_block %.loc18_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc18_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc18_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = impl_self_witness @D.as.J.impl.1010f3.2.%D.ref, @J, @J(constants.%ptr) [symbolic = @D.as.J.impl.1010f3.2.%.loc18_34 (constants.%.7a3)]
// CHECK:STDOUT:   impl_decl @D.as.J.impl.d69d6f.1 [concrete] {
// CHECK:STDOUT:     %T.patt.loc22_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %N.ref: %N.type.b76 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22_15.1 [symbolic = %T.loc22_15.2 (constants.%T)]
// CHECK:STDOUT:     %N.type.loc22_31.1: type = facet_type <@N, @N(constants.%T)> [symbolic = %N.type.loc22_31.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:     %.loc22_17.1: type = splice_block %.loc22_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc22_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc22_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22: <witness> = impl_self_witness @D.as.J.impl.d69d6f.1.%D.ref, @J, @J(constants.%T) [symbolic = @D.as.J.impl.d69d6f.1.%.loc22_33 (constants.%.1a5)]
// CHECK:STDOUT:   impl_decl @D.as.J.impl.d69d6f.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc26_15.1: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %N.ref: %N.type.b76 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_15.1 [symbolic = %T.loc26_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc26_31.1: type = ptr_type %T.ref [symbolic = %ptr.loc26_31.2 (constants.%ptr)]
// CHECK:STDOUT:     %N.type.loc26_32.1: type = facet_type <@N, @N(constants.%ptr)> [symbolic = %N.type.loc26_32.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:     %.loc26_17.1: type = splice_block %.loc26_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc26_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc26_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26: <witness> = impl_self_witness @D.as.J.impl.d69d6f.2.%D.ref, @J, @J(constants.%ptr) [symbolic = @D.as.J.impl.d69d6f.2.%.loc26_34 (constants.%.7a3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(imports.%Main.import_ref.b3bc94.2: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.5c8)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.5c8) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.9f1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ec8
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(imports.%Main.import_ref.b3bc94.7: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.694
// CHECK:STDOUT:     extend imports.%Main.import_ref.b19
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.J.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.849 impls imports.%Main.import_ref.1936e1.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.J.type.require(imports.%Main.import_ref.b3bc94.6: type, imports.%Main.import_ref.006c33.2: @N.%N.type (%N.type.d682d6.1)) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.J.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.5c8)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.bdb(imports.%Main.import_ref.b3bc94.3: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.5c8)]
// CHECK:STDOUT:   %.1: <witness> = impl_self_witness constants.%D, @J, @J(%T) [symbolic = %.1 (constants.%.1a5)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness imports.%J.impl_witness_table.494, @D.as.J.impl.bdb(%T) [symbolic = %J.impl_witness (constants.%J.impl_witness.cb2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.2e2cf0.1 as imports.%Main.import_ref.1936e1.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.fa6(imports.%Main.import_ref.b3bc94.4: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%ptr)> [symbolic = %J.type (constants.%J.type.c71)]
// CHECK:STDOUT:   %.1: <witness> = impl_self_witness constants.%D, @J, @J(%ptr) [symbolic = %.1 (constants.%.7a3)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness imports.%J.impl_witness_table.dd7, @D.as.J.impl.fa6(%T) [symbolic = %J.impl_witness (constants.%J.impl_witness.90a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.2e2cf0.2 as imports.%Main.import_ref.5a0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.5a5643.1(%T.loc6_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc6_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc6_31.2: type = facet_type <@J, @J(%T.loc6_15.2)> [symbolic = %J.type.loc6_31.2 (constants.%J.type.5c8)]
// CHECK:STDOUT:   %.loc6_32: <witness> = impl_self_witness constants.%D, @J, @J(%T.loc6_15.2) [symbolic = %.loc6_32 (constants.%.1a5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc6_31.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.5a5643.2(%T.loc10_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc10_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc10_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc10_15.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc10_31.2: type = facet_type <@J, @J(%T.loc10_15.2)> [symbolic = %J.type.loc10_31.2 (constants.%J.type.5c8)]
// CHECK:STDOUT:   %.loc10_33: <witness> = impl_self_witness constants.%D, @J, @J(%T.loc10_15.2) [symbolic = %.loc10_33 (constants.%.1a5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc10_31.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %J.type.loc10_31.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.1010f3.1(%T.loc14_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc14_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc14_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc14_31.2: type = ptr_type %T.loc14_15.2 [symbolic = %ptr.loc14_31.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc14_32.2: type = facet_type <@J, @J(%ptr.loc14_31.2)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.c71)]
// CHECK:STDOUT:   %.loc14_33: <witness> = impl_self_witness constants.%D, @J, @J(%ptr.loc14_31.2) [symbolic = %.loc14_33 (constants.%.7a3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc14_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.1010f3.2(%T.loc18_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc18_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc18_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc18_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc18_31.2: type = ptr_type %T.loc18_15.2 [symbolic = %ptr.loc18_31.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc18_32.2: type = facet_type <@J, @J(%ptr.loc18_31.2)> [symbolic = %J.type.loc18_32.2 (constants.%J.type.c71)]
// CHECK:STDOUT:   %.loc18_34: <witness> = impl_self_witness constants.%D, @J, @J(%ptr.loc18_31.2) [symbolic = %.loc18_34 (constants.%.7a3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc18_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %J.type.loc18_32.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.d69d6f.1(%T.loc22_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc22_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc22_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc22_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc22_15.2 (constants.%T)]
// CHECK:STDOUT:   %N.type.loc22_31.2: type = facet_type <@N, @N(%T.loc22_15.2)> [symbolic = %N.type.loc22_31.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %.loc22_33: <witness> = impl_self_witness constants.%D, @J, @J(%T.loc22_15.2) [symbolic = %.loc22_33 (constants.%.1a5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %N.type.loc22_31.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %N.type.loc22_31.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.d69d6f.2(%T.loc26_15.1: type) {
// CHECK:STDOUT:   %T.patt.loc26_15.2: %pattern_type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc26_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc26_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc26_31.2: type = ptr_type %T.loc26_15.2 [symbolic = %ptr.loc26_31.2 (constants.%ptr)]
// CHECK:STDOUT:   %N.type.loc26_32.2: type = facet_type <@N, @N(%ptr.loc26_31.2)> [symbolic = %N.type.loc26_32.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:   %.loc26_34: <witness> = impl_self_witness constants.%D, @J, @J(%ptr.loc26_31.2) [symbolic = %.loc26_34 (constants.%.7a3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %N.type.loc26_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %N.type.loc26_32.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.54a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.5c8
// CHECK:STDOUT:   %Self => constants.%Self.9f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.9f1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.bdb(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.5c8
// CHECK:STDOUT:   %.1 => constants.%.1a5
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.cb2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.fa6(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.c71
// CHECK:STDOUT:   %.1 => constants.%.7a3
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.90a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.5a5643.1(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_15.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc6_31.2 => constants.%J.type.5c8
// CHECK:STDOUT:   %.loc6_32 => constants.%.1a5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.5a5643.2(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc10_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc10_15.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc10_31.2 => constants.%J.type.5c8
// CHECK:STDOUT:   %.loc10_33 => constants.%.1a5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.1010f3.1(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc14_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc14_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc14_31.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc14_32.2 => constants.%J.type.c71
// CHECK:STDOUT:   %.loc14_33 => constants.%.7a3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.1010f3.2(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc18_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc18_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc18_31.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc18_32.2 => constants.%J.type.c71
// CHECK:STDOUT:   %.loc18_34 => constants.%.7a3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.e1f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.J.type.require(constants.%T, constants.%Self.e1f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.e1f
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %J.type => constants.%J.type.5c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%D.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.5c8
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.J.type.require(constants.%T, constants.%D.type.facet) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%D.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%D
// CHECK:STDOUT:   %J.type => constants.%J.type.5c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.d69d6f.1(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc22_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc22_15.2 => constants.%T
// CHECK:STDOUT:   %N.type.loc22_31.2 => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %.loc22_33 => constants.%.1a5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr, constants.%D.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.c71
// CHECK:STDOUT:   %require_complete => constants.%require_complete.769
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.J.type.require(constants.%ptr, constants.%D.type.facet) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%D.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%D
// CHECK:STDOUT:   %J.type => constants.%J.type.c71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.d69d6f.2(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc26_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc26_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc26_31.2 => constants.%ptr
// CHECK:STDOUT:   %N.type.loc26_32.2 => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %.loc26_34 => constants.%.7a3
// CHECK:STDOUT: }
// CHECK:STDOUT:

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

interface I(T:! type) {}
interface J(T:! type) {
  extend require impls I(T);
}

// --- basic_import_generic_interface.impl.carbon
impl library "[[@TEST_NAME]]";

impl {} as J({}) {}

// --- basic_import_generic_constraint.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {}
constraint J(T:! type) {
  extend require impls I(T);
}

// --- basic_import_generic_constraint.impl.carbon
impl library "[[@TEST_NAME]]";

impl {} as J({}) {}

// --- import_generic.carbon

library "[[@TEST_NAME]]";

class C {}
interface I(T:! type) {}

// Has both declaration and definition.
impl forall [T:! type] C as I(T);
impl forall [T:! type] C as I(T) {}

// Only has definition.
impl forall [T:! type] C as I(T*) {}

constraint N(T:! type) {
  extend require impls I(T);
}

// --- fail_import_generic.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] C as I(T);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C as I(T);

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] C as I(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C as I(T) {}

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] C as I(T*);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C as I(T*);

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] C as I(T*) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C as I(T*) {}

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] C as N(T);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C as N(T);
// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] C as N(T*);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C as N(T*);

// --- import_generic_with_different_specific.carbon

library "[[@TEST_NAME]]";

class C {}
interface I(T:! type) {}

impl forall [T:! type] C as I(T) {}

constraint N(T:! type) {
  extend require impls I(T);
}

// --- import_generic_with_different_specific.impl.carbon
impl library "[[@TEST_NAME]]";

// These have different specifics than any impl decl in the api file, so they
// are allowed.
impl forall [T:! type] C as I(T*) {}
impl forall [T:! type] C as N(T**) {}

// --- fail_import_generic_decl.carbon

library "[[@TEST_NAME]]";

class D {}
interface J(T:! type) {}

constraint N(T:! type) {
  extend require impls J(T);
}

// CHECK:STDERR: fail_import_generic_decl.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl forall [T:! type] D as J(T);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] D as J(T);

// CHECK:STDERR: fail_import_generic_decl.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl forall [T:! type] D as J(T*);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] D as J(T*);

// --- fail_import_generic_decl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] D as J(T);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] D as J(T);

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] D as J(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] D as J(T) {}

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] D as J(T*);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] D as J(T*);

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] D as J(T*) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] D as J(T*) {}

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] D as N(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] D as N(T) {}

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] D as N(T*) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] D as N(T*) {}

// CHECK:STDOUT: --- basic_import_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %J.type.2b8: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.2b8 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.8ec: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.f68: %J.type.8ec = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.f68 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.070 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc3_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type.2b8 = interface_decl @J [concrete = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc3_13.2: type) {
// CHECK:STDOUT:   %T.loc3_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc3_13.1)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self.loc3_23.2: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc4_13.1)> [symbolic = %J.type (constants.%J.type.8ec)]
// CHECK:STDOUT:   %Self.loc4_23.2: @J.%J.type (%J.type.8ec) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.f68)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.1)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_23.1: @J.%J.type (%J.type.8ec) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.f68)]
// CHECK:STDOUT:     %J.require0.decl = require_decl @J.require0 [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @J.%Self.loc4_23.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @J.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc5_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.070)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_23.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.require0 {
// CHECK:STDOUT:       require @J.require0.%Self.as_type impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.require0(@J.%T.loc4_13.2: type, @J.%Self.loc4_23.1: @J.%J.type (%J.type.8ec)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.8ec)]
// CHECK:STDOUT:   %Self: @J.require0.%J.type (%J.type.8ec) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.f68)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type.loc5_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.070)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.require0(constants.%T, constants.%Self.f68) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.8ec
// CHECK:STDOUT:   %Self => constants.%Self.f68
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type.loc5_27.2 => constants.%I.type.070
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_interface.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.2b8: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.2b8 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.8ec: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.f68: %J.type.8ec = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.f68 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.070 [symbolic]
// CHECK:STDOUT:   %J.type.457: type = facet_type <@J, @J(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.07b: %J.type.457 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.type.399: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.32d: %I.type.399 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %I.type.399 [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness file.%J.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//basic_import_generic_interface, I, unloaded
// CHECK:STDOUT:   %Main.J: %J.type.2b8 = import_ref Main//basic_import_generic_interface, J, loaded [concrete = constants.%J.generic]
// CHECK:STDOUT:   %Main.import_ref.769 = import_ref Main//basic_import_generic_interface, loc3_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.1: type = import_ref Main//basic_import_generic_interface, loc3_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.ce5: type = import_ref Main//basic_import_generic_interface, loc5_18, loaded [symbolic = @J.require0.%Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.2: type = import_ref Main//basic_import_generic_interface, loc4_13, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.cc4: @J.%J.type (%J.type.8ec) = import_ref Main//basic_import_generic_interface, loc4_23, loaded [symbolic = @J.%Self (constants.%Self.f68)]
// CHECK:STDOUT:   %Main.import_ref.b3b = import_ref Main//basic_import_generic_interface, loc4_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.3: type = import_ref Main//basic_import_generic_interface, loc4_13, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc1_46.1 = import <none>
// CHECK:STDOUT:   %default.import.loc1_46.2 = import <none>
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %.loc3_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc3_7.2: type = converted %.loc3_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %.loc3_15: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc3_16: type = converted %.loc3_15, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.type: type = facet_type <@J, @J(constants.%empty_struct_type)> [concrete = constants.%J.type.457]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @empty_struct_type.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(imports.%Main.import_ref.efcd44.3: type) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.8ec)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.8ec) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.f68)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.b3b
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.require0 {
// CHECK:STDOUT:       require imports.%Main.import_ref.ce5 impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.efcd44.1: type) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.769
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.require0(imports.%Main.import_ref.efcd44.2: type, imports.%Main.import_ref.cc4: @J.%J.type (%J.type.8ec)) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.8ec)]
// CHECK:STDOUT:   %Self: @J.require0.%J.type (%J.type.8ec) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.f68)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.J.impl: %.loc3_7.2 as %J.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self => constants.%Self.269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.require0(constants.%T, constants.%Self.f68) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.8ec
// CHECK:STDOUT:   %Self => constants.%Self.f68
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.457
// CHECK:STDOUT:   %Self => constants.%Self.07b
// CHECK:STDOUT:   %I.type => constants.%I.type.399
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.399
// CHECK:STDOUT:   %Self => constants.%Self.32d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %J.type.267: type = generic_named_constaint_type @J [concrete]
// CHECK:STDOUT:   %empty_struct: %J.type.267 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.b8d: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa1: %J.type.b8d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.aa1 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.070 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc3_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type.267 = constraint_decl @J [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc3_13.2: type) {
// CHECK:STDOUT:   %T.loc3_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc3_13.1)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self.loc3_23.2: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @J(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc4_14.1)> [symbolic = %J.type (constants.%J.type.b8d)]
// CHECK:STDOUT:   %Self.loc4_24.2: @J.%J.type (%J.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc4_24.2 (constants.%Self.aa1)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_14.1)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc4_24.1: @J.%J.type (%J.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc4_24.2 (constants.%Self.aa1)]
// CHECK:STDOUT:     %J.require0.decl = require_decl @J.require0 [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @J.%Self.loc4_24.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @J.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc5_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.070)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_24.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.require0 {
// CHECK:STDOUT:       require @J.require0.%Self.as_type impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.require0(@J.%T.loc4_14.2: type, @J.%Self.loc4_24.1: @J.%J.type (%J.type.b8d)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.b8d)]
// CHECK:STDOUT:   %Self: @J.require0.%J.type (%J.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type.loc5_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.070)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc3_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.require0(constants.%T, constants.%Self.aa1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.b8d
// CHECK:STDOUT:   %Self => constants.%Self.aa1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type.loc5_27.2 => constants.%I.type.070
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_constraint.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.267: type = generic_named_constaint_type @J [concrete]
// CHECK:STDOUT:   %empty_struct.b6f: %J.type.267 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.b8d23b.1: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa1546.1: %J.type.b8d23b.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.aa1546.1 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.070 [symbolic]
// CHECK:STDOUT:   %J.type.b8d23b.2: type = facet_type <@J, @J(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %I.type.399: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.aa1546.2: %J.type.b8d23b.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.32d: %I.type.399 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %I.type.399 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//basic_import_generic_constraint, I, unloaded
// CHECK:STDOUT:   %Main.J: %J.type.267 = import_ref Main//basic_import_generic_constraint, J, loaded [concrete = constants.%empty_struct.b6f]
// CHECK:STDOUT:   %Main.import_ref.769 = import_ref Main//basic_import_generic_constraint, loc3_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.1: type = import_ref Main//basic_import_generic_constraint, loc3_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.f92: type = import_ref Main//basic_import_generic_constraint, loc5_18, loaded [symbolic = @J.require0.%Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.2: type = import_ref Main//basic_import_generic_constraint, loc4_14, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.d4d: @J.%J.type (%J.type.b8d23b.1) = import_ref Main//basic_import_generic_constraint, loc4_24, loaded [symbolic = @J.%Self (constants.%Self.aa1546.1)]
// CHECK:STDOUT:   %Main.import_ref.388 = import_ref Main//basic_import_generic_constraint, loc4_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.3: type = import_ref Main//basic_import_generic_constraint, loc4_14, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc1_47.1 = import <none>
// CHECK:STDOUT:   %default.import.loc1_47.2 = import <none>
// CHECK:STDOUT:   impl_decl @empty_struct_type.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %.loc3_7.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:     %.loc3_7.2: type = converted %.loc3_7.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.ref: %J.type.267 = name_ref J, imports.%Main.J [concrete = constants.%empty_struct.b6f]
// CHECK:STDOUT:     %.loc3_15: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:     %.loc3_16: type = converted %.loc3_15, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.type: type = facet_type <@J, @J(constants.%empty_struct_type)> [concrete = constants.%J.type.b8d23b.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @empty_struct_type.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.efcd44.1: type) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.769
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @J(imports.%Main.import_ref.efcd44.3: type) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.b8d23b.1)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.b8d23b.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1546.1)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.388
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.require0 {
// CHECK:STDOUT:       require imports.%Main.import_ref.f92 impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.require0(imports.%Main.import_ref.efcd44.2: type, imports.%Main.import_ref.d4d: @J.%J.type (%J.type.b8d23b.1)) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.b8d23b.1)]
// CHECK:STDOUT:   %Self: @J.require0.%J.type (%J.type.b8d23b.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1546.1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_struct_type.as.I.impl: %.loc3_7.2 as %J.type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self => constants.%Self.269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.require0(constants.%T, constants.%Self.aa1546.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.b8d23b.1
// CHECK:STDOUT:   %Self => constants.%Self.aa1546.1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.b8d23b.2
// CHECK:STDOUT:   %Self => constants.%Self.aa1546.2
// CHECK:STDOUT:   %I.type => constants.%I.type.399
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.require0(constants.%empty_struct_type, constants.%Self.aa1546.1) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %J.type => constants.%J.type.b8d23b.2
// CHECK:STDOUT:   %Self => constants.%Self.aa1546.1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.399
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.399
// CHECK:STDOUT:   %Self => constants.%Self.32d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.f13: <witness> = impl_witness file.%I.impl_witness_table.loc8, @C.as.I.impl.f3e(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.c94: <witness> = require_complete_type %I.type.070 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.229: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %Self.6d0: %I.type.229 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.555: <witness> = require_complete_type %I.type.229 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.a0f: <witness> = impl_witness file.%I.impl_witness_table.loc12, @C.as.I.impl.1fd(%T) [symbolic]
// CHECK:STDOUT:   %N.type.673: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.673 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.b8d: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa1: %N.type.b8d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.aa1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.f3e [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc8: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.070)]
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc8 = impl_witness_table (), @C.as.I.impl.f3e [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc8: <witness> = impl_witness %I.impl_witness_table.loc8, @C.as.I.impl.f3e(constants.%T) [symbolic = @C.as.I.impl.f3e.%I.impl_witness (constants.%I.impl_witness.f13)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.f3e [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc9: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc9 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc9: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.070)]
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc9: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.1fd [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc12_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.229)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc12_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc12 = impl_witness_table (), @C.as.I.impl.1fd [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc12: <witness> = impl_witness %I.impl_witness_table.loc12, @C.as.I.impl.1fd(constants.%T) [symbolic = @C.as.I.impl.1fd.%I.impl_witness (constants.%I.impl_witness.a0f)]
// CHECK:STDOUT:   %N.decl: %N.type.673 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc14_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc14_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc5_13.1)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self.loc5_23.2: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_23.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc14_14.2: type) {
// CHECK:STDOUT:   %T.loc14_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc14_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc14_14.1)> [symbolic = %N.type (constants.%N.type.b8d)]
// CHECK:STDOUT:   %Self.loc14_24.2: @N.%N.type (%N.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc14_24.2 (constants.%Self.aa1)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc14_14.1)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.c94)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc14_24.1: @N.%N.type (%N.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc14_24.2 (constants.%Self.aa1)]
// CHECK:STDOUT:     %N.require0.decl = require_decl @N.require0 [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @N.%Self.loc14_24.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc14_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc15_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc15_27.2 (constants.%I.type.070)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc14_24.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.require0 {
// CHECK:STDOUT:       require @N.require0.%Self.as_type impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.require0(@N.%T.loc14_14.2: type, @N.%Self.loc14_24.1: @N.%N.type (%N.type.b8d)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.b8d)]
// CHECK:STDOUT:   %Self: @N.require0.%N.type (%N.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type.loc15_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc15_27.2 (constants.%I.type.070)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.f3e(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_14.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.070)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table.loc8, @C.as.I.impl.f3e(%T.loc8_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness.f13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc8_32.2 [symbolic = %require_complete (constants.%require_complete.c94)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref.loc8 as %I.type.loc8_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%I.impl_witness.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.1fd(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc12_32.2: type = ptr_type %T.loc12_14.2 [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc12_33.2: type = facet_type <@I, @I(%ptr.loc12_32.2)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.229)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc12_33.2 [symbolic = %require_complete (constants.%require_complete.555)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table.loc12, @C.as.I.impl.1fd(%T.loc12_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness.a0f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc12_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%I.impl_witness.loc12
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
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.f3e(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.070
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.f13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.229
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.6d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.1fd(constants.%T) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc12_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc12_33.2 => constants.%I.type.229
// CHECK:STDOUT:   %require_complete => constants.%require_complete.555
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.a0f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.loc14_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.require0(constants.%T, constants.%Self.aa1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.b8d
// CHECK:STDOUT:   %Self => constants.%Self.aa1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type.loc15_27.2 => constants.%I.type.070
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness.f13: <witness> = impl_witness imports.%I.impl_witness_table.478, @C.as.I.impl.f3ed6b.1(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.c94: <witness> = require_complete_type %I.type.070 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.229: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %Self.6d0: %I.type.229 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.a0f: <witness> = impl_witness imports.%I.impl_witness_table.af9, @C.as.I.impl.1fddff.1(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.555: <witness> = require_complete_type %I.type.229 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.type.673: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.673 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.b8d23b.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa1: %N.type.b8d23b.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.aa1 [symbolic]
// CHECK:STDOUT:   %N.type.b8d23b.2: type = facet_type <@N, @N(%ptr)> [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//import_generic, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//import_generic, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.N: %N.type.673 = import_ref Main//import_generic, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.769 = import_ref Main//import_generic, loc5_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.1: type = import_ref Main//import_generic, loc5_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e46 = import_ref Main//import_generic, loc8_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//import_generic, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %I.impl_witness_table.478 = impl_witness_table (), @C.as.I.impl.f3ed6b.1 [concrete]
// CHECK:STDOUT:   %Main.import_ref.29aca8.1: type = import_ref Main//import_generic, loc8_24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.46444d.1: type = import_ref Main//import_generic, loc8_32, loaded [symbolic = @C.as.I.impl.f3ed6b.1.%I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.2: type = import_ref Main//import_generic, loc8_14, loaded [symbolic = @C.as.I.impl.f3ed6b.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.4f8 = import_ref Main//import_generic, loc12_35, unloaded
// CHECK:STDOUT:   %I.impl_witness_table.af9 = impl_witness_table (), @C.as.I.impl.1fddff.1 [concrete]
// CHECK:STDOUT:   %Main.import_ref.29aca8.2: type = import_ref Main//import_generic, loc12_24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.7b6: type = import_ref Main//import_generic, loc12_33, loaded [symbolic = @C.as.I.impl.1fddff.1.%I.type (constants.%I.type.229)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.3: type = import_ref Main//import_generic, loc12_14, loaded [symbolic = @C.as.I.impl.1fddff.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.f92: type = import_ref Main//import_generic, loc15_18, loaded [symbolic = @N.require0.%Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.4: type = import_ref Main//import_generic, loc14_14, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.d4d: @N.%N.type (%N.type.b8d23b.1) = import_ref Main//import_generic, loc14_24, loaded [symbolic = @N.%Self (constants.%Self.aa1)]
// CHECK:STDOUT:   %Main.import_ref.388 = import_ref Main//import_generic, loc14_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.5: type = import_ref Main//import_generic, loc14_14, loaded [symbolic = @N.%T (constants.%T)]
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
// CHECK:STDOUT:   impl_decl @C.as.I.impl.f3ed6b.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.070)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.f3ed6b.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc14_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.070)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc14_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.1fddff.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_14.1 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc20_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc20_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc20_33.2 (constants.%I.type.229)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc20_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.1fddff.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc26_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc26_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc26_33.2 (constants.%I.type.229)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc26_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.3f8440.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.673 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc32_14.1 [symbolic = %T.loc32_14.2 (constants.%T)]
// CHECK:STDOUT:     %N.type.loc32_32.1: type = facet_type <@N, @N(constants.%T)> [symbolic = %N.type.loc32_32.2 (constants.%N.type.b8d23b.1)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc32_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc32_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.3f8440.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.673 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc37_14.1 [symbolic = %T.loc37_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc37_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc37_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %N.type.loc37_33.1: type = facet_type <@N, @N(constants.%ptr)> [symbolic = %N.type.loc37_33.2 (constants.%N.type.b8d23b.2)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc37_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc37_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.efcd44.1: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.769
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(imports.%Main.import_ref.efcd44.5: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.b8d23b.1)]
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.b8d23b.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.c94)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.388
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.require0 {
// CHECK:STDOUT:       require imports.%Main.import_ref.f92 impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.require0(imports.%Main.import_ref.efcd44.4: type, imports.%Main.import_ref.d4d: @N.%N.type (%N.type.b8d23b.1)) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.b8d23b.1)]
// CHECK:STDOUT:   %Self: @N.require0.%N.type (%N.type.b8d23b.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.f3ed6b.1(imports.%Main.import_ref.efcd44.2: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table.478, @C.as.I.impl.f3ed6b.1(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.f13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.c94)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.29aca8.1 as imports.%Main.import_ref.46444d.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.e46
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.1fddff.1(imports.%Main.import_ref.efcd44.3: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%ptr)> [symbolic = %I.type (constants.%I.type.229)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.555)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table.af9, @C.as.I.impl.1fddff.1(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.a0f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.29aca8.2 as imports.%Main.import_ref.7b6 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.4f8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.f3ed6b.2(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_14.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.070)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc8_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.f3ed6b.3(%T.loc14_14.1: type) {
// CHECK:STDOUT:   %T.loc14_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc14_32.2: type = facet_type <@I, @I(%T.loc14_14.2)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.070)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc14_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.1fddff.2(%T.loc20_14.1: type) {
// CHECK:STDOUT:   %T.loc20_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc20_32.2: type = ptr_type %T.loc20_14.2 [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc20_33.2: type = facet_type <@I, @I(%ptr.loc20_32.2)> [symbolic = %I.type.loc20_33.2 (constants.%I.type.229)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc20_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.1fddff.3(%T.loc26_14.1: type) {
// CHECK:STDOUT:   %T.loc26_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc26_32.2: type = ptr_type %T.loc26_14.2 [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc26_33.2: type = facet_type <@I, @I(%ptr.loc26_32.2)> [symbolic = %I.type.loc26_33.2 (constants.%I.type.229)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc26_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.3f8440.1(%T.loc32_14.1: type) {
// CHECK:STDOUT:   %T.loc32_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc32_14.2 (constants.%T)]
// CHECK:STDOUT:   %N.type.loc32_32.2: type = facet_type <@N, @N(%T.loc32_14.2)> [symbolic = %N.type.loc32_32.2 (constants.%N.type.b8d23b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc32_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.3f8440.2(%T.loc37_14.1: type) {
// CHECK:STDOUT:   %T.loc37_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc37_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc37_32.2: type = ptr_type %T.loc37_14.2 [symbolic = %ptr.loc37_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %N.type.loc37_33.2: type = facet_type <@N, @N(%ptr.loc37_32.2)> [symbolic = %N.type.loc37_33.2 (constants.%N.type.b8d23b.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc37_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "import_generic.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self => constants.%Self.269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.f3ed6b.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.f13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.229
// CHECK:STDOUT:   %Self => constants.%Self.6d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.1fddff.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.229
// CHECK:STDOUT:   %require_complete => constants.%require_complete.555
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.a0f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.f3ed6b.2(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.070
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.f3ed6b.3(constants.%T) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc14_32.2 => constants.%I.type.070
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.1fddff.2(constants.%T) {
// CHECK:STDOUT:   %T.loc20_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc20_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc20_33.2 => constants.%I.type.229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.1fddff.3(constants.%T) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc26_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc26_33.2 => constants.%I.type.229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.require0(constants.%T, constants.%Self.aa1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.b8d23b.1
// CHECK:STDOUT:   %Self => constants.%Self.aa1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.3f8440.1(constants.%T) {
// CHECK:STDOUT:   %T.loc32_14.2 => constants.%T
// CHECK:STDOUT:   %N.type.loc32_32.2 => constants.%N.type.b8d23b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.require0(constants.%ptr, constants.%Self.aa1) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %N.type => constants.%N.type.b8d23b.2
// CHECK:STDOUT:   %Self => constants.%Self.aa1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.3f8440.2(constants.%T) {
// CHECK:STDOUT:   %T.loc37_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc37_32.2 => constants.%ptr
// CHECK:STDOUT:   %N.type.loc37_33.2 => constants.%N.type.b8d23b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_generic_with_different_specific.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.070 [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @C.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %N.type.673: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.673 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.b8d: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa1: %N.type.b8d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.aa1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc7_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc7_32.2 (constants.%I.type.070)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T) [symbolic = @C.as.I.impl.%I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:   %N.decl: %N.type.673 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc9_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc5_13.1)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self.loc5_23.2: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_23.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc9_14.2: type) {
// CHECK:STDOUT:   %T.loc9_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc9_14.1)> [symbolic = %N.type (constants.%N.type.b8d)]
// CHECK:STDOUT:   %Self.loc9_24.2: @N.%N.type (%N.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc9_24.2 (constants.%Self.aa1)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc9_14.1)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc9_24.1: @N.%N.type (%N.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc9_24.2 (constants.%Self.aa1)]
// CHECK:STDOUT:     %N.require0.decl = require_decl @N.require0 [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @N.%Self.loc9_24.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc9_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc10_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc10_27.2 (constants.%I.type.070)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc9_24.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.require0 {
// CHECK:STDOUT:       require @N.require0.%Self.as_type impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.require0(@N.%T.loc9_14.2: type, @N.%Self.loc9_24.1: @N.%N.type (%N.type.b8d)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.b8d)]
// CHECK:STDOUT:   %Self: @N.require0.%N.type (%N.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type.loc10_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc10_27.2 (constants.%I.type.070)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc7_32.2: type = facet_type <@I, @I(%T.loc7_14.2)> [symbolic = %I.type.loc7_32.2 (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_32.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @C.as.I.impl(%T.loc7_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%I.impl_witness
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
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc7_32.2 => constants.%I.type.070
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.loc9_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.require0(constants.%T, constants.%Self.aa1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.b8d
// CHECK:STDOUT:   %Self => constants.%Self.aa1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type.loc10_27.2 => constants.%I.type.070
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_generic_with_different_specific.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness.f13: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl.f3e(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.c94: <witness> = require_complete_type %I.type.070 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ptr.4f0: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.229: type = facet_type <@I, @I(%ptr.4f0)> [symbolic]
// CHECK:STDOUT:   %Self.6d0: %I.type.229 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.555: <witness> = require_complete_type %I.type.229 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.a0f: <witness> = impl_witness file.%I.impl_witness_table.loc5, @C.as.I.impl.1fd(%T) [symbolic]
// CHECK:STDOUT:   %N.type.673: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.673 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.b8d23b.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa1546.1: %N.type.b8d23b.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.aa1546.1 [symbolic]
// CHECK:STDOUT:   %ptr.3f2: type = ptr_type %ptr.4f0 [symbolic]
// CHECK:STDOUT:   %N.type.b8d23b.2: type = facet_type <@N, @N(%ptr.3f2)> [symbolic]
// CHECK:STDOUT:   %I.type.71a: type = facet_type <@I, @I(%ptr.3f2)> [symbolic]
// CHECK:STDOUT:   %Self.aa1546.2: %N.type.b8d23b.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.08c: <witness> = require_complete_type %I.type.71a [symbolic]
// CHECK:STDOUT:   %require_complete.a37: <witness> = require_complete_type %N.type.b8d23b.2 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.524: <witness> = impl_witness file.%I.impl_witness_table.loc6, @C.as.I.impl.3f8(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//import_generic_with_different_specific, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//import_generic_with_different_specific, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.N: %N.type.673 = import_ref Main//import_generic_with_different_specific, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.769 = import_ref Main//import_generic_with_different_specific, loc5_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.1: type = import_ref Main//import_generic_with_different_specific, loc5_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e46 = import_ref Main//import_generic_with_different_specific, loc7_34, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic_with_different_specific, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//import_generic_with_different_specific, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.f3e [concrete]
// CHECK:STDOUT:   %Main.import_ref.29a: type = import_ref Main//import_generic_with_different_specific, loc7_24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.46444d.1: type = import_ref Main//import_generic_with_different_specific, loc7_32, loaded [symbolic = @C.as.I.impl.f3e.%I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.2: type = import_ref Main//import_generic_with_different_specific, loc7_14, loaded [symbolic = @C.as.I.impl.f3e.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.f92: type = import_ref Main//import_generic_with_different_specific, loc10_18, loaded [symbolic = @N.require0.%Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.3: type = import_ref Main//import_generic_with_different_specific, loc9_14, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.d4d: @N.%N.type (%N.type.b8d23b.1) = import_ref Main//import_generic_with_different_specific, loc9_24, loaded [symbolic = @N.%Self (constants.%Self.aa1546.1)]
// CHECK:STDOUT:   %Main.import_ref.388 = import_ref Main//import_generic_with_different_specific, loc9_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.4: type = import_ref Main//import_generic_with_different_specific, loc9_14, loaded [symbolic = @N.%T (constants.%T)]
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
// CHECK:STDOUT:   impl_decl @C.as.I.impl.1fd [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_14.1 [symbolic = %T.loc5_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc5_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc5_32.2 (constants.%ptr.4f0)]
// CHECK:STDOUT:     %I.type.loc5_33.1: type = facet_type <@I, @I(constants.%ptr.4f0)> [symbolic = %I.type.loc5_33.2 (constants.%I.type.229)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc5 = impl_witness_table (), @C.as.I.impl.1fd [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc5: <witness> = impl_witness %I.impl_witness_table.loc5, @C.as.I.impl.1fd(constants.%T) [symbolic = @C.as.I.impl.1fd.%I.impl_witness (constants.%I.impl_witness.a0f)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.3f8 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.673 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_14.1 [symbolic = %T.loc6_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc6_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc6_32.2 (constants.%ptr.4f0)]
// CHECK:STDOUT:     %ptr.loc6_33.1: type = ptr_type %ptr.loc6_32.1 [symbolic = %ptr.loc6_33.2 (constants.%ptr.3f2)]
// CHECK:STDOUT:     %N.type.loc6_34.1: type = facet_type <@N, @N(constants.%ptr.3f2)> [symbolic = %N.type.loc6_34.2 (constants.%N.type.b8d23b.2)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc6 = impl_witness_table (), @C.as.I.impl.3f8 [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc6: <witness> = impl_witness %I.impl_witness_table.loc6, @C.as.I.impl.3f8(constants.%T) [symbolic = @C.as.I.impl.3f8.%I.impl_witness (constants.%I.impl_witness.524)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.efcd44.1: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.769
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(imports.%Main.import_ref.efcd44.4: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.b8d23b.1)]
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.b8d23b.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1546.1)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.c94)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.388
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.require0 {
// CHECK:STDOUT:       require imports.%Main.import_ref.f92 impls <@I, @I(constants.%T)>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.require0(imports.%Main.import_ref.efcd44.3: type, imports.%Main.import_ref.d4d: @N.%N.type (%N.type.b8d23b.1)) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.b8d23b.1)]
// CHECK:STDOUT:   %Self: @N.require0.%N.type (%N.type.b8d23b.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1546.1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.f3e(imports.%Main.import_ref.efcd44.2: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.c94)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl.f3e(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.f13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.29a as imports.%Main.import_ref.46444d.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.e46
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.1fd(%T.loc5_14.1: type) {
// CHECK:STDOUT:   %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc5_32.2: type = ptr_type %T.loc5_14.2 [symbolic = %ptr.loc5_32.2 (constants.%ptr.4f0)]
// CHECK:STDOUT:   %I.type.loc5_33.2: type = facet_type <@I, @I(%ptr.loc5_32.2)> [symbolic = %I.type.loc5_33.2 (constants.%I.type.229)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc5_33.2 [symbolic = %require_complete (constants.%require_complete.555)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table.loc5, @C.as.I.impl.1fd(%T.loc5_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness.a0f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc5_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%I.impl_witness.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.3f8(%T.loc6_14.1: type) {
// CHECK:STDOUT:   %T.loc6_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc6_32.2: type = ptr_type %T.loc6_14.2 [symbolic = %ptr.loc6_32.2 (constants.%ptr.4f0)]
// CHECK:STDOUT:   %ptr.loc6_33.2: type = ptr_type %ptr.loc6_32.2 [symbolic = %ptr.loc6_33.2 (constants.%ptr.3f2)]
// CHECK:STDOUT:   %N.type.loc6_34.2: type = facet_type <@N, @N(%ptr.loc6_33.2)> [symbolic = %N.type.loc6_34.2 (constants.%N.type.b8d23b.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %N.type.loc6_34.2 [symbolic = %require_complete (constants.%require_complete.a37)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table.loc6, @C.as.I.impl.3f8(%T.loc6_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness.524)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc6_34.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%I.impl_witness.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.2c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self => constants.%Self.269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.f3e(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c94
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.f13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr.4f0) {
// CHECK:STDOUT:   %T => constants.%ptr.4f0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.229
// CHECK:STDOUT:   %Self => constants.%Self.6d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.1fd(constants.%T) {
// CHECK:STDOUT:   %T.loc5_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc5_32.2 => constants.%ptr.4f0
// CHECK:STDOUT:   %I.type.loc5_33.2 => constants.%I.type.229
// CHECK:STDOUT:   %require_complete => constants.%require_complete.555
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.a0f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.require0(constants.%T, constants.%Self.aa1546.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.b8d23b.1
// CHECK:STDOUT:   %Self => constants.%Self.aa1546.1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr.3f2) {
// CHECK:STDOUT:   %T => constants.%ptr.3f2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type => constants.%N.type.b8d23b.2
// CHECK:STDOUT:   %Self => constants.%Self.aa1546.2
// CHECK:STDOUT:   %I.type => constants.%I.type.71a
// CHECK:STDOUT:   %require_complete => constants.%require_complete.08c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.require0(constants.%ptr.3f2, constants.%Self.aa1546.1) {
// CHECK:STDOUT:   %T => constants.%ptr.3f2
// CHECK:STDOUT:   %N.type => constants.%N.type.b8d23b.2
// CHECK:STDOUT:   %Self => constants.%Self.aa1546.1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.71a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr.3f2) {
// CHECK:STDOUT:   %T => constants.%ptr.3f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.3f8(constants.%T) {
// CHECK:STDOUT:   %T.loc6_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc6_32.2 => constants.%ptr.4f0
// CHECK:STDOUT:   %ptr.loc6_33.2 => constants.%ptr.3f2
// CHECK:STDOUT:   %N.type.loc6_34.2 => constants.%N.type.b8d23b.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.a37
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.524
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %J.type.2b8: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.2b8 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.8ec: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.f68: %J.type.8ec = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %N.type.673: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.673 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.b8d: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa1: %N.type.b8d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.aa1 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %J.type.8ec [symbolic]
// CHECK:STDOUT:   %J.impl_witness.224: <witness> = impl_witness file.%J.impl_witness_table.loc15, @D.as.J.impl.b47(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.4fa: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %J.impl_witness.537: <witness> = impl_witness file.%J.impl_witness_table.loc21, @D.as.J.impl.265(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %J.decl: %J.type.2b8 = interface_decl @J [concrete = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: %N.type.673 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.b47 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_14.1 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc15_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc15_32.2 (constants.%J.type.8ec)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table.loc15 = impl_witness_table (), @D.as.J.impl.b47 [concrete]
// CHECK:STDOUT:   %J.impl_witness.loc15: <witness> = impl_witness %J.impl_witness_table.loc15, @D.as.J.impl.b47(constants.%T) [symbolic = @D.as.J.impl.b47.%J.impl_witness (constants.%J.impl_witness.224)]
// CHECK:STDOUT:   impl_decl @D.as.J.impl.265 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_14.1 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc21_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc21_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc21_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc21_33.2 (constants.%J.type.4fa)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc21_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table.loc21 = impl_witness_table (), @D.as.J.impl.265 [concrete]
// CHECK:STDOUT:   %J.impl_witness.loc21: <witness> = impl_witness %J.impl_witness_table.loc21, @D.as.J.impl.265(constants.%T) [symbolic = @D.as.J.impl.265.%J.impl_witness (constants.%J.impl_witness.537)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc5_13.1)> [symbolic = %J.type (constants.%J.type.8ec)]
// CHECK:STDOUT:   %Self.loc5_23.2: @J.%J.type (%J.type.8ec) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.f68)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @J.%J.type (%J.type.8ec) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.f68)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_23.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc7_14.2: type) {
// CHECK:STDOUT:   %T.loc7_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc7_14.1)> [symbolic = %N.type (constants.%N.type.b8d)]
// CHECK:STDOUT:   %Self.loc7_24.2: @N.%N.type (%N.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc7_24.2 (constants.%Self.aa1)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc7_14.1)> [symbolic = %J.type (constants.%J.type.8ec)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %J.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc7_24.1: @N.%N.type (%N.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc7_24.2 (constants.%Self.aa1)]
// CHECK:STDOUT:     %N.require0.decl = require_decl @N.require0 [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls <@J, @J(constants.%T)>
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @N.%Self.loc7_24.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %J.ref: %J.type.2b8 = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc7_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %J.type.loc8_27.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc8_27.2 (constants.%J.type.8ec)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc7_24.1
// CHECK:STDOUT:     .J = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.require0 {
// CHECK:STDOUT:       require @N.require0.%Self.as_type impls <@J, @J(constants.%T)>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.require0(@N.%T.loc7_14.2: type, @N.%Self.loc7_24.1: @N.%N.type (%N.type.b8d)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.b8d)]
// CHECK:STDOUT:   %Self: @N.require0.%N.type (%N.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %J.type.loc8_27.2: type = facet_type <@J, @J(%T)> [symbolic = %J.type.loc8_27.2 (constants.%J.type.8ec)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.b47(%T.loc15_14.1: type) {
// CHECK:STDOUT:   %T.loc15_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_14.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc15_32.2: type = facet_type <@J, @J(%T.loc15_14.2)> [symbolic = %J.type.loc15_32.2 (constants.%J.type.8ec)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness file.%J.impl_witness_table.loc15, @D.as.J.impl.b47(%T.loc15_14.2) [symbolic = %J.impl_witness (constants.%J.impl_witness.224)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc15_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.265(%T.loc21_14.1: type) {
// CHECK:STDOUT:   %T.loc21_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc21_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc21_32.2: type = ptr_type %T.loc21_14.2 [symbolic = %ptr.loc21_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc21_33.2: type = facet_type <@J, @J(%ptr.loc21_32.2)> [symbolic = %J.type.loc21_33.2 (constants.%J.type.4fa)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness file.%J.impl_witness_table.loc21, @D.as.J.impl.265(%T.loc21_14.2) [symbolic = %J.impl_witness (constants.%J.impl_witness.537)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc21_33.1;
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
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.8ec
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.f68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.require0(constants.%T, constants.%Self.aa1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.b8d
// CHECK:STDOUT:   %Self => constants.%Self.aa1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %J.type.loc8_27.2 => constants.%J.type.8ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.b47(constants.%T) {
// CHECK:STDOUT:   %T.loc15_14.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc15_32.2 => constants.%J.type.8ec
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.224
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.265(constants.%T) {
// CHECK:STDOUT:   %T.loc21_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc21_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc21_33.2 => constants.%J.type.4fa
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.537
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.2b8: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.2b8 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.8ec: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.f68: %J.type.8ec = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.impl_witness.224: <witness> = impl_witness imports.%J.impl_witness_table.bc9, @D.as.J.impl.b470bf.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.4fa: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %J.impl_witness.537: <witness> = impl_witness imports.%J.impl_witness_table.bd9, @D.as.J.impl.265db6.1(%T) [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.type.673: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.673 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.b8d23b.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa1: %N.type.b8d23b.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.aa1 [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %J.type.8ec [symbolic]
// CHECK:STDOUT:   %N.type.b8d23b.2: type = facet_type <@N, @N(%ptr)> [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//import_generic_decl, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.J: %J.type.2b8 = import_ref Main//import_generic_decl, J, loaded [concrete = constants.%J.generic]
// CHECK:STDOUT:   %Main.N: %N.type.673 = import_ref Main//import_generic_decl, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.b3b = import_ref Main//import_generic_decl, loc5_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.1: type = import_ref Main//import_generic_decl, loc5_13, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic_decl, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.cab = import_ref Main//import_generic_decl, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %J.impl_witness_table.bc9 = impl_witness_table (), @D.as.J.impl.b470bf.1 [concrete]
// CHECK:STDOUT:   %Main.import_ref.aa9f8a.1: type = import_ref Main//import_generic_decl, loc15_24, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.03fb8f.1: type = import_ref Main//import_generic_decl, loc15_32, loaded [symbolic = @D.as.J.impl.b470bf.1.%J.type (constants.%J.type.8ec)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.2: type = import_ref Main//import_generic_decl, loc15_14, loaded [symbolic = @D.as.J.impl.b470bf.1.%T (constants.%T)]
// CHECK:STDOUT:   %J.impl_witness_table.bd9 = impl_witness_table (), @D.as.J.impl.265db6.1 [concrete]
// CHECK:STDOUT:   %Main.import_ref.aa9f8a.2: type = import_ref Main//import_generic_decl, loc21_24, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.a00: type = import_ref Main//import_generic_decl, loc21_33, loaded [symbolic = @D.as.J.impl.265db6.1.%J.type (constants.%J.type.4fa)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.3: type = import_ref Main//import_generic_decl, loc21_14, loaded [symbolic = @D.as.J.impl.265db6.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.f92: type = import_ref Main//import_generic_decl, loc8_18, loaded [symbolic = @N.require0.%Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.4: type = import_ref Main//import_generic_decl, loc7_14, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.d4d: @N.%N.type (%N.type.b8d23b.1) = import_ref Main//import_generic_decl, loc7_24, loaded [symbolic = @N.%Self (constants.%Self.aa1)]
// CHECK:STDOUT:   %Main.import_ref.388 = import_ref Main//import_generic_decl, loc7_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.5: type = import_ref Main//import_generic_decl, loc7_14, loaded [symbolic = @N.%T (constants.%T)]
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
// CHECK:STDOUT:   impl_decl @D.as.J.impl.b470bf.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc8_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc8_32.2 (constants.%J.type.8ec)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.b470bf.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc14_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.8ec)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc14_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.265db6.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_14.1 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc20_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc20_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc20_33.2 (constants.%J.type.4fa)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc20_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.265db6.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc26_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc26_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc26_33.2 (constants.%J.type.4fa)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc26_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.3b0484.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %N.ref: %N.type.673 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc32_14.1 [symbolic = %T.loc32_14.2 (constants.%T)]
// CHECK:STDOUT:     %N.type.loc32_32.1: type = facet_type <@N, @N(constants.%T)> [symbolic = %N.type.loc32_32.2 (constants.%N.type.b8d23b.1)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc32_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc32_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.3b0484.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %N.ref: %N.type.673 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc38_14.1 [symbolic = %T.loc38_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc38_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc38_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %N.type.loc38_33.1: type = facet_type <@N, @N(constants.%ptr)> [symbolic = %N.type.loc38_33.2 (constants.%N.type.b8d23b.2)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc38_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc38_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(imports.%Main.import_ref.efcd44.1: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.8ec)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.8ec) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.f68)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.b3b
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(imports.%Main.import_ref.efcd44.5: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.b8d23b.1)]
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.b8d23b.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.8ec)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %J.type [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.388
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.require0 {
// CHECK:STDOUT:       require imports.%Main.import_ref.f92 impls <@J, @J(constants.%T)>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.require0(imports.%Main.import_ref.efcd44.4: type, imports.%Main.import_ref.d4d: @N.%N.type (%N.type.b8d23b.1)) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.b8d23b.1)]
// CHECK:STDOUT:   %Self: @N.require0.%N.type (%N.type.b8d23b.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.8ec)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.b470bf.1(imports.%Main.import_ref.efcd44.2: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.8ec)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness imports.%J.impl_witness_table.bc9, @D.as.J.impl.b470bf.1(%T) [symbolic = %J.impl_witness (constants.%J.impl_witness.224)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.aa9f8a.1 as imports.%Main.import_ref.03fb8f.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.265db6.1(imports.%Main.import_ref.efcd44.3: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%ptr)> [symbolic = %J.type (constants.%J.type.4fa)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness imports.%J.impl_witness_table.bd9, @D.as.J.impl.265db6.1(%T) [symbolic = %J.impl_witness (constants.%J.impl_witness.537)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.aa9f8a.2 as imports.%Main.import_ref.a00;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.b470bf.2(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc8_32.2: type = facet_type <@J, @J(%T.loc8_14.2)> [symbolic = %J.type.loc8_32.2 (constants.%J.type.8ec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc8_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.b470bf.3(%T.loc14_14.1: type) {
// CHECK:STDOUT:   %T.loc14_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc14_32.2: type = facet_type <@J, @J(%T.loc14_14.2)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.8ec)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc14_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.265db6.2(%T.loc20_14.1: type) {
// CHECK:STDOUT:   %T.loc20_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc20_32.2: type = ptr_type %T.loc20_14.2 [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc20_33.2: type = facet_type <@J, @J(%ptr.loc20_32.2)> [symbolic = %J.type.loc20_33.2 (constants.%J.type.4fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc20_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.265db6.3(%T.loc26_14.1: type) {
// CHECK:STDOUT:   %T.loc26_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc26_32.2: type = ptr_type %T.loc26_14.2 [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc26_33.2: type = facet_type <@J, @J(%ptr.loc26_32.2)> [symbolic = %J.type.loc26_33.2 (constants.%J.type.4fa)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc26_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.3b0484.1(%T.loc32_14.1: type) {
// CHECK:STDOUT:   %T.loc32_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc32_14.2 (constants.%T)]
// CHECK:STDOUT:   %N.type.loc32_32.2: type = facet_type <@N, @N(%T.loc32_14.2)> [symbolic = %N.type.loc32_32.2 (constants.%N.type.b8d23b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %N.type.loc32_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.3b0484.2(%T.loc38_14.1: type) {
// CHECK:STDOUT:   %T.loc38_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc38_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc38_32.2: type = ptr_type %T.loc38_14.2 [symbolic = %ptr.loc38_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %N.type.loc38_33.2: type = facet_type <@N, @N(%ptr.loc38_32.2)> [symbolic = %N.type.loc38_33.2 (constants.%N.type.b8d23b.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %N.type.loc38_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.cab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.8ec
// CHECK:STDOUT:   %Self => constants.%Self.f68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.b470bf.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.8ec
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.224
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.265db6.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.4fa
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.537
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.b470bf.2(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc8_32.2 => constants.%J.type.8ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.b470bf.3(constants.%T) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc14_32.2 => constants.%J.type.8ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.265db6.2(constants.%T) {
// CHECK:STDOUT:   %T.loc20_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc20_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc20_33.2 => constants.%J.type.4fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.265db6.3(constants.%T) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc26_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc26_33.2 => constants.%J.type.4fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.require0(constants.%T, constants.%Self.aa1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.b8d23b.1
// CHECK:STDOUT:   %Self => constants.%Self.aa1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %J.type => constants.%J.type.8ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.3b0484.1(constants.%T) {
// CHECK:STDOUT:   %T.loc32_14.2 => constants.%T
// CHECK:STDOUT:   %N.type.loc32_32.2 => constants.%N.type.b8d23b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.require0(constants.%ptr, constants.%Self.aa1) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %N.type => constants.%N.type.b8d23b.2
// CHECK:STDOUT:   %Self => constants.%Self.aa1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %J.type => constants.%J.type.4fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.3b0484.2(constants.%T) {
// CHECK:STDOUT:   %T.loc38_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc38_32.2 => constants.%ptr
// CHECK:STDOUT:   %N.type.loc38_33.2 => constants.%N.type.b8d23b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

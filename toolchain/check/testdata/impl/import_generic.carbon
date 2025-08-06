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

// --- import_generic.carbon

library "[[@TEST_NAME]]";

class C {}
interface I(T:! type) {}

// Has both declaration and definition.
impl forall [T:! type] C as I(T);
impl forall [T:! type] C as I(T) {}

// Only has definition.
impl forall [T:! type] C as I(T*) {}

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

// --- fail_import_generic_decl.carbon

library "[[@TEST_NAME]]";

class D {}
interface J(T:! type) {}

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

// CHECK:STDOUT: --- import_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.209: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.1a9: <witness> = impl_witness file.%I.impl_witness_table.loc8, @C.as.I.impl.084(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.cfe: <witness> = require_complete_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.0e2: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %Self.fb4: %I.type.0e2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.0e6: <witness> = require_complete_type %I.type.0e2 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.6ad: <witness> = impl_witness file.%I.impl_witness_table.loc12, @C.as.I.impl.cd2(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.084 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc8: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc8 = impl_witness_table (), @C.as.I.impl.084 [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc8: <witness> = impl_witness %I.impl_witness_table.loc8, @C.as.I.impl.084(constants.%T) [symbolic = @C.as.I.impl.084.%I.impl_witness (constants.%I.impl_witness.1a9)]
// CHECK:STDOUT:   impl_decl @C.as.I.impl.084 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc9: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc9 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc9: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:     %T.loc9: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.cd2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc12_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:     %T.loc12_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc12 = impl_witness_table (), @C.as.I.impl.cd2 [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc12: <witness> = impl_witness %I.impl_witness_table.loc12, @C.as.I.impl.cd2(constants.%T) [symbolic = @C.as.I.impl.cd2.%I.impl_witness (constants.%I.impl_witness.6ad)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc5_13.1)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.325) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.325) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.084(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_14.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table.loc8, @C.as.I.impl.084(%T.loc8_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness.1a9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc8_32.2 [symbolic = %require_complete (constants.%require_complete.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref.loc8 as %I.type.loc8_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%I.impl_witness.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.cd2(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc12_32.2: type = ptr_type %T.loc12_14.2 [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc12_33.2: type = facet_type <@I, @I(%ptr.loc12_32.2)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc12_33.2 [symbolic = %require_complete (constants.%require_complete.0e6)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table.loc12, @C.as.I.impl.cd2(%T.loc12_14.2) [symbolic = %I.impl_witness (constants.%I.impl_witness.6ad)]
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self.2 => constants.%Self.209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.084(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.325
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.1a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.0e2
// CHECK:STDOUT:   %Self.2 => constants.%Self.fb4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.cd2(constants.%T) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc12_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc12_33.2 => constants.%I.type.0e2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0e6
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.6ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.209: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness.1a9: <witness> = impl_witness imports.%I.impl_witness_table.028, @C.as.I.impl.08450a.1(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.cfe: <witness> = require_complete_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.0e2: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.0e6: <witness> = require_complete_type %I.type.0e2 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.6ad: <witness> = impl_witness imports.%I.impl_witness_table.e14, @C.as.I.impl.cd2fdc.1(%T) [symbolic]
// CHECK:STDOUT:   %Self.fb4: %I.type.0e2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//import_generic, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//import_generic, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//import_generic, loc5_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.884 = import_ref Main//import_generic, inst32 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.cc6 = import_ref Main//import_generic, loc8_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//import_generic, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//import_generic, loc8_14, loaded [symbolic = @C.as.I.impl.08450a.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.29aca8.1: type = import_ref Main//import_generic, loc8_24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.4be: type = import_ref Main//import_generic, loc8_32, loaded [symbolic = @C.as.I.impl.08450a.1.%I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %I.impl_witness_table.028 = impl_witness_table (), @C.as.I.impl.08450a.1 [concrete]
// CHECK:STDOUT:   %Main.import_ref.fc2 = import_ref Main//import_generic, loc12_35, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//import_generic, loc12_14, loaded [symbolic = @C.as.I.impl.cd2fdc.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.29aca8.2: type = import_ref Main//import_generic, loc12_24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.3e2: type = import_ref Main//import_generic, loc12_33, loaded [symbolic = @C.as.I.impl.cd2fdc.1.%I.type (constants.%I.type.0e2)]
// CHECK:STDOUT:   %I.impl_witness_table.e14 = impl_witness_table (), @C.as.I.impl.cd2fdc.1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_30.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_30.2 = import <none>
// CHECK:STDOUT:   impl_decl @C.as.I.impl.08450a.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.08450a.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc14_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:     %T.loc14_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.cd2fdc.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_14.1 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc20_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc20_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc20_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:     %T.loc20_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.cd2fdc.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc26_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc26_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc26_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:     %T.loc26_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.5ab3ec.1: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.325) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.209)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.884
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.08450a.1(imports.%Main.import_ref.5ab3ec.2: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table.028, @C.as.I.impl.08450a.1(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.1a9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.29aca8.1 as imports.%Main.import_ref.4be {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.cc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.cd2fdc.1(imports.%Main.import_ref.5ab3ec.3: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%ptr)> [symbolic = %I.type (constants.%I.type.0e2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.0e6)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table.e14, @C.as.I.impl.cd2fdc.1(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.6ad)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.29aca8.2 as imports.%Main.import_ref.3e2 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.fc2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.08450a.2(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_14.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc8_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.08450a.3(%T.loc14_14.1: type) {
// CHECK:STDOUT:   %T.loc14_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc14_32.2: type = facet_type <@I, @I(%T.loc14_14.2)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc14_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.cd2fdc.2(%T.loc20_14.1: type) {
// CHECK:STDOUT:   %T.loc20_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc20_32.2: type = ptr_type %T.loc20_14.2 [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc20_33.2: type = facet_type <@I, @I(%ptr.loc20_32.2)> [symbolic = %I.type.loc20_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc20_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.cd2fdc.3(%T.loc26_14.1: type) {
// CHECK:STDOUT:   %T.loc26_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc26_32.2: type = ptr_type %T.loc26_14.2 [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc26_33.2: type = facet_type <@I, @I(%ptr.loc26_32.2)> [symbolic = %I.type.loc26_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc26_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self => constants.%Self.209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.08450a.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.1a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.0e2
// CHECK:STDOUT:   %Self => constants.%Self.fb4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.cd2fdc.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.0e2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0e6
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.6ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.08450a.2(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.325
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.08450a.3(constants.%T) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc14_32.2 => constants.%I.type.325
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.cd2fdc.2(constants.%T) {
// CHECK:STDOUT:   %T.loc20_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc20_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc20_33.2 => constants.%I.type.0e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.cd2fdc.3(constants.%T) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc26_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc26_33.2 => constants.%I.type.0e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %J.type.2b8: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.2b8 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.b72: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %J.type.b72 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %J.impl_witness.809: <witness> = impl_witness file.%J.impl_witness_table.loc11, @D.as.J.impl.199(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.628: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %J.impl_witness.5a8: <witness> = impl_witness file.%J.impl_witness_table.loc17, @D.as.J.impl.dfd(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %J.decl: %J.type.2b8 = interface_decl @J [concrete = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.199 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc11_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc11_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table.loc11 = impl_witness_table (), @D.as.J.impl.199 [concrete]
// CHECK:STDOUT:   %J.impl_witness.loc11: <witness> = impl_witness %J.impl_witness_table.loc11, @D.as.J.impl.199(constants.%T) [symbolic = @D.as.J.impl.199.%J.impl_witness (constants.%J.impl_witness.809)]
// CHECK:STDOUT:   impl_decl @D.as.J.impl.dfd [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc17_14.1 [symbolic = %T.loc17_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc17_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc17_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc17_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc17_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:     %T.loc17_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table.loc17 = impl_witness_table (), @D.as.J.impl.dfd [concrete]
// CHECK:STDOUT:   %J.impl_witness.loc17: <witness> = impl_witness %J.impl_witness_table.loc17, @D.as.J.impl.dfd(constants.%T) [symbolic = @D.as.J.impl.dfd.%J.impl_witness (constants.%J.impl_witness.5a8)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc5_13.2: type) {
// CHECK:STDOUT:   %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc5_13.1)> [symbolic = %J.type (constants.%J.type.b72)]
// CHECK:STDOUT:   %Self.2: @J.%J.type (%J.type.b72) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @J.%J.type (%J.type.b72) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.199(%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T.loc11_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc11_32.2: type = facet_type <@J, @J(%T.loc11_14.2)> [symbolic = %J.type.loc11_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness file.%J.impl_witness_table.loc11, @D.as.J.impl.199(%T.loc11_14.2) [symbolic = %J.impl_witness (constants.%J.impl_witness.809)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc11_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.dfd(%T.loc17_14.1: type) {
// CHECK:STDOUT:   %T.loc17_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc17_32.2: type = ptr_type %T.loc17_14.2 [symbolic = %ptr.loc17_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc17_33.2: type = facet_type <@J, @J(%ptr.loc17_32.2)> [symbolic = %J.type.loc17_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness file.%J.impl_witness_table.loc17, @D.as.J.impl.dfd(%T.loc17_14.2) [symbolic = %J.impl_witness (constants.%J.impl_witness.5a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc17_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.199(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc11_32.2 => constants.%J.type.b72
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.809
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.dfd(constants.%T) {
// CHECK:STDOUT:   %T.loc17_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc17_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc17_33.2 => constants.%J.type.628
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.5a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.2b8: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.2b8 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.b72: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %J.type.b72 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.impl_witness.809: <witness> = impl_witness imports.%J.impl_witness_table.891, @D.as.J.impl.199bba.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.628: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %J.impl_witness.5a8: <witness> = impl_witness imports.%J.impl_witness_table.487, @D.as.J.impl.dfd2f7.1(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//import_generic_decl, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.J: %J.type.2b8 = import_ref Main//import_generic_decl, J, loaded [concrete = constants.%J.generic]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//import_generic_decl, loc5_13, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.ff5 = import_ref Main//import_generic_decl, inst32 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic_decl, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.cab = import_ref Main//import_generic_decl, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//import_generic_decl, loc11_14, loaded [symbolic = @D.as.J.impl.199bba.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.aa9f8a.1: type = import_ref Main//import_generic_decl, loc11_24, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.ded: type = import_ref Main//import_generic_decl, loc11_32, loaded [symbolic = @D.as.J.impl.199bba.1.%J.type (constants.%J.type.b72)]
// CHECK:STDOUT:   %J.impl_witness_table.891 = impl_witness_table (), @D.as.J.impl.199bba.1 [concrete]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//import_generic_decl, loc17_14, loaded [symbolic = @D.as.J.impl.dfd2f7.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.aa9f8a.2: type = import_ref Main//import_generic_decl, loc17_24, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.0d3: type = import_ref Main//import_generic_decl, loc17_33, loaded [symbolic = @D.as.J.impl.dfd2f7.1.%J.type (constants.%J.type.628)]
// CHECK:STDOUT:   %J.impl_witness_table.487 = impl_witness_table (), @D.as.J.impl.dfd2f7.1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_35.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_35.2 = import <none>
// CHECK:STDOUT:   impl_decl @D.as.J.impl.199bba.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc8_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc8_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.199bba.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc14_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:     %T.loc14_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.dfd2f7.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_14.1 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc20_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc20_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc20_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:     %T.loc20_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.dfd2f7.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc26_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc26_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc26_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:     %T.loc26_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(imports.%Main.import_ref.5ab3ec.1: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.b72)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.b72) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ff5
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.199bba.1(imports.%Main.import_ref.5ab3ec.2: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.b72)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness imports.%J.impl_witness_table.891, @D.as.J.impl.199bba.1(%T) [symbolic = %J.impl_witness (constants.%J.impl_witness.809)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.aa9f8a.1 as imports.%Main.import_ref.ded;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.dfd2f7.1(imports.%Main.import_ref.5ab3ec.3: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%ptr)> [symbolic = %J.type (constants.%J.type.628)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness imports.%J.impl_witness_table.487, @D.as.J.impl.dfd2f7.1(%T) [symbolic = %J.impl_witness (constants.%J.impl_witness.5a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.aa9f8a.2 as imports.%Main.import_ref.0d3;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.199bba.2(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc8_32.2: type = facet_type <@J, @J(%T.loc8_14.2)> [symbolic = %J.type.loc8_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc8_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.199bba.3(%T.loc14_14.1: type) {
// CHECK:STDOUT:   %T.loc14_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc14_32.2: type = facet_type <@J, @J(%T.loc14_14.2)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc14_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.dfd2f7.2(%T.loc20_14.1: type) {
// CHECK:STDOUT:   %T.loc20_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc20_32.2: type = ptr_type %T.loc20_14.2 [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc20_33.2: type = facet_type <@J, @J(%ptr.loc20_32.2)> [symbolic = %J.type.loc20_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc20_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.dfd2f7.3(%T.loc26_14.1: type) {
// CHECK:STDOUT:   %T.loc26_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc26_32.2: type = ptr_type %T.loc26_14.2 [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc26_33.2: type = facet_type <@J, @J(%ptr.loc26_32.2)> [symbolic = %J.type.loc26_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc26_33.1 {
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.199bba.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.b72
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.809
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.dfd2f7.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.628
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.5a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.199bba.2(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc8_32.2 => constants.%J.type.b72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.199bba.3(constants.%T) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc14_32.2 => constants.%J.type.b72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.dfd2f7.2(constants.%T) {
// CHECK:STDOUT:   %T.loc20_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc20_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc20_33.2 => constants.%J.type.628
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.dfd2f7.3(constants.%T) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc26_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc26_33.2 => constants.%J.type.628
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_generic.carbon

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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.209: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %impl_witness.eff: <witness> = impl_witness (), @impl.084(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.cfe: <witness> = require_complete_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.0e2: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %Self.fb4: %I.type.0e2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.0e6: <witness> = require_complete_type %I.type.0e2 [symbolic]
// CHECK:STDOUT:   %impl_witness.465: <witness> = impl_witness (), @impl.cd2(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc5_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.084 [concrete] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc8: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness (), @impl.084(constants.%T) [symbolic = @impl.084.%impl_witness (constants.%impl_witness.eff)]
// CHECK:STDOUT:   impl_decl @impl.084 [concrete] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc9: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc9 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc9: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:     %T.loc9: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.cd2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc12_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_32.1: type = ptr_type %T [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc12_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:     %T.loc12_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc12: <witness> = impl_witness (), @impl.cd2(constants.%T) [symbolic = @impl.cd2.%impl_witness (constants.%impl_witness.465)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc5_13.2)> [symbolic = %I.type (constants.%I.type.325)]
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
// CHECK:STDOUT: generic impl @impl.084(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_14.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.084(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness.eff)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.084.%I.type.loc8_32.2 (%I.type.325) [symbolic = %require_complete (constants.%require_complete.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref.loc8 as %I.type.loc8_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.cd2(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc12_32.2: type = ptr_type @impl.cd2.%T.loc12_14.2 (%T) [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc12_33.2: type = facet_type <@I, @I(%ptr.loc12_32.2)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.cd2.%I.type.loc12_33.2 (%I.type.0e2) [symbolic = %require_complete (constants.%require_complete.0e6)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.cd2(%T.loc12_14.2) [symbolic = %impl_witness (constants.%impl_witness.465)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc12_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self.2 => constants.%Self.209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T.loc5_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.084(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T.patt
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.325
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.eff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.084.%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.084(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%ptr
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.0e2
// CHECK:STDOUT:   %Self.2 => constants.%Self.fb4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.cd2(constants.%T) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc12_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc12_33.2 => constants.%I.type.0e2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0e6
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.465
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.cd2.%ptr.loc12_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.cd2(%T.loc12_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.209: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness.eff58b.1: <witness> = impl_witness (), @impl.08450a.1(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.cfe: <witness> = require_complete_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.0e2: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.0e6: <witness> = require_complete_type %I.type.0e2 [symbolic]
// CHECK:STDOUT:   %impl_witness.46542c.1: <witness> = impl_witness (), @impl.cd2fdc.1(%T) [symbolic]
// CHECK:STDOUT:   %Self.fb4: %I.type.0e2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %impl_witness.eff58b.2: <witness> = impl_witness (), @impl.08450a.2(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.eff58b.3: <witness> = impl_witness (), @impl.08450a.3(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.46542c.2: <witness> = impl_witness (), @impl.cd2fdc.2(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.46542c.3: <witness> = impl_witness (), @impl.cd2fdc.3(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//import_generic, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//import_generic, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//import_generic, loc5_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.884 = import_ref Main//import_generic, inst30 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.003 = import_ref Main//import_generic, loc8_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2c4 = import_ref Main//import_generic, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//import_generic, loc8_14, loaded [symbolic = @impl.08450a.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.29aca8.1: type = import_ref Main//import_generic, loc8_24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.4be: type = import_ref Main//import_generic, loc8_32, loaded [symbolic = @impl.08450a.1.%I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Main.import_ref.4e1 = import_ref Main//import_generic, loc12_35, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//import_generic, loc12_14, loaded [symbolic = @impl.cd2fdc.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.29aca8.2: type = import_ref Main//import_generic, loc12_24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.3e2: type = import_ref Main//import_generic, loc12_33, loaded [symbolic = @impl.cd2fdc.1.%I.type (constants.%I.type.0e2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_30.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_30.2 = import <none>
// CHECK:STDOUT:   impl_decl @impl.08450a.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness (), @impl.08450a.2(constants.%T) [symbolic = @impl.08450a.2.%impl_witness (constants.%impl_witness.eff58b.2)]
// CHECK:STDOUT:   impl_decl @impl.08450a.3 [concrete] {
// CHECK:STDOUT:     %T.patt.loc14_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc14_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:     %T.loc14_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness (), @impl.08450a.3(constants.%T) [symbolic = @impl.08450a.3.%impl_witness (constants.%impl_witness.eff58b.3)]
// CHECK:STDOUT:   impl_decl @impl.cd2fdc.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc20_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_14.1 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc20_32.1: type = ptr_type %T [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc20_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc20_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:     %T.loc20_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc20: <witness> = impl_witness (), @impl.cd2fdc.2(constants.%T) [symbolic = @impl.cd2fdc.2.%impl_witness (constants.%impl_witness.46542c.2)]
// CHECK:STDOUT:   impl_decl @impl.cd2fdc.3 [concrete] {
// CHECK:STDOUT:     %T.patt.loc26_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc26_32.1: type = ptr_type %T [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc26_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc26_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:     %T.loc26_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc26: <witness> = impl_witness (), @impl.cd2fdc.3(constants.%T) [symbolic = @impl.cd2fdc.3.%impl_witness (constants.%impl_witness.46542c.3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.5ab3ec.1: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
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
// CHECK:STDOUT: generic impl @impl.08450a.1(imports.%Main.import_ref.5ab3ec.2: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.08450a.1(%T) [symbolic = %impl_witness (constants.%impl_witness.eff58b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.08450a.1.%I.type (%I.type.325) [symbolic = %require_complete (constants.%require_complete.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.29aca8.1 as imports.%Main.import_ref.4be {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.003
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.cd2fdc.1(imports.%Main.import_ref.5ab3ec.3: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %ptr: type = ptr_type @impl.cd2fdc.1.%T (%T) [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%ptr)> [symbolic = %I.type (constants.%I.type.0e2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.cd2fdc.1.%I.type (%I.type.0e2) [symbolic = %require_complete (constants.%require_complete.0e6)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.cd2fdc.1(%T) [symbolic = %impl_witness (constants.%impl_witness.46542c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.29aca8.2 as imports.%Main.import_ref.3e2 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.4e1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.08450a.2(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_14.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.08450a.2(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness.eff58b.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc8_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.08450a.3(%T.loc14_14.1: type) {
// CHECK:STDOUT:   %T.loc14_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc14_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %I.type.loc14_32.2: type = facet_type <@I, @I(%T.loc14_14.2)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.325)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.08450a.3.%I.type.loc14_32.2 (%I.type.325) [symbolic = %require_complete (constants.%require_complete.cfe)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.08450a.3(%T.loc14_14.2) [symbolic = %impl_witness (constants.%impl_witness.eff58b.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc14_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.cd2fdc.2(%T.loc20_14.1: type) {
// CHECK:STDOUT:   %T.loc20_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc20_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc20_32.2: type = ptr_type @impl.cd2fdc.2.%T.loc20_14.2 (%T) [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc20_33.2: type = facet_type <@I, @I(%ptr.loc20_32.2)> [symbolic = %I.type.loc20_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.cd2fdc.2(%T.loc20_14.2) [symbolic = %impl_witness (constants.%impl_witness.46542c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc20_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.cd2fdc.3(%T.loc26_14.1: type) {
// CHECK:STDOUT:   %T.loc26_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc26_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc26_32.2: type = ptr_type @impl.cd2fdc.3.%T.loc26_14.2 (%T) [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc26_33.2: type = facet_type <@I, @I(%ptr.loc26_32.2)> [symbolic = %I.type.loc26_33.2 (constants.%I.type.0e2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.cd2fdc.3.%I.type.loc26_33.2 (%I.type.0e2) [symbolic = %require_complete (constants.%require_complete.0e6)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.cd2fdc.3(%T.loc26_14.2) [symbolic = %impl_witness (constants.%impl_witness.46542c.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc26_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc26
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
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self => constants.%Self.209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.08450a.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.eff58b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.08450a.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.08450a.1(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.0e2
// CHECK:STDOUT:   %Self => constants.%Self.fb4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.cd2fdc.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.0e2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0e6
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.46542c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.cd2fdc.1.%ptr) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.cd2fdc.1(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.08450a.2(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T.patt
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.325
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.eff58b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.08450a.2.%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.08450a.2(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.08450a.3(constants.%T) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc14_14.2 => constants.%T.patt
// CHECK:STDOUT:   %I.type.loc14_32.2 => constants.%I.type.325
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cfe
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.eff58b.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.08450a.3.%T.loc14_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.08450a.3(%T.loc14_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.cd2fdc.2(constants.%T) {
// CHECK:STDOUT:   %T.loc20_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc20_14.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc20_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc20_33.2 => constants.%I.type.0e2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.46542c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.cd2fdc.2.%ptr.loc20_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.cd2fdc.2(%T.loc20_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.cd2fdc.3(constants.%T) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc26_14.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc26_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc26_33.2 => constants.%I.type.0e2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.0e6
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.46542c.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.cd2fdc.3.%ptr.loc26_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.cd2fdc.3(%T.loc26_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.2b8: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.2b8 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.b72: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %J.type.b72 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %impl_witness.0ef: <witness> = impl_witness (), @impl.199(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.628: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %impl_witness.b80: <witness> = impl_witness (), @impl.dfd(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %J.decl: %J.type.2b8 = interface_decl @J [concrete = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt.loc5_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.199 [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc11_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc11_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc11: <witness> = impl_witness (), @impl.199(constants.%T) [symbolic = @impl.199.%impl_witness (constants.%impl_witness.0ef)]
// CHECK:STDOUT:   impl_decl @impl.dfd [concrete] {
// CHECK:STDOUT:     %T.patt.loc17_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc17_14.1 [symbolic = %T.loc17_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc17_32.1: type = ptr_type %T [symbolic = %ptr.loc17_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc17_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc17_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:     %T.loc17_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness (), @impl.dfd(constants.%T) [symbolic = @impl.dfd.%impl_witness (constants.%impl_witness.b80)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc5_13.2)> [symbolic = %J.type (constants.%J.type.b72)]
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
// CHECK:STDOUT: generic impl @impl.199(%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T.loc11_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %J.type.loc11_32.2: type = facet_type <@J, @J(%T.loc11_14.2)> [symbolic = %J.type.loc11_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.199(%T.loc11_14.2) [symbolic = %impl_witness (constants.%impl_witness.0ef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc11_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.dfd(%T.loc17_14.1: type) {
// CHECK:STDOUT:   %T.loc17_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc17_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc17_32.2: type = ptr_type @impl.dfd.%T.loc17_14.2 (%T) [symbolic = %ptr.loc17_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc17_33.2: type = facet_type <@J, @J(%ptr.loc17_32.2)> [symbolic = %J.type.loc17_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.dfd(%T.loc17_14.2) [symbolic = %impl_witness (constants.%impl_witness.b80)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc17_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(%T.loc5_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.199(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T.patt
// CHECK:STDOUT:   %J.type.loc11_32.2 => constants.%J.type.b72
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.0ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.199.%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.199(%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%ptr
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.dfd(constants.%T) {
// CHECK:STDOUT:   %T.loc17_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc17_14.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc17_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc17_33.2 => constants.%J.type.628
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.dfd.%ptr.loc17_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.dfd(%T.loc17_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.2b8: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.2b8 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.b72: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.252: %J.type.b72 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %impl_witness.0ef94b.1: <witness> = impl_witness (), @impl.199bba.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.628: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %impl_witness.b80f53.1: <witness> = impl_witness (), @impl.dfd2f7.1(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.0ef94b.2: <witness> = impl_witness (), @impl.199bba.2(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.287: <witness> = require_complete_type %J.type.b72 [symbolic]
// CHECK:STDOUT:   %impl_witness.0ef94b.3: <witness> = impl_witness (), @impl.199bba.3(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.b80f53.2: <witness> = impl_witness (), @impl.dfd2f7.2(%T) [symbolic]
// CHECK:STDOUT:   %Self.fbe: %J.type.628 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.c60: <witness> = require_complete_type %J.type.628 [symbolic]
// CHECK:STDOUT:   %impl_witness.b80f53.3: <witness> = impl_witness (), @impl.dfd2f7.3(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//import_generic_decl, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.J: %J.type.2b8 = import_ref Main//import_generic_decl, J, loaded [concrete = constants.%J.generic]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//import_generic_decl, loc5_13, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.ff5 = import_ref Main//import_generic_decl, inst30 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic_decl, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.cab = import_ref Main//import_generic_decl, inst15 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//import_generic_decl, loc11_14, loaded [symbolic = @impl.199bba.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.aa9f8a.1: type = import_ref Main//import_generic_decl, loc11_24, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.ded: type = import_ref Main//import_generic_decl, loc11_32, loaded [symbolic = @impl.199bba.1.%J.type (constants.%J.type.b72)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//import_generic_decl, loc17_14, loaded [symbolic = @impl.dfd2f7.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.aa9f8a.2: type = import_ref Main//import_generic_decl, loc17_24, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.0d3: type = import_ref Main//import_generic_decl, loc17_33, loaded [symbolic = @impl.dfd2f7.1.%J.type (constants.%J.type.628)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_35.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_35.2 = import <none>
// CHECK:STDOUT:   impl_decl @impl.199bba.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc8_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc8_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness (), @impl.199bba.2(constants.%T) [symbolic = @impl.199bba.2.%impl_witness (constants.%impl_witness.0ef94b.2)]
// CHECK:STDOUT:   impl_decl @impl.199bba.3 [concrete] {
// CHECK:STDOUT:     %T.patt.loc14_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc14_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:     %T.loc14_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness (), @impl.199bba.3(constants.%T) [symbolic = @impl.199bba.3.%impl_witness (constants.%impl_witness.0ef94b.3)]
// CHECK:STDOUT:   impl_decl @impl.dfd2f7.2 [concrete] {
// CHECK:STDOUT:     %T.patt.loc20_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_14.1 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc20_32.1: type = ptr_type %T [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc20_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc20_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:     %T.loc20_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc20: <witness> = impl_witness (), @impl.dfd2f7.2(constants.%T) [symbolic = @impl.dfd2f7.2.%impl_witness (constants.%impl_witness.b80f53.2)]
// CHECK:STDOUT:   impl_decl @impl.dfd2f7.3 [concrete] {
// CHECK:STDOUT:     %T.patt.loc26_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc26_32.1: type = ptr_type %T [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc26_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc26_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:     %T.loc26_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc26: <witness> = impl_witness (), @impl.dfd2f7.3(constants.%T) [symbolic = @impl.dfd2f7.3.%impl_witness (constants.%impl_witness.b80f53.3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(imports.%Main.import_ref.5ab3ec.1: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.b72)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.b72) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.252)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.ff5
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.199bba.1(imports.%Main.import_ref.5ab3ec.2: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.b72)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.199bba.1(%T) [symbolic = %impl_witness (constants.%impl_witness.0ef94b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.aa9f8a.1 as imports.%Main.import_ref.ded;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.dfd2f7.1(imports.%Main.import_ref.5ab3ec.3: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %ptr: type = ptr_type @impl.dfd2f7.1.%T (%T) [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%ptr)> [symbolic = %J.type (constants.%J.type.628)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.dfd2f7.1(%T) [symbolic = %impl_witness (constants.%impl_witness.b80f53.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.aa9f8a.2 as imports.%Main.import_ref.0d3;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.199bba.2(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %J.type.loc8_32.2: type = facet_type <@J, @J(%T.loc8_14.2)> [symbolic = %J.type.loc8_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.199bba.2(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness.0ef94b.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc8_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.199bba.3(%T.loc14_14.1: type) {
// CHECK:STDOUT:   %T.loc14_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc14_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %J.type.loc14_32.2: type = facet_type <@J, @J(%T.loc14_14.2)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.b72)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.199bba.3.%J.type.loc14_32.2 (%J.type.b72) [symbolic = %require_complete (constants.%require_complete.287)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.199bba.3(%T.loc14_14.2) [symbolic = %impl_witness (constants.%impl_witness.0ef94b.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc14_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.dfd2f7.2(%T.loc20_14.1: type) {
// CHECK:STDOUT:   %T.loc20_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc20_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc20_32.2: type = ptr_type @impl.dfd2f7.2.%T.loc20_14.2 (%T) [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc20_33.2: type = facet_type <@J, @J(%ptr.loc20_32.2)> [symbolic = %J.type.loc20_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.dfd2f7.2(%T.loc20_14.2) [symbolic = %impl_witness (constants.%impl_witness.b80f53.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc20_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.dfd2f7.3(%T.loc26_14.1: type) {
// CHECK:STDOUT:   %T.loc26_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc26_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc26_32.2: type = ptr_type @impl.dfd2f7.3.%T.loc26_14.2 (%T) [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc26_33.2: type = facet_type <@J, @J(%ptr.loc26_32.2)> [symbolic = %J.type.loc26_33.2 (constants.%J.type.628)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.dfd2f7.3.%J.type.loc26_33.2 (%J.type.628) [symbolic = %require_complete (constants.%require_complete.c60)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.dfd2f7.3(%T.loc26_14.2) [symbolic = %impl_witness (constants.%impl_witness.b80f53.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc26_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc26
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
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.b72
// CHECK:STDOUT:   %Self => constants.%Self.252
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.199bba.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %J.type => constants.%J.type.b72
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.0ef94b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.199bba.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.199bba.1(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.628
// CHECK:STDOUT:   %Self => constants.%Self.fbe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.dfd2f7.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.628
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b80f53.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.dfd2f7.1.%ptr) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.dfd2f7.1(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.199bba.2(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T.patt
// CHECK:STDOUT:   %J.type.loc8_32.2 => constants.%J.type.b72
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.0ef94b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.199bba.2.%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.199bba.2(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.199bba.3(constants.%T) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc14_14.2 => constants.%T.patt
// CHECK:STDOUT:   %J.type.loc14_32.2 => constants.%J.type.b72
// CHECK:STDOUT:   %require_complete => constants.%require_complete.287
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.0ef94b.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.199bba.3.%T.loc14_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.199bba.3(%T.loc14_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.dfd2f7.2(constants.%T) {
// CHECK:STDOUT:   %T.loc20_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc20_14.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc20_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc20_33.2 => constants.%J.type.628
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b80f53.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.dfd2f7.2.%ptr.loc20_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.dfd2f7.2(%T.loc20_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.dfd2f7.3(constants.%T) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc26_14.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc26_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc26_33.2 => constants.%J.type.628
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c60
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.b80f53.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.dfd2f7.3.%ptr.loc26_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.dfd2f7.3(%T.loc26_14.2) {}
// CHECK:STDOUT:

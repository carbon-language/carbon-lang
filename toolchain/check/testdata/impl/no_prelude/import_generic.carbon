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

// CHECK:STDERR: fail_import_generic_decl.carbon:[[@LINE+4]]:1: error: impl declared but not defined [MissingImplDefinition]
// CHECK:STDERR: impl forall [T:! type] D as J(T);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] D as J(T);

// CHECK:STDERR: fail_import_generic_decl.carbon:[[@LINE+4]]:1: error: impl declared but not defined [MissingImplDefinition]
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

// CHECK:STDERR: fail_import_generic_decl.impl.carbon:[[@LINE+3]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] D as J(T*) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl forall [T:! type] D as J(T*) {}

// CHECK:STDOUT: --- import_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [template]
// CHECK:STDOUT:   %I.type.bc4: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.bc4 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.ffb: <witness> = require_complete_type %I.type.bc4 [symbolic]
// CHECK:STDOUT:   %impl_witness.455: <witness> = impl_witness (), @impl.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.51f: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.96d: <witness> = require_complete_type %I.type.51f [symbolic]
// CHECK:STDOUT:   %impl_witness.26c: <witness> = impl_witness (), @impl.2(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [template = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc5_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc5_13.1, runtime_param<invalid> [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc5_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [template] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_14.1, runtime_param<invalid> [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref.loc8: %I.type.dac = name_ref I, file.%I.decl [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.bc4)]
// CHECK:STDOUT:     %T.param.loc8: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0, %T.param.loc8 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness (), @impl.1(constants.%T) [symbolic = @impl.1.%impl_witness (constants.%impl_witness.455)]
// CHECK:STDOUT:   impl_decl @impl.1 [template] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_14.1, runtime_param<invalid> [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref.loc9: %I.type.dac = name_ref I, file.%I.decl [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc9 [symbolic = constants.%T]
// CHECK:STDOUT:     %I.type.loc9: type = facet_type <@I, @I(constants.%T)> [symbolic = constants.%I.type.bc4]
// CHECK:STDOUT:     %T.param.loc9: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc9: type = bind_symbolic_name T, 0, %T.param.loc9 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {
// CHECK:STDOUT:     %T.patt.loc12_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_14.1, runtime_param<invalid> [symbolic = %T.patt.loc12_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_32.1: type = ptr_type %T [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc12_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.51f)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc12_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc12: <witness> = impl_witness (), @impl.2(constants.%T) [symbolic = @impl.2.%impl_witness (constants.%impl_witness.26c)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc5_13.2)> [symbolic = %I.type (constants.%I.type.bc4)]
// CHECK:STDOUT:   %Self.2: %I.type.bc4 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.bc4) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_14.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.bc4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.1.%I.type.loc8_32.2 (%I.type.bc4) [symbolic = %require_complete (constants.%require_complete.ffb)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness.455)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref.loc8 as %I.type.loc8_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc12_32.2: type = ptr_type @impl.2.%T.loc12_14.2 (%T) [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc12_33.2: type = facet_type <@I, @I(%ptr.loc12_32.2)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.51f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.2.%I.type.loc12_33.2 (%I.type.51f) [symbolic = %require_complete (constants.%require_complete.96d)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2(%T.loc12_14.2) [symbolic = %impl_witness (constants.%impl_witness.26c)]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.bc4
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T.loc5_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.bc4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ffb
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.455
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.1.%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%ptr
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.51f
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc12_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc12_33.2 => constants.%I.type.51f
// CHECK:STDOUT:   %require_complete => constants.%require_complete.96d
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.26c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.2.%ptr.loc12_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%T.loc12_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [template]
// CHECK:STDOUT:   %I.type.9d1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.9d1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %require_complete.48b: <witness> = require_complete_type %I.type.9d1 [symbolic]
// CHECK:STDOUT:   %impl_witness.8fd: <witness> = impl_witness (), @impl.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.ffc: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.6c6: <witness> = require_complete_type %I.type.ffc [symbolic]
// CHECK:STDOUT:   %impl_witness.4b1: <witness> = impl_witness (), @impl.2(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.cbc10c.1: <witness> = impl_witness (), @impl.3(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.cbc10c.2: <witness> = impl_witness (), @impl.4(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.aad5cc.1: <witness> = impl_witness (), @impl.5(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.aad5cc.2: <witness> = impl_witness (), @impl.6(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.0ac: type = import_ref Main//import_generic, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.848: %I.type.dac = import_ref Main//import_generic, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.ea4 = import_ref Main//import_generic, loc8_33, unloaded
// CHECK:STDOUT:   %import_ref.9c1 = import_ref Main//import_generic, inst31 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref Main//import_generic, loc4_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.2c4 = import_ref Main//import_generic, inst14 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.3de5b0.1: type = import_ref Main//import_generic, loc8_24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.c59: type = import_ref Main//import_generic, loc8_32, loaded [symbolic = @impl.1.%I.type (constants.%I.type.9d1)]
// CHECK:STDOUT:   %import_ref.c38 = import_ref Main//import_generic, loc12_35, unloaded
// CHECK:STDOUT:   %import_ref.3de5b0.2: type = import_ref Main//import_generic, loc12_24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.39d: type = import_ref Main//import_generic, loc12_33, loaded [symbolic = @impl.2.%I.type (constants.%I.type.ffc)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.0ac
// CHECK:STDOUT:     .I = imports.%import_ref.848
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_14.1, runtime_param<invalid> [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.0ac [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%import_ref.848 [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.9d1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness (), @impl.3(constants.%T) [symbolic = @impl.3.%impl_witness (constants.%impl_witness.cbc10c.1)]
// CHECK:STDOUT:   impl_decl @impl.4 [template] {
// CHECK:STDOUT:     %T.patt.loc14_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc14_14.1, runtime_param<invalid> [symbolic = %T.patt.loc14_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.0ac [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%import_ref.848 [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc14_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.9d1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc14_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness (), @impl.4(constants.%T) [symbolic = @impl.4.%impl_witness (constants.%impl_witness.cbc10c.2)]
// CHECK:STDOUT:   impl_decl @impl.5 [template] {
// CHECK:STDOUT:     %T.patt.loc20_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc20_14.1, runtime_param<invalid> [symbolic = %T.patt.loc20_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.0ac [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%import_ref.848 [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_14.1 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc20_32.1: type = ptr_type %T [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc20_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc20_33.2 (constants.%I.type.ffc)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc20_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc20: <witness> = impl_witness (), @impl.5(constants.%T) [symbolic = @impl.5.%impl_witness (constants.%impl_witness.aad5cc.1)]
// CHECK:STDOUT:   impl_decl @impl.6 [template] {
// CHECK:STDOUT:     %T.patt.loc26_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc26_14.1, runtime_param<invalid> [symbolic = %T.patt.loc26_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.0ac [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%import_ref.848 [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_14.1 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc26_32.1: type = ptr_type %T [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc26_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc26_33.2 (constants.%I.type.ffc)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc26_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc26: <witness> = impl_witness (), @impl.6(constants.%T) [symbolic = @impl.6.%impl_witness (constants.%impl_witness.aad5cc.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.9d1)]
// CHECK:STDOUT:   %Self: %I.type.9d1 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9c1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(constants.%T: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.9d1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.1.%I.type (%I.type.9d1) [symbolic = %require_complete (constants.%require_complete.48b)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%T) [symbolic = %impl_witness (constants.%impl_witness.8fd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.3de5b0.1 as imports.%import_ref.c59 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.ea4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%T: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %ptr: type = ptr_type @impl.2.%T (%T) [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%ptr)> [symbolic = %I.type (constants.%I.type.ffc)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.2.%I.type (%I.type.ffc) [symbolic = %require_complete (constants.%require_complete.6c6)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2(%T) [symbolic = %impl_witness (constants.%impl_witness.4b1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.3de5b0.2 as imports.%import_ref.39d {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.c38
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_14.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.9d1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.3.%I.type.loc8_32.2 (%I.type.9d1) [symbolic = %require_complete (constants.%require_complete.48b)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.3(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness.cbc10c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc8_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(%T.loc14_14.1: type) {
// CHECK:STDOUT:   %T.loc14_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc14_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %I.type.loc14_32.2: type = facet_type <@I, @I(%T.loc14_14.2)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.9d1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.4.%I.type.loc14_32.2 (%I.type.9d1) [symbolic = %require_complete (constants.%require_complete.48b)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.4(%T.loc14_14.2) [symbolic = %impl_witness (constants.%impl_witness.cbc10c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc14_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(%T.loc20_14.1: type) {
// CHECK:STDOUT:   %T.loc20_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc20_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc20_32.2: type = ptr_type @impl.5.%T.loc20_14.2 (%T) [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc20_33.2: type = facet_type <@I, @I(%ptr.loc20_32.2)> [symbolic = %I.type.loc20_33.2 (constants.%I.type.ffc)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.5.%I.type.loc20_33.2 (%I.type.ffc) [symbolic = %require_complete (constants.%require_complete.6c6)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.5(%T.loc20_14.2) [symbolic = %impl_witness (constants.%impl_witness.aad5cc.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc20_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(%T.loc26_14.1: type) {
// CHECK:STDOUT:   %T.loc26_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc26_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc26_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc26_32.2: type = ptr_type @impl.6.%T.loc26_14.2 (%T) [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc26_33.2: type = facet_type <@I, @I(%ptr.loc26_32.2)> [symbolic = %I.type.loc26_33.2 (constants.%I.type.ffc)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.6.%I.type.loc26_33.2 (%I.type.ffc) [symbolic = %require_complete (constants.%require_complete.6c6)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.6(%T.loc26_14.2) [symbolic = %impl_witness (constants.%impl_witness.aad5cc.2)]
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
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2c4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.9d1
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.9d1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.48b
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.8fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %T.patt => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.ffc
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.ffc
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6c6
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.4b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.2.%ptr) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.9d1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.48b
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.cbc10c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.3.%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%T) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc14_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc14_32.2 => constants.%I.type.9d1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.48b
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.cbc10c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.4.%T.loc14_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(%T.loc14_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%T) {
// CHECK:STDOUT:   %T.loc20_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc20_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc20_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc20_33.2 => constants.%I.type.ffc
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6c6
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.aad5cc.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.5.%ptr.loc20_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(%T.loc20_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%T) {
// CHECK:STDOUT:   %T.loc26_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc26_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc26_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc26_33.2 => constants.%I.type.ffc
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6c6
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.aad5cc.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.6.%ptr.loc26_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(%T.loc26_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.2b8: type = generic_interface_type @J [template]
// CHECK:STDOUT:   %J.generic: %J.type.2b8 = struct_value () [template]
// CHECK:STDOUT:   %J.type.d4b: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %J.type.d4b = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.7f4: <witness> = require_complete_type %J.type.d4b [symbolic]
// CHECK:STDOUT:   %impl_witness.6f1: <witness> = impl_witness (), @impl.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.a4c: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.664: <witness> = require_complete_type %J.type.a4c [symbolic]
// CHECK:STDOUT:   %impl_witness.87e: <witness> = impl_witness (), @impl.2(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %J.decl: %J.type.2b8 = interface_decl @J [template = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt.loc5_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc5_13.1, runtime_param<invalid> [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc5_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [template] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_14.1, runtime_param<invalid> [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, file.%J.decl [template = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc11_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc11_32.2 (constants.%J.type.d4b)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc11: <witness> = impl_witness (), @impl.1(constants.%T) [symbolic = @impl.1.%impl_witness (constants.%impl_witness.6f1)]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {
// CHECK:STDOUT:     %T.patt.loc17_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc17_14.1, runtime_param<invalid> [symbolic = %T.patt.loc17_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, file.%J.decl [template = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc17_14.1 [symbolic = %T.loc17_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc17_32.1: type = ptr_type %T [symbolic = %ptr.loc17_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc17_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc17_33.2 (constants.%J.type.a4c)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc17_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc17_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness (), @impl.2(constants.%T) [symbolic = @impl.2.%impl_witness (constants.%impl_witness.87e)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc5_13.2)> [symbolic = %J.type (constants.%J.type.d4b)]
// CHECK:STDOUT:   %Self.2: %J.type.d4b = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @J.%J.type (%J.type.d4b) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T.loc11_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %J.type.loc11_32.2: type = facet_type <@J, @J(%T.loc11_14.2)> [symbolic = %J.type.loc11_32.2 (constants.%J.type.d4b)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.1.%J.type.loc11_32.2 (%J.type.d4b) [symbolic = %require_complete (constants.%require_complete.7f4)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%T.loc11_14.2) [symbolic = %impl_witness (constants.%impl_witness.6f1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc11_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(%T.loc17_14.1: type) {
// CHECK:STDOUT:   %T.loc17_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc17_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc17_32.2: type = ptr_type @impl.2.%T.loc17_14.2 (%T) [symbolic = %ptr.loc17_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc17_33.2: type = facet_type <@J, @J(%ptr.loc17_32.2)> [symbolic = %J.type.loc17_33.2 (constants.%J.type.a4c)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.2.%J.type.loc17_33.2 (%J.type.a4c) [symbolic = %require_complete (constants.%require_complete.664)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2(%T.loc17_14.2) [symbolic = %impl_witness (constants.%impl_witness.87e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc17_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.d4b
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(%T.loc5_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc11_32.2 => constants.%J.type.d4b
// CHECK:STDOUT:   %require_complete => constants.%require_complete.7f4
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.6f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.1.%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%ptr
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.a4c
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T.loc17_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc17_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc17_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc17_33.2 => constants.%J.type.a4c
// CHECK:STDOUT:   %require_complete => constants.%require_complete.664
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.87e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.2.%ptr.loc17_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%T.loc17_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.2b8: type = generic_interface_type @J [template]
// CHECK:STDOUT:   %J.generic: %J.type.2b8 = struct_value () [template]
// CHECK:STDOUT:   %J.type.dab: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %J.type.dab = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %require_complete.5ad: <witness> = require_complete_type %J.type.dab [symbolic]
// CHECK:STDOUT:   %impl_witness.11f: <witness> = impl_witness (), @impl.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.919: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.c6c: <witness> = require_complete_type %J.type.919 [symbolic]
// CHECK:STDOUT:   %impl_witness.e2f: <witness> = impl_witness (), @impl.2(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.8e29ed.1: <witness> = impl_witness (), @impl.3(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.8e29ed.2: <witness> = impl_witness (), @impl.4(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.64a655.1: <witness> = impl_witness (), @impl.5(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.64a655.2: <witness> = impl_witness (), @impl.6(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.275: type = import_ref Main//import_generic_decl, D, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.51e: %J.type.2b8 = import_ref Main//import_generic_decl, J, loaded [template = constants.%J.generic]
// CHECK:STDOUT:   %import_ref.b49 = import_ref Main//import_generic_decl, inst31 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref Main//import_generic_decl, loc4_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.cab = import_ref Main//import_generic_decl, inst14 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.3b992b.1: type = import_ref Main//import_generic_decl, loc11_24, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.bdf: type = import_ref Main//import_generic_decl, loc11_32, loaded [symbolic = @impl.1.%J.type (constants.%J.type.dab)]
// CHECK:STDOUT:   %import_ref.3b992b.2: type = import_ref Main//import_generic_decl, loc17_24, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.857: type = import_ref Main//import_generic_decl, loc17_33, loaded [symbolic = @impl.2.%J.type (constants.%J.type.919)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .D = imports.%import_ref.275
// CHECK:STDOUT:     .J = imports.%import_ref.51e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_14.1, runtime_param<invalid> [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.275 [template = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%import_ref.51e [template = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc8_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc8_32.2 (constants.%J.type.dab)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness (), @impl.3(constants.%T) [symbolic = @impl.3.%impl_witness (constants.%impl_witness.8e29ed.1)]
// CHECK:STDOUT:   impl_decl @impl.4 [template] {
// CHECK:STDOUT:     %T.patt.loc14_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc14_14.1, runtime_param<invalid> [symbolic = %T.patt.loc14_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.275 [template = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%import_ref.51e [template = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_14.1 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc14_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.dab)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc14_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc14: <witness> = impl_witness (), @impl.4(constants.%T) [symbolic = @impl.4.%impl_witness (constants.%impl_witness.8e29ed.2)]
// CHECK:STDOUT:   impl_decl @impl.5 [template] {
// CHECK:STDOUT:     %T.patt.loc20_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc20_14.1, runtime_param<invalid> [symbolic = %T.patt.loc20_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.275 [template = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%import_ref.51e [template = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_14.1 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc20_32.1: type = ptr_type %T [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc20_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc20_33.2 (constants.%J.type.919)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc20_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc20: <witness> = impl_witness (), @impl.5(constants.%T) [symbolic = @impl.5.%impl_witness (constants.%impl_witness.64a655.1)]
// CHECK:STDOUT:   impl_decl @impl.6 [template] {
// CHECK:STDOUT:     %T.patt.loc25_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc25_14.1, runtime_param<invalid> [symbolic = %T.patt.loc25_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%import_ref.275 [template = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.2b8 = name_ref J, imports.%import_ref.51e [template = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc25_14.1 [symbolic = %T.loc25_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc25_32.1: type = ptr_type %T [symbolic = %ptr.loc25_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc25_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc25_33.2 (constants.%J.type.919)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc25_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc25_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc25: <witness> = impl_witness (), @impl.6(constants.%T) [symbolic = @impl.6.%impl_witness (constants.%impl_witness.64a655.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(constants.%T: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.dab)]
// CHECK:STDOUT:   %Self: %J.type.dab = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.b49
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(constants.%T: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.dab)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.1.%J.type (%J.type.dab) [symbolic = %require_complete (constants.%require_complete.5ad)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%T) [symbolic = %impl_witness (constants.%impl_witness.11f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.3b992b.1 as imports.%import_ref.bdf;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%T: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %ptr: type = ptr_type @impl.2.%T (%T) [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%ptr)> [symbolic = %J.type (constants.%J.type.919)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.2.%J.type (%J.type.919) [symbolic = %require_complete (constants.%require_complete.c6c)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2(%T) [symbolic = %impl_witness (constants.%impl_witness.e2f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.3b992b.2 as imports.%import_ref.857;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %J.type.loc8_32.2: type = facet_type <@J, @J(%T.loc8_14.2)> [symbolic = %J.type.loc8_32.2 (constants.%J.type.dab)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.3.%J.type.loc8_32.2 (%J.type.dab) [symbolic = %require_complete (constants.%require_complete.5ad)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.3(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness.8e29ed.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc8_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.4(%T.loc14_14.1: type) {
// CHECK:STDOUT:   %T.loc14_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc14_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %J.type.loc14_32.2: type = facet_type <@J, @J(%T.loc14_14.2)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.dab)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.4.%J.type.loc14_32.2 (%J.type.dab) [symbolic = %require_complete (constants.%require_complete.5ad)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.4(%T.loc14_14.2) [symbolic = %impl_witness (constants.%impl_witness.8e29ed.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc14_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(%T.loc20_14.1: type) {
// CHECK:STDOUT:   %T.loc20_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc20_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc20_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc20_32.2: type = ptr_type @impl.5.%T.loc20_14.2 (%T) [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc20_33.2: type = facet_type <@J, @J(%ptr.loc20_32.2)> [symbolic = %J.type.loc20_33.2 (constants.%J.type.919)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.5.%J.type.loc20_33.2 (%J.type.919) [symbolic = %require_complete (constants.%require_complete.c6c)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.5(%T.loc20_14.2) [symbolic = %impl_witness (constants.%impl_witness.64a655.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc20_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(%T.loc25_14.1: type) {
// CHECK:STDOUT:   %T.loc25_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc25_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc25_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc25_32.2: type = ptr_type @impl.6.%T.loc25_14.2 (%T) [symbolic = %ptr.loc25_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc25_33.2: type = facet_type <@J, @J(%ptr.loc25_32.2)> [symbolic = %J.type.loc25_33.2 (constants.%J.type.919)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.6.%J.type.loc25_33.2 (%J.type.919) [symbolic = %require_complete (constants.%require_complete.c6c)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.6(%T.loc25_14.2) [symbolic = %impl_witness (constants.%impl_witness.64a655.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc25_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.cab
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.dab
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.dab
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5ad
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.11f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %T.patt => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.919
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.919
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c6c
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.e2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.2.%ptr) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc8_32.2 => constants.%J.type.dab
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5ad
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.8e29ed.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.3.%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(constants.%T) {
// CHECK:STDOUT:   %T.loc14_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc14_14.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc14_32.2 => constants.%J.type.dab
// CHECK:STDOUT:   %require_complete => constants.%require_complete.5ad
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.8e29ed.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.4.%T.loc14_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.4(%T.loc14_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%T) {
// CHECK:STDOUT:   %T.loc20_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc20_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc20_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc20_33.2 => constants.%J.type.919
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c6c
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.64a655.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.5.%ptr.loc20_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(%T.loc20_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%T) {
// CHECK:STDOUT:   %T.loc25_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc25_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc25_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc25_33.2 => constants.%J.type.919
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c6c
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.64a655.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(@impl.6.%ptr.loc25_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(%T.loc25_14.2) {}
// CHECK:STDOUT:

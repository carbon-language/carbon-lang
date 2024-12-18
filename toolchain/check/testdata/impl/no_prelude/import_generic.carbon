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

// CHECK:STDERR: import_generic.carbon:[[@LINE+4]]:1: error: no definition found for declaration of impl [MissingImplDefinition]
// CHECK:STDERR: impl forall [T:! type] C as I(T);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C as I(T);
impl forall [T:! type] C as I(T*) {}

// --- import_generic.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: import_generic.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] C as I(T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] C as I(T) {}

// --- fail_import_generic.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_generic.impl.carbon:[[@LINE+3]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl forall [T:! type] C as I(T*) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl forall [T:! type] C as I(T*) {}

// CHECK:STDOUT: --- import_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %I.type.2 [symbolic]
// CHECK:STDOUT:   %impl_witness.1: <witness> = impl_witness (), @impl.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.3: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %I.type.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (), @impl.2(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.1 = interface_decl @I [template = constants.%I.generic] {
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
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc11_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc11_32.2 (constants.%I.type.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc11: <witness> = impl_witness (), @impl.1(constants.%T) [symbolic = @impl.1.%impl_witness (constants.%impl_witness.1)]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {
// CHECK:STDOUT:     %T.patt.loc12_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc12_14.1, runtime_param<invalid> [symbolic = %T.patt.loc12_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_14.1 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_32.1: type = ptr_type %T [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc12_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.3)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc12_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc12: <witness> = impl_witness (), @impl.2(constants.%T) [symbolic = @impl.2.%impl_witness (constants.%impl_witness.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5_13.1: type) {
// CHECK:STDOUT:   %T.loc5_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc5_13.2)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self.2: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
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
// CHECK:STDOUT:   %I.type.loc11_32.2: type = facet_type <@I, @I(%T.loc11_14.2)> [symbolic = %I.type.loc11_32.2 (constants.%I.type.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.1.%I.type.loc11_32.2 (%I.type.2) [symbolic = %require_complete (constants.%require_complete.1)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%T.loc11_14.2) [symbolic = %impl_witness (constants.%impl_witness.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc11_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(%T.loc12_14.1: type) {
// CHECK:STDOUT:   %T.loc12_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc12_32.2: type = ptr_type @impl.2.%T.loc12_14.2 (%T) [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc12_33.2: type = facet_type <@I, @I(%ptr.loc12_32.2)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.2.%I.type.loc12_33.2 (%I.type.3) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2(%T.loc12_14.2) [symbolic = %impl_witness (constants.%impl_witness.2)]
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
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T.loc5_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc11_32.2 => constants.%I.type.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.1.%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_13.2 => constants.%ptr
// CHECK:STDOUT:   %T.patt.loc5_13.2 => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T.loc12_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc12_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc12_33.2 => constants.%I.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.2.%ptr.loc12_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%T.loc12_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %I.type.2 [symbolic]
// CHECK:STDOUT:   %impl_witness.1: <witness> = impl_witness (), @impl.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.3: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %I.type.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (), @impl.2(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.3: <witness> = impl_witness (), @impl.3(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//import_generic, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: %I.type.1 = import_ref Main//import_generic, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//import_generic, inst31 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.5: <witness> = import_ref Main//import_generic, loc4_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//import_generic, inst14 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.7: type = import_ref Main//import_generic, loc11_24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.8: type = import_ref Main//import_generic, loc11_32, loaded [symbolic = @impl.1.%I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//import_generic, loc12_35, unloaded
// CHECK:STDOUT:   %import_ref.10: type = import_ref Main//import_generic, loc12_24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.11: type = import_ref Main//import_generic, loc12_33, loaded [symbolic = @impl.2.%I.type (constants.%I.type.3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .I = imports.%import_ref.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %T.patt.loc8_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_14.1, runtime_param<invalid> [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, imports.%import_ref.2 [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.3(constants.%T) [symbolic = @impl.3.%impl_witness (constants.%impl_witness.3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.4
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(constants.%T: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.1.%I.type (%I.type.2) [symbolic = %require_complete (constants.%require_complete.1)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%T) [symbolic = %impl_witness (constants.%impl_witness.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.7 as imports.%import_ref.8;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%T: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %ptr: type = ptr_type @impl.2.%T (%T) [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%ptr)> [symbolic = %I.type (constants.%I.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.2.%I.type (%I.type.3) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2(%T) [symbolic = %impl_witness (constants.%impl_witness.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.10 as imports.%import_ref.11 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_14.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.3.%I.type.loc8_32.2 (%I.type.2) [symbolic = %require_complete (constants.%require_complete.1)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.3(%T.loc8_14.2) [symbolic = %impl_witness (constants.%impl_witness.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc8_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "import_generic.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %T.patt => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.2.%ptr) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.3.%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(%T.loc8_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %require_complete.1: <witness> = require_complete_type %I.type.2 [symbolic]
// CHECK:STDOUT:   %impl_witness.1: <witness> = impl_witness (), @impl.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.3: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %I.type.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (), @impl.2(%T) [symbolic]
// CHECK:STDOUT:   %impl_witness.3: <witness> = impl_witness (), @impl.3(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//import_generic, C, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2: %I.type.1 = import_ref Main//import_generic, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//import_generic, inst31 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.5: <witness> = import_ref Main//import_generic, loc4_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//import_generic, inst14 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.7: type = import_ref Main//import_generic, loc11_24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.8: type = import_ref Main//import_generic, loc11_32, loaded [symbolic = @impl.1.%I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//import_generic, loc12_35, unloaded
// CHECK:STDOUT:   %import_ref.10: type = import_ref Main//import_generic, loc12_24, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.11: type = import_ref Main//import_generic, loc12_33, loaded [symbolic = @impl.2.%I.type (constants.%I.type.3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .I = imports.%import_ref.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_6.1 = import <invalid>
// CHECK:STDOUT:   %default.import.loc2_6.2 = import <invalid>
// CHECK:STDOUT:   impl_decl @impl.3 [template] {
// CHECK:STDOUT:     %T.patt.loc7_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc7_14.1, runtime_param<invalid> [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%import_ref.1 [template = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, imports.%import_ref.2 [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_14.1 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc7_32.1: type = ptr_type %T [symbolic = %ptr.loc7_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc7_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc7_33.2 (constants.%I.type.3)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.3(constants.%T) [symbolic = @impl.3.%impl_witness (constants.%impl_witness.3)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.4
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.1(constants.%T: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.1.%I.type (%I.type.2) [symbolic = %require_complete (constants.%require_complete.1)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.1(%T) [symbolic = %impl_witness (constants.%impl_witness.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.7 as imports.%import_ref.8;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%T: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %ptr: type = ptr_type @impl.2.%T (%T) [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%ptr)> [symbolic = %I.type (constants.%I.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.2.%I.type (%I.type.3) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.2(%T) [symbolic = %impl_witness (constants.%impl_witness.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.10 as imports.%import_ref.11 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(%T.loc7_14.1: type) {
// CHECK:STDOUT:   %T.loc7_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc7_32.2: type = ptr_type @impl.3.%T.loc7_14.2 (%T) [symbolic = %ptr.loc7_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc7_33.2: type = facet_type <@I, @I(%ptr.loc7_32.2)> [symbolic = %I.type.loc7_33.2 (constants.%I.type.3)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.3.%I.type.loc7_33.2 (%I.type.3) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (), @impl.3(%T.loc7_14.2) [symbolic = %impl_witness (constants.%impl_witness.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "import_generic.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.1
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
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.2.%ptr) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%T) {
// CHECK:STDOUT:   %T.loc7_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_14.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc7_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc7_33.2 => constants.%I.type.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.3.%ptr.loc7_32.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(%T.loc7_14.2) {}
// CHECK:STDOUT:

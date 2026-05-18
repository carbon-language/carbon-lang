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

class C {}

impl C as I({}) {}

impl C as J({}) {}

// --- basic_import_generic_constraint.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {}
constraint J(T:! type) {
  extend require impls I(T);
}

// --- basic_import_generic_constraint.impl.carbon
impl library "[[@TEST_NAME]]";

class C {}

impl C as J({}) {}

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
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.cd7: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.cd7 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ad: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.cb6: %I.type.1ad = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %J.type.29e: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.29e = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.e8a: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0fd: %J.type.e8a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.0fd [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.cd7 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_17.1: type = splice_block %.loc3_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type.29e = interface_decl @J [concrete = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc3_14.2: type) {
// CHECK:STDOUT:   %T.loc3_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc3_14.1)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Self.loc3_23.2: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.cb6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.cb6)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc4_14.1)> [symbolic = %J.type (constants.%J.type.e8a)]
// CHECK:STDOUT:   %Self.loc4_23.2: @J.%J.type (%J.type.e8a) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.0fd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_23.1: @J.%J.type (%J.type.e8a) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.0fd)]
// CHECK:STDOUT:     %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %J.WithSelf.Self.as_type.impls.I.type.require.decl = require_decl @J.WithSelf.Self.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc5_18.1 impls %I.type.loc5_27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc5_18.1: type = facet_access_type @J.%Self.loc4_23.1 [symbolic = %Self.as_type.loc5_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.cd7 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @J.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc5_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc5: type = specific_constant @J.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc5_27.1, @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.0fd) [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_23.1
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
// CHECK:STDOUT: generic require @J.WithSelf.Self.as_type.impls.I.type.require(@J.%T.loc4_14.2: type, @J.%Self.loc4_23.1: @J.%J.type (%J.type.e8a)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.e8a)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.as_type.impls.I.type.require.%J.type (%J.type.e8a) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0fd)]
// CHECK:STDOUT:   %Self.as_type.loc5_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type.loc5_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.1ad)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.cb6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.0fd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.0fd) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.e8a
// CHECK:STDOUT:   %Self => constants.%Self.0fd
// CHECK:STDOUT:   %Self.as_type.loc5_18.2 => constants.%Self.as_type
// CHECK:STDOUT:   %I.type.loc5_27.2 => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_interface.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type.cd7: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.cd7 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1ad: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0a1: %I.type.1ad = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.656: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.903: %I.type.656 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.656 = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %J.type.29e: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.29e = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.e8a: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.5c9: %J.type.e8a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.5c9 [symbolic]
// CHECK:STDOUT:   %J.type.2ce: type = facet_type <@J, @J(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %complete_type.248: <witness> = complete_type_witness %I.type.656 [concrete]
// CHECK:STDOUT:   %Self.cb2: %J.type.2ce = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type.2ce = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.cd7 = import_ref Main//basic_import_generic_interface, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.J: %J.type.29e = import_ref Main//basic_import_generic_interface, J, loaded [concrete = constants.%J.generic]
// CHECK:STDOUT:   %Main.import_ref.1f9 = import_ref Main//basic_import_generic_interface, loc3_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//basic_import_generic_interface, loc3_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.bc0 = import_ref Main//basic_import_generic_interface, loc5_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.45d: type = import_ref Main//basic_import_generic_interface, loc5_18, loaded [symbolic = @J.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %Main.import_ref.959: type = import_ref Main//basic_import_generic_interface, loc5_27, loaded [symbolic = @J.WithSelf.Self.as_type.impls.I.type.require.%I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//basic_import_generic_interface, loc4_14, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.2e7566.2: @J.%J.type (%J.type.e8a) = import_ref Main//basic_import_generic_interface, loc4_23, loaded [symbolic = @J.%Self (constants.%Self.5c9)]
// CHECK:STDOUT:   %Main.import_ref.bcf = import_ref Main//basic_import_generic_interface, loc4_23, unloaded
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
// CHECK:STDOUT:     %I.ref: %I.type.cd7 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %.loc5_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_15: type = converted %.loc5_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%empty_struct_type)> [concrete = constants.%I.type.656]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: %J.type.29e = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %.loc7_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_15: type = converted %.loc7_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.type: type = facet_type <@J, @J(constants.%empty_struct_type)> [concrete = constants.%J.type.2ce]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.2: type) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.1f9
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(imports.%Main.import_ref.b3bc94.5: type) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.e8a)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.e8a) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.5c9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.bcf
// CHECK:STDOUT:     extend imports.%Main.import_ref.bc0
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.45d impls imports.%Main.import_ref.959
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.WithSelf.Self.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.4: type, imports.%Main.import_ref.2e7566.2: @J.%J.type (%J.type.e8a)) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.e8a)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.as_type.impls.I.type.require.%J.type (%J.type.e8a) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.5c9)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ad)]
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
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %Self => constants.%Self.0a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.0a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.656
// CHECK:STDOUT:   %Self => constants.%Self.903
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%Self.0a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.5c9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.5c9) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.e8a
// CHECK:STDOUT:   %Self => constants.%Self.5c9
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.2ce
// CHECK:STDOUT:   %Self => constants.%Self.cb2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.656
// CHECK:STDOUT:   %require_complete => constants.%complete_type.248
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %J.type => constants.%J.type.2ce
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.656
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%Self.5c9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.656
// CHECK:STDOUT:   %require_complete => constants.%complete_type.248
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.656
// CHECK:STDOUT:   %require_complete => constants.%complete_type.248
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.cd7: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.cd7 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ad: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.cb6: %I.type.1ad = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %J.type.593: type = generic_named_constaint_type @J [concrete]
// CHECK:STDOUT:   %empty_struct: %J.type.593 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.d68: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.107: %J.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.107 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.cd7 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_17.1: type = splice_block %.loc3_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc3_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc3_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type.593 = constraint_decl @J [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_18.1: type = splice_block %.loc4_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc3_14.2: type) {
// CHECK:STDOUT:   %T.loc3_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc3_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc3_14.1)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Self.loc3_23.2: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.cb6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc3_23.1: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self.loc3_23.2 (constants.%Self.cb6)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc3_23.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @J(%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T.loc4_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc4_15.1)> [symbolic = %J.type (constants.%J.type.d68)]
// CHECK:STDOUT:   %Self.loc4_24.2: @J.%J.type (%J.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc4_24.2 (constants.%Self.107)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc4_24.1: @J.%J.type (%J.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc4_24.2 (constants.%Self.107)]
// CHECK:STDOUT:     %J.WithSelf.decl = constraint_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %J.WithSelf.Self.as_type.impls.I.type.require.decl = require_decl @J.WithSelf.Self.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc5_18.1 impls %I.type.loc5_27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc5_18.1: type = facet_access_type @J.%Self.loc4_24.1 [symbolic = %Self.as_type.loc5_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.cd7 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @J.%T.loc4_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc5_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc5: type = specific_constant @J.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc5_27.1, @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.107) [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_24.1
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
// CHECK:STDOUT: generic require @J.WithSelf.Self.as_type.impls.I.type.require(@J.%T.loc4_15.2: type, @J.%Self.loc4_24.1: @J.%J.type (%J.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.d68)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.as_type.impls.I.type.require.%J.type (%J.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.107)]
// CHECK:STDOUT:   %Self.as_type.loc5_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type.loc5_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc5_27.2 (constants.%I.type.1ad)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc3_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %Self.loc3_23.2 => constants.%Self.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.cb6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.107) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.107) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.107
// CHECK:STDOUT:   %Self.as_type.loc5_18.2 => constants.%Self.as_type
// CHECK:STDOUT:   %I.type.loc5_27.2 => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_constraint.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.type.593: type = generic_named_constaint_type @J [concrete]
// CHECK:STDOUT:   %empty_struct.2a5: %J.type.593 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.d682d6.1: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f642.1: %J.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.type.1ad: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0a1: %I.type.1ad = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e1f642.1 [symbolic]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.d682d6.2: type = facet_type <@J, @J(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %I.type.656: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.903: %I.type.656 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.248: <witness> = complete_type_witness %I.type.656 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.e1f642.2: %J.type.d682d6.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.656 = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//basic_import_generic_constraint, I, unloaded
// CHECK:STDOUT:   %Main.J: %J.type.593 = import_ref Main//basic_import_generic_constraint, J, loaded [concrete = constants.%empty_struct.2a5]
// CHECK:STDOUT:   %Main.import_ref.f80 = import_ref Main//basic_import_generic_constraint, loc5_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.1f9 = import_ref Main//basic_import_generic_constraint, loc3_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//basic_import_generic_constraint, loc3_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.3d8: type = import_ref Main//basic_import_generic_constraint, loc5_18, loaded [symbolic = @J.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %Main.import_ref.959: type = import_ref Main//basic_import_generic_constraint, loc5_27, loaded [symbolic = @J.WithSelf.Self.as_type.impls.I.type.require.%I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//basic_import_generic_constraint, loc4_15, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.49ba3f.2: @J.%J.type (%J.type.d682d6.1) = import_ref Main//basic_import_generic_constraint, loc4_24, loaded [symbolic = @J.%Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Main.import_ref.6e8 = import_ref Main//basic_import_generic_constraint, loc4_24, unloaded
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
// CHECK:STDOUT:     %J.ref: %J.type.593 = name_ref J, imports.%Main.J [concrete = constants.%empty_struct.2a5]
// CHECK:STDOUT:     %.loc5_14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:     %.loc5_15: type = converted %.loc5_14, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.type: type = facet_type <@J, @J(constants.%empty_struct_type)> [concrete = constants.%J.type.d682d6.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.3: type) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.1f9
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @J(imports.%Main.import_ref.b3bc94.5: type) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.6e8
// CHECK:STDOUT:     extend imports.%Main.import_ref.f80
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.3d8 impls imports.%Main.import_ref.959
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.WithSelf.Self.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.4: type, imports.%Main.import_ref.49ba3f.2: @J.%J.type (%J.type.d682d6.1)) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.as_type.impls.I.type.require.%J.type (%J.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ad)]
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
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %Self => constants.%Self.0a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.0a1) {
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
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%empty_struct_type) {
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
// CHECK:STDOUT:   %I.type => constants.%I.type.656
// CHECK:STDOUT:   %require_complete => constants.%complete_type.248
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.656
// CHECK:STDOUT:   %Self => constants.%Self.903
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%Self.0a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.as_type.impls.I.type.require(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %J.type => constants.%J.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.656
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%Self.e1f642.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.656
// CHECK:STDOUT:   %require_complete => constants.%complete_type.248
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
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.cd7: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.cd7 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ad: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.cb6: %I.type.1ad = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.a4e: <witness> = impl_witness @C.as.I.impl.cb7.%I.impl_witness_table, @C.as.I.impl.cb7(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.fa0: <witness> = require_complete_type %I.type.1ad [symbolic]
// CHECK:STDOUT:   %I.facet.1dd: %I.type.1ad = facet_value %C, (%I.impl_witness.a4e) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.f4b: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %I.impl_witness.a27: <witness> = impl_witness @C.as.I.impl.a2d.%I.impl_witness_table, @C.as.I.impl.a2d(%T) [symbolic]
// CHECK:STDOUT:   %Self.e6e: %I.type.f4b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.13d: <witness> = require_complete_type %I.type.f4b [symbolic]
// CHECK:STDOUT:   %I.facet.585: %I.type.f4b = facet_value %C, (%I.impl_witness.a27) [symbolic]
// CHECK:STDOUT:   %N.type.197: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.197 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d68: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.107: %N.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.107 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.cd7 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_17.1: type = splice_block %.loc5_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.cb7 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc8: %I.type.cd7 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc8: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.cb7 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc9: %I.type.cd7 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc9 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc9: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc9: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.a2d [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.cd7 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_15.1 [symbolic = %T.loc12_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc12_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc12_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.f4b)]
// CHECK:STDOUT:     %.loc12_18.1: type = splice_block %.loc12_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc12_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: %N.type.197 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_18.1: type = splice_block %.loc14_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc14_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc5_14.1)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Self.loc5_23.2: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.cb6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.cb6)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_23.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc14_15.2: type) {
// CHECK:STDOUT:   %T.loc14_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc14_15.1)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self.loc14_24.2: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc14_24.2 (constants.%Self.107)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc14_24.1: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc14_24.2 (constants.%Self.107)]
// CHECK:STDOUT:     %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.as_type.impls.I.type.require.decl = require_decl @N.WithSelf.Self.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc15_18.1 impls %I.type.loc15_27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc15_18.1: type = facet_access_type @N.%Self.loc14_24.1 [symbolic = %Self.as_type.loc15_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.cd7 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc14_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc15_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc15_27.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc15: type = specific_constant @N.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc15_27.1, @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.107) [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc14_24.1
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
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.I.type.require(@N.%T.loc14_15.2: type, @N.%Self.loc14_24.1: @N.%N.type (%N.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.I.type.require.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.107)]
// CHECK:STDOUT:   %Self.as_type.loc15_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc15_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type.loc15_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc15_27.2 (constants.%I.type.1ad)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.cb7(%T.loc8_15.1: type) {
// CHECK:STDOUT:   %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_15.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:   %I.impl_witness.loc8_33.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.cb7(%T.loc8_15.2) [symbolic = %I.impl_witness.loc8_33.2 (constants.%I.impl_witness.a4e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc8_32.2 [symbolic = %require_complete (constants.%require_complete.fa0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref.loc8 as %I.type.loc8_32.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.cb7 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc8_33.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.cb7(constants.%T) [symbolic = %I.impl_witness.loc8_33.2 (constants.%I.impl_witness.a4e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc8_32.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc8_33.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.a2d(%T.loc12_15.1: type) {
// CHECK:STDOUT:   %T.loc12_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc12_32.2: type = ptr_type %T.loc12_15.2 [symbolic = %ptr.loc12_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc12_33.2: type = facet_type <@I, @I(%ptr.loc12_32.2)> [symbolic = %I.type.loc12_33.2 (constants.%I.type.f4b)]
// CHECK:STDOUT:   %I.impl_witness.loc12_35.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.a2d(%T.loc12_15.2) [symbolic = %I.impl_witness.loc12_35.2 (constants.%I.impl_witness.a27)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc12_33.2 [symbolic = %require_complete (constants.%require_complete.13d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc12_33.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.a2d [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc12_35.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.a2d(constants.%T) [symbolic = %I.impl_witness.loc12_35.2 (constants.%I.impl_witness.a27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc12_33.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc12_35.1
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
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.cb6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.cb7(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.1ad
// CHECK:STDOUT:   %I.impl_witness.loc8_33.2 => constants.%I.impl_witness.a4e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%I.facet.1dd) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.f4b
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.e6e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.a2d(constants.%T) {
// CHECK:STDOUT:   %T.loc12_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc12_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc12_33.2 => constants.%I.type.f4b
// CHECK:STDOUT:   %I.impl_witness.loc12_35.2 => constants.%I.impl_witness.a27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr, constants.%Self.cb6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr, constants.%I.facet.585) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.loc14_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.107) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.107) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.107
// CHECK:STDOUT:   %Self.as_type.loc15_18.2 => constants.%Self.as_type
// CHECK:STDOUT:   %I.type.loc15_27.2 => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.cd7: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.cd7 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ad: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0a1: %I.type.1ad = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness.f7e: <witness> = impl_witness imports.%I.impl_witness_table.d56, @C.as.I.impl.775(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.fa0: <witness> = require_complete_type %I.type.1ad [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.f4b: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %Self.0b3: %I.type.f4b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.fca: <witness> = impl_witness imports.%I.impl_witness_table.4eb, @C.as.I.impl.fcf(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.13d: <witness> = require_complete_type %I.type.f4b [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.type.197: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.197 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e1f [symbolic]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%ptr)> [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//import_generic, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.I: %I.type.cd7 = import_ref Main//import_generic, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.N: %N.type.197 = import_ref Main//import_generic, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.1f9 = import_ref Main//import_generic, loc5_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//import_generic, loc5_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//import_generic, loc8_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.d96 = import_ref Main//import_generic, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %I.impl_witness_table.d56 = impl_witness_table (), @C.as.I.impl.775 [concrete]
// CHECK:STDOUT:   %Main.import_ref.f0fb11.1: type = import_ref Main//import_generic, loc8_24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.959e86.1: type = import_ref Main//import_generic, loc8_32, loaded [symbolic = @C.as.I.impl.775.%I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Main.import_ref.1eb = import_ref Main//import_generic, loc8_32, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//import_generic, loc8_15, loaded [symbolic = @C.as.I.impl.775.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.890 = import_ref Main//import_generic, loc12_35, unloaded
// CHECK:STDOUT:   %I.impl_witness_table.4eb = impl_witness_table (), @C.as.I.impl.fcf [concrete]
// CHECK:STDOUT:   %Main.import_ref.f0fb11.2: type = import_ref Main//import_generic, loc12_24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.7b1: type = import_ref Main//import_generic, loc12_33, loaded [symbolic = @C.as.I.impl.fcf.%I.type (constants.%I.type.f4b)]
// CHECK:STDOUT:   %Main.import_ref.674 = import_ref Main//import_generic, loc12_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//import_generic, loc12_15, loaded [symbolic = @C.as.I.impl.fcf.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.7ef = import_ref Main//import_generic, loc15_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.3d8: type = import_ref Main//import_generic, loc15_18, loaded [symbolic = @N.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %Main.import_ref.959e86.2: type = import_ref Main//import_generic, loc15_27, loaded [symbolic = @N.WithSelf.Self.as_type.impls.I.type.require.%I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.6: type = import_ref Main//import_generic, loc14_15, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.49ba3f.2: @N.%N.type (%N.type.d682d6.1) = import_ref Main//import_generic, loc14_24, loaded [symbolic = @N.%Self (constants.%Self.e1f)]
// CHECK:STDOUT:   %Main.import_ref.6e8 = import_ref Main//import_generic, loc14_24, unloaded
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
// CHECK:STDOUT:   impl_decl @C.as.I.impl.cb7ff8.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.cd7 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc8_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.cb7ff8.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.cd7 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_15.1 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc14_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:     %.loc14_18.1: type = splice_block %.loc14_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc14_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.a2d5c5.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.cd7 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_15.1 [symbolic = %T.loc20_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc20_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc20_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc20_33.2 (constants.%I.type.f4b)]
// CHECK:STDOUT:     %.loc20_18.1: type = splice_block %.loc20_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc20_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc20_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.a2d5c5.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.cd7 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_15.1 [symbolic = %T.loc26_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc26_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc26_33.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc26_33.2 (constants.%I.type.f4b)]
// CHECK:STDOUT:     %.loc26_18.1: type = splice_block %.loc26_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc26_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc26_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.186c6d.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.197 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc32_15.1 [symbolic = %T.loc32_15.2 (constants.%T)]
// CHECK:STDOUT:     %N.type.loc32_32.1: type = facet_type <@N, @N(constants.%T)> [symbolic = %N.type.loc32_32.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:     %.loc32_18.1: type = splice_block %.loc32_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc32_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc32_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc32_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.186c6d.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.197 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc37_15.1 [symbolic = %T.loc37_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc37_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc37_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %N.type.loc37_33.1: type = facet_type <@N, @N(constants.%ptr)> [symbolic = %N.type.loc37_33.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:     %.loc37_18.1: type = splice_block %.loc37_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc37_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc37_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc37_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.2: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.1f9
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(imports.%Main.import_ref.b3bc94.7: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.6e8
// CHECK:STDOUT:     extend imports.%Main.import_ref.7ef
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.3d8 impls imports.%Main.import_ref.959e86.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.6: type, imports.%Main.import_ref.49ba3f.2: @N.%N.type (%N.type.d682d6.1)) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.I.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.775(imports.%Main.import_ref.b3bc94.3: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table.d56, @C.as.I.impl.775(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.f7e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.fa0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.f0fb11.1 as imports.%Main.import_ref.959e86.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend imports.%Main.import_ref.1eb
// CHECK:STDOUT:     witness = imports.%Main.import_ref.1cb
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.fcf(imports.%Main.import_ref.b3bc94.4: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%ptr)> [symbolic = %I.type (constants.%I.type.f4b)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table.4eb, @C.as.I.impl.fcf(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.fca)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.13d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.f0fb11.2 as imports.%Main.import_ref.7b1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend imports.%Main.import_ref.674
// CHECK:STDOUT:     witness = imports.%Main.import_ref.890
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.cb7ff8.1(%T.loc8_15.1: type) {
// CHECK:STDOUT:   %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc8_32.2: type = facet_type <@I, @I(%T.loc8_15.2)> [symbolic = %I.type.loc8_32.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc8_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.cb7ff8.2(%T.loc14_15.1: type) {
// CHECK:STDOUT:   %T.loc14_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc14_32.2: type = facet_type <@I, @I(%T.loc14_15.2)> [symbolic = %I.type.loc14_32.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc14_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc14_32.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.a2d5c5.1(%T.loc20_15.1: type) {
// CHECK:STDOUT:   %T.loc20_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc20_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc20_32.2: type = ptr_type %T.loc20_15.2 [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc20_33.2: type = facet_type <@I, @I(%ptr.loc20_32.2)> [symbolic = %I.type.loc20_33.2 (constants.%I.type.f4b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc20_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.a2d5c5.2(%T.loc26_15.1: type) {
// CHECK:STDOUT:   %T.loc26_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc26_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc26_32.2: type = ptr_type %T.loc26_15.2 [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc26_33.2: type = facet_type <@I, @I(%ptr.loc26_32.2)> [symbolic = %I.type.loc26_33.2 (constants.%I.type.f4b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc26_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc26_33.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.186c6d.1(%T.loc32_15.1: type) {
// CHECK:STDOUT:   %T.loc32_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc32_15.2 (constants.%T)]
// CHECK:STDOUT:   %N.type.loc32_32.2: type = facet_type <@N, @N(%T.loc32_15.2)> [symbolic = %N.type.loc32_32.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc32_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.186c6d.2(%T.loc37_15.1: type) {
// CHECK:STDOUT:   %T.loc37_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc37_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc37_32.2: type = ptr_type %T.loc37_15.2 [symbolic = %ptr.loc37_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %N.type.loc37_33.2: type = facet_type <@N, @N(%ptr.loc37_32.2)> [symbolic = %N.type.loc37_33.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc37_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "import_generic.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.d96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %Self => constants.%Self.0a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.0a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.775(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.f7e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.f4b
// CHECK:STDOUT:   %Self => constants.%Self.0b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.fcf(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.f4b
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.fca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.cb7ff8.1(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc8_32.2 => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.cb7ff8.2(constants.%T) {
// CHECK:STDOUT:   %T.loc14_15.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc14_32.2 => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.a2d5c5.1(constants.%T) {
// CHECK:STDOUT:   %T.loc20_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc20_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc20_33.2 => constants.%I.type.f4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.a2d5c5.2(constants.%T) {
// CHECK:STDOUT:   %T.loc26_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc26_32.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc26_33.2 => constants.%I.type.f4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
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
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %require_complete => constants.%require_complete.fa0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.186c6d.1(constants.%T) {
// CHECK:STDOUT:   %T.loc32_15.2 => constants.%T
// CHECK:STDOUT:   %N.type.loc32_32.2 => constants.%N.type.d682d6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.f4b
// CHECK:STDOUT:   %require_complete => constants.%require_complete.13d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%ptr, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.f4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.186c6d.2(constants.%T) {
// CHECK:STDOUT:   %T.loc37_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc37_32.2 => constants.%ptr
// CHECK:STDOUT:   %N.type.loc37_33.2 => constants.%N.type.d682d6.2
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
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.cd7: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.cd7 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ad: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.cb6: %I.type.1ad = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.1ad [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.1ad = facet_value %C, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %N.type.197: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.197 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d68: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.107: %N.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.107 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.cd7 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_17.1: type = splice_block %.loc5_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.cd7 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_15.1 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc7_32.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc7_32.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:     %.loc7_18.1: type = splice_block %.loc7_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: %N.type.197 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc5_14.1)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Self.loc5_23.2: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.cb6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.cb6)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_23.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc9_15.2: type) {
// CHECK:STDOUT:   %T.loc9_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc9_15.1)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self.loc9_24.2: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc9_24.2 (constants.%Self.107)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc9_24.1: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc9_24.2 (constants.%Self.107)]
// CHECK:STDOUT:     %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.as_type.impls.I.type.require.decl = require_decl @N.WithSelf.Self.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc10_18.1 impls %I.type.loc10_27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc10_18.1: type = facet_access_type @N.%Self.loc9_24.1 [symbolic = %Self.as_type.loc10_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.cd7 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc9_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc10_27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc10_27.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10: type = specific_constant @N.WithSelf.Self.as_type.impls.I.type.require.%I.type.loc10_27.1, @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.107) [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc9_24.1
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
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.I.type.require(@N.%T.loc9_15.2: type, @N.%Self.loc9_24.1: @N.%N.type (%N.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.I.type.require.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.107)]
// CHECK:STDOUT:   %Self.as_type.loc10_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type.loc10_27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc10_27.2 (constants.%I.type.1ad)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc7_15.1: type) {
// CHECK:STDOUT:   %T.loc7_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc7_32.2: type = facet_type <@I, @I(%T.loc7_15.2)> [symbolic = %I.type.loc7_32.2 (constants.%I.type.1ad)]
// CHECK:STDOUT:   %I.impl_witness.loc7_34.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%T.loc7_15.2) [symbolic = %I.impl_witness.loc7_34.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc7_32.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc7_32.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc7_34.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T) [symbolic = %I.impl_witness.loc7_34.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc7_32.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc7_34.1
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
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.cb6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc7_15.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc7_32.2 => constants.%I.type.1ad
// CHECK:STDOUT:   %I.impl_witness.loc7_34.2 => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.loc9_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.107) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%T, constants.%Self.107) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.107
// CHECK:STDOUT:   %Self.as_type.loc10_18.2 => constants.%Self.as_type
// CHECK:STDOUT:   %I.type.loc10_27.2 => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_generic_with_different_specific.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.cd7: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.cd7 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ad: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0a1: %I.type.1ad = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness.f7e: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl.775(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.fa0: <witness> = require_complete_type %I.type.1ad [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.f4b: type = facet_type <@I, @I(%ptr.e8f)> [symbolic]
// CHECK:STDOUT:   %I.impl_witness.a27: <witness> = impl_witness @C.as.I.impl.a2d.%I.impl_witness_table, @C.as.I.impl.a2d(%T) [symbolic]
// CHECK:STDOUT:   %Self.0b3: %I.type.f4b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.13d: <witness> = require_complete_type %I.type.f4b [symbolic]
// CHECK:STDOUT:   %I.facet.585: %I.type.f4b = facet_value %C, (%I.impl_witness.a27) [symbolic]
// CHECK:STDOUT:   %N.type.197: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.197 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f642.1: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e1f642.1 [symbolic]
// CHECK:STDOUT:   %ptr.125: type = ptr_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%ptr.125)> [symbolic]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %I.type.aa2: type = facet_type <@I, @I(%ptr.125)> [symbolic]
// CHECK:STDOUT:   %require_complete.584: <witness> = require_complete_type %I.type.aa2 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.8cd: <witness> = impl_witness @C.as.I.impl.186.%I.impl_witness_table, @C.as.I.impl.186(%T) [symbolic]
// CHECK:STDOUT:   %Self.e1f642.2: %N.type.d682d6.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.533: <witness> = require_complete_type %N.type.d682d6.2 [symbolic]
// CHECK:STDOUT:   %I.facet.192: %I.type.aa2 = facet_value %C, (%I.impl_witness.8cd) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//import_generic_with_different_specific, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.I: %I.type.cd7 = import_ref Main//import_generic_with_different_specific, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.N: %N.type.197 = import_ref Main//import_generic_with_different_specific, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.1f9 = import_ref Main//import_generic_with_different_specific, loc5_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//import_generic_with_different_specific, loc5_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//import_generic_with_different_specific, loc7_34, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic_with_different_specific, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.d96 = import_ref Main//import_generic_with_different_specific, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.775 [concrete]
// CHECK:STDOUT:   %Main.import_ref.f0f: type = import_ref Main//import_generic_with_different_specific, loc7_24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.959e86.1: type = import_ref Main//import_generic_with_different_specific, loc7_32, loaded [symbolic = @C.as.I.impl.775.%I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Main.import_ref.1eb = import_ref Main//import_generic_with_different_specific, loc7_32, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//import_generic_with_different_specific, loc7_15, loaded [symbolic = @C.as.I.impl.775.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.7ef = import_ref Main//import_generic_with_different_specific, loc10_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.3d8: type = import_ref Main//import_generic_with_different_specific, loc10_18, loaded [symbolic = @N.WithSelf.Self.as_type.impls.I.type.require.%Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %Main.import_ref.959e86.2: type = import_ref Main//import_generic_with_different_specific, loc10_27, loaded [symbolic = @N.WithSelf.Self.as_type.impls.I.type.require.%I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.5: type = import_ref Main//import_generic_with_different_specific, loc9_15, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.49ba3f.2: @N.%N.type (%N.type.d682d6.1) = import_ref Main//import_generic_with_different_specific, loc9_24, loaded [symbolic = @N.%Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Main.import_ref.6e8 = import_ref Main//import_generic_with_different_specific, loc9_24, unloaded
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
// CHECK:STDOUT:   impl_decl @C.as.I.impl.a2d [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.cd7 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_15.1 [symbolic = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc5_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc5_32.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %I.type.loc5_33.1: type = facet_type <@I, @I(constants.%ptr.e8f)> [symbolic = %I.type.loc5_33.2 (constants.%I.type.f4b)]
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.186 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.197 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_15.1 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc6_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc6_32.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %ptr.loc6_33.1: type = ptr_type %ptr.loc6_32.1 [symbolic = %ptr.loc6_33.2 (constants.%ptr.125)]
// CHECK:STDOUT:     %N.type.loc6_34.1: type = facet_type <@N, @N(constants.%ptr.125)> [symbolic = %N.type.loc6_34.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.2: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ad) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.0a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.1f9
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(imports.%Main.import_ref.b3bc94.6: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.6e8
// CHECK:STDOUT:     extend imports.%Main.import_ref.7ef
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.3d8 impls imports.%Main.import_ref.959e86.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.5: type, imports.%Main.import_ref.49ba3f.2: @N.%N.type (%N.type.d682d6.1)) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.I.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.775(imports.%Main.import_ref.b3bc94.3: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ad)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl.775(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.f7e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.fa0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.f0f as imports.%Main.import_ref.959e86.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend imports.%Main.import_ref.1eb
// CHECK:STDOUT:     witness = imports.%Main.import_ref.1cb
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.a2d(%T.loc5_15.1: type) {
// CHECK:STDOUT:   %T.loc5_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc5_32.2: type = ptr_type %T.loc5_15.2 [symbolic = %ptr.loc5_32.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %I.type.loc5_33.2: type = facet_type <@I, @I(%ptr.loc5_32.2)> [symbolic = %I.type.loc5_33.2 (constants.%I.type.f4b)]
// CHECK:STDOUT:   %I.impl_witness.loc5_35.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.a2d(%T.loc5_15.2) [symbolic = %I.impl_witness.loc5_35.2 (constants.%I.impl_witness.a27)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc5_33.2 [symbolic = %require_complete (constants.%require_complete.13d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc5_33.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.a2d [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc5_35.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.a2d(constants.%T) [symbolic = %I.impl_witness.loc5_35.2 (constants.%I.impl_witness.a27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %I.type.loc5_33.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc5_35.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.186(%T.loc6_15.1: type) {
// CHECK:STDOUT:   %T.loc6_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc6_32.2: type = ptr_type %T.loc6_15.2 [symbolic = %ptr.loc6_32.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %ptr.loc6_33.2: type = ptr_type %ptr.loc6_32.2 [symbolic = %ptr.loc6_33.2 (constants.%ptr.125)]
// CHECK:STDOUT:   %N.type.loc6_34.2: type = facet_type <@N, @N(%ptr.loc6_33.2)> [symbolic = %N.type.loc6_34.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:   %I.impl_witness.loc6_36.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.186(%T.loc6_15.2) [symbolic = %I.impl_witness.loc6_36.2 (constants.%I.impl_witness.8cd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %N.type.loc6_34.2 [symbolic = %require_complete (constants.%require_complete.533)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc6_34.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.186 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc6_36.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.186(constants.%T) [symbolic = %I.impl_witness.loc6_36.2 (constants.%I.impl_witness.8cd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %N.type.loc6_34.1
// CHECK:STDOUT:     witness = %I.impl_witness.loc6_36.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.d96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %Self => constants.%Self.0a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.0a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.775(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.f7e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr.e8f) {
// CHECK:STDOUT:   %T => constants.%ptr.e8f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.f4b
// CHECK:STDOUT:   %Self => constants.%Self.0b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.a2d(constants.%T) {
// CHECK:STDOUT:   %T.loc5_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc5_32.2 => constants.%ptr.e8f
// CHECK:STDOUT:   %I.type.loc5_33.2 => constants.%I.type.f4b
// CHECK:STDOUT:   %I.impl_witness.loc5_35.2 => constants.%I.impl_witness.a27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr.e8f, constants.%Self.0a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr.e8f, constants.%I.facet.585) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
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
// CHECK:STDOUT:   %I.type => constants.%I.type.1ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr.125) {
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
// CHECK:STDOUT:   %I.type => constants.%I.type.aa2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.584
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr.125) {
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.I.type.require(constants.%ptr.125, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.aa2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.186(constants.%T) {
// CHECK:STDOUT:   %T.loc6_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc6_32.2 => constants.%ptr.e8f
// CHECK:STDOUT:   %ptr.loc6_33.2 => constants.%ptr.125
// CHECK:STDOUT:   %N.type.loc6_34.2 => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %I.impl_witness.loc6_36.2 => constants.%I.impl_witness.8cd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr.125, constants.%Self.e1f642.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT:   %I.type => constants.%I.type.aa2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.584
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr.125, constants.%I.facet.192) {
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
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.29e: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.29e = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.e8a: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0fd: %J.type.e8a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %N.type.197: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.197 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d68: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.107: %N.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.107 [symbolic]
// CHECK:STDOUT:   %J.impl_witness.e79: <witness> = impl_witness @D.as.J.impl.79d.%J.impl_witness_table, @D.as.J.impl.79d(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.b26: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %J.impl_witness.63c: <witness> = impl_witness @D.as.J.impl.5bc.%J.impl_witness_table, @D.as.J.impl.5bc(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %J.decl: %J.type.29e = interface_decl @J [concrete = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_17.1: type = splice_block %.loc5_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: %N.type.197 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_18.1: type = splice_block %.loc7_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.79d [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.29e = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_15.1 [symbolic = %T.loc15_15.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc15_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc15_32.2 (constants.%J.type.e8a)]
// CHECK:STDOUT:     %.loc15_18.1: type = splice_block %.loc15_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.5bc [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.29e = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_15.1 [symbolic = %T.loc21_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc21_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc21_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc21_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc21_33.2 (constants.%J.type.b26)]
// CHECK:STDOUT:     %.loc21_18.1: type = splice_block %.loc21_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc21_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc21_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc5_14.1)> [symbolic = %J.type (constants.%J.type.e8a)]
// CHECK:STDOUT:   %Self.loc5_23.2: @J.%J.type (%J.type.e8a) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.0fd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_23.1: @J.%J.type (%J.type.e8a) = symbolic_binding Self, 1 [symbolic = %Self.loc5_23.2 (constants.%Self.0fd)]
// CHECK:STDOUT:     %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_23.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc7_15.2: type) {
// CHECK:STDOUT:   %T.loc7_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc7_15.1)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self.loc7_24.2: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc7_24.2 (constants.%Self.107)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc7_24.1: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc7_24.2 (constants.%Self.107)]
// CHECK:STDOUT:     %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.as_type.impls.J.type.require.decl = require_decl @N.WithSelf.Self.as_type.impls.J.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc8_18.1 impls %J.type.loc8_27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc8_18.1: type = facet_access_type @N.%Self.loc7_24.1 [symbolic = %Self.as_type.loc8_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %J.ref: %J.type.29e = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc7_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %J.type.loc8_27.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc8_27.2 (constants.%J.type.e8a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc8: type = specific_constant @N.WithSelf.Self.as_type.impls.J.type.require.%J.type.loc8_27.1, @N.WithSelf.Self.as_type.impls.J.type.require(constants.%T, constants.%Self.107) [symbolic = %J.type (constants.%J.type.e8a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc7_24.1
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
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.J.type.require(@N.%T.loc7_15.2: type, @N.%Self.loc7_24.1: @N.%N.type (%N.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.J.type.require.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.107)]
// CHECK:STDOUT:   %Self.as_type.loc8_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_18.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %J.type.loc8_27.2: type = facet_type <@J, @J(%T)> [symbolic = %J.type.loc8_27.2 (constants.%J.type.e8a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.79d(%T.loc15_15.1: type) {
// CHECK:STDOUT:   %T.loc15_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_15.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc15_32.2: type = facet_type <@J, @J(%T.loc15_15.2)> [symbolic = %J.type.loc15_32.2 (constants.%J.type.e8a)]
// CHECK:STDOUT:   %J.impl_witness.loc15_33.2: <witness> = impl_witness %J.impl_witness_table, @D.as.J.impl.79d(%T.loc15_15.2) [symbolic = %J.impl_witness.loc15_33.2 (constants.%J.impl_witness.e79)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc15_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.5bc(%T.loc21_15.1: type) {
// CHECK:STDOUT:   %T.loc21_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc21_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc21_32.2: type = ptr_type %T.loc21_15.2 [symbolic = %ptr.loc21_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc21_33.2: type = facet_type <@J, @J(%ptr.loc21_32.2)> [symbolic = %J.type.loc21_33.2 (constants.%J.type.b26)]
// CHECK:STDOUT:   %J.impl_witness.loc21_34.2: <witness> = impl_witness %J.impl_witness_table, @D.as.J.impl.5bc(%T.loc21_15.2) [symbolic = %J.impl_witness.loc21_34.2 (constants.%J.impl_witness.63c)]
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
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.e8a
// CHECK:STDOUT:   %Self.loc5_23.2 => constants.%Self.0fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.0fd) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.loc7_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.107) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.J.type.require(constants.%T, constants.%Self.107) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.107
// CHECK:STDOUT:   %Self.as_type.loc8_18.2 => constants.%Self.as_type
// CHECK:STDOUT:   %J.type.loc8_27.2 => constants.%J.type.e8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.79d(constants.%T) {
// CHECK:STDOUT:   %T.loc15_15.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc15_32.2 => constants.%J.type.e8a
// CHECK:STDOUT:   %J.impl_witness.loc15_33.2 => constants.%J.impl_witness.e79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.5bc(constants.%T) {
// CHECK:STDOUT:   %T.loc21_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc21_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc21_33.2 => constants.%J.type.b26
// CHECK:STDOUT:   %J.impl_witness.loc21_34.2 => constants.%J.impl_witness.63c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.29e: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.29e = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.e8a: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.5c9: %J.type.e8a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.impl_witness.029: <witness> = impl_witness imports.%J.impl_witness_table.35f, @D.as.J.impl.824(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.b26: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %J.impl_witness.641: <witness> = impl_witness imports.%J.impl_witness_table.ec4, @D.as.J.impl.e87(%T) [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.type.197: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.197 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.564: <witness> = require_complete_type %J.type.e8a [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.e1f [symbolic]
// CHECK:STDOUT:   %D.type.facet: %type = facet_value %D, () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%ptr)> [symbolic]
// CHECK:STDOUT:   %require_complete.efc: <witness> = require_complete_type %J.type.b26 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//import_generic_decl, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.J: %J.type.29e = import_ref Main//import_generic_decl, J, loaded [concrete = constants.%J.generic]
// CHECK:STDOUT:   %Main.N: %N.type.197 = import_ref Main//import_generic_decl, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.bcf = import_ref Main//import_generic_decl, loc5_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//import_generic_decl, loc5_14, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic_decl, loc4_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.4db = import_ref Main//import_generic_decl, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %J.impl_witness_table.35f = impl_witness_table (), @D.as.J.impl.824 [concrete]
// CHECK:STDOUT:   %Main.import_ref.8e545e.1: type = import_ref Main//import_generic_decl, loc15_24, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.882362.1: type = import_ref Main//import_generic_decl, loc15_32, loaded [symbolic = @D.as.J.impl.824.%J.type (constants.%J.type.e8a)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//import_generic_decl, loc15_15, loaded [symbolic = @D.as.J.impl.824.%T (constants.%T)]
// CHECK:STDOUT:   %J.impl_witness_table.ec4 = impl_witness_table (), @D.as.J.impl.e87 [concrete]
// CHECK:STDOUT:   %Main.import_ref.8e545e.2: type = import_ref Main//import_generic_decl, loc21_24, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.1c0: type = import_ref Main//import_generic_decl, loc21_33, loaded [symbolic = @D.as.J.impl.e87.%J.type (constants.%J.type.b26)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//import_generic_decl, loc21_15, loaded [symbolic = @D.as.J.impl.e87.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.01f = import_ref Main//import_generic_decl, loc8_28, unloaded
// CHECK:STDOUT:   %Main.import_ref.3d8: type = import_ref Main//import_generic_decl, loc8_18, loaded [symbolic = @N.WithSelf.Self.as_type.impls.J.type.require.%Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %Main.import_ref.882362.2: type = import_ref Main//import_generic_decl, loc8_27, loaded [symbolic = @N.WithSelf.Self.as_type.impls.J.type.require.%J.type (constants.%J.type.e8a)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.6: type = import_ref Main//import_generic_decl, loc7_15, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.49ba3f.2: @N.%N.type (%N.type.d682d6.1) = import_ref Main//import_generic_decl, loc7_24, loaded [symbolic = @N.%Self (constants.%Self.e1f)]
// CHECK:STDOUT:   %Main.import_ref.6e8 = import_ref Main//import_generic_decl, loc7_24, unloaded
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
// CHECK:STDOUT:   impl_decl @D.as.J.impl.79d5fb.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.29e = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc8_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc8_32.2 (constants.%J.type.e8a)]
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.79d5fb.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.29e = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_15.1 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc14_32.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.e8a)]
// CHECK:STDOUT:     %.loc14_18.1: type = splice_block %.loc14_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc14_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.5bc982.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.29e = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_15.1 [symbolic = %T.loc20_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc20_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc20_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc20_33.2 (constants.%J.type.b26)]
// CHECK:STDOUT:     %.loc20_18.1: type = splice_block %.loc20_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc20_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc20_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.5bc982.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.29e = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26_15.1 [symbolic = %T.loc26_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc26_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc26_33.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc26_33.2 (constants.%J.type.b26)]
// CHECK:STDOUT:     %.loc26_18.1: type = splice_block %.loc26_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc26_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc26_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.27c174.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %N.ref: %N.type.197 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc32_15.1 [symbolic = %T.loc32_15.2 (constants.%T)]
// CHECK:STDOUT:     %N.type.loc32_32.1: type = facet_type <@N, @N(constants.%T)> [symbolic = %N.type.loc32_32.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:     %.loc32_18.1: type = splice_block %.loc32_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc32_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc32_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc32_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.27c174.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %N.ref: %N.type.197 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc38_15.1 [symbolic = %T.loc38_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc38_32.1: type = ptr_type %T.ref [symbolic = %ptr.loc38_32.2 (constants.%ptr)]
// CHECK:STDOUT:     %N.type.loc38_33.1: type = facet_type <@N, @N(constants.%ptr)> [symbolic = %N.type.loc38_33.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:     %.loc38_18.1: type = splice_block %.loc38_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc38_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc38_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc38_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(imports.%Main.import_ref.b3bc94.2: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.e8a)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.e8a) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.5c9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.bcf
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(imports.%Main.import_ref.b3bc94.7: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.6e8
// CHECK:STDOUT:     extend imports.%Main.import_ref.01f
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.J.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.3d8 impls imports.%Main.import_ref.882362.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.J.type.require(imports.%Main.import_ref.b3bc94.6: type, imports.%Main.import_ref.49ba3f.2: @N.%N.type (%N.type.d682d6.1)) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.J.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.e8a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.824(imports.%Main.import_ref.b3bc94.3: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.e8a)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness imports.%J.impl_witness_table.35f, @D.as.J.impl.824(%T) [symbolic = %J.impl_witness (constants.%J.impl_witness.029)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.8e545e.1 as imports.%Main.import_ref.882362.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.e87(imports.%Main.import_ref.b3bc94.4: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%ptr)> [symbolic = %J.type (constants.%J.type.b26)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness imports.%J.impl_witness_table.ec4, @D.as.J.impl.e87(%T) [symbolic = %J.impl_witness (constants.%J.impl_witness.641)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.8e545e.2 as imports.%Main.import_ref.1c0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.79d5fb.1(%T.loc8_15.1: type) {
// CHECK:STDOUT:   %T.loc8_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc8_32.2: type = facet_type <@J, @J(%T.loc8_15.2)> [symbolic = %J.type.loc8_32.2 (constants.%J.type.e8a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc8_32.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.79d5fb.2(%T.loc14_15.1: type) {
// CHECK:STDOUT:   %T.loc14_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc14_15.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc14_32.2: type = facet_type <@J, @J(%T.loc14_15.2)> [symbolic = %J.type.loc14_32.2 (constants.%J.type.e8a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc14_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %J.type.loc14_32.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.5bc982.1(%T.loc20_15.1: type) {
// CHECK:STDOUT:   %T.loc20_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc20_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc20_32.2: type = ptr_type %T.loc20_15.2 [symbolic = %ptr.loc20_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc20_33.2: type = facet_type <@J, @J(%ptr.loc20_32.2)> [symbolic = %J.type.loc20_33.2 (constants.%J.type.b26)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc20_33.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.5bc982.2(%T.loc26_15.1: type) {
// CHECK:STDOUT:   %T.loc26_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc26_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc26_32.2: type = ptr_type %T.loc26_15.2 [symbolic = %ptr.loc26_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc26_33.2: type = facet_type <@J, @J(%ptr.loc26_32.2)> [symbolic = %J.type.loc26_33.2 (constants.%J.type.b26)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc26_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %J.type.loc26_33.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.27c174.1(%T.loc32_15.1: type) {
// CHECK:STDOUT:   %T.loc32_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc32_15.2 (constants.%T)]
// CHECK:STDOUT:   %N.type.loc32_32.2: type = facet_type <@N, @N(%T.loc32_15.2)> [symbolic = %N.type.loc32_32.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %N.type.loc32_32.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %N.type.loc32_32.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.27c174.2(%T.loc38_15.1: type) {
// CHECK:STDOUT:   %T.loc38_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc38_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc38_32.2: type = ptr_type %T.loc38_15.2 [symbolic = %ptr.loc38_32.2 (constants.%ptr)]
// CHECK:STDOUT:   %N.type.loc38_33.2: type = facet_type <@N, @N(%ptr.loc38_32.2)> [symbolic = %N.type.loc38_33.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %N.type.loc38_33.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %N.type.loc38_33.1
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.4db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.e8a
// CHECK:STDOUT:   %Self => constants.%Self.5c9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.5c9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.824(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.e8a
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.029
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.e87(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.b26
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.641
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.79d5fb.1(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc8_32.2 => constants.%J.type.e8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.79d5fb.2(constants.%T) {
// CHECK:STDOUT:   %T.loc14_15.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc14_32.2 => constants.%J.type.e8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.5bc982.1(constants.%T) {
// CHECK:STDOUT:   %T.loc20_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc20_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc20_33.2 => constants.%J.type.b26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.5bc982.2(constants.%T) {
// CHECK:STDOUT:   %T.loc26_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc26_32.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc26_33.2 => constants.%J.type.b26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
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
// CHECK:STDOUT:   %J.type => constants.%J.type.e8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%D.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.e8a
// CHECK:STDOUT:   %require_complete => constants.%require_complete.564
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.J.type.require(constants.%T, constants.%D.type.facet) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%D.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%D
// CHECK:STDOUT:   %J.type => constants.%J.type.e8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.27c174.1(constants.%T) {
// CHECK:STDOUT:   %T.loc32_15.2 => constants.%T
// CHECK:STDOUT:   %N.type.loc32_32.2 => constants.%N.type.d682d6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr, constants.%D.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.b26
// CHECK:STDOUT:   %require_complete => constants.%require_complete.efc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.J.type.require(constants.%ptr, constants.%D.type.facet) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%D.type.facet
// CHECK:STDOUT:   %Self.as_type => constants.%D
// CHECK:STDOUT:   %J.type => constants.%J.type.b26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.27c174.2(constants.%T) {
// CHECK:STDOUT:   %T.loc38_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc38_32.2 => constants.%ptr
// CHECK:STDOUT:   %N.type.loc38_33.2 => constants.%N.type.d682d6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

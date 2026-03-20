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
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.fdb: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %J.type.885: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.885 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.04e: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.a60: %J.type.04e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.a60 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.609 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9.1: type = splice_block %.loc9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type.885 = interface_decl @J [concrete = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18.1: type = splice_block %.loc18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16.2: type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc7.2: type) {
// CHECK:STDOUT:   %T.loc7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc7.1)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self.loc11.2: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc11.2 (constants.%Self.fdb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11.1: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc11.2 (constants.%Self.fdb)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc16.2: type) {
// CHECK:STDOUT:   %T.loc16.1: type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc16.1)> [symbolic = %J.type (constants.%J.type.04e)]
// CHECK:STDOUT:   %Self.loc20.2: @J.%J.type (%J.type.04e) = symbolic_binding Self, 1 [symbolic = %Self.loc20.2 (constants.%Self.a60)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc20.1: @J.%J.type (%J.type.04e) = symbolic_binding Self, 1 [symbolic = %Self.loc20.2 (constants.%Self.a60)]
// CHECK:STDOUT:     %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %J.WithSelf.Self.binding.as_type.impls.I.type.require.decl = require_decl @J.WithSelf.Self.binding.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls %I.type.loc27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @J.%Self.loc20.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @J.%T.loc16.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc27.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc28: type = specific_constant @J.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type.loc27.1, @J.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.a60) [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc20.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     extend @J.WithSelf.%.loc28
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.WithSelf.Self.binding.as_type.impls.I.type.require {
// CHECK:STDOUT:       require @J.WithSelf.Self.binding.as_type.impls.I.type.require.%Self.as_type impls @J.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type.loc27.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.WithSelf.Self.binding.as_type.impls.I.type.require(@J.%T.loc16.2: type, @J.%Self.loc20.1: @J.%J.type (%J.type.04e)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.04e)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.binding.as_type.impls.I.type.require.%J.type (%J.type.04e) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.a60)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type.loc27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc27.2 (constants.%I.type.1ab)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self.loc11.2 => constants.%Self.fdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.fdb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.loc16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.a60) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.a60) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.04e
// CHECK:STDOUT:   %Self => constants.%Self.a60
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type.loc27.2 => constants.%I.type.1ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_interface.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c1b: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.ab5: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.025: %I.type.ab5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.ab5 = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %J.type.885: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.885 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.04e: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c25: %J.type.04e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.c25 [symbolic]
// CHECK:STDOUT:   %J.type.d6f: type = facet_type <@J, @J(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness @C.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %complete_type.5b1: <witness> = complete_type_witness %I.type.ab5 [concrete]
// CHECK:STDOUT:   %Self.9e4: %J.type.d6f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type.d6f = facet_value %C, (%J.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.609 = import_ref Main//basic_import_generic_interface, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.J: %J.type.885 = import_ref Main//basic_import_generic_interface, J, loaded [concrete = constants.%J.generic]
// CHECK:STDOUT:   %Main.import_ref.3fa = import_ref Main//basic_import_generic_interface, loc11, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//basic_import_generic_interface, loc7, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.7bf = import_ref Main//basic_import_generic_interface, loc28, unloaded
// CHECK:STDOUT:   %Main.import_ref.2ff: type = import_ref Main//basic_import_generic_interface, loc23, loaded [symbolic = @J.WithSelf.Self.binding.as_type.impls.I.type.require.%Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Main.import_ref.358: type = import_ref Main//basic_import_generic_interface, loc27, loaded [symbolic = @J.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//basic_import_generic_interface, loc16, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.99f327.2: @J.%J.type (%J.type.04e) = import_ref Main//basic_import_generic_interface, loc20, loaded [symbolic = @J.%Self (constants.%Self.c25)]
// CHECK:STDOUT:   %Main.import_ref.e9b = import_ref Main//basic_import_generic_interface, loc20, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.5: type = import_ref Main//basic_import_generic_interface, loc16, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc4.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4.2 = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %.loc15: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc16: type = converted %.loc15, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%empty_struct_type)> [concrete = constants.%I.type.ab5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: %J.type.885 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %.loc25: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc26: type = converted %.loc25, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.type: type = facet_type <@J, @J(constants.%empty_struct_type)> [concrete = constants.%J.type.d6f]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.2: type) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3fa
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
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.04e)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.04e) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c25)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.e9b
// CHECK:STDOUT:     extend imports.%Main.import_ref.7bf
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.WithSelf.Self.binding.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.2ff impls imports.%Main.import_ref.358
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.WithSelf.Self.binding.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.4: type, imports.%Main.import_ref.99f327.2: @J.%J.type (%J.type.04e)) [from "basic_import_generic_interface.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.04e)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.binding.as_type.impls.I.type.require.%J.type (%J.type.04e) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c25)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %I.type {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.J.impl: %C.ref as %J.type {
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (), @C.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
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
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.c1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
// CHECK:STDOUT:   %Self => constants.%Self.025
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%Self.c1b) {
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
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.c25) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.c25) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.04e
// CHECK:STDOUT:   %Self => constants.%Self.c25
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.d6f
// CHECK:STDOUT:   %Self => constants.%Self.9e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
// CHECK:STDOUT:   %require_complete => constants.%complete_type.5b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %J.type => constants.%J.type.d6f
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%Self.c25) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
// CHECK:STDOUT:   %require_complete => constants.%complete_type.5b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%J.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
// CHECK:STDOUT:   %require_complete => constants.%complete_type.5b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.fdb: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %J.type.d08: type = generic_named_constaint_type @J [concrete]
// CHECK:STDOUT:   %empty_struct: %J.type.d08 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.d68: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.31c: %J.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.31c [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.609 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9.1: type = splice_block %.loc9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: %J.type.d08 = constraint_decl @J [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18.1: type = splice_block %.loc18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16.2: type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc7.2: type) {
// CHECK:STDOUT:   %T.loc7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc7.1)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self.loc11.2: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc11.2 (constants.%Self.fdb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11.1: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc11.2 (constants.%Self.fdb)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @J(%T.loc16.2: type) {
// CHECK:STDOUT:   %T.loc16.1: type = symbolic_binding T, 0 [symbolic = %T.loc16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc16.1)> [symbolic = %J.type (constants.%J.type.d68)]
// CHECK:STDOUT:   %Self.loc20.2: @J.%J.type (%J.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc20.2 (constants.%Self.31c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc20.1: @J.%J.type (%J.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc20.2 (constants.%Self.31c)]
// CHECK:STDOUT:     %J.WithSelf.decl = constraint_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %J.WithSelf.Self.binding.as_type.impls.I.type.require.decl = require_decl @J.WithSelf.Self.binding.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls %I.type.loc27.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @J.%Self.loc20.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @J.%T.loc16.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc27.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc27.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc28: type = specific_constant @J.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type.loc27.1, @J.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.31c) [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc20.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     extend @J.WithSelf.%.loc28
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.WithSelf.Self.binding.as_type.impls.I.type.require {
// CHECK:STDOUT:       require @J.WithSelf.Self.binding.as_type.impls.I.type.require.%Self.as_type impls @J.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type.loc27.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.WithSelf.Self.binding.as_type.impls.I.type.require(@J.%T.loc16.2: type, @J.%Self.loc20.1: @J.%J.type (%J.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.d68)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.binding.as_type.impls.I.type.require.%J.type (%J.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.31c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type.loc27.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc27.2 (constants.%I.type.1ab)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self.loc11.2 => constants.%Self.fdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.fdb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T.loc16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.31c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.31c) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.31c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type.loc27.2 => constants.%I.type.1ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_import_generic_constraint.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.type.d08: type = generic_named_constaint_type @J [concrete]
// CHECK:STDOUT:   %empty_struct.d2a: %J.type.d08 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.d682d6.1: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f642.1: %J.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c1b: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.e1f642.1 [symbolic]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.d682d6.2: type = facet_type <@J, @J(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.e1f642.2: %J.type.d682d6.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.type.ab5: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.025: %I.type.ab5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.5b1: <witness> = complete_type_witness %I.type.ab5 [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.ab5 = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//basic_import_generic_constraint, I, unloaded
// CHECK:STDOUT:   %Main.J: %J.type.d08 = import_ref Main//basic_import_generic_constraint, J, loaded [concrete = constants.%empty_struct.d2a]
// CHECK:STDOUT:   %Main.import_ref.7b1 = import_ref Main//basic_import_generic_constraint, loc28, unloaded
// CHECK:STDOUT:   %Main.import_ref.3fa = import_ref Main//basic_import_generic_constraint, loc11, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//basic_import_generic_constraint, loc7, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.213: type = import_ref Main//basic_import_generic_constraint, loc23, loaded [symbolic = @J.WithSelf.Self.binding.as_type.impls.I.type.require.%Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Main.import_ref.358: type = import_ref Main//basic_import_generic_constraint, loc27, loaded [symbolic = @J.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//basic_import_generic_constraint, loc16, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.2fc394.2: @J.%J.type (%J.type.d682d6.1) = import_ref Main//basic_import_generic_constraint, loc20, loaded [symbolic = @J.%Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Main.import_ref.830 = import_ref Main//basic_import_generic_constraint, loc20, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.5: type = import_ref Main//basic_import_generic_constraint, loc16, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc4.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4.2 = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: %J.type.d08 = name_ref J, imports.%Main.J [concrete = constants.%empty_struct.d2a]
// CHECK:STDOUT:     %.loc15: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:     %.loc16: type = converted %.loc15, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %J.type: type = facet_type <@J, @J(constants.%empty_struct_type)> [concrete = constants.%J.type.d682d6.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.3: type) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3fa
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
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.830
// CHECK:STDOUT:     extend imports.%Main.import_ref.7b1
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @J.WithSelf.Self.binding.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.213 impls imports.%Main.import_ref.358
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @J.WithSelf.Self.binding.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.4: type, imports.%Main.import_ref.2fc394.2: @J.%J.type (%J.type.d682d6.1)) [from "basic_import_generic_constraint.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @J.WithSelf.Self.binding.as_type.impls.I.type.require.%J.type (%J.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %J.type {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
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
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.c1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.e1f642.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.e1f642.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
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
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%Self.e1f642.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
// CHECK:STDOUT:   %require_complete => constants.%complete_type.5b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
// CHECK:STDOUT:   %Self => constants.%Self.025
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%Self.c1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
// CHECK:STDOUT:   %require_complete => constants.%complete_type.5b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%empty_struct_type, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %J.type => constants.%J.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
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
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.fdb: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.9f5: <witness> = impl_witness @C.as.I.impl.431.%I.impl_witness_table, @C.as.I.impl.431(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.e36: <witness> = require_complete_type %I.type.1ab [symbolic]
// CHECK:STDOUT:   %I.facet.b43: %I.type.1ab = facet_value %C, (%I.impl_witness.9f5) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.e19: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %I.impl_witness.18a: <witness> = impl_witness @C.as.I.impl.815.%I.impl_witness_table, @C.as.I.impl.815(%T) [symbolic]
// CHECK:STDOUT:   %Self.eff: %I.type.e19 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.ffd: <witness> = require_complete_type %I.type.e19 [symbolic]
// CHECK:STDOUT:   %I.facet.064: %I.type.e19 = facet_value %C, (%I.impl_witness.18a) [symbolic]
// CHECK:STDOUT:   %N.type.b47: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b47 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d68: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.31c: %N.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.31c [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.609 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13.1: type = splice_block %.loc13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11.2: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.431 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc26: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc28: type = name_ref T, %T.loc20.1 [symbolic = %T.loc20.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc29.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc29.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:     %.loc22.1: type = splice_block %.loc22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20.1: type = symbolic_binding T, 0 [symbolic = %T.loc20.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.431 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc38: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref.loc40: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref.loc42: type = name_ref T, %T.loc34 [symbolic = %T.loc20.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc43: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc29.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:     %.loc36.1: type = splice_block %.loc36.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc36.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc34: type = symbolic_binding T, 0 [symbolic = %T.loc20.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.815 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc49.1 [symbolic = %T.loc49.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc58.1: type = ptr_type %T.ref [symbolic = %ptr.loc58.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc59.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc59.2 (constants.%I.type.e19)]
// CHECK:STDOUT:     %.loc51.1: type = splice_block %.loc51.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc51.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc49.1: type = symbolic_binding T, 0 [symbolic = %T.loc49.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: %N.type.b47 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc67.1: type = splice_block %.loc67.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc67.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc65.2: type = symbolic_binding T, 0 [symbolic = %T.loc65.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc11.2: type) {
// CHECK:STDOUT:   %T.loc11.1: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc11.1)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self.loc15.2: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc15.2 (constants.%Self.fdb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15.1: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc15.2 (constants.%Self.fdb)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc65.2: type) {
// CHECK:STDOUT:   %T.loc65.1: type = symbolic_binding T, 0 [symbolic = %T.loc65.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc65.1)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self.loc69.2: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc69.2 (constants.%Self.31c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc69.1: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc69.2 (constants.%Self.31c)]
// CHECK:STDOUT:     %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.binding.as_type.impls.I.type.require.decl = require_decl @N.WithSelf.Self.binding.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls %I.type.loc76.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @N.%Self.loc69.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc65.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc76.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc76.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc77: type = specific_constant @N.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type.loc76.1, @N.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.31c) [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc69.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     extend @N.WithSelf.%.loc77
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.binding.as_type.impls.I.type.require {
// CHECK:STDOUT:       require @N.WithSelf.Self.binding.as_type.impls.I.type.require.%Self.as_type impls @N.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type.loc76.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.binding.as_type.impls.I.type.require(@N.%T.loc65.2: type, @N.%Self.loc69.1: @N.%N.type (%N.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.binding.as_type.impls.I.type.require.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.31c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type.loc76.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc76.2 (constants.%I.type.1ab)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.431(%T.loc20.1: type) {
// CHECK:STDOUT:   %T.loc20.2: type = symbolic_binding T, 0 [symbolic = %T.loc20.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc29.2: type = facet_type <@I, @I(%T.loc20.2)> [symbolic = %I.type.loc29.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:   %I.impl_witness.loc30.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.431(%T.loc20.2) [symbolic = %I.impl_witness.loc30.2 (constants.%I.impl_witness.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc29.2 [symbolic = %require_complete (constants.%require_complete.e36)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref.loc24 as %I.type.loc29.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.431 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc30.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.431(constants.%T) [symbolic = %I.impl_witness.loc30.2 (constants.%I.impl_witness.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %I.impl_witness.loc30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.815(%T.loc49.1: type) {
// CHECK:STDOUT:   %T.loc49.2: type = symbolic_binding T, 0 [symbolic = %T.loc49.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc58.2: type = ptr_type %T.loc49.2 [symbolic = %ptr.loc58.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc59.2: type = facet_type <@I, @I(%ptr.loc58.2)> [symbolic = %I.type.loc59.2 (constants.%I.type.e19)]
// CHECK:STDOUT:   %I.impl_witness.loc60.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.815(%T.loc49.2) [symbolic = %I.impl_witness.loc60.2 (constants.%I.impl_witness.18a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc59.2 [symbolic = %require_complete (constants.%require_complete.ffd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc59.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.815 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc60.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.815(constants.%T) [symbolic = %I.impl_witness.loc60.2 (constants.%I.impl_witness.18a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %I.impl_witness.loc60.1
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
// CHECK:STDOUT:   %T.loc11.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self.loc15.2 => constants.%Self.fdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.fdb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.431(constants.%T) {
// CHECK:STDOUT:   %T.loc20.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc29.2 => constants.%I.type.1ab
// CHECK:STDOUT:   %I.impl_witness.loc30.2 => constants.%I.impl_witness.9f5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%I.facet.b43) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T.loc11.1 => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.e19
// CHECK:STDOUT:   %Self.loc15.2 => constants.%Self.eff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.815(constants.%T) {
// CHECK:STDOUT:   %T.loc49.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc58.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc59.2 => constants.%I.type.e19
// CHECK:STDOUT:   %I.impl_witness.loc60.2 => constants.%I.impl_witness.18a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr, constants.%Self.fdb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr, constants.%I.facet.064) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.loc65.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.31c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.31c) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.31c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type.loc76.2 => constants.%I.type.1ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c1b: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness.9f5: <witness> = impl_witness imports.%I.impl_witness_table.2ac, @C.as.I.impl.431d5e.1(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.e36: <witness> = require_complete_type %I.type.1ab [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.e19: type = facet_type <@I, @I(%ptr)> [symbolic]
// CHECK:STDOUT:   %Self.dda: %I.type.e19 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.18a: <witness> = impl_witness imports.%I.impl_witness_table.a23, @C.as.I.impl.8159c7.1(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.ffd: <witness> = require_complete_type %I.type.e19 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.type.b47: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b47 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f642.1: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.e1f642.1 [symbolic]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%ptr)> [symbolic]
// CHECK:STDOUT:   %Self.e1f642.2: %N.type.d682d6.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//import_generic, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.I: %I.type.609 = import_ref Main//import_generic, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.N: %N.type.b47 = import_ref Main//import_generic, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.3fa = import_ref Main//import_generic, loc15, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//import_generic, loc11, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e08 = import_ref Main//import_generic, loc30, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic, loc7, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//import_generic, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %I.impl_witness_table.2ac = impl_witness_table (), @C.as.I.impl.431d5e.1 [concrete]
// CHECK:STDOUT:   %Main.import_ref.61cd7d.1: type = import_ref Main//import_generic, loc24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.3582b1.1: type = import_ref Main//import_generic, loc29, loaded [symbolic = @C.as.I.impl.431d5e.1.%I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//import_generic, loc20, loaded [symbolic = @C.as.I.impl.431d5e.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.2c7 = import_ref Main//import_generic, loc60, unloaded
// CHECK:STDOUT:   %I.impl_witness_table.a23 = impl_witness_table (), @C.as.I.impl.8159c7.1 [concrete]
// CHECK:STDOUT:   %Main.import_ref.61cd7d.2: type = import_ref Main//import_generic, loc53, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.ada: type = import_ref Main//import_generic, loc59, loaded [symbolic = @C.as.I.impl.8159c7.1.%I.type (constants.%I.type.e19)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//import_generic, loc49, loaded [symbolic = @C.as.I.impl.8159c7.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.a5e = import_ref Main//import_generic, loc77, unloaded
// CHECK:STDOUT:   %Main.import_ref.213: type = import_ref Main//import_generic, loc72, loaded [symbolic = @N.WithSelf.Self.binding.as_type.impls.I.type.require.%Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Main.import_ref.3582b1.2: type = import_ref Main//import_generic, loc76, loaded [symbolic = @N.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.6: type = import_ref Main//import_generic, loc65, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.2fc394.2: @N.%N.type (%N.type.d682d6.1) = import_ref Main//import_generic, loc69, loaded [symbolic = @N.%Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Main.import_ref.830 = import_ref Main//import_generic, loc69, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.7: type = import_ref Main//import_generic, loc65, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .N = imports.%Main.N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc4.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4.2 = import <none>
// CHECK:STDOUT:   impl_decl @C.as.I.impl.431d5e.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8.1 [symbolic = %T.loc8.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc17.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc17.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:     %.loc10.1: type = splice_block %.loc10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8.1: type = symbolic_binding T, 0 [symbolic = %T.loc8.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.431d5e.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22.1 [symbolic = %T.loc22.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc31.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc31.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:     %.loc24.1: type = splice_block %.loc24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22.1: type = symbolic_binding T, 0 [symbolic = %T.loc22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.8159c7.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc37.1 [symbolic = %T.loc37.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc46.1: type = ptr_type %T.ref [symbolic = %ptr.loc46.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc47.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc47.2 (constants.%I.type.e19)]
// CHECK:STDOUT:     %.loc39.1: type = splice_block %.loc39.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc39.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc37.1: type = symbolic_binding T, 0 [symbolic = %T.loc37.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.8159c7.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc52.1 [symbolic = %T.loc52.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc61.1: type = ptr_type %T.ref [symbolic = %ptr.loc61.2 (constants.%ptr)]
// CHECK:STDOUT:     %I.type.loc62.1: type = facet_type <@I, @I(constants.%ptr)> [symbolic = %I.type.loc62.2 (constants.%I.type.e19)]
// CHECK:STDOUT:     %.loc54.1: type = splice_block %.loc54.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc54.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc52.1: type = symbolic_binding T, 0 [symbolic = %T.loc52.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.c9332a.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.b47 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc68.1 [symbolic = %T.loc68.2 (constants.%T)]
// CHECK:STDOUT:     %N.type.loc77.1: type = facet_type <@N, @N(constants.%T)> [symbolic = %N.type.loc77.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:     %.loc70.1: type = splice_block %.loc70.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc70.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc68.1: type = symbolic_binding T, 0 [symbolic = %T.loc68.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.c9332a.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.b47 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc82.1 [symbolic = %T.loc82.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc91.1: type = ptr_type %T.ref [symbolic = %ptr.loc91.2 (constants.%ptr)]
// CHECK:STDOUT:     %N.type.loc92.1: type = facet_type <@N, @N(constants.%ptr)> [symbolic = %N.type.loc92.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:     %.loc84.1: type = splice_block %.loc84.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc84.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc82.1: type = symbolic_binding T, 0 [symbolic = %T.loc82.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.2: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3fa
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
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.830
// CHECK:STDOUT:     extend imports.%Main.import_ref.a5e
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.binding.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.213 impls imports.%Main.import_ref.3582b1.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.binding.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.6: type, imports.%Main.import_ref.2fc394.2: @N.%N.type (%N.type.d682d6.1)) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.binding.as_type.impls.I.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.431d5e.1(imports.%Main.import_ref.b3bc94.3: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table.2ac, @C.as.I.impl.431d5e.1(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.e36)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.61cd7d.1 as imports.%Main.import_ref.3582b1.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.e08
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.8159c7.1(imports.%Main.import_ref.b3bc94.4: type) [from "import_generic.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%ptr)> [symbolic = %I.type (constants.%I.type.e19)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table.a23, @C.as.I.impl.8159c7.1(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.18a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.ffd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.61cd7d.2 as imports.%Main.import_ref.ada {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.2c7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.431d5e.2(%T.loc8.1: type) {
// CHECK:STDOUT:   %T.loc8.2: type = symbolic_binding T, 0 [symbolic = %T.loc8.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc17.2: type = facet_type <@I, @I(%T.loc8.2)> [symbolic = %I.type.loc17.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc17.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.431d5e.3(%T.loc22.1: type) {
// CHECK:STDOUT:   %T.loc22.2: type = symbolic_binding T, 0 [symbolic = %T.loc22.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc31.2: type = facet_type <@I, @I(%T.loc22.2)> [symbolic = %I.type.loc31.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc31.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.8159c7.2(%T.loc37.1: type) {
// CHECK:STDOUT:   %T.loc37.2: type = symbolic_binding T, 0 [symbolic = %T.loc37.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc46.2: type = ptr_type %T.loc37.2 [symbolic = %ptr.loc46.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc47.2: type = facet_type <@I, @I(%ptr.loc46.2)> [symbolic = %I.type.loc47.2 (constants.%I.type.e19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc47.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.8159c7.3(%T.loc52.1: type) {
// CHECK:STDOUT:   %T.loc52.2: type = symbolic_binding T, 0 [symbolic = %T.loc52.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc61.2: type = ptr_type %T.loc52.2 [symbolic = %ptr.loc61.2 (constants.%ptr)]
// CHECK:STDOUT:   %I.type.loc62.2: type = facet_type <@I, @I(%ptr.loc61.2)> [symbolic = %I.type.loc62.2 (constants.%I.type.e19)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc62.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.c9332a.1(%T.loc68.1: type) {
// CHECK:STDOUT:   %T.loc68.2: type = symbolic_binding T, 0 [symbolic = %T.loc68.2 (constants.%T)]
// CHECK:STDOUT:   %N.type.loc77.2: type = facet_type <@N, @N(%T.loc68.2)> [symbolic = %N.type.loc77.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc77.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.c9332a.2(%T.loc82.1: type) {
// CHECK:STDOUT:   %T.loc82.2: type = symbolic_binding T, 0 [symbolic = %T.loc82.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc91.2: type = ptr_type %T.loc82.2 [symbolic = %ptr.loc91.2 (constants.%ptr)]
// CHECK:STDOUT:   %N.type.loc92.2: type = facet_type <@N, @N(%ptr.loc91.2)> [symbolic = %N.type.loc92.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc92.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "import_generic.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.743
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.c1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.431d5e.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.9f5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.e19
// CHECK:STDOUT:   %Self => constants.%Self.dda
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.8159c7.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.e19
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.18a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.431d5e.2(constants.%T) {
// CHECK:STDOUT:   %T.loc8.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc17.2 => constants.%I.type.1ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.431d5e.3(constants.%T) {
// CHECK:STDOUT:   %T.loc22.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc31.2 => constants.%I.type.1ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.8159c7.2(constants.%T) {
// CHECK:STDOUT:   %T.loc37.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc46.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc47.2 => constants.%I.type.e19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.8159c7.3(constants.%T) {
// CHECK:STDOUT:   %T.loc52.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc61.2 => constants.%ptr
// CHECK:STDOUT:   %I.type.loc62.2 => constants.%I.type.e19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.e1f642.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.e1f642.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %require_complete => constants.%require_complete.e36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.c9332a.1(constants.%T) {
// CHECK:STDOUT:   %T.loc68.2 => constants.%T
// CHECK:STDOUT:   %N.type.loc77.2 => constants.%N.type.d682d6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr, constants.%Self.e1f642.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.e19
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ffd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %I.type => constants.%I.type.e19
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ffd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%ptr, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.e19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.c9332a.2(constants.%T) {
// CHECK:STDOUT:   %T.loc82.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc91.2 => constants.%ptr
// CHECK:STDOUT:   %N.type.loc92.2 => constants.%N.type.d682d6.2
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
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.fdb: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table, @C.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.1ab [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.1ab = facet_value %C, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %N.type.b47: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b47 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d68: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.31c: %N.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.31c [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: %I.type.609 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13.1: type = splice_block %.loc13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11.2: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20.1 [symbolic = %T.loc20.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc29.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc29.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:     %.loc22.1: type = splice_block %.loc22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20.1: type = symbolic_binding T, 0 [symbolic = %T.loc20.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: %N.type.b47 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc37.1: type = splice_block %.loc37.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc37.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc35.2: type = symbolic_binding T, 0 [symbolic = %T.loc35.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc11.2: type) {
// CHECK:STDOUT:   %T.loc11.1: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc11.1)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self.loc15.2: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc15.2 (constants.%Self.fdb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15.1: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc15.2 (constants.%Self.fdb)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc35.2: type) {
// CHECK:STDOUT:   %T.loc35.1: type = symbolic_binding T, 0 [symbolic = %T.loc35.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc35.1)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self.loc39.2: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc39.2 (constants.%Self.31c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc39.1: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc39.2 (constants.%Self.31c)]
// CHECK:STDOUT:     %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.binding.as_type.impls.I.type.require.decl = require_decl @N.WithSelf.Self.binding.as_type.impls.I.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls %I.type.loc46.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @N.%Self.loc39.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc35.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %I.type.loc46.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc46.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc47: type = specific_constant @N.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type.loc46.1, @N.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.31c) [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc39.1
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     extend @N.WithSelf.%.loc47
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.binding.as_type.impls.I.type.require {
// CHECK:STDOUT:       require @N.WithSelf.Self.binding.as_type.impls.I.type.require.%Self.as_type impls @N.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type.loc46.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.binding.as_type.impls.I.type.require(@N.%T.loc35.2: type, @N.%Self.loc39.1: @N.%N.type (%N.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.binding.as_type.impls.I.type.require.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.31c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type.loc46.2: type = facet_type <@I, @I(%T)> [symbolic = %I.type.loc46.2 (constants.%I.type.1ab)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl(%T.loc20.1: type) {
// CHECK:STDOUT:   %T.loc20.2: type = symbolic_binding T, 0 [symbolic = %T.loc20.2 (constants.%T)]
// CHECK:STDOUT:   %I.type.loc29.2: type = facet_type <@I, @I(%T.loc20.2)> [symbolic = %I.type.loc29.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:   %I.impl_witness.loc30.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(%T.loc20.2) [symbolic = %I.impl_witness.loc30.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc29.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc29.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc30.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl(constants.%T) [symbolic = %I.impl_witness.loc30.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %I.impl_witness.loc30.1
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
// CHECK:STDOUT:   %T.loc11.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self.loc15.2 => constants.%Self.fdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.fdb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc20.2 => constants.%T
// CHECK:STDOUT:   %I.type.loc29.2 => constants.%I.type.1ab
// CHECK:STDOUT:   %I.impl_witness.loc30.2 => constants.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.loc35.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.31c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.31c) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.31c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type.loc46.2 => constants.%I.type.1ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_generic_with_different_specific.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c1b: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness.9f5: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl.431(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.e36: <witness> = require_complete_type %I.type.1ab [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %I.type.e19: type = facet_type <@I, @I(%ptr.e8f)> [symbolic]
// CHECK:STDOUT:   %I.impl_witness.18a: <witness> = impl_witness @C.as.I.impl.815.%I.impl_witness_table, @C.as.I.impl.815(%T) [symbolic]
// CHECK:STDOUT:   %Self.dda: %I.type.e19 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.ffd: <witness> = require_complete_type %I.type.e19 [symbolic]
// CHECK:STDOUT:   %I.facet.064: %I.type.e19 = facet_value %C, (%I.impl_witness.18a) [symbolic]
// CHECK:STDOUT:   %N.type.b47: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b47 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f642.1: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.e1f642.1 [symbolic]
// CHECK:STDOUT:   %ptr.125: type = ptr_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%ptr.125)> [symbolic]
// CHECK:STDOUT:   %Self.e1f642.2: %N.type.d682d6.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.type.08a: type = facet_type <@I, @I(%ptr.125)> [symbolic]
// CHECK:STDOUT:   %require_complete.ea6: <witness> = require_complete_type %I.type.08a [symbolic]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %I.impl_witness.d57: <witness> = impl_witness @C.as.I.impl.c93.%I.impl_witness_table, @C.as.I.impl.c93(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.533: <witness> = require_complete_type %N.type.d682d6.2 [symbolic]
// CHECK:STDOUT:   %I.facet.3e9: %I.type.08a = facet_value %C, (%I.impl_witness.d57) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//import_generic_with_different_specific, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.I: %I.type.609 = import_ref Main//import_generic_with_different_specific, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.N: %N.type.b47 = import_ref Main//import_generic_with_different_specific, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.3fa = import_ref Main//import_generic_with_different_specific, loc15, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//import_generic_with_different_specific, loc11, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.e08 = import_ref Main//import_generic_with_different_specific, loc30, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic_with_different_specific, loc7, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.743 = import_ref Main//import_generic_with_different_specific, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.431 [concrete]
// CHECK:STDOUT:   %Main.import_ref.61c: type = import_ref Main//import_generic_with_different_specific, loc24, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.3582b1.1: type = import_ref Main//import_generic_with_different_specific, loc29, loaded [symbolic = @C.as.I.impl.431.%I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//import_generic_with_different_specific, loc20, loaded [symbolic = @C.as.I.impl.431.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.a5e = import_ref Main//import_generic_with_different_specific, loc47, unloaded
// CHECK:STDOUT:   %Main.import_ref.213: type = import_ref Main//import_generic_with_different_specific, loc42, loaded [symbolic = @N.WithSelf.Self.binding.as_type.impls.I.type.require.%Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Main.import_ref.3582b1.2: type = import_ref Main//import_generic_with_different_specific, loc46, loaded [symbolic = @N.WithSelf.Self.binding.as_type.impls.I.type.require.%I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.5: type = import_ref Main//import_generic_with_different_specific, loc35, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.2fc394.2: @N.%N.type (%N.type.d682d6.1) = import_ref Main//import_generic_with_different_specific, loc39, loaded [symbolic = @N.%Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Main.import_ref.830 = import_ref Main//import_generic_with_different_specific, loc39, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.6: type = import_ref Main//import_generic_with_different_specific, loc35, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .N = imports.%Main.N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc4.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4.2 = import <none>
// CHECK:STDOUT:   impl_decl @C.as.I.impl.815 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8.1 [symbolic = %T.loc8.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc17.1: type = ptr_type %T.ref [symbolic = %ptr.loc17.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %I.type.loc18.1: type = facet_type <@I, @I(constants.%ptr.e8f)> [symbolic = %I.type.loc18.2 (constants.%I.type.e19)]
// CHECK:STDOUT:     %.loc10.1: type = splice_block %.loc10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8.1: type = symbolic_binding T, 0 [symbolic = %T.loc8.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.I.impl.c93 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %N.ref: %N.type.b47 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc24.1 [symbolic = %T.loc24.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc33.1: type = ptr_type %T.ref [symbolic = %ptr.loc33.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %ptr.loc34.1: type = ptr_type %ptr.loc33.1 [symbolic = %ptr.loc34.2 (constants.%ptr.125)]
// CHECK:STDOUT:     %N.type.loc35.1: type = facet_type <@N, @N(constants.%ptr.125)> [symbolic = %N.type.loc35.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:     %.loc26.1: type = splice_block %.loc26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc24.1: type = symbolic_binding T, 0 [symbolic = %T.loc24.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.2: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3fa
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
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.830
// CHECK:STDOUT:     extend imports.%Main.import_ref.a5e
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.binding.as_type.impls.I.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.213 impls imports.%Main.import_ref.3582b1.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.binding.as_type.impls.I.type.require(imports.%Main.import_ref.b3bc94.5: type, imports.%Main.import_ref.2fc394.2: @N.%N.type (%N.type.d682d6.1)) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.binding.as_type.impls.I.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.431(imports.%Main.import_ref.b3bc94.3: type) [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table, @C.as.I.impl.431(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness.9f5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.e36)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.61c as imports.%Main.import_ref.3582b1.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Main.import_ref.e08
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.815(%T.loc8.1: type) {
// CHECK:STDOUT:   %T.loc8.2: type = symbolic_binding T, 0 [symbolic = %T.loc8.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc17.2: type = ptr_type %T.loc8.2 [symbolic = %ptr.loc17.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %I.type.loc18.2: type = facet_type <@I, @I(%ptr.loc17.2)> [symbolic = %I.type.loc18.2 (constants.%I.type.e19)]
// CHECK:STDOUT:   %I.impl_witness.loc19.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.815(%T.loc8.2) [symbolic = %I.impl_witness.loc19.2 (constants.%I.impl_witness.18a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc18.2 [symbolic = %require_complete (constants.%require_complete.ffd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %I.type.loc18.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.815 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc19.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.815(constants.%T) [symbolic = %I.impl_witness.loc19.2 (constants.%I.impl_witness.18a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %I.impl_witness.loc19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.I.impl.c93(%T.loc24.1: type) {
// CHECK:STDOUT:   %T.loc24.2: type = symbolic_binding T, 0 [symbolic = %T.loc24.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc33.2: type = ptr_type %T.loc24.2 [symbolic = %ptr.loc33.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %ptr.loc34.2: type = ptr_type %ptr.loc33.2 [symbolic = %ptr.loc34.2 (constants.%ptr.125)]
// CHECK:STDOUT:   %N.type.loc35.2: type = facet_type <@N, @N(%ptr.loc34.2)> [symbolic = %N.type.loc35.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:   %I.impl_witness.loc36.2: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.c93(%T.loc24.2) [symbolic = %I.impl_witness.loc36.2 (constants.%I.impl_witness.d57)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %N.type.loc35.2 [symbolic = %require_complete (constants.%require_complete.533)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %C.ref as %N.type.loc35.1 {
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (), @C.as.I.impl.c93 [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc36.1: <witness> = impl_witness %I.impl_witness_table, @C.as.I.impl.c93(constants.%T) [symbolic = %I.impl_witness.loc36.2 (constants.%I.impl_witness.d57)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = %I.impl_witness.loc36.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "import_generic_with_different_specific.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.743
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.c1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.431(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.9f5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr.e8f) {
// CHECK:STDOUT:   %T => constants.%ptr.e8f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.e19
// CHECK:STDOUT:   %Self => constants.%Self.dda
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.815(constants.%T) {
// CHECK:STDOUT:   %T.loc8.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc17.2 => constants.%ptr.e8f
// CHECK:STDOUT:   %I.type.loc18.2 => constants.%I.type.e19
// CHECK:STDOUT:   %I.impl_witness.loc19.2 => constants.%I.impl_witness.18a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr.e8f, constants.%Self.c1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr.e8f, constants.%I.facet.064) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.e1f642.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%T, constants.%Self.e1f642.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
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
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr.125, constants.%Self.e1f642.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT:   %I.type => constants.%I.type.08a
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ea6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%ptr.125) {
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr.125, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT:   %I.type => constants.%I.type.08a
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ea6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.I.type.require(constants.%ptr.125, constants.%C.type.facet) {
// CHECK:STDOUT:   %T => constants.%ptr.125
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %I.type => constants.%I.type.08a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.I.impl.c93(constants.%T) {
// CHECK:STDOUT:   %T.loc24.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc33.2 => constants.%ptr.e8f
// CHECK:STDOUT:   %ptr.loc34.2 => constants.%ptr.125
// CHECK:STDOUT:   %N.type.loc35.2 => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %I.impl_witness.loc36.2 => constants.%I.impl_witness.d57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%ptr.125, constants.%I.facet.3e9) {
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
// CHECK:STDOUT:   %J.type.885: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.885 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.04e: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.a60: %J.type.04e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %N.type.b47: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b47 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d68: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.31c: %N.type.d68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.31c [symbolic]
// CHECK:STDOUT:   %J.impl_witness.861: <witness> = impl_witness @D.as.J.impl.b25.%J.impl_witness_table, @D.as.J.impl.b25(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.80f: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %J.impl_witness.f4f: <witness> = impl_witness @D.as.J.impl.f05.%J.impl_witness_table, @D.as.J.impl.f05(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %J.decl: %J.type.885 = interface_decl @J [concrete = constants.%J.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13.1: type = splice_block %.loc13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11.2: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: %N.type.b47 = constraint_decl @N [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22.1: type = splice_block %.loc22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20.2: type = symbolic_binding T, 0 [symbolic = %T.loc20.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.b25 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.885 = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc37.1 [symbolic = %T.loc37.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc46.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc46.2 (constants.%J.type.04e)]
// CHECK:STDOUT:     %.loc39.1: type = splice_block %.loc39.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc39.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc37.1: type = symbolic_binding T, 0 [symbolic = %T.loc37.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.f05 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.885 = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc51.1 [symbolic = %T.loc51.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc60.1: type = ptr_type %T.ref [symbolic = %ptr.loc60.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc61.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc61.2 (constants.%J.type.80f)]
// CHECK:STDOUT:     %.loc53.1: type = splice_block %.loc53.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc53.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc51.1: type = symbolic_binding T, 0 [symbolic = %T.loc51.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(%T.loc11.2: type) {
// CHECK:STDOUT:   %T.loc11.1: type = symbolic_binding T, 0 [symbolic = %T.loc11.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T.loc11.1)> [symbolic = %J.type (constants.%J.type.04e)]
// CHECK:STDOUT:   %Self.loc15.2: @J.%J.type (%J.type.04e) = symbolic_binding Self, 1 [symbolic = %Self.loc15.2 (constants.%Self.a60)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15.1: @J.%J.type (%J.type.04e) = symbolic_binding Self, 1 [symbolic = %Self.loc15.2 (constants.%Self.a60)]
// CHECK:STDOUT:     %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%T.loc20.2: type) {
// CHECK:STDOUT:   %T.loc20.1: type = symbolic_binding T, 0 [symbolic = %T.loc20.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T.loc20.1)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self.loc24.2: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc24.2 (constants.%Self.31c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc24.1: @N.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self.loc24.2 (constants.%Self.31c)]
// CHECK:STDOUT:     %N.WithSelf.decl = constraint_with_self_decl @N [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.binding.as_type.impls.J.type.require.decl = require_decl @N.WithSelf.Self.binding.as_type.impls.J.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type impls %J.type.loc31.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type: type = facet_access_type @N.%Self.loc24.1 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %J.ref: %J.type.885 = name_ref J, file.%J.decl [concrete = constants.%J.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @N.%T.loc20.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %J.type.loc31.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc31.2 (constants.%J.type.04e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc32: type = specific_constant @N.WithSelf.Self.binding.as_type.impls.J.type.require.%J.type.loc31.1, @N.WithSelf.Self.binding.as_type.impls.J.type.require(constants.%T, constants.%Self.31c) [symbolic = %J.type (constants.%J.type.04e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc24.1
// CHECK:STDOUT:     .J = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .J = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     extend @N.WithSelf.%.loc32
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.binding.as_type.impls.J.type.require {
// CHECK:STDOUT:       require @N.WithSelf.Self.binding.as_type.impls.J.type.require.%Self.as_type impls @N.WithSelf.Self.binding.as_type.impls.J.type.require.%J.type.loc31.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.binding.as_type.impls.J.type.require(@N.%T.loc20.2: type, @N.%Self.loc24.1: @N.%N.type (%N.type.d68)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d68)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.binding.as_type.impls.J.type.require.%N.type (%N.type.d68) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.31c)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %J.type.loc31.2: type = facet_type <@J, @J(%T)> [symbolic = %J.type.loc31.2 (constants.%J.type.04e)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.b25(%T.loc37.1: type) {
// CHECK:STDOUT:   %T.loc37.2: type = symbolic_binding T, 0 [symbolic = %T.loc37.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc46.2: type = facet_type <@J, @J(%T.loc37.2)> [symbolic = %J.type.loc46.2 (constants.%J.type.04e)]
// CHECK:STDOUT:   %J.impl_witness.loc47.2: <witness> = impl_witness %J.impl_witness_table, @D.as.J.impl.b25(%T.loc37.2) [symbolic = %J.impl_witness.loc47.2 (constants.%J.impl_witness.861)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc46.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.f05(%T.loc51.1: type) {
// CHECK:STDOUT:   %T.loc51.2: type = symbolic_binding T, 0 [symbolic = %T.loc51.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc60.2: type = ptr_type %T.loc51.2 [symbolic = %ptr.loc60.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc61.2: type = facet_type <@J, @J(%ptr.loc60.2)> [symbolic = %J.type.loc61.2 (constants.%J.type.80f)]
// CHECK:STDOUT:   %J.impl_witness.loc62.2: <witness> = impl_witness %J.impl_witness_table, @D.as.J.impl.f05(%T.loc51.2) [symbolic = %J.impl_witness.loc62.2 (constants.%J.impl_witness.f4f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc61.1;
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
// CHECK:STDOUT:   %T.loc11.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.04e
// CHECK:STDOUT:   %Self.loc15.2 => constants.%Self.a60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.a60) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T.loc20.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.31c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.J.type.require(constants.%T, constants.%Self.31c) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d68
// CHECK:STDOUT:   %Self => constants.%Self.31c
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %J.type.loc31.2 => constants.%J.type.04e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.b25(constants.%T) {
// CHECK:STDOUT:   %T.loc37.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc46.2 => constants.%J.type.04e
// CHECK:STDOUT:   %J.impl_witness.loc47.2 => constants.%J.impl_witness.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T.loc11.1 => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.f05(constants.%T) {
// CHECK:STDOUT:   %T.loc51.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc60.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc61.2 => constants.%J.type.80f
// CHECK:STDOUT:   %J.impl_witness.loc62.2 => constants.%J.impl_witness.f4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_generic_decl.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %J.type.885: type = generic_interface_type @J [concrete]
// CHECK:STDOUT:   %J.generic: %J.type.885 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.04e: type = facet_type <@J, @J(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c25: %J.type.04e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J.impl_witness.861: <witness> = impl_witness imports.%J.impl_witness_table.50b, @D.as.J.impl.b25a4e.1(%T) [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %J.type.80f: type = facet_type <@J, @J(%ptr)> [symbolic]
// CHECK:STDOUT:   %J.impl_witness.f4f: <witness> = impl_witness imports.%J.impl_witness_table.a37, @D.as.J.impl.f050b9.1(%T) [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %N.type.b47: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b47 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e1f642.1: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.cc6: <witness> = require_complete_type %J.type.04e [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.e1f642.1 [symbolic]
// CHECK:STDOUT:   %D.type.facet: %type = facet_value %D, () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%ptr)> [symbolic]
// CHECK:STDOUT:   %Self.e1f642.2: %N.type.d682d6.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %require_complete.19b: <witness> = require_complete_type %J.type.80f [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.D: type = import_ref Main//import_generic_decl, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.J: %J.type.885 = import_ref Main//import_generic_decl, J, loaded [concrete = constants.%J.generic]
// CHECK:STDOUT:   %Main.N: %N.type.b47 = import_ref Main//import_generic_decl, N, loaded [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Main.import_ref.e9b = import_ref Main//import_generic_decl, loc15, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//import_generic_decl, loc11, loaded [symbolic = @J.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//import_generic_decl, loc7, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.1d5 = import_ref Main//import_generic_decl, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %J.impl_witness_table.50b = impl_witness_table (), @D.as.J.impl.b25a4e.1 [concrete]
// CHECK:STDOUT:   %Main.import_ref.4ae308.1: type = import_ref Main//import_generic_decl, loc41, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.f2a55c.1: type = import_ref Main//import_generic_decl, loc46, loaded [symbolic = @D.as.J.impl.b25a4e.1.%J.type (constants.%J.type.04e)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//import_generic_decl, loc37, loaded [symbolic = @D.as.J.impl.b25a4e.1.%T (constants.%T)]
// CHECK:STDOUT:   %J.impl_witness_table.a37 = impl_witness_table (), @D.as.J.impl.f050b9.1 [concrete]
// CHECK:STDOUT:   %Main.import_ref.4ae308.2: type = import_ref Main//import_generic_decl, loc55, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.import_ref.739: type = import_ref Main//import_generic_decl, loc61, loaded [symbolic = @D.as.J.impl.f050b9.1.%J.type (constants.%J.type.80f)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//import_generic_decl, loc51, loaded [symbolic = @D.as.J.impl.f050b9.1.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.9d6 = import_ref Main//import_generic_decl, loc32, unloaded
// CHECK:STDOUT:   %Main.import_ref.213: type = import_ref Main//import_generic_decl, loc27, loaded [symbolic = @N.WithSelf.Self.binding.as_type.impls.J.type.require.%Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %Main.import_ref.f2a55c.2: type = import_ref Main//import_generic_decl, loc31, loaded [symbolic = @N.WithSelf.Self.binding.as_type.impls.J.type.require.%J.type (constants.%J.type.04e)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.6: type = import_ref Main//import_generic_decl, loc20, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.2fc394.2: @N.%N.type (%N.type.d682d6.1) = import_ref Main//import_generic_decl, loc24, loaded [symbolic = @N.%Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Main.import_ref.830 = import_ref Main//import_generic_decl, loc24, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.7: type = import_ref Main//import_generic_decl, loc20, loaded [symbolic = @N.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .J = imports.%Main.J
// CHECK:STDOUT:     .N = imports.%Main.N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc4.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4.2 = import <none>
// CHECK:STDOUT:   impl_decl @D.as.J.impl.b25a4e.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.885 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8.1 [symbolic = %T.loc8.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc17.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc17.2 (constants.%J.type.04e)]
// CHECK:STDOUT:     %.loc10.1: type = splice_block %.loc10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8.1: type = symbolic_binding T, 0 [symbolic = %T.loc8.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.b25a4e.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.885 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22.1 [symbolic = %T.loc22.2 (constants.%T)]
// CHECK:STDOUT:     %J.type.loc31.1: type = facet_type <@J, @J(constants.%T)> [symbolic = %J.type.loc31.2 (constants.%J.type.04e)]
// CHECK:STDOUT:     %.loc24.1: type = splice_block %.loc24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22.1: type = symbolic_binding T, 0 [symbolic = %T.loc22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.f050b9.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.885 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc37.1 [symbolic = %T.loc37.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc46.1: type = ptr_type %T.ref [symbolic = %ptr.loc46.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc47.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc47.2 (constants.%J.type.80f)]
// CHECK:STDOUT:     %.loc39.1: type = splice_block %.loc39.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc39.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc37.1: type = symbolic_binding T, 0 [symbolic = %T.loc37.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.f050b9.3 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %J.ref: %J.type.885 = name_ref J, imports.%Main.J [concrete = constants.%J.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc52.1 [symbolic = %T.loc52.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc61.1: type = ptr_type %T.ref [symbolic = %ptr.loc61.2 (constants.%ptr)]
// CHECK:STDOUT:     %J.type.loc62.1: type = facet_type <@J, @J(constants.%ptr)> [symbolic = %J.type.loc62.2 (constants.%J.type.80f)]
// CHECK:STDOUT:     %.loc54.1: type = splice_block %.loc54.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc54.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc52.1: type = symbolic_binding T, 0 [symbolic = %T.loc52.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.e28c26.1 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %N.ref: %N.type.b47 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc68.1 [symbolic = %T.loc68.2 (constants.%T)]
// CHECK:STDOUT:     %N.type.loc77.1: type = facet_type <@N, @N(constants.%T)> [symbolic = %N.type.loc77.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:     %.loc70.1: type = splice_block %.loc70.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc70.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc68.1: type = symbolic_binding T, 0 [symbolic = %T.loc68.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @D.as.J.impl.e28c26.2 [concrete] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:     %N.ref: %N.type.b47 = name_ref N, imports.%Main.N [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc83.1 [symbolic = %T.loc83.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc92.1: type = ptr_type %T.ref [symbolic = %ptr.loc92.2 (constants.%ptr)]
// CHECK:STDOUT:     %N.type.loc93.1: type = facet_type <@N, @N(constants.%ptr)> [symbolic = %N.type.loc93.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:     %.loc85.1: type = splice_block %.loc85.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc85.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc83.1: type = symbolic_binding T, 0 [symbolic = %T.loc83.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @J(imports.%Main.import_ref.b3bc94.2: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.04e)]
// CHECK:STDOUT:   %Self: @J.%J.type (%J.type.04e) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c25)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.e9b
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
// CHECK:STDOUT:   %Self: @N.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.830
// CHECK:STDOUT:     extend imports.%Main.import_ref.9d6
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.binding.as_type.impls.J.type.require {
// CHECK:STDOUT:       require imports.%Main.import_ref.213 impls imports.%Main.import_ref.f2a55c.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.binding.as_type.impls.J.type.require(imports.%Main.import_ref.b3bc94.6: type, imports.%Main.import_ref.2fc394.2: @N.%N.type (%N.type.d682d6.1)) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%T)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.binding.as_type.impls.J.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e1f642.1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.04e)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.b25a4e.1(imports.%Main.import_ref.b3bc94.3: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%T)> [symbolic = %J.type (constants.%J.type.04e)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness imports.%J.impl_witness_table.50b, @D.as.J.impl.b25a4e.1(%T) [symbolic = %J.impl_witness (constants.%J.impl_witness.861)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.4ae308.1 as imports.%Main.import_ref.f2a55c.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.f050b9.1(imports.%Main.import_ref.b3bc94.4: type) [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %J.type: type = facet_type <@J, @J(%ptr)> [symbolic = %J.type (constants.%J.type.80f)]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness imports.%J.impl_witness_table.a37, @D.as.J.impl.f050b9.1(%T) [symbolic = %J.impl_witness (constants.%J.impl_witness.f4f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Main.import_ref.4ae308.2 as imports.%Main.import_ref.739;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.b25a4e.2(%T.loc8.1: type) {
// CHECK:STDOUT:   %T.loc8.2: type = symbolic_binding T, 0 [symbolic = %T.loc8.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc17.2: type = facet_type <@J, @J(%T.loc8.2)> [symbolic = %J.type.loc17.2 (constants.%J.type.04e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc17.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.b25a4e.3(%T.loc22.1: type) {
// CHECK:STDOUT:   %T.loc22.2: type = symbolic_binding T, 0 [symbolic = %T.loc22.2 (constants.%T)]
// CHECK:STDOUT:   %J.type.loc31.2: type = facet_type <@J, @J(%T.loc22.2)> [symbolic = %J.type.loc31.2 (constants.%J.type.04e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc31.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.f050b9.2(%T.loc37.1: type) {
// CHECK:STDOUT:   %T.loc37.2: type = symbolic_binding T, 0 [symbolic = %T.loc37.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc46.2: type = ptr_type %T.loc37.2 [symbolic = %ptr.loc46.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc47.2: type = facet_type <@J, @J(%ptr.loc46.2)> [symbolic = %J.type.loc47.2 (constants.%J.type.80f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc47.1;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.f050b9.3(%T.loc52.1: type) {
// CHECK:STDOUT:   %T.loc52.2: type = symbolic_binding T, 0 [symbolic = %T.loc52.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc61.2: type = ptr_type %T.loc52.2 [symbolic = %ptr.loc61.2 (constants.%ptr)]
// CHECK:STDOUT:   %J.type.loc62.2: type = facet_type <@J, @J(%ptr.loc61.2)> [symbolic = %J.type.loc62.2 (constants.%J.type.80f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %J.type.loc62.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.e28c26.1(%T.loc68.1: type) {
// CHECK:STDOUT:   %T.loc68.2: type = symbolic_binding T, 0 [symbolic = %T.loc68.2 (constants.%T)]
// CHECK:STDOUT:   %N.type.loc77.2: type = facet_type <@N, @N(%T.loc68.2)> [symbolic = %N.type.loc77.2 (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %N.type.loc77.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.J.impl.e28c26.2(%T.loc83.1: type) {
// CHECK:STDOUT:   %T.loc83.2: type = symbolic_binding T, 0 [symbolic = %T.loc83.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc92.2: type = ptr_type %T.loc83.2 [symbolic = %ptr.loc92.2 (constants.%ptr)]
// CHECK:STDOUT:   %N.type.loc93.2: type = facet_type <@N, @N(%ptr.loc92.2)> [symbolic = %N.type.loc93.2 (constants.%N.type.d682d6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %D.ref as %N.type.loc93.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "fail_import_generic_decl.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.1d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %J.type => constants.%J.type.04e
// CHECK:STDOUT:   %Self => constants.%Self.c25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%T, constants.%Self.c25) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.b25a4e.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.04e
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.f050b9.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.80f
// CHECK:STDOUT:   %J.impl_witness => constants.%J.impl_witness.f4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.b25a4e.2(constants.%T) {
// CHECK:STDOUT:   %T.loc8.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc17.2 => constants.%J.type.04e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.b25a4e.3(constants.%T) {
// CHECK:STDOUT:   %T.loc22.2 => constants.%T
// CHECK:STDOUT:   %J.type.loc31.2 => constants.%J.type.04e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.f050b9.2(constants.%T) {
// CHECK:STDOUT:   %T.loc37.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc46.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc47.2 => constants.%J.type.80f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.f050b9.3(constants.%T) {
// CHECK:STDOUT:   %T.loc52.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc61.2 => constants.%ptr
// CHECK:STDOUT:   %J.type.loc62.2 => constants.%J.type.80f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%Self.e1f642.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.04e
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.J.type.require(constants.%T, constants.%Self.e1f642.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %J.type => constants.%J.type.04e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%T, constants.%D.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %J.type => constants.%J.type.04e
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.J.type.require(constants.%T, constants.%D.type.facet) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%D.type.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%D
// CHECK:STDOUT:   %J.type => constants.%J.type.04e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.e28c26.1(constants.%T) {
// CHECK:STDOUT:   %T.loc68.2 => constants.%T
// CHECK:STDOUT:   %N.type.loc77.2 => constants.%N.type.d682d6.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%ptr) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%Self.e1f642.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr, constants.%Self.e1f642.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.80f
// CHECK:STDOUT:   %require_complete => constants.%require_complete.19b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%ptr, constants.%D.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %J.type => constants.%J.type.80f
// CHECK:STDOUT:   %require_complete => constants.%require_complete.19b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.binding.as_type.impls.J.type.require(constants.%ptr, constants.%D.type.facet) {
// CHECK:STDOUT:   %T => constants.%ptr
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.2
// CHECK:STDOUT:   %Self => constants.%D.type.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%D
// CHECK:STDOUT:   %J.type => constants.%J.type.80f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.J.impl.e28c26.2(constants.%T) {
// CHECK:STDOUT:   %T.loc83.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc92.2 => constants.%ptr
// CHECK:STDOUT:   %N.type.loc93.2 => constants.%N.type.d682d6.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

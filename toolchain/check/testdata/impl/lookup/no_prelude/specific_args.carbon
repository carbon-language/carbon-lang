// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/no_prelude/specific_args.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/no_prelude/specific_args.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) { fn F[self: Self](); }
class C(T:! type) {}

class X {}

// --- impl_in_interface_args.carbon

library "[[@TEST_NAME]]";
import library "types";

class InInterfaceArgs {}

impl X as I(InInterfaceArgs) { fn F[self: Self]() {} }

// --- use_impl_in_interface_args.carbon

library "[[@TEST_NAME]]";
import library "types";
import library "impl_in_interface_args";

fn G(x: X) { x.(I(InInterfaceArgs).F)(); }

// --- impl_in_class_args.carbon

library "[[@TEST_NAME]]";
import library "types";

class InClassArgs {}

impl C(InClassArgs) as I(X) { fn F[self: Self]() {} }

// --- use_impl_in_class_args.carbon

library "[[@TEST_NAME]]";
import library "types";
import library "impl_in_class_args";

fn H(c: C(InClassArgs)) { c.(I(X).F)(); }

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc5_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.2)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self.2: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.325) [symbolic = %I.assoc_type (constants.%I.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_43.2: @I.%I.assoc_type (%I.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc4_43.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.325) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type) = fn_decl @F [symbolic = @I.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%Self.as_type.loc4_36.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.%Self.as_type.loc4_36.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.%Self.as_type.loc4_36.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_36.1: type = splice_block %.loc4_36.3 [symbolic = %Self.as_type.loc4_36.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc4_36.2: @F.%I.type (%I.type.325) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @F.%I.type (%I.type.325) = name_ref Self, %.loc4_36.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc4_36.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_36.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc4_36.3: type = converted %Self.ref, %Self.as_type.loc4_36.2 [symbolic = %Self.as_type.loc4_36.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.%Self.as_type.loc4_36.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_43.1: @I.%I.assoc_type (%I.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc4_43.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc4_43.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc5_9.1: type) {
// CHECK:STDOUT:   %T.loc5_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc5_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%T.loc4_13.1: type, @I.%Self.1: @I.%I.type (%I.type.325)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_36.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_36.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Self.as_type.loc4_36.1 (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_36.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T.loc4_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc5_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc5_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_in_interface_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InInterfaceArgs: type = class_type @InInterfaceArgs [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.2ae: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.bb2: %F.type.2ae = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %I.assoc_type.955: type = assoc_entity_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %assoc0.249: %I.assoc_type.955 = assoc_entity element0, imports.%Main.import_ref.479 [symbolic]
// CHECK:STDOUT:   %I.type.e45: type = facet_type <@I, @I(%InInterfaceArgs)> [concrete]
// CHECK:STDOUT:   %F.type.14f: type = fn_type @F.1, @I(%InInterfaceArgs) [concrete]
// CHECK:STDOUT:   %F.b81: %F.type.14f = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.9f3: type = assoc_entity_type %I.type.e45 [concrete]
// CHECK:STDOUT:   %assoc0.055: %I.assoc_type.9f3 = assoc_entity element0, imports.%Main.import_ref.479 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.b63: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.d5f: %F.type.b63 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.e45 = facet_value %X, (%impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.acf = import_ref Main//types, inst65 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.884 = import_ref Main//types, inst25 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.cbe = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.F: @I.%F.type (%F.type.2ae) = import_ref Main//types, F, loaded [symbolic = @I.%F (constants.%F.bb2)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.38e: @I.%I.type (%I.type.325) = import_ref Main//types, inst25 [no loc], loaded [symbolic = @I.%Self (constants.%Self)]
// CHECK:STDOUT:   %Main.import_ref.479 = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .X = imports.%Main.X
// CHECK:STDOUT:     .InInterfaceArgs = %InInterfaceArgs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %InInterfaceArgs.decl: type = class_decl @InInterfaceArgs [concrete = constants.%InInterfaceArgs] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, file.%InInterfaceArgs.decl [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [concrete = constants.%I.type.e45]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.5ab3ec.1: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.2ae)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.2ae) = struct_value () [symbolic = %F (constants.%F.bb2)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.325) [symbolic = %I.assoc_type (constants.%I.assoc_type.955)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.955) = assoc_entity element0, imports.%Main.import_ref.479 [symbolic = %assoc0 (constants.%assoc0.249)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.884
// CHECK:STDOUT:     .F = imports.%Main.import_ref.cbe
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %X.ref as %I.type {
// CHECK:STDOUT:   %F.decl: %F.type.b63 = fn_decl @F.2 [concrete = constants.%F.d5f] {
// CHECK:STDOUT:     %self.patt: %X = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %X = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %X = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%X.ref [concrete = constants.%X]
// CHECK:STDOUT:     %self: %X = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InInterfaceArgs {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%InInterfaceArgs
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.acf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.38e: @I.%I.type (%I.type.325)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %X]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%InInterfaceArgs) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %T.patt => constants.%InInterfaceArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.e45
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.14f
// CHECK:STDOUT:   %F => constants.%F.b81
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.9f3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.055
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%InInterfaceArgs, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.e45
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_impl_in_interface_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.2ae: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.bb2: %F.type.2ae = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %I.assoc_type.955: type = assoc_entity_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %assoc0.62c: %I.assoc_type.955 = assoc_entity element0, imports.%Main.import_ref.e54 [symbolic]
// CHECK:STDOUT:   %InInterfaceArgs: type = class_type @InInterfaceArgs [concrete]
// CHECK:STDOUT:   %I.type.e45: type = facet_type <@I, @I(%InInterfaceArgs)> [concrete]
// CHECK:STDOUT:   %F.type.14f: type = fn_type @F.1, @I(%InInterfaceArgs) [concrete]
// CHECK:STDOUT:   %F.b81: %F.type.14f = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.9f3: type = assoc_entity_type %I.type.e45 [concrete]
// CHECK:STDOUT:   %assoc0.0fc: %I.assoc_type.9f3 = assoc_entity element0, imports.%Main.import_ref.e54 [concrete]
// CHECK:STDOUT:   %assoc0.249: %I.assoc_type.955 = assoc_entity element0, imports.%Main.import_ref.479 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Main.import_ref.7f5) [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.e45 = facet_value %X, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.f71: type = fn_type_with_self_type %F.type.14f, %I.facet [concrete]
// CHECK:STDOUT:   %F.type.b63: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.d5f: %F.type.b63 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.InInterfaceArgs: type = import_ref Main//impl_in_interface_args, InInterfaceArgs, loaded [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.acf = import_ref Main//types, inst65 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.884 = import_ref Main//types, inst25 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.cd3: @I.%I.assoc_type (%I.assoc_type.955) = import_ref Main//types, loc4_43, loaded [symbolic = @I.%assoc0 (constants.%assoc0.249)]
// CHECK:STDOUT:   %Main.F = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.38e: @I.%I.type (%I.type.325) = import_ref Main//types, inst25 [no loc], loaded [symbolic = @I.%Self (constants.%Self)]
// CHECK:STDOUT:   %Main.import_ref.e54: @I.%F.type (%F.type.2ae) = import_ref Main//types, loc4_43, loaded [symbolic = @I.%F (constants.%F.bb2)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_interface_args, loc5_24, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.bf8 = import_ref Main//impl_in_interface_args, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.f9f: <witness> = import_ref Main//impl_in_interface_args, loc7_30, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %Main.import_ref.956: type = import_ref Main//impl_in_interface_args, loc7_6, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.e8c: type = import_ref Main//impl_in_interface_args, loc7_28, loaded [concrete = constants.%I.type.e45]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .X = imports.%Main.X
// CHECK:STDOUT:     .InInterfaceArgs = imports.%Main.InInterfaceArgs
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %x.patt: %X = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %X = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:     %x: %X = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.5ab3ec.1: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.2ae)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.2ae) = struct_value () [symbolic = %F (constants.%F.bb2)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.325) [symbolic = %I.assoc_type (constants.%I.assoc_type.955)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.955) = assoc_entity element0, imports.%Main.import_ref.e54 [symbolic = %assoc0 (constants.%assoc0.62c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.884
// CHECK:STDOUT:     .F = imports.%Main.import_ref.cd3
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%Main.import_ref.956 as imports.%Main.import_ref.e8c [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.f9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.acf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InInterfaceArgs [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.bf8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %X = name_ref x, %x
// CHECK:STDOUT:   %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:   %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, imports.%Main.InInterfaceArgs [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [concrete = constants.%I.type.e45]
// CHECK:STDOUT:   %.loc6: %I.assoc_type.9f3 = specific_constant imports.%Main.import_ref.cd3, @I(constants.%InInterfaceArgs) [concrete = constants.%assoc0.0fc]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.9f3 = name_ref F, %.loc6 [concrete = constants.%assoc0.0fc]
// CHECK:STDOUT:   %impl.elem0: %.f71 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%F.d5f]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %bound_method(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.2: type, imports.%Main.import_ref.38e: @I.%I.type (%I.type.325)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %X]() [from "impl_in_interface_args.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%InInterfaceArgs) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %T.patt => constants.%InInterfaceArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.e45
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.14f
// CHECK:STDOUT:   %F => constants.%F.b81
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.9f3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_in_class_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InClassArgs: type = class_type @InClassArgs [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.23b: type = class_type @C, @C(%InClassArgs) [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.2ae: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.bb2: %F.type.2ae = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %I.assoc_type.955: type = assoc_entity_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %assoc0.249: %I.assoc_type.955 = assoc_entity element0, imports.%Main.import_ref.479 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %I.type.45c: type = facet_type <@I, @I(%X)> [concrete]
// CHECK:STDOUT:   %F.type.56a: type = fn_type @F.1, @I(%X) [concrete]
// CHECK:STDOUT:   %F.a79: %F.type.56a = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.a7e: type = assoc_entity_type %I.type.45c [concrete]
// CHECK:STDOUT:   %assoc0.7c6: %I.assoc_type.a7e = assoc_entity element0, imports.%Main.import_ref.479 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.fab: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.7ab: %F.type.fab = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.45c = facet_value %C.23b, (%impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//types, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//types, loc5_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.4c0 = import_ref Main//types, inst60 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.884 = import_ref Main//types, inst25 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.cbe = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.F: @I.%F.type (%F.type.2ae) = import_ref Main//types, F, loaded [symbolic = @I.%F (constants.%F.bb2)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.38e: @I.%I.type (%I.type.325) = import_ref Main//types, inst25 [no loc], loaded [symbolic = @I.%Self (constants.%Self)]
// CHECK:STDOUT:   %Main.import_ref.479 = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.acf = import_ref Main//types, inst65 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .X = imports.%Main.X
// CHECK:STDOUT:     .InClassArgs = %InClassArgs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %InClassArgs.decl: type = class_decl @InClassArgs [concrete = constants.%InClassArgs] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %InClassArgs.ref: type = name_ref InClassArgs, file.%InClassArgs.decl [concrete = constants.%InClassArgs]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%InClassArgs) [concrete = constants.%C.23b]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%X)> [concrete = constants.%I.type.45c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.5ab3ec.2: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.2ae)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.2ae) = struct_value () [symbolic = %F (constants.%F.bb2)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.325) [symbolic = %I.assoc_type (constants.%I.assoc_type.955)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.955) = assoc_entity element0, imports.%Main.import_ref.479 [symbolic = %assoc0 (constants.%assoc0.249)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.884
// CHECK:STDOUT:     .F = imports.%Main.import_ref.cbe
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %I.type {
// CHECK:STDOUT:   %F.decl: %F.type.fab = fn_decl @F.2 [concrete = constants.%F.7ab] {
// CHECK:STDOUT:     %self.patt: %C.23b = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C.23b = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C.23b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%C [concrete = constants.%C.23b]
// CHECK:STDOUT:     %self: %C.23b = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InClassArgs {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%InClassArgs
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.5ab3ec.1: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.4c0
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.acf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.3: type, imports.%Main.import_ref.38e: @I.%I.type (%I.type.325)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %C.23b]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%InClassArgs) {
// CHECK:STDOUT:   %T => constants.%InClassArgs
// CHECK:STDOUT:   %T.patt => constants.%InClassArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %T.patt => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.45c
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.56a
// CHECK:STDOUT:   %F => constants.%F.a79
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.a7e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7c6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.45c
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C.23b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_impl_in_class_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %InClassArgs: type = class_type @InClassArgs [concrete]
// CHECK:STDOUT:   %C.23b: type = class_type @C, @C(%InClassArgs) [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.2ae: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.bb2: %F.type.2ae = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %I.assoc_type.955: type = assoc_entity_type %I.type.325 [symbolic]
// CHECK:STDOUT:   %assoc0.62c: %I.assoc_type.955 = assoc_entity element0, imports.%Main.import_ref.e54 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %I.type.45c: type = facet_type <@I, @I(%X)> [concrete]
// CHECK:STDOUT:   %F.type.56a: type = fn_type @F.1, @I(%X) [concrete]
// CHECK:STDOUT:   %F.a79: %F.type.56a = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.a7e: type = assoc_entity_type %I.type.45c [concrete]
// CHECK:STDOUT:   %assoc0.ba0: %I.assoc_type.a7e = assoc_entity element0, imports.%Main.import_ref.e54 [concrete]
// CHECK:STDOUT:   %assoc0.249: %I.assoc_type.955 = assoc_entity element0, imports.%Main.import_ref.479 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%Main.import_ref.ae4) [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.45c = facet_value %C.23b, (%impl_witness) [concrete]
// CHECK:STDOUT:   %.0e8: type = fn_type_with_self_type %F.type.56a, %I.facet [concrete]
// CHECK:STDOUT:   %F.type.fab: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.7ab: %F.type.fab = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//types, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.InClassArgs: type = import_ref Main//impl_in_class_args, InClassArgs, loaded [concrete = constants.%InClassArgs]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//types, loc5_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.4c0 = import_ref Main//types, inst60 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_class_args, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.683 = import_ref Main//impl_in_class_args, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.884 = import_ref Main//types, inst25 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.cd3: @I.%I.assoc_type (%I.assoc_type.955) = import_ref Main//types, loc4_43, loaded [symbolic = @I.%assoc0 (constants.%assoc0.249)]
// CHECK:STDOUT:   %Main.F = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.38e: @I.%I.type (%I.type.325) = import_ref Main//types, inst25 [no loc], loaded [symbolic = @I.%Self (constants.%Self)]
// CHECK:STDOUT:   %Main.import_ref.e54: @I.%F.type (%F.type.2ae) = import_ref Main//types, loc4_43, loaded [symbolic = @I.%F (constants.%F.bb2)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.acf = import_ref Main//types, inst65 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.6de: <witness> = import_ref Main//impl_in_class_args, loc7_29, loaded [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %Main.import_ref.d6e: type = import_ref Main//impl_in_class_args, loc7_19, loaded [concrete = constants.%C.23b]
// CHECK:STDOUT:   %Main.import_ref.208: type = import_ref Main//impl_in_class_args, loc7_27, loaded [concrete = constants.%I.type.45c]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .X = imports.%Main.X
// CHECK:STDOUT:     .InClassArgs = imports.%Main.InClassArgs
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %c.patt: %C.23b = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C.23b = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.23b = value_param call_param0
// CHECK:STDOUT:     %.loc6_22: type = splice_block %C [concrete = constants.%C.23b] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:       %InClassArgs.ref: type = name_ref InClassArgs, imports.%Main.InClassArgs [concrete = constants.%InClassArgs]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%InClassArgs) [concrete = constants.%C.23b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.23b = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.5ab3ec.2: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.2ae)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.2ae) = struct_value () [symbolic = %F (constants.%F.bb2)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.325) [symbolic = %I.assoc_type (constants.%I.assoc_type.955)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.955) = assoc_entity element0, imports.%Main.import_ref.e54 [symbolic = %assoc0 (constants.%assoc0.62c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.884
// CHECK:STDOUT:     .F = imports.%Main.import_ref.cd3
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%Main.import_ref.d6e as imports.%Main.import_ref.208 [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.5ab3ec.1: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.4c0
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InClassArgs [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.683
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.acf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param_patt: %C.23b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.23b = name_ref c, %c
// CHECK:STDOUT:   %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%X)> [concrete = constants.%I.type.45c]
// CHECK:STDOUT:   %.loc6_34: %I.assoc_type.a7e = specific_constant imports.%Main.import_ref.cd3, @I(constants.%X) [concrete = constants.%assoc0.ba0]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.a7e = name_ref F, %.loc6_34 [concrete = constants.%assoc0.ba0]
// CHECK:STDOUT:   %impl.elem0: %.0e8 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%F.7ab]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.3: type, imports.%Main.import_ref.38e: @I.%I.type (%I.type.325)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self: %I.type.325 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type (%Self.as_type)]();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %C.23b]() [from "impl_in_class_args.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%InClassArgs) {
// CHECK:STDOUT:   %T => constants.%InClassArgs
// CHECK:STDOUT:   %T.patt => constants.%InClassArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %T.patt => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.45c
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.56a
// CHECK:STDOUT:   %F => constants.%F.a79
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.a7e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ba0
// CHECK:STDOUT: }
// CHECK:STDOUT:

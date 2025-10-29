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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/specific_args.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/specific_args.carbon

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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.dbc: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.F.decl [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc5_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.1)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%T.loc4_13.1) [symbolic = %I.F.type (constants.%I.F.type)]
// CHECK:STDOUT:   %I.F: @I.%I.F.type (%I.F.type) = struct_value () [symbolic = %I.F (constants.%I.F)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T.loc4_13.1) [symbolic = %I.assoc_type (constants.%I.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_43.2: @I.%I.assoc_type (%I.assoc_type) = assoc_entity element0, %I.F.decl [symbolic = %assoc0.loc4_43.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %I.F.decl: @I.%I.F.type (%I.F.type) = fn_decl @I.F [symbolic = @I.%I.F (constants.%I.F)] {
// CHECK:STDOUT:       %self.patt: @I.F.%pattern_type (%pattern_type.dbc) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @I.F.%pattern_type (%pattern_type.dbc) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @I.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_36.1: type = splice_block %.loc4_36.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc4_36.2: @I.F.%I.type (%I.type.070) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @I.F.%I.type (%I.type.070) = name_ref Self, %.loc4_36.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc4_36.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @I.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_43.1: @I.%I.assoc_type (%I.assoc_type) = assoc_entity element0, %I.F.decl [symbolic = %assoc0.loc4_43.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc4_43.1
// CHECK:STDOUT:     witness = (%I.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc5_9.2: type) {
// CHECK:STDOUT:   %T.loc5_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%T.loc4_13.2: type, @I.%Self.1: @I.%I.type (%I.type.070)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.F.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.dbc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.F.%Self.binding.as_type (%Self.binding.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dbc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc5_9.1 => constants.%T
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.76d: type = fn_type @I.F, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.F.dde: %I.F.type.76d = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.269 [symbolic]
// CHECK:STDOUT:   %pattern_type.dbc: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.assoc_type.b65: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.254: %I.assoc_type.b65 = assoc_entity element0, imports.%Main.import_ref.e2f [symbolic]
// CHECK:STDOUT:   %I.type.c61: type = facet_type <@I, @I(%InInterfaceArgs)> [concrete]
// CHECK:STDOUT:   %Self.bca: %I.type.c61 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.14f: type = fn_type @I.F, @I(%InInterfaceArgs) [concrete]
// CHECK:STDOUT:   %I.F.b81: %I.F.type.14f = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.521: type = assoc_entity_type @I, @I(%InInterfaceArgs) [concrete]
// CHECK:STDOUT:   %assoc0.557: %I.assoc_type.521 = assoc_entity element0, imports.%Main.import_ref.e2f [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F: %X.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.c61 = facet_value %X, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.acf = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.1: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.769 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.2de = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.F: @I.%I.F.type (%I.F.type.76d) = import_ref Main//types, F, loaded [symbolic = @I.%I.F (constants.%I.F.dde)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.2: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.781: @I.%I.type (%I.type.070) = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], loaded [symbolic = @I.%Self (constants.%Self.269)]
// CHECK:STDOUT:   %Main.import_ref.e2f = import_ref Main//types, loc4_43, unloaded
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
// CHECK:STDOUT:   impl_decl @X.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, file.%InInterfaceArgs.decl [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [concrete = constants.%I.type.c61]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@X.as.I.impl.%X.as.I.impl.F.decl), @X.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.efcd44.1: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%T) [symbolic = %I.F.type (constants.%I.F.type.76d)]
// CHECK:STDOUT:   %I.F: @I.%I.F.type (%I.F.type.76d) = struct_value () [symbolic = %I.F (constants.%I.F.dde)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T) [symbolic = %I.assoc_type (constants.%I.assoc_type.b65)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.b65) = assoc_entity element0, imports.%Main.import_ref.e2f [symbolic = %assoc0 (constants.%assoc0.254)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.769
// CHECK:STDOUT:     .F = imports.%Main.import_ref.2de
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.I.impl: %X.ref as %I.type {
// CHECK:STDOUT:   %X.as.I.impl.F.decl: %X.as.I.impl.F.type = fn_decl @X.as.I.impl.F [concrete = constants.%X.as.I.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.019 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.019 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %X = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @X.as.I.impl.%X.ref [concrete = constants.%X]
// CHECK:STDOUT:     %self: %X = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %X.as.I.impl.F.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InInterfaceArgs {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: generic fn @I.F(imports.%Main.import_ref.efcd44.2: type, imports.%Main.import_ref.781: @I.%I.type (%I.type.070)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.F.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.dbc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.I.impl.F(%self.param: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%T, constants.%Self.269) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self => constants.%Self.269
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dbc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%InInterfaceArgs) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.c61
// CHECK:STDOUT:   %Self => constants.%Self.bca
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.14f
// CHECK:STDOUT:   %I.F => constants.%I.F.b81
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.521
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.557
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%InInterfaceArgs, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.c61
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.019
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_impl_in_interface_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.76d: type = fn_type @I.F, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.F.dde: %I.F.type.76d = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.269 [symbolic]
// CHECK:STDOUT:   %pattern_type.dbc: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.assoc_type.b65: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.436: %I.assoc_type.b65 = assoc_entity element0, imports.%Main.import_ref.21c [symbolic]
// CHECK:STDOUT:   %InInterfaceArgs: type = class_type @InInterfaceArgs [concrete]
// CHECK:STDOUT:   %I.type.c61: type = facet_type <@I, @I(%InInterfaceArgs)> [concrete]
// CHECK:STDOUT:   %Self.bca: %I.type.c61 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.14f: type = fn_type @I.F, @I(%InInterfaceArgs) [concrete]
// CHECK:STDOUT:   %I.F.b81: %I.F.type.14f = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.521: type = assoc_entity_type @I, @I(%InInterfaceArgs) [concrete]
// CHECK:STDOUT:   %assoc0.ad2: %I.assoc_type.521 = assoc_entity element0, imports.%Main.import_ref.21c [concrete]
// CHECK:STDOUT:   %assoc0.254: %I.assoc_type.b65 = assoc_entity element0, imports.%Main.import_ref.e2f [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.c61 = facet_value %X, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %.ed4: type = fn_type_with_self_type %I.F.type.14f, %I.facet [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F: %X.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.InInterfaceArgs: type = import_ref Main//impl_in_interface_args, InInterfaceArgs, loaded [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.acf = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.1: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.769 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.715: @I.%I.assoc_type (%I.assoc_type.b65) = import_ref Main//types, loc4_43, loaded [symbolic = @I.%assoc0 (constants.%assoc0.254)]
// CHECK:STDOUT:   %Main.F = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.2: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.781: @I.%I.type (%I.type.070) = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], loaded [symbolic = @I.%Self (constants.%Self.269)]
// CHECK:STDOUT:   %Main.import_ref.21c: @I.%I.F.type (%I.F.type.76d) = import_ref Main//types, loc4_43, loaded [symbolic = @I.%I.F (constants.%I.F.dde)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_interface_args, loc5_24, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.bf8 = import_ref Main//impl_in_interface_args, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e2f = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.7c8: <witness> = import_ref Main//impl_in_interface_args, loc7_30, loaded [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.956: type = import_ref Main//impl_in_interface_args, loc7_6, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.c00: type = import_ref Main//impl_in_interface_args, loc7_28, loaded [concrete = constants.%I.type.c61]
// CHECK:STDOUT:   %Main.import_ref.1ac: %X.as.I.impl.F.type = import_ref Main//impl_in_interface_args, loc7_51, loaded [concrete = constants.%X.as.I.impl.F]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.1ac), @X.as.I.impl [concrete]
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
// CHECK:STDOUT:     %x.patt: %pattern_type.019 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.019 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:     %x: %X = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.efcd44.1: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%T) [symbolic = %I.F.type (constants.%I.F.type.76d)]
// CHECK:STDOUT:   %I.F: @I.%I.F.type (%I.F.type.76d) = struct_value () [symbolic = %I.F (constants.%I.F.dde)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T) [symbolic = %I.assoc_type (constants.%I.assoc_type.b65)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.b65) = assoc_entity element0, imports.%Main.import_ref.21c [symbolic = %assoc0 (constants.%assoc0.436)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.769
// CHECK:STDOUT:     .F = imports.%Main.import_ref.715
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.I.impl: imports.%Main.import_ref.956 as imports.%Main.import_ref.c00 [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.7c8
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
// CHECK:STDOUT: fn @G(%x.param: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %X = name_ref x, %x
// CHECK:STDOUT:   %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:   %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, imports.%Main.InInterfaceArgs [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [concrete = constants.%I.type.c61]
// CHECK:STDOUT:   %.loc6: %I.assoc_type.521 = specific_constant imports.%Main.import_ref.715, @I(constants.%InInterfaceArgs) [concrete = constants.%assoc0.ad2]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.521 = name_ref F, %.loc6 [concrete = constants.%assoc0.ad2]
// CHECK:STDOUT:   %impl.elem0: %.ed4 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%X.as.I.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %X.as.I.impl.F.call: init %empty_tuple.type = call %bound_method(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(imports.%Main.import_ref.efcd44.2: type, imports.%Main.import_ref.781: @I.%I.type (%I.type.070)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.F.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.dbc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.as.I.impl.F [from "impl_in_interface_args.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%T, constants.%Self.269) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self => constants.%Self.269
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dbc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%InInterfaceArgs) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.c61
// CHECK:STDOUT:   %Self => constants.%Self.bca
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.14f
// CHECK:STDOUT:   %I.F => constants.%I.F.b81
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.521
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ad2
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.23b: type = class_type @C, @C(%InClassArgs) [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.76d: type = fn_type @I.F, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.F.dde: %I.F.type.76d = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.269 [symbolic]
// CHECK:STDOUT:   %pattern_type.dbc: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.assoc_type.b65: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.254: %I.assoc_type.b65 = assoc_entity element0, imports.%Main.import_ref.e2f [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %I.type.95a: type = facet_type <@I, @I(%X)> [concrete]
// CHECK:STDOUT:   %Self.c72: %I.type.95a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.56a: type = fn_type @I.F, @I(%X) [concrete]
// CHECK:STDOUT:   %I.F.a79: %I.F.type.56a = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.2db: type = assoc_entity_type @I, @I(%X) [concrete]
// CHECK:STDOUT:   %assoc0.a52: %I.assoc_type.2db = assoc_entity element0, imports.%Main.import_ref.e2f [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.e06: type = pattern_type %C.23b [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.95a = facet_value %C.23b, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//types, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.efcd44.1: type = import_ref Main//types, loc5_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.034 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.2: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.769 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.2de = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.F: @I.%I.F.type (%I.F.type.76d) = import_ref Main//types, F, loaded [symbolic = @I.%I.F (constants.%I.F.dde)]
// CHECK:STDOUT:   %Main.import_ref.efcd44.3: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.781: @I.%I.type (%I.type.070) = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], loaded [symbolic = @I.%Self (constants.%Self.269)]
// CHECK:STDOUT:   %Main.import_ref.e2f = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.acf = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
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
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:     %InClassArgs.ref: type = name_ref InClassArgs, file.%InClassArgs.decl [concrete = constants.%InClassArgs]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%InClassArgs) [concrete = constants.%C.23b]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%X)> [concrete = constants.%I.type.95a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@C.as.I.impl.%C.as.I.impl.F.decl), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.efcd44.2: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%T) [symbolic = %I.F.type (constants.%I.F.type.76d)]
// CHECK:STDOUT:   %I.F: @I.%I.F.type (%I.F.type.76d) = struct_value () [symbolic = %I.F (constants.%I.F.dde)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T) [symbolic = %I.assoc_type (constants.%I.assoc_type.b65)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.b65) = assoc_entity element0, imports.%Main.import_ref.e2f [symbolic = %assoc0 (constants.%assoc0.254)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.769
// CHECK:STDOUT:     .F = imports.%Main.import_ref.2de
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C as %I.type {
// CHECK:STDOUT:   %C.as.I.impl.F.decl: %C.as.I.impl.F.type = fn_decl @C.as.I.impl.F [concrete = constants.%C.as.I.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.e06 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e06 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C.23b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.I.impl.%C [concrete = constants.%C.23b]
// CHECK:STDOUT:     %self: %C.23b = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InClassArgs {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%InClassArgs
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.efcd44.1: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.034
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
// CHECK:STDOUT: generic fn @I.F(imports.%Main.import_ref.efcd44.3: type, imports.%Main.import_ref.781: @I.%I.type (%I.type.070)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.F.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.dbc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F(%self.param: %C.23b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%InClassArgs) {
// CHECK:STDOUT:   %T => constants.%InClassArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%T, constants.%Self.269) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self => constants.%Self.269
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dbc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.95a
// CHECK:STDOUT:   %Self => constants.%Self.c72
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.56a
// CHECK:STDOUT:   %I.F => constants.%I.F.a79
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.2db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%X, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.95a
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C.23b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e06
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %InClassArgs: type = class_type @InClassArgs [concrete]
// CHECK:STDOUT:   %C.23b: type = class_type @C, @C(%InClassArgs) [concrete]
// CHECK:STDOUT:   %pattern_type.e06: type = pattern_type %C.23b [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.76d: type = fn_type @I.F, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.F.dde: %I.F.type.76d = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.269 [symbolic]
// CHECK:STDOUT:   %pattern_type.dbc: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %I.assoc_type.b65: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.436: %I.assoc_type.b65 = assoc_entity element0, imports.%Main.import_ref.21c [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %I.type.95a: type = facet_type <@I, @I(%X)> [concrete]
// CHECK:STDOUT:   %Self.c72: %I.type.95a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.56a: type = fn_type @I.F, @I(%X) [concrete]
// CHECK:STDOUT:   %I.F.a79: %I.F.type.56a = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.2db: type = assoc_entity_type @I, @I(%X) [concrete]
// CHECK:STDOUT:   %assoc0.abd: %I.assoc_type.2db = assoc_entity element0, imports.%Main.import_ref.21c [concrete]
// CHECK:STDOUT:   %assoc0.254: %I.assoc_type.b65 = assoc_entity element0, imports.%Main.import_ref.e2f [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.95a = facet_value %C.23b, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %.aa6: type = fn_type_with_self_type %I.F.type.56a, %I.facet [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.dac = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//types, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.InClassArgs: type = import_ref Main//impl_in_class_args, InClassArgs, loaded [concrete = constants.%InClassArgs]
// CHECK:STDOUT:   %Main.import_ref.efcd44.1: type = import_ref Main//types, loc5_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.034 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_class_args, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.683 = import_ref Main//impl_in_class_args, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.2: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.769 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.715: @I.%I.assoc_type (%I.assoc_type.b65) = import_ref Main//types, loc4_43, loaded [symbolic = @I.%assoc0 (constants.%assoc0.254)]
// CHECK:STDOUT:   %Main.F = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.efcd44.3: type = import_ref Main//types, loc4_13, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.781: @I.%I.type (%I.type.070) = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], loaded [symbolic = @I.%Self (constants.%Self.269)]
// CHECK:STDOUT:   %Main.import_ref.21c: @I.%I.F.type (%I.F.type.76d) = import_ref Main//types, loc4_43, loaded [symbolic = @I.%I.F (constants.%I.F.dde)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.acf = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e2f = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.56c: <witness> = import_ref Main//impl_in_class_args, loc7_29, loaded [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.d6e: type = import_ref Main//impl_in_class_args, loc7_19, loaded [concrete = constants.%C.23b]
// CHECK:STDOUT:   %Main.import_ref.cc1: type = import_ref Main//impl_in_class_args, loc7_27, loaded [concrete = constants.%I.type.95a]
// CHECK:STDOUT:   %Main.import_ref.849: %C.as.I.impl.F.type = import_ref Main//impl_in_class_args, loc7_50, loaded [concrete = constants.%C.as.I.impl.F]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.849), @C.as.I.impl [concrete]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.e06 = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.e06 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.23b = value_param call_param0
// CHECK:STDOUT:     %.loc6_22: type = splice_block %C [concrete = constants.%C.23b] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:       %InClassArgs.ref: type = name_ref InClassArgs, imports.%Main.InClassArgs [concrete = constants.%InClassArgs]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%InClassArgs) [concrete = constants.%C.23b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.23b = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.efcd44.2: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%T) [symbolic = %I.F.type (constants.%I.F.type.76d)]
// CHECK:STDOUT:   %I.F: @I.%I.F.type (%I.F.type.76d) = struct_value () [symbolic = %I.F (constants.%I.F.dde)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T) [symbolic = %I.assoc_type (constants.%I.assoc_type.b65)]
// CHECK:STDOUT:   %assoc0: @I.%I.assoc_type (%I.assoc_type.b65) = assoc_entity element0, imports.%Main.import_ref.21c [symbolic = %assoc0 (constants.%assoc0.436)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.769
// CHECK:STDOUT:     .F = imports.%Main.import_ref.715
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: imports.%Main.import_ref.d6e as imports.%Main.import_ref.cc1 [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.56c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.efcd44.1: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.034
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
// CHECK:STDOUT: fn @H(%c.param: %C.23b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.23b = name_ref c, %c
// CHECK:STDOUT:   %I.ref: %I.type.dac = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%X)> [concrete = constants.%I.type.95a]
// CHECK:STDOUT:   %.loc6_34: %I.assoc_type.2db = specific_constant imports.%Main.import_ref.715, @I(constants.%X) [concrete = constants.%assoc0.abd]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.2db = name_ref F, %.loc6_34 [concrete = constants.%assoc0.abd]
// CHECK:STDOUT:   %impl.elem0: %.aa6 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%C.as.I.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.I.impl.F.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(imports.%Main.import_ref.efcd44.3: type, imports.%Main.import_ref.781: @I.%I.type (%I.type.070)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self: @I.F.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.269)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.dbc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F [from "impl_in_class_args.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%InClassArgs) {
// CHECK:STDOUT:   %T => constants.%InClassArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%T, constants.%Self.269) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self => constants.%Self.269
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dbc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.95a
// CHECK:STDOUT:   %Self => constants.%Self.c72
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.56a
// CHECK:STDOUT:   %I.F => constants.%I.F.a79
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.2db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.abd
// CHECK:STDOUT: }
// CHECK:STDOUT:

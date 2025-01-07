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

interface I(T:! type) { fn F(); }
class C(T:! type) {}

class X {}

// --- impl_in_interface_args.carbon

library "[[@TEST_NAME]]";
import library "types";

class InInterfaceArgs {}

impl X as I(InInterfaceArgs) { fn F() {} }

// --- use_impl_in_interface_args.carbon

library "[[@TEST_NAME]]";
import library "types";
import library "impl_in_interface_args";

fn G(x: X) { x.(I(InInterfaceArgs).F)(); }

// --- impl_in_class_args.carbon

library "[[@TEST_NAME]]";
import library "types";

class InClassArgs {}

impl C(InClassArgs) as I(X) { fn F() {} }

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
// CHECK:STDOUT:   %I.type.dac1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.bc4b: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.bc4b = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type.bc4b, %F.type [symbolic]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.dac1 = interface_decl @I [template = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc5_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc5_9.1, runtime_param<invalid> [symbolic = %T.patt.loc5_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc5_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc5_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.2)> [symbolic = %I.type (constants.%I.type.bc4b)]
// CHECK:STDOUT:   %Self.2: %I.type.bc4b = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.bc4b), @I.%F.type (%F.type) [symbolic = %F.assoc_type (constants.%F.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_31.2: @I.%F.assoc_type (%F.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc4_31.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.bc4b) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type) = fn_decl @F [symbolic = @I.%F (constants.%F)] {} {}
// CHECK:STDOUT:     %assoc0.loc4_31.1: @I.%F.assoc_type (%F.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc4_31.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc4_31.1
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
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%T.loc4_13.1: type, @I.%Self.1: @I.%I.type (%I.type.bc4b)) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%Self) {}
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
// CHECK:STDOUT:   %InInterfaceArgs: type = class_type @InInterfaceArgs [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %I.type.dac1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.9d16: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.9d16 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.8ee5: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.1297: %F.type.8ee5 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.3511: type = assoc_entity_type %I.type.9d16, %F.type.8ee5 [symbolic]
// CHECK:STDOUT:   %assoc0.fd0b: %F.assoc_type.3511 = assoc_entity element0, imports.%import_ref.8d80 [symbolic]
// CHECK:STDOUT:   %I.type.da1a: type = facet_type <@I, @I(%InInterfaceArgs)> [template]
// CHECK:STDOUT:   %F.type.5dce: type = fn_type @F.1, @I(%InInterfaceArgs) [template]
// CHECK:STDOUT:   %F.55d9: %F.type.5dce = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.8008: type = assoc_entity_type %I.type.da1a, %F.type.5dce [template]
// CHECK:STDOUT:   %assoc0.d7d0: %F.assoc_type.8008 = assoc_entity element0, imports.%import_ref.8d80 [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.2a03: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.eb4f: %F.type.2a03 = struct_value () [template]
// CHECK:STDOUT:   %I.facet: %I.type.9d16 = facet_value %X, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.8480: %I.type.dac1 = import_ref Main//types, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.2081 = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %import_ref.58fd: type = import_ref Main//types, X, loaded [template = constants.%X]
// CHECK:STDOUT:   %import_ref.8f24: <witness> = import_ref Main//types, loc7_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.acf4 = import_ref Main//types, inst54 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9c13 = import_ref Main//types, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.3552 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT:   %import_ref.938d: @I.%F.type (%F.type.8ee5) = import_ref Main//types, F, loaded [symbolic = @I.%F (constants.%F.1297)]
// CHECK:STDOUT:   %import_ref.8d80 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.8480
// CHECK:STDOUT:     .C = imports.%import_ref.2081
// CHECK:STDOUT:     .X = imports.%import_ref.58fd
// CHECK:STDOUT:     .InInterfaceArgs = %InInterfaceArgs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %InInterfaceArgs.decl: type = class_decl @InInterfaceArgs [template = constants.%InInterfaceArgs] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%import_ref.58fd [template = constants.%X]
// CHECK:STDOUT:     %I.ref: %I.type.dac1 = name_ref I, imports.%import_ref.8480 [template = constants.%I.generic]
// CHECK:STDOUT:     %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, file.%InInterfaceArgs.decl [template = constants.%InInterfaceArgs]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [template = constants.%I.type.da1a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.9d16)]
// CHECK:STDOUT:   %Self: %I.type.9d16 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.8ee5)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.8ee5) = struct_value () [symbolic = %F (constants.%F.1297)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.9d16), @I.%F.type (%F.type.8ee5) [symbolic = %F.assoc_type (constants.%F.assoc_type.3511)]
// CHECK:STDOUT:   %assoc0: @I.%F.assoc_type (%F.assoc_type.3511) = assoc_entity element0, imports.%import_ref.8d80 [symbolic = %assoc0 (constants.%assoc0.fd0b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9c13
// CHECK:STDOUT:     .F = imports.%import_ref.3552
// CHECK:STDOUT:     witness = (imports.%import_ref.938d)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %X.ref as %I.type {
// CHECK:STDOUT:   %F.decl: %F.type.2a03 = fn_decl @F.2 [template = constants.%F.eb4f] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InInterfaceArgs {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%InInterfaceArgs
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.acf4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type, constants.%Self: %I.type.9d16) [from "types.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
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
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%InInterfaceArgs) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %T.patt => constants.%InInterfaceArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.da1a
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.5dce
// CHECK:STDOUT:   %F => constants.%F.55d9
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.8008
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d7d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%InInterfaceArgs, constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_impl_in_interface_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %I.type.dac1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac1 = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.9d16: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.9d16 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.8ee5: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.1297: %F.type.8ee5 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.3511: type = assoc_entity_type %I.type.9d16, %F.type.8ee5 [symbolic]
// CHECK:STDOUT:   %assoc0.fd0b7a.1: %F.assoc_type.3511 = assoc_entity element0, imports.%import_ref.8d80ba.1 [symbolic]
// CHECK:STDOUT:   %InInterfaceArgs: type = class_type @InInterfaceArgs [template]
// CHECK:STDOUT:   %I.type.da1a: type = facet_type <@I, @I(%InInterfaceArgs)> [template]
// CHECK:STDOUT:   %F.type.5dce: type = fn_type @F.1, @I(%InInterfaceArgs) [template]
// CHECK:STDOUT:   %F.55d9: %F.type.5dce = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.8008: type = assoc_entity_type %I.type.da1a, %F.type.5dce [template]
// CHECK:STDOUT:   %assoc0.d7d0: %F.assoc_type.8008 = assoc_entity element0, imports.%import_ref.8d80ba.1 [template]
// CHECK:STDOUT:   %assoc0.fd0b7a.2: %F.assoc_type.3511 = assoc_entity element0, imports.%import_ref.8d80ba.2 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.a9d9) [template]
// CHECK:STDOUT:   %F.type.4618: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.4798: %F.type.4618 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.8480: %I.type.dac1 = import_ref Main//types, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.2081 = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %import_ref.58fd: type = import_ref Main//types, X, loaded [template = constants.%X]
// CHECK:STDOUT:   %import_ref.4a40: type = import_ref Main//impl_in_interface_args, InInterfaceArgs, loaded [template = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.acf4 = import_ref Main//types, inst54 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9c13 = import_ref Main//types, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.336d: @I.%F.assoc_type (%F.assoc_type.3511) = import_ref Main//types, loc4_31, loaded [symbolic = @I.%assoc0 (constants.%assoc0.fd0b7a.2)]
// CHECK:STDOUT:   %import_ref.3d25 = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %import_ref.8d80ba.1 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_interface_args, loc5_24, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.bf89 = import_ref Main//impl_in_interface_args, inst18 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9b8b: <witness> = import_ref Main//impl_in_interface_args, loc7_30, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.0ae0: type = import_ref Main//impl_in_interface_args, loc7_6, loaded [template = constants.%X]
// CHECK:STDOUT:   %import_ref.9276: type = import_ref Main//impl_in_interface_args, loc7_28, loaded [template = constants.%I.type.da1a]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.8480
// CHECK:STDOUT:     .C = imports.%import_ref.2081
// CHECK:STDOUT:     .X = imports.%import_ref.58fd
// CHECK:STDOUT:     .InInterfaceArgs = imports.%import_ref.4a40
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %X = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %X = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %X = value_param runtime_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%import_ref.58fd [template = constants.%X]
// CHECK:STDOUT:     %x: %X = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.9d16)]
// CHECK:STDOUT:   %Self: %I.type.9d16 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.8ee5)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.8ee5) = struct_value () [symbolic = %F (constants.%F.1297)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.9d16), @I.%F.type (%F.type.8ee5) [symbolic = %F.assoc_type (constants.%F.assoc_type.3511)]
// CHECK:STDOUT:   %assoc0: @I.%F.assoc_type (%F.assoc_type.3511) = assoc_entity element0, imports.%import_ref.8d80ba.1 [symbolic = %assoc0 (constants.%assoc0.fd0b7a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9c13
// CHECK:STDOUT:     .F = imports.%import_ref.336d
// CHECK:STDOUT:     witness = (imports.%import_ref.3d25)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.0ae0 as imports.%import_ref.9276 [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.9b8b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.acf4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InInterfaceArgs [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.bf89
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %X = name_ref x, %x
// CHECK:STDOUT:   %I.ref: %I.type.dac1 = name_ref I, imports.%import_ref.8480 [template = constants.%I.generic]
// CHECK:STDOUT:   %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, imports.%import_ref.4a40 [template = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [template = constants.%I.type.da1a]
// CHECK:STDOUT:   %.loc6: %F.assoc_type.8008 = specific_constant imports.%import_ref.336d, @I(constants.%InInterfaceArgs) [template = constants.%assoc0.d7d0]
// CHECK:STDOUT:   %F.ref: %F.assoc_type.8008 = name_ref F, %.loc6 [template = constants.%assoc0.d7d0]
// CHECK:STDOUT:   %impl.elem0: %F.type.5dce = impl_witness_access constants.%impl_witness, element0 [template = constants.%F.4798]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type, constants.%Self: %I.type.9d16) [from "types.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() [from "impl_in_interface_args.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%InInterfaceArgs) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %T.patt => constants.%InInterfaceArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.da1a
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.5dce
// CHECK:STDOUT:   %F => constants.%F.55d9
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.8008
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d7d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_in_class_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InClassArgs: type = class_type @InClassArgs [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.5b11: type = class_type @C, @C(%InClassArgs) [template]
// CHECK:STDOUT:   %I.type.dac1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.9d16: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.9d16 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.8ee5: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.1297: %F.type.8ee5 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.3511: type = assoc_entity_type %I.type.9d16, %F.type.8ee5 [symbolic]
// CHECK:STDOUT:   %assoc0.fd0b: %F.assoc_type.3511 = assoc_entity element0, imports.%import_ref.8d80 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %I.type.914c: type = facet_type <@I, @I(%X)> [template]
// CHECK:STDOUT:   %F.type.ea3f: type = fn_type @F.1, @I(%X) [template]
// CHECK:STDOUT:   %F.e0ec: %F.type.ea3f = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.f411: type = assoc_entity_type %I.type.914c, %F.type.ea3f [template]
// CHECK:STDOUT:   %assoc0.7c34: %F.assoc_type.f411 = assoc_entity element0, imports.%import_ref.8d80 [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.73dc: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.e0f9: %F.type.73dc = struct_value () [template]
// CHECK:STDOUT:   %I.facet: %I.type.9d16 = facet_value %C.5b11, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.8480: %I.type.dac1 = import_ref Main//types, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.26e8: %C.type = import_ref Main//types, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.58fd: type = import_ref Main//types, X, loaded [template = constants.%X]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.495d = import_ref Main//types, inst49 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9c13 = import_ref Main//types, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.3552 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT:   %import_ref.938d: @I.%F.type (%F.type.8ee5) = import_ref Main//types, F, loaded [symbolic = @I.%F (constants.%F.1297)]
// CHECK:STDOUT:   %import_ref.8d80 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//types, loc7_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.acf4 = import_ref Main//types, inst54 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.8480
// CHECK:STDOUT:     .C = imports.%import_ref.26e8
// CHECK:STDOUT:     .X = imports.%import_ref.58fd
// CHECK:STDOUT:     .InClassArgs = %InClassArgs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %InClassArgs.decl: type = class_decl @InClassArgs [template = constants.%InClassArgs] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%import_ref.26e8 [template = constants.%C.generic]
// CHECK:STDOUT:     %InClassArgs.ref: type = name_ref InClassArgs, file.%InClassArgs.decl [template = constants.%InClassArgs]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%InClassArgs) [template = constants.%C.5b11]
// CHECK:STDOUT:     %I.ref: %I.type.dac1 = name_ref I, imports.%import_ref.8480 [template = constants.%I.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%import_ref.58fd [template = constants.%X]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%X)> [template = constants.%I.type.914c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.9d16)]
// CHECK:STDOUT:   %Self: %I.type.9d16 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.8ee5)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.8ee5) = struct_value () [symbolic = %F (constants.%F.1297)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.9d16), @I.%F.type (%F.type.8ee5) [symbolic = %F.assoc_type (constants.%F.assoc_type.3511)]
// CHECK:STDOUT:   %assoc0: @I.%F.assoc_type (%F.assoc_type.3511) = assoc_entity element0, imports.%import_ref.8d80 [symbolic = %assoc0 (constants.%assoc0.fd0b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9c13
// CHECK:STDOUT:     .F = imports.%import_ref.3552
// CHECK:STDOUT:     witness = (imports.%import_ref.938d)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %I.type {
// CHECK:STDOUT:   %F.decl: %F.type.73dc = fn_decl @F.2 [template = constants.%F.e0f9] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InClassArgs {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%InClassArgs
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%T: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.495d
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.acf4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type, constants.%Self: %I.type.9d16) [from "types.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %T.patt => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.914c
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.ea3f
// CHECK:STDOUT:   %F => constants.%F.e0ec
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.f411
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7c34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X, constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_impl_in_class_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %InClassArgs: type = class_type @InClassArgs [template]
// CHECK:STDOUT:   %C.5b11: type = class_type @C, @C(%InClassArgs) [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %I.type.dac1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.9d16: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.9d16 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.8ee5: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.1297: %F.type.8ee5 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.3511: type = assoc_entity_type %I.type.9d16, %F.type.8ee5 [symbolic]
// CHECK:STDOUT:   %assoc0.fd0b7a.1: %F.assoc_type.3511 = assoc_entity element0, imports.%import_ref.8d80ba.1 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %I.type.914c: type = facet_type <@I, @I(%X)> [template]
// CHECK:STDOUT:   %F.type.ea3f: type = fn_type @F.1, @I(%X) [template]
// CHECK:STDOUT:   %F.e0ec: %F.type.ea3f = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.f411: type = assoc_entity_type %I.type.914c, %F.type.ea3f [template]
// CHECK:STDOUT:   %assoc0.7c34: %F.assoc_type.f411 = assoc_entity element0, imports.%import_ref.8d80ba.1 [template]
// CHECK:STDOUT:   %assoc0.fd0b7a.2: %F.assoc_type.3511 = assoc_entity element0, imports.%import_ref.8d80ba.2 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.8f2f) [template]
// CHECK:STDOUT:   %F.type.5634: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.6f35: %F.type.5634 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.8480: %I.type.dac1 = import_ref Main//types, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.26e8: %C.type = import_ref Main//types, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.58fd: type = import_ref Main//types, X, loaded [template = constants.%X]
// CHECK:STDOUT:   %import_ref.4d5c: type = import_ref Main//impl_in_class_args, InClassArgs, loaded [template = constants.%InClassArgs]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.495d = import_ref Main//types, inst49 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_class_args, loc5_20, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.6836 = import_ref Main//impl_in_class_args, inst18 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9c13 = import_ref Main//types, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.336d: @I.%F.assoc_type (%F.assoc_type.3511) = import_ref Main//types, loc4_31, loaded [symbolic = @I.%assoc0 (constants.%assoc0.fd0b7a.2)]
// CHECK:STDOUT:   %import_ref.3d25 = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %import_ref.8d80ba.1 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.3: <witness> = import_ref Main//types, loc7_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.acf4 = import_ref Main//types, inst54 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.74e7: <witness> = import_ref Main//impl_in_class_args, loc7_29, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.829a: type = import_ref Main//impl_in_class_args, loc7_19, loaded [template = constants.%C.5b11]
// CHECK:STDOUT:   %import_ref.0226: type = import_ref Main//impl_in_class_args, loc7_27, loaded [template = constants.%I.type.914c]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.8480
// CHECK:STDOUT:     .C = imports.%import_ref.26e8
// CHECK:STDOUT:     .X = imports.%import_ref.58fd
// CHECK:STDOUT:     .InClassArgs = imports.%import_ref.4d5c
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %c.patt: %C.5b11 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C.5b11 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.5b11 = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_22: type = splice_block %C [template = constants.%C.5b11] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, imports.%import_ref.26e8 [template = constants.%C.generic]
// CHECK:STDOUT:       %InClassArgs.ref: type = name_ref InClassArgs, imports.%import_ref.4d5c [template = constants.%InClassArgs]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%InClassArgs) [template = constants.%C.5b11]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.5b11 = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.9d16)]
// CHECK:STDOUT:   %Self: %I.type.9d16 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.8ee5)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.8ee5) = struct_value () [symbolic = %F (constants.%F.1297)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.9d16), @I.%F.type (%F.type.8ee5) [symbolic = %F.assoc_type (constants.%F.assoc_type.3511)]
// CHECK:STDOUT:   %assoc0: @I.%F.assoc_type (%F.assoc_type.3511) = assoc_entity element0, imports.%import_ref.8d80ba.1 [symbolic = %assoc0 (constants.%assoc0.fd0b7a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9c13
// CHECK:STDOUT:     .F = imports.%import_ref.336d
// CHECK:STDOUT:     witness = (imports.%import_ref.3d25)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.829a as imports.%import_ref.0226 [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.74e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%T: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.495d
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InClassArgs [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.6836
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.acf4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param_patt: %C.5b11) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.5b11 = name_ref c, %c
// CHECK:STDOUT:   %I.ref: %I.type.dac1 = name_ref I, imports.%import_ref.8480 [template = constants.%I.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, imports.%import_ref.58fd [template = constants.%X]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%X)> [template = constants.%I.type.914c]
// CHECK:STDOUT:   %.loc6_34: %F.assoc_type.f411 = specific_constant imports.%import_ref.336d, @I(constants.%X) [template = constants.%assoc0.7c34]
// CHECK:STDOUT:   %F.ref: %F.assoc_type.f411 = name_ref F, %.loc6_34 [template = constants.%assoc0.7c34]
// CHECK:STDOUT:   %impl.elem0: %F.type.ea3f = impl_witness_access constants.%impl_witness, element0 [template = constants.%F.6f35]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type, constants.%Self: %I.type.9d16) [from "types.carbon"] {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() [from "impl_in_class_args.carbon"];
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
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %T.patt => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.914c
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.ea3f
// CHECK:STDOUT:   %F => constants.%F.e0ec
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.f411
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7c34
// CHECK:STDOUT: }
// CHECK:STDOUT:

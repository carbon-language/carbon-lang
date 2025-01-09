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
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [template]
// CHECK:STDOUT:   %I.type.bc4: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.bc4 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type.bc4, %F.type [symbolic]
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
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [template = constants.%I.generic] {
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
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.2)> [symbolic = %I.type (constants.%I.type.bc4)]
// CHECK:STDOUT:   %Self.2: %I.type.bc4 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.bc4), @I.%F.type (%F.type) [symbolic = %F.assoc_type (constants.%F.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc4_31.2: @I.%F.assoc_type (%F.assoc_type) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc4_31.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.bc4) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
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
// CHECK:STDOUT: generic fn @F(@I.%T.loc4_13.1: type, @I.%Self.1: @I.%I.type (%I.type.bc4)) {
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
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.9d1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.9d1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.8ee: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.129: %F.type.8ee = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.351: type = assoc_entity_type %I.type.9d1, %F.type.8ee [symbolic]
// CHECK:STDOUT:   %assoc0.fd0: %F.assoc_type.351 = assoc_entity element0, imports.%import_ref.8d8 [symbolic]
// CHECK:STDOUT:   %I.type.da1: type = facet_type <@I, @I(%InInterfaceArgs)> [template]
// CHECK:STDOUT:   %F.type.5dc: type = fn_type @F.1, @I(%InInterfaceArgs) [template]
// CHECK:STDOUT:   %F.55d: %F.type.5dc = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.800: type = assoc_entity_type %I.type.da1, %F.type.5dc [template]
// CHECK:STDOUT:   %assoc0.d7d: %F.assoc_type.800 = assoc_entity element0, imports.%import_ref.8d8 [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.2a0: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.eb4: %F.type.2a0 = struct_value () [template]
// CHECK:STDOUT:   %I.facet: %I.type.9d1 = facet_value %X, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.848: %I.type.dac = import_ref Main//types, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.208 = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %import_ref.58f: type = import_ref Main//types, X, loaded [template = constants.%X]
// CHECK:STDOUT:   %import_ref.8f2: <witness> = import_ref Main//types, loc7_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.acf = import_ref Main//types, inst54 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9c1 = import_ref Main//types, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.355 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT:   %import_ref.938: @I.%F.type (%F.type.8ee) = import_ref Main//types, F, loaded [symbolic = @I.%F (constants.%F.129)]
// CHECK:STDOUT:   %import_ref.8d8 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.848
// CHECK:STDOUT:     .C = imports.%import_ref.208
// CHECK:STDOUT:     .X = imports.%import_ref.58f
// CHECK:STDOUT:     .InInterfaceArgs = %InInterfaceArgs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %InInterfaceArgs.decl: type = class_decl @InInterfaceArgs [template = constants.%InInterfaceArgs] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%import_ref.58f [template = constants.%X]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%import_ref.848 [template = constants.%I.generic]
// CHECK:STDOUT:     %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, file.%InInterfaceArgs.decl [template = constants.%InInterfaceArgs]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [template = constants.%I.type.da1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.9d1)]
// CHECK:STDOUT:   %Self: %I.type.9d1 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.8ee)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.8ee) = struct_value () [symbolic = %F (constants.%F.129)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.9d1), @I.%F.type (%F.type.8ee) [symbolic = %F.assoc_type (constants.%F.assoc_type.351)]
// CHECK:STDOUT:   %assoc0: @I.%F.assoc_type (%F.assoc_type.351) = assoc_entity element0, imports.%import_ref.8d8 [symbolic = %assoc0 (constants.%assoc0.fd0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9c1
// CHECK:STDOUT:     .F = imports.%import_ref.355
// CHECK:STDOUT:     witness = (imports.%import_ref.938)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %X.ref as %I.type {
// CHECK:STDOUT:   %F.decl: %F.type.2a0 = fn_decl @F.2 [template = constants.%F.eb4] {} {}
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
// CHECK:STDOUT:   .Self = imports.%import_ref.acf
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type, constants.%Self: %I.type.9d1) [from "types.carbon"] {
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
// CHECK:STDOUT:   %I.type => constants.%I.type.da1
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.5dc
// CHECK:STDOUT:   %F => constants.%F.55d
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.800
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d7d
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
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.9d1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.9d1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.8ee: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.129: %F.type.8ee = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.351: type = assoc_entity_type %I.type.9d1, %F.type.8ee [symbolic]
// CHECK:STDOUT:   %assoc0.fd0b7a.1: %F.assoc_type.351 = assoc_entity element0, imports.%import_ref.8d80ba.1 [symbolic]
// CHECK:STDOUT:   %InInterfaceArgs: type = class_type @InInterfaceArgs [template]
// CHECK:STDOUT:   %I.type.da1: type = facet_type <@I, @I(%InInterfaceArgs)> [template]
// CHECK:STDOUT:   %F.type.5dc: type = fn_type @F.1, @I(%InInterfaceArgs) [template]
// CHECK:STDOUT:   %F.55d: %F.type.5dc = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.800: type = assoc_entity_type %I.type.da1, %F.type.5dc [template]
// CHECK:STDOUT:   %assoc0.d7d: %F.assoc_type.800 = assoc_entity element0, imports.%import_ref.8d80ba.1 [template]
// CHECK:STDOUT:   %assoc0.fd0b7a.2: %F.assoc_type.351 = assoc_entity element0, imports.%import_ref.8d80ba.2 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.a9d) [template]
// CHECK:STDOUT:   %F.type.461: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.479: %F.type.461 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.848: %I.type.dac = import_ref Main//types, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.208 = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %import_ref.58f: type = import_ref Main//types, X, loaded [template = constants.%X]
// CHECK:STDOUT:   %import_ref.4a4: type = import_ref Main//impl_in_interface_args, InInterfaceArgs, loaded [template = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.acf = import_ref Main//types, inst54 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9c1 = import_ref Main//types, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.336: @I.%F.assoc_type (%F.assoc_type.351) = import_ref Main//types, loc4_31, loaded [symbolic = @I.%assoc0 (constants.%assoc0.fd0b7a.2)]
// CHECK:STDOUT:   %import_ref.3d2 = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %import_ref.8d80ba.1 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_interface_args, loc5_24, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.bf8 = import_ref Main//impl_in_interface_args, inst18 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9b8: <witness> = import_ref Main//impl_in_interface_args, loc7_30, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.0ae: type = import_ref Main//impl_in_interface_args, loc7_6, loaded [template = constants.%X]
// CHECK:STDOUT:   %import_ref.927: type = import_ref Main//impl_in_interface_args, loc7_28, loaded [template = constants.%I.type.da1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.848
// CHECK:STDOUT:     .C = imports.%import_ref.208
// CHECK:STDOUT:     .X = imports.%import_ref.58f
// CHECK:STDOUT:     .InInterfaceArgs = imports.%import_ref.4a4
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %x.patt: %X = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %X = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %X = value_param runtime_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%import_ref.58f [template = constants.%X]
// CHECK:STDOUT:     %x: %X = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.9d1)]
// CHECK:STDOUT:   %Self: %I.type.9d1 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.8ee)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.8ee) = struct_value () [symbolic = %F (constants.%F.129)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.9d1), @I.%F.type (%F.type.8ee) [symbolic = %F.assoc_type (constants.%F.assoc_type.351)]
// CHECK:STDOUT:   %assoc0: @I.%F.assoc_type (%F.assoc_type.351) = assoc_entity element0, imports.%import_ref.8d80ba.1 [symbolic = %assoc0 (constants.%assoc0.fd0b7a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9c1
// CHECK:STDOUT:     .F = imports.%import_ref.336
// CHECK:STDOUT:     witness = (imports.%import_ref.3d2)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.0ae as imports.%import_ref.927 [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.9b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.acf
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InInterfaceArgs [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.bf8
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param_patt: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %X = name_ref x, %x
// CHECK:STDOUT:   %I.ref: %I.type.dac = name_ref I, imports.%import_ref.848 [template = constants.%I.generic]
// CHECK:STDOUT:   %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, imports.%import_ref.4a4 [template = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [template = constants.%I.type.da1]
// CHECK:STDOUT:   %.loc6: %F.assoc_type.800 = specific_constant imports.%import_ref.336, @I(constants.%InInterfaceArgs) [template = constants.%assoc0.d7d]
// CHECK:STDOUT:   %F.ref: %F.assoc_type.800 = name_ref F, %.loc6 [template = constants.%assoc0.d7d]
// CHECK:STDOUT:   %impl.elem0: %F.type.5dc = impl_witness_access constants.%impl_witness, element0 [template = constants.%F.479]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type, constants.%Self: %I.type.9d1) [from "types.carbon"] {
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
// CHECK:STDOUT:   %I.type => constants.%I.type.da1
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.5dc
// CHECK:STDOUT:   %F => constants.%F.55d
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.800
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d7d
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
// CHECK:STDOUT:   %C.5b1: type = class_type @C, @C(%InClassArgs) [template]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [template]
// CHECK:STDOUT:   %I.type.9d1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.9d1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.8ee: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.129: %F.type.8ee = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.351: type = assoc_entity_type %I.type.9d1, %F.type.8ee [symbolic]
// CHECK:STDOUT:   %assoc0.fd0: %F.assoc_type.351 = assoc_entity element0, imports.%import_ref.8d8 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %I.type.914: type = facet_type <@I, @I(%X)> [template]
// CHECK:STDOUT:   %F.type.ea3: type = fn_type @F.1, @I(%X) [template]
// CHECK:STDOUT:   %F.e0e: %F.type.ea3 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.f41: type = assoc_entity_type %I.type.914, %F.type.ea3 [template]
// CHECK:STDOUT:   %assoc0.7c3: %F.assoc_type.f41 = assoc_entity element0, imports.%import_ref.8d8 [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template]
// CHECK:STDOUT:   %F.type.73d: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.e0f: %F.type.73d = struct_value () [template]
// CHECK:STDOUT:   %I.facet: %I.type.9d1 = facet_value %C.5b1, %impl_witness [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.848: %I.type.dac = import_ref Main//types, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.26e: %C.type = import_ref Main//types, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.58f: type = import_ref Main//types, X, loaded [template = constants.%X]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.495 = import_ref Main//types, inst49 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9c1 = import_ref Main//types, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.355 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT:   %import_ref.938: @I.%F.type (%F.type.8ee) = import_ref Main//types, F, loaded [symbolic = @I.%F (constants.%F.129)]
// CHECK:STDOUT:   %import_ref.8d8 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//types, loc7_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.acf = import_ref Main//types, inst54 [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.848
// CHECK:STDOUT:     .C = imports.%import_ref.26e
// CHECK:STDOUT:     .X = imports.%import_ref.58f
// CHECK:STDOUT:     .InClassArgs = %InClassArgs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %InClassArgs.decl: type = class_decl @InClassArgs [template = constants.%InClassArgs] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, imports.%import_ref.26e [template = constants.%C.generic]
// CHECK:STDOUT:     %InClassArgs.ref: type = name_ref InClassArgs, file.%InClassArgs.decl [template = constants.%InClassArgs]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%InClassArgs) [template = constants.%C.5b1]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, imports.%import_ref.848 [template = constants.%I.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%import_ref.58f [template = constants.%X]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%X)> [template = constants.%I.type.914]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.9d1)]
// CHECK:STDOUT:   %Self: %I.type.9d1 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.8ee)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.8ee) = struct_value () [symbolic = %F (constants.%F.129)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.9d1), @I.%F.type (%F.type.8ee) [symbolic = %F.assoc_type (constants.%F.assoc_type.351)]
// CHECK:STDOUT:   %assoc0: @I.%F.assoc_type (%F.assoc_type.351) = assoc_entity element0, imports.%import_ref.8d8 [symbolic = %assoc0 (constants.%assoc0.fd0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9c1
// CHECK:STDOUT:     .F = imports.%import_ref.355
// CHECK:STDOUT:     witness = (imports.%import_ref.938)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C as %I.type {
// CHECK:STDOUT:   %F.decl: %F.type.73d = fn_decl @F.2 [template = constants.%F.e0f] {} {}
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
// CHECK:STDOUT:     .Self = imports.%import_ref.495
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.acf
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type, constants.%Self: %I.type.9d1) [from "types.carbon"] {
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
// CHECK:STDOUT:   %I.type => constants.%I.type.914
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.ea3
// CHECK:STDOUT:   %F => constants.%F.e0e
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.f41
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7c3
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
// CHECK:STDOUT:   %C.5b1: type = class_type @C, @C(%InClassArgs) [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [template]
// CHECK:STDOUT:   %I.type.9d1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.9d1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.8ee: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.129: %F.type.8ee = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.351: type = assoc_entity_type %I.type.9d1, %F.type.8ee [symbolic]
// CHECK:STDOUT:   %assoc0.fd0b7a.1: %F.assoc_type.351 = assoc_entity element0, imports.%import_ref.8d80ba.1 [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %I.type.914: type = facet_type <@I, @I(%X)> [template]
// CHECK:STDOUT:   %F.type.ea3: type = fn_type @F.1, @I(%X) [template]
// CHECK:STDOUT:   %F.e0e: %F.type.ea3 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.f41: type = assoc_entity_type %I.type.914, %F.type.ea3 [template]
// CHECK:STDOUT:   %assoc0.7c3: %F.assoc_type.f41 = assoc_entity element0, imports.%import_ref.8d80ba.1 [template]
// CHECK:STDOUT:   %assoc0.fd0b7a.2: %F.assoc_type.351 = assoc_entity element0, imports.%import_ref.8d80ba.2 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (imports.%import_ref.8f2f) [template]
// CHECK:STDOUT:   %F.type.563: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.6f3: %F.type.563 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.848: %I.type.dac = import_ref Main//types, I, loaded [template = constants.%I.generic]
// CHECK:STDOUT:   %import_ref.26e: %C.type = import_ref Main//types, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.58f: type = import_ref Main//types, X, loaded [template = constants.%X]
// CHECK:STDOUT:   %import_ref.4d5: type = import_ref Main//impl_in_class_args, InClassArgs, loaded [template = constants.%InClassArgs]
// CHECK:STDOUT:   %import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.495 = import_ref Main//types, inst49 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_class_args, loc5_20, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.683 = import_ref Main//impl_in_class_args, inst18 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.9c1 = import_ref Main//types, inst26 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.336: @I.%F.assoc_type (%F.assoc_type.351) = import_ref Main//types, loc4_31, loaded [symbolic = @I.%assoc0 (constants.%assoc0.fd0b7a.2)]
// CHECK:STDOUT:   %import_ref.3d2 = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %import_ref.8d80ba.1 = import_ref Main//types, loc4_31, unloaded
// CHECK:STDOUT:   %import_ref.8f24d3.3: <witness> = import_ref Main//types, loc7_10, loaded [template = constants.%complete_type]
// CHECK:STDOUT:   %import_ref.acf = import_ref Main//types, inst54 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.74e: <witness> = import_ref Main//impl_in_class_args, loc7_29, loaded [template = constants.%impl_witness]
// CHECK:STDOUT:   %import_ref.829: type = import_ref Main//impl_in_class_args, loc7_19, loaded [template = constants.%C.5b1]
// CHECK:STDOUT:   %import_ref.022: type = import_ref Main//impl_in_class_args, loc7_27, loaded [template = constants.%I.type.914]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.848
// CHECK:STDOUT:     .C = imports.%import_ref.26e
// CHECK:STDOUT:     .X = imports.%import_ref.58f
// CHECK:STDOUT:     .InClassArgs = imports.%import_ref.4d5
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %c.patt: %C.5b1 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C.5b1 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.5b1 = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_22: type = splice_block %C [template = constants.%C.5b1] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, imports.%import_ref.26e [template = constants.%C.generic]
// CHECK:STDOUT:       %InClassArgs.ref: type = name_ref InClassArgs, imports.%import_ref.4d5 [template = constants.%InClassArgs]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%InClassArgs) [template = constants.%C.5b1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.5b1 = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(constants.%T: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.9d1)]
// CHECK:STDOUT:   %Self: %I.type.9d1 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T) [symbolic = %F.type (constants.%F.type.8ee)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.8ee) = struct_value () [symbolic = %F (constants.%F.129)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.9d1), @I.%F.type (%F.type.8ee) [symbolic = %F.assoc_type (constants.%F.assoc_type.351)]
// CHECK:STDOUT:   %assoc0: @I.%F.assoc_type (%F.assoc_type.351) = assoc_entity element0, imports.%import_ref.8d80ba.1 [symbolic = %assoc0 (constants.%assoc0.fd0b7a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.9c1
// CHECK:STDOUT:     .F = imports.%import_ref.336
// CHECK:STDOUT:     witness = (imports.%import_ref.3d2)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: imports.%import_ref.829 as imports.%import_ref.022 [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.74e
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
// CHECK:STDOUT:     .Self = imports.%import_ref.495
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24d3.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InClassArgs [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.683
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.acf
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.8f24d3.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param_patt: %C.5b1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.5b1 = name_ref c, %c
// CHECK:STDOUT:   %I.ref: %I.type.dac = name_ref I, imports.%import_ref.848 [template = constants.%I.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, imports.%import_ref.58f [template = constants.%X]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%X)> [template = constants.%I.type.914]
// CHECK:STDOUT:   %.loc6_34: %F.assoc_type.f41 = specific_constant imports.%import_ref.336, @I(constants.%X) [template = constants.%assoc0.7c3]
// CHECK:STDOUT:   %F.ref: %F.assoc_type.f41 = name_ref F, %.loc6_34 [template = constants.%assoc0.7c3]
// CHECK:STDOUT:   %impl.elem0: %F.type.ea3 = impl_witness_access constants.%impl_witness, element0 [template = constants.%F.6f3]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %impl.elem0()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(constants.%T: type, constants.%Self: %I.type.9d1) [from "types.carbon"] {
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
// CHECK:STDOUT:   %I.type => constants.%I.type.914
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.ea3
// CHECK:STDOUT:   %F => constants.%F.e0e
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.f41
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7c3
// CHECK:STDOUT: }
// CHECK:STDOUT:

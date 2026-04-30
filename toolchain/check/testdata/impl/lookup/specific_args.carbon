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

impl X as I(InInterfaceArgs) { fn F[unused self: Self]() {} }

// --- use_impl_in_interface_args.carbon

library "[[@TEST_NAME]]";
import library "types";
import library "impl_in_interface_args";

fn G(x: X) { x.(I(InInterfaceArgs).F)(); }

// --- impl_in_class_args.carbon

library "[[@TEST_NAME]]";
import library "types";

class InClassArgs {}

impl C(InClassArgs) as I(X) { fn F[unused self: Self]() {} }

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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.f2f: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F: %I.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [symbolic]
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
// CHECK:STDOUT:   %I.decl: %I.type.609 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_13.1: type = splice_block %.loc5_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_14.1)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self.loc4_23.2: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_23.1: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F)] {
// CHECK:STDOUT:       %self.param_patt: @I.WithSelf.F.%pattern_type (%pattern_type.f2f) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @I.WithSelf.F.%pattern_type (%pattern_type.f2f) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @I.WithSelf.F.%Self.as_type.loc4_36.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_36.1: type = splice_block %.loc4_36.3 [symbolic = %Self.as_type.loc4_36.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc4_36.2: @I.WithSelf.F.%I.type (%I.type.1ab) = specific_constant @I.%Self.loc4_23.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @I.WithSelf.F.%I.type (%I.type.1ab) = name_ref Self, %.loc4_36.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc4_36.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_36.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc4_36.3: type = converted %Self.ref, %Self.as_type.loc4_36.2 [symbolic = %Self.as_type.loc4_36.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @I.WithSelf.F.%Self.as_type.loc4_36.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc4_43.1: @I.WithSelf.%I.assoc_type (%I.assoc_type) = assoc_entity element0, %I.WithSelf.F.decl [symbolic = %assoc0.loc4_43.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_23.1
// CHECK:STDOUT:     .F = @I.WithSelf.%assoc0.loc4_43.1
// CHECK:STDOUT:     witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc5_10.2: type) {
// CHECK:STDOUT:   %T.loc5_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_10.1 (constants.%T)]
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
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%T.loc4_14.2: type, @I.%Self.loc4_23.1: @I.%I.type (%I.type.1ab)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_36.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_36.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc4_36.1 [symbolic = %pattern_type (constants.%pattern_type.f2f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.F.%Self.as_type.loc4_36.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_36.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc5_10.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_in_interface_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %InInterfaceArgs: type = class_type @InInterfaceArgs [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c1b: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.efc: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self.c1b) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.213: %I.WithSelf.F.type.efc = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.c1b [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %I.type.39c: type = facet_type <@I, @I(%InInterfaceArgs)> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @X.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.ccd: %I.type.39c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.8da: type = fn_type @I.WithSelf.F, @I.WithSelf(%InInterfaceArgs, %Self.c1b) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.1cf: %I.WithSelf.F.type.8da = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.6ad: type = assoc_entity_type @I, @I(%InInterfaceArgs) [concrete]
// CHECK:STDOUT:   %assoc0.c3e: %I.assoc_type.6ad = assoc_entity element0, imports.%Main.import_ref.a9d [concrete]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F: %X.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.39c = facet_value %X, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.705: type = fn_type @I.WithSelf.F, @I.WithSelf(%InInterfaceArgs, %I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.f79: %I.WithSelf.F.type.705 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.609 = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.8d2 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.108 = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.F: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.efc) = import_ref Main//types, F, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.213)]
// CHECK:STDOUT:   %Main.import_ref.a9d = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.45ead6.2: @I.%I.type (%I.type.1ab) = import_ref Main//types, loc4_23, loaded [symbolic = @I.%Self (constants.%Self.c1b)]
// CHECK:STDOUT:   %Main.import_ref.3fa = import_ref Main//types, loc4_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
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
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, file.%InInterfaceArgs.decl [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [concrete = constants.%I.type.39c]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.3: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3fa
// CHECK:STDOUT:     .F = imports.%Main.import_ref.108
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.I.impl: %X.ref as %I.type {
// CHECK:STDOUT:   %X.as.I.impl.F.decl: %X.as.I.impl.F.type = fn_decl @X.as.I.impl.F [concrete = constants.%X.as.I.impl.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.05f = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.05f = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %X = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @X.as.I.impl.%X.ref [concrete = constants.%X]
// CHECK:STDOUT:     %self: %X = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%X.as.I.impl.F.decl), @X.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %X.as.I.impl.F.decl
// CHECK:STDOUT:   witness = %I.impl_witness
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.45ead6.2: @I.%I.type (%I.type.1ab)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.831)]
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.c1b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self.c1b) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.831
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%InInterfaceArgs) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.39c
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%InInterfaceArgs, constants.%Self.c1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.39c
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.8da
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.1cf
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.6ad
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c3e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%InInterfaceArgs, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.39c
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.705
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.f79
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.6ad
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c3e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%InInterfaceArgs, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.39c
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.05f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_impl_in_interface_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c1b: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.76c: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.efc: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self.c1b) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.213: %I.WithSelf.F.type.efc = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.c1b [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %InInterfaceArgs: type = class_type @InInterfaceArgs [concrete]
// CHECK:STDOUT:   %I.type.39c: type = facet_type <@I, @I(%InInterfaceArgs)> [concrete]
// CHECK:STDOUT:   %Self.ccd: %I.type.39c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.8da: type = fn_type @I.WithSelf.F, @I.WithSelf(%InInterfaceArgs, %Self.c1b) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.1cf: %I.WithSelf.F.type.8da = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.6ad: type = assoc_entity_type @I, @I(%InInterfaceArgs) [concrete]
// CHECK:STDOUT:   %assoc0.8de: %I.assoc_type.6ad = assoc_entity element0, imports.%Main.import_ref.e65 [concrete]
// CHECK:STDOUT:   %assoc0.d22: %I.assoc_type.76c = assoc_entity element0, imports.%Main.import_ref.a9d [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.39c = facet_value %X, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.25e: type = fn_type @I.WithSelf.F, @I.WithSelf(%InInterfaceArgs, %I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.511: %I.WithSelf.F.type.25e = struct_value () [concrete]
// CHECK:STDOUT:   %.9ab: type = fn_type_with_self_type %I.WithSelf.F.type.25e, %I.facet [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F: %X.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.609 = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.InInterfaceArgs: type = import_ref Main//impl_in_interface_args, InInterfaceArgs, loaded [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.8d2 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.480: @I.WithSelf.%I.assoc_type (%I.assoc_type.76c) = import_ref Main//types, loc4_43, loaded [symbolic = @I.WithSelf.%assoc0 (constants.%assoc0.d22)]
// CHECK:STDOUT:   %Main.F = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.e65: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.efc) = import_ref Main//types, loc4_43, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.213)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.45ead6.2: @I.%I.type (%I.type.1ab) = import_ref Main//types, loc4_23, loaded [symbolic = @I.%Self (constants.%Self.c1b)]
// CHECK:STDOUT:   %Main.import_ref.3fa = import_ref Main//types, loc4_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_interface_args, loc5_24, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ec3 = import_ref Main//impl_in_interface_args, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.a9d = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.d1b: <witness> = import_ref Main//impl_in_interface_args, loc7_30, loaded [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.044: type = import_ref Main//impl_in_interface_args, loc7_6, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.acb: type = import_ref Main//impl_in_interface_args, loc7_28, loaded [concrete = constants.%I.type.39c]
// CHECK:STDOUT:   %Main.import_ref.95d: %X.as.I.impl.F.type = import_ref Main//impl_in_interface_args, loc7_58, loaded [concrete = constants.%X.as.I.impl.F]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.95d), @X.as.I.impl [concrete]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type.05f = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.05f = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:     %x: %X = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.3: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3fa
// CHECK:STDOUT:     .F = imports.%Main.import_ref.480
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.I.impl: imports.%Main.import_ref.044 as imports.%Main.import_ref.acb [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.d1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InInterfaceArgs [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ec3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %X = name_ref x, %x
// CHECK:STDOUT:   %I.ref: %I.type.609 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:   %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, imports.%Main.InInterfaceArgs [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [concrete = constants.%I.type.39c]
// CHECK:STDOUT:   %.loc6: %I.assoc_type.6ad = specific_constant imports.%Main.import_ref.480, @I.WithSelf(constants.%InInterfaceArgs, constants.%Self.c1b) [concrete = constants.%assoc0.8de]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.6ad = name_ref F, %.loc6 [concrete = constants.%assoc0.8de]
// CHECK:STDOUT:   %impl.elem0: %.9ab = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%X.as.I.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %X.as.I.impl.F.call: init %empty_tuple.type = call %bound_method(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.45ead6.2: @I.%I.type (%I.type.1ab)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.831)]
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.c1b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self.c1b) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.831
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%InInterfaceArgs) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.39c
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%InInterfaceArgs, constants.%Self.c1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.39c
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.8da
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.1cf
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.6ad
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%InInterfaceArgs, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.39c
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.25e
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.511
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.6ad
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8de
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
// CHECK:STDOUT:   %C.83f: type = class_type @C, @C(%InClassArgs) [concrete]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c1b: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.efc: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self.c1b) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.213: %I.WithSelf.F.type.efc = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.c1b [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %I.type.03e: type = facet_type <@I, @I(%X)> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.9c0: %I.type.03e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.bfe: type = fn_type @I.WithSelf.F, @I.WithSelf(%X, %Self.c1b) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.0d9: %I.WithSelf.F.type.bfe = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.46b: type = assoc_entity_type @I, @I(%X) [concrete]
// CHECK:STDOUT:   %assoc0.e51: %I.assoc_type.46b = assoc_entity element0, imports.%Main.import_ref.a9d [concrete]
// CHECK:STDOUT:   %pattern_type.65d: type = pattern_type %C.83f [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.03e = facet_value %C.83f, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.0fc: type = fn_type @I.WithSelf.F, @I.WithSelf(%X, %I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.b98: %I.WithSelf.F.type.0fc = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.609 = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//types, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.820 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//types, loc5_10, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.108 = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.F: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.efc) = import_ref Main//types, F, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.213)]
// CHECK:STDOUT:   %Main.import_ref.a9d = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.45ead6.2: @I.%I.type (%I.type.1ab) = import_ref Main//types, loc4_23, loaded [symbolic = @I.%Self (constants.%Self.c1b)]
// CHECK:STDOUT:   %Main.import_ref.3fa = import_ref Main//types, loc4_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.8d2 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
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
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%InClassArgs) [concrete = constants.%C.83f]
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%X)> [concrete = constants.%I.type.03e]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.4: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3fa
// CHECK:STDOUT:     .F = imports.%Main.import_ref.108
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C as %I.type {
// CHECK:STDOUT:   %C.as.I.impl.F.decl: %C.as.I.impl.F.type = fn_decl @C.as.I.impl.F [concrete = constants.%C.as.I.impl.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.65d = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.65d = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C.83f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.I.impl.%C [concrete = constants.%C.83f]
// CHECK:STDOUT:     %self: %C.83f = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%C.as.I.impl.F.decl), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:   witness = %I.impl_witness
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
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.b3bc94.1: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.820
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.45ead6.2: @I.%I.type (%I.type.1ab)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.831)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F(%self.param: %C.83f) {
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.c1b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self.c1b) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.831
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.03e
// CHECK:STDOUT:   %Self => constants.%Self.9c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%X, constants.%Self.c1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.03e
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.bfe
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.0d9
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.46b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%X, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.03e
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.0fc
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.b98
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.46b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%X, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.03e
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C.83f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.65d
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
// CHECK:STDOUT:   %C.83f: type = class_type @C, @C(%InClassArgs) [concrete]
// CHECK:STDOUT:   %pattern_type.65d: type = pattern_type %C.83f [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c1b: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.76c: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.efc: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self.c1b) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.213: %I.WithSelf.F.type.efc = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.c1b [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %I.type.03e: type = facet_type <@I, @I(%X)> [concrete]
// CHECK:STDOUT:   %Self.9c0: %I.type.03e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.bfe: type = fn_type @I.WithSelf.F, @I.WithSelf(%X, %Self.c1b) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.0d9: %I.WithSelf.F.type.bfe = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.46b: type = assoc_entity_type @I, @I(%X) [concrete]
// CHECK:STDOUT:   %assoc0.683: %I.assoc_type.46b = assoc_entity element0, imports.%Main.import_ref.e65 [concrete]
// CHECK:STDOUT:   %assoc0.d22: %I.assoc_type.76c = assoc_entity element0, imports.%Main.import_ref.a9d [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.03e = facet_value %C.83f, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.2d3: type = fn_type @I.WithSelf.F, @I.WithSelf(%X, %I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.841: %I.WithSelf.F.type.2d3 = struct_value () [concrete]
// CHECK:STDOUT:   %.ac4: type = fn_type_with_self_type %I.WithSelf.F.type.2d3, %I.facet [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.609 = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//types, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.InClassArgs: type = import_ref Main//impl_in_class_args, InClassArgs, loaded [concrete = constants.%InClassArgs]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.820 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//types, loc5_10, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_class_args, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.9b7 = import_ref Main//impl_in_class_args, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.480: @I.WithSelf.%I.assoc_type (%I.assoc_type.76c) = import_ref Main//types, loc4_43, loaded [symbolic = @I.WithSelf.%assoc0 (constants.%assoc0.d22)]
// CHECK:STDOUT:   %Main.F = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.e65: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.efc) = import_ref Main//types, loc4_43, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.213)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.45ead6.2: @I.%I.type (%I.type.1ab) = import_ref Main//types, loc4_23, loaded [symbolic = @I.%Self (constants.%Self.c1b)]
// CHECK:STDOUT:   %Main.import_ref.3fa = import_ref Main//types, loc4_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.8d2 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.a9d = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ca: <witness> = import_ref Main//impl_in_class_args, loc7_29, loaded [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.7f4: type = import_ref Main//impl_in_class_args, loc7_19, loaded [concrete = constants.%C.83f]
// CHECK:STDOUT:   %Main.import_ref.15e: type = import_ref Main//impl_in_class_args, loc7_27, loaded [concrete = constants.%I.type.03e]
// CHECK:STDOUT:   %Main.import_ref.e57: %C.as.I.impl.F.type = import_ref Main//impl_in_class_args, loc7_57, loaded [concrete = constants.%C.as.I.impl.F]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.e57), @C.as.I.impl [concrete]
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
// CHECK:STDOUT:     %c.param_patt: %pattern_type.65d = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.65d = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.83f = value_param call_param0
// CHECK:STDOUT:     %.loc6_22: type = splice_block %C [concrete = constants.%C.83f] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:       %InClassArgs.ref: type = name_ref InClassArgs, imports.%Main.InClassArgs [concrete = constants.%InClassArgs]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%InClassArgs) [concrete = constants.%C.83f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.83f = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.4: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.3fa
// CHECK:STDOUT:     .F = imports.%Main.import_ref.480
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: imports.%Main.import_ref.7f4 as imports.%Main.import_ref.15e [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.5ca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.b3bc94.1: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.820
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InClassArgs [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.9b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param: %C.83f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.83f = name_ref c, %c
// CHECK:STDOUT:   %I.ref: %I.type.609 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%X)> [concrete = constants.%I.type.03e]
// CHECK:STDOUT:   %.loc6_34: %I.assoc_type.46b = specific_constant imports.%Main.import_ref.480, @I.WithSelf(constants.%X, constants.%Self.c1b) [concrete = constants.%assoc0.683]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.46b = name_ref F, %.loc6_34 [concrete = constants.%assoc0.683]
// CHECK:STDOUT:   %impl.elem0: %.ac4 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%C.as.I.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.I.impl.F.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.45ead6.2: @I.%I.type (%I.type.1ab)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c1b)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.831)]
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.c1b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self.c1b) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.831
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.03e
// CHECK:STDOUT:   %Self => constants.%Self.9c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%X, constants.%Self.c1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.03e
// CHECK:STDOUT:   %Self => constants.%Self.c1b
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.bfe
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.0d9
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.46b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.683
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%X, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.03e
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.2d3
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.841
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.46b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.683
// CHECK:STDOUT: }
// CHECK:STDOUT:

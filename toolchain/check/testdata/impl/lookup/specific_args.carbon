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
// CHECK:STDOUT:   %I.type.751: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.751 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.834: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.834 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.67b: type = pattern_type %Self.as_type [symbolic]
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
// CHECK:STDOUT:   %I.decl: %I.type.751 = interface_decl @I [concrete = constants.%I.generic] {
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
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_14.1)> [symbolic = %I.type (constants.%I.type.834)]
// CHECK:STDOUT:   %Self.loc4_23.2: @I.%I.type (%I.type.834) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_23.1: @I.%I.type (%I.type.834) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F)] {
// CHECK:STDOUT:       %self.param_patt: @I.WithSelf.F.%pattern_type (%pattern_type.67b) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @I.WithSelf.F.%pattern_type (%pattern_type.67b) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @I.WithSelf.F.%Self.as_type.loc4_36.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc4_36.1: type = splice_block %.loc4_36.3 [symbolic = %Self.as_type.loc4_36.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc4_36.2: @I.WithSelf.F.%I.type (%I.type.834) = specific_constant @I.%Self.loc4_23.1, @I(constants.%T) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @I.WithSelf.F.%I.type (%I.type.834) = name_ref Self, %.loc4_36.2 [symbolic = %Self (constants.%Self)]
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
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%T.loc4_14.2: type, @I.%Self.loc4_23.1: @I.%I.type (%I.type.834)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.834)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.834) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_36.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_36.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc4_36.1 [symbolic = %pattern_type (constants.%pattern_type.67b)]
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
// CHECK:STDOUT:   %I.type => constants.%I.type.834
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_36.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.67b
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
// CHECK:STDOUT:   %I.type.cc2: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.cc2 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.658: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.8ab: %I.type.658 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.20e: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self.8ab) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.f1b: %I.WithSelf.F.type.20e = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.8ab [symbolic]
// CHECK:STDOUT:   %pattern_type.1b9: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %I.type.9ce: type = facet_type <@I, @I(%InInterfaceArgs)> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @X.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.723: %I.type.9ce = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.399: type = fn_type @I.WithSelf.F, @I.WithSelf(%InInterfaceArgs, %Self.8ab) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.6e5: %I.WithSelf.F.type.399 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.26e: type = assoc_entity_type @I, @I(%InInterfaceArgs) [concrete]
// CHECK:STDOUT:   %assoc0.102: %I.assoc_type.26e = assoc_entity element0, imports.%Main.import_ref.a02 [concrete]
// CHECK:STDOUT:   %pattern_type.4d0: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F: %X.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.9ce = facet_value %X, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.df8: type = fn_type @I.WithSelf.F, @I.WithSelf(%InInterfaceArgs, %I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.7f3: %I.WithSelf.F.type.df8 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.cc2 = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.bd9 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c2c = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.F: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.20e) = import_ref Main//types, F, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.f1b)]
// CHECK:STDOUT:   %Main.import_ref.a02 = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.cd6899.2: @I.%I.type (%I.type.658) = import_ref Main//types, loc4_23, loaded [symbolic = @I.%Self (constants.%Self.8ab)]
// CHECK:STDOUT:   %Main.import_ref.fbf = import_ref Main//types, loc4_23, unloaded
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
// CHECK:STDOUT:     %I.ref: %I.type.cc2 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, file.%InInterfaceArgs.decl [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [concrete = constants.%I.type.9ce]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.3: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.658)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.658) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.8ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.fbf
// CHECK:STDOUT:     .F = imports.%Main.import_ref.c2c
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.I.impl: %X.ref as %I.type {
// CHECK:STDOUT:   %X.as.I.impl.F.decl: %X.as.I.impl.F.type = fn_decl @X.as.I.impl.F [concrete = constants.%X.as.I.impl.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4d0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.4d0 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   extend %I.type
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
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.bd9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.cd6899.2: @I.%I.type (%I.type.658)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.658)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.658) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.8ab)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.1b9)]
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.8ab) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self.8ab) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.658
// CHECK:STDOUT:   %Self => constants.%Self.8ab
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%InInterfaceArgs) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.9ce
// CHECK:STDOUT:   %Self => constants.%Self.723
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%InInterfaceArgs, constants.%Self.8ab) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.9ce
// CHECK:STDOUT:   %Self => constants.%Self.8ab
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.399
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.6e5
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.26e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%InInterfaceArgs, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.9ce
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.df8
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.7f3
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.26e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%InInterfaceArgs, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.9ce
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%X
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_impl_in_interface_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.7a4: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.074: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.074 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.718: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.3fc: %I.type.718 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.ba7: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.ed9: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self.3fc) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.329: %I.WithSelf.F.type.ed9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.3fc [symbolic]
// CHECK:STDOUT:   %pattern_type.6a5: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %InInterfaceArgs: type = class_type @InInterfaceArgs [concrete]
// CHECK:STDOUT:   %I.type.c68: type = facet_type <@I, @I(%InInterfaceArgs)> [concrete]
// CHECK:STDOUT:   %Self.efe: %I.type.c68 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.a8d: type = fn_type @I.WithSelf.F, @I.WithSelf(%InInterfaceArgs, %Self.3fc) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.284: %I.WithSelf.F.type.a8d = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.0d9: type = assoc_entity_type @I, @I(%InInterfaceArgs) [concrete]
// CHECK:STDOUT:   %assoc0.e7a: %I.assoc_type.0d9 = assoc_entity element0, imports.%Main.import_ref.143 [concrete]
// CHECK:STDOUT:   %assoc0.b1f: %I.assoc_type.ba7 = assoc_entity element0, imports.%Main.import_ref.a02 [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.c68 = facet_value %X, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.577: type = fn_type @I.WithSelf.F, @I.WithSelf(%InInterfaceArgs, %I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.3d1: %I.WithSelf.F.type.577 = struct_value () [concrete]
// CHECK:STDOUT:   %.bf3: type = fn_type_with_self_type %I.WithSelf.F.type.577, %I.facet [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F [concrete]
// CHECK:STDOUT:   %X.as.I.impl.F: %X.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.074 = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C = import_ref Main//types, C, unloaded
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.InInterfaceArgs: type = import_ref Main//impl_in_interface_args, InInterfaceArgs, loaded [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.bd9 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e4b: @I.WithSelf.%I.assoc_type (%I.assoc_type.ba7) = import_ref Main//types, loc4_43, loaded [symbolic = @I.WithSelf.%assoc0 (constants.%assoc0.b1f)]
// CHECK:STDOUT:   %Main.F = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.143: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.ed9) = import_ref Main//types, loc4_43, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.329)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.6c3b78.2: @I.%I.type (%I.type.718) = import_ref Main//types, loc4_23, loaded [symbolic = @I.%Self (constants.%Self.3fc)]
// CHECK:STDOUT:   %Main.import_ref.fbf = import_ref Main//types, loc4_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_interface_args, loc5_24, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.377 = import_ref Main//impl_in_interface_args, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.a02 = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.05a: <witness> = import_ref Main//impl_in_interface_args, loc7_30, loaded [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.720: type = import_ref Main//impl_in_interface_args, loc7_6, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.fe5: type = import_ref Main//impl_in_interface_args, loc7_28, loaded [concrete = constants.%I.type.c68]
// CHECK:STDOUT:   %Main.import_ref.80b: %X.as.I.impl.F.type = import_ref Main//impl_in_interface_args, loc7_58, loaded [concrete = constants.%X.as.I.impl.F]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.80b), @X.as.I.impl [concrete]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7a4 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.7a4 = at_binding_pattern x, %x.param_patt [concrete]
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
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.718)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.718) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.3fc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.fbf
// CHECK:STDOUT:     .F = imports.%Main.import_ref.e4b
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @X.as.I.impl: imports.%Main.import_ref.720 as imports.%Main.import_ref.fe5 [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.05a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.bd9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InInterfaceArgs [from "impl_in_interface_args.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.377
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x.param: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %X = name_ref x, %x
// CHECK:STDOUT:   %I.ref: %I.type.074 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:   %InInterfaceArgs.ref: type = name_ref InInterfaceArgs, imports.%Main.InInterfaceArgs [concrete = constants.%InInterfaceArgs]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%InInterfaceArgs)> [concrete = constants.%I.type.c68]
// CHECK:STDOUT:   %.loc6: %I.assoc_type.0d9 = specific_constant imports.%Main.import_ref.e4b, @I.WithSelf(constants.%InInterfaceArgs, constants.%Self.3fc) [concrete = constants.%assoc0.e7a]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.0d9 = name_ref F, %.loc6 [concrete = constants.%assoc0.e7a]
// CHECK:STDOUT:   %impl.elem0: %.bf3 = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%X.as.I.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %X.as.I.impl.F.call: init %empty_tuple.type = call %bound_method(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.b3bc94.2: type, imports.%Main.import_ref.6c3b78.2: @I.%I.type (%I.type.718)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.718)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.718) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.3fc)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.6a5)]
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.3fc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self.3fc) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.718
// CHECK:STDOUT:   %Self => constants.%Self.3fc
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6a5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%InInterfaceArgs) {
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.c68
// CHECK:STDOUT:   %Self => constants.%Self.efe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%InInterfaceArgs, constants.%Self.3fc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.c68
// CHECK:STDOUT:   %Self => constants.%Self.3fc
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.a8d
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.284
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.0d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e7a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%InInterfaceArgs, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%InInterfaceArgs
// CHECK:STDOUT:   %I.type => constants.%I.type.c68
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.577
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.3d1
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.0d9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e7a
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
// CHECK:STDOUT:   %C.1c7: type = class_type @C, @C(%InClassArgs) [concrete]
// CHECK:STDOUT:   %I.type.55c: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.55c = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.5e0: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.d99: %I.type.5e0 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.415: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self.d99) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.346: %I.WithSelf.F.type.415 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.d99 [symbolic]
// CHECK:STDOUT:   %pattern_type.497: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %I.type.9ef: type = facet_type <@I, @I(%X)> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.a4e: %I.type.9ef = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.739: type = fn_type @I.WithSelf.F, @I.WithSelf(%X, %Self.d99) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.427: %I.WithSelf.F.type.739 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.24d: type = assoc_entity_type @I, @I(%X) [concrete]
// CHECK:STDOUT:   %assoc0.4f0: %I.assoc_type.24d = assoc_entity element0, imports.%Main.import_ref.a02 [concrete]
// CHECK:STDOUT:   %pattern_type.7bf: type = pattern_type %C.1c7 [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.9ef = facet_value %C.1c7, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.95f: type = fn_type @I.WithSelf.F, @I.WithSelf(%X, %I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.468: %I.WithSelf.F.type.95f = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.55c = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//types, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.998 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//types, loc5_10, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.c2c = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.F: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.415) = import_ref Main//types, F, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.346)]
// CHECK:STDOUT:   %Main.import_ref.a02 = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.878b50.2: @I.%I.type (%I.type.5e0) = import_ref Main//types, loc4_23, loaded [symbolic = @I.%Self (constants.%Self.d99)]
// CHECK:STDOUT:   %Main.import_ref.fbf = import_ref Main//types, loc4_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.bd9 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
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
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%InClassArgs) [concrete = constants.%C.1c7]
// CHECK:STDOUT:     %I.ref: %I.type.55c = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%X)> [concrete = constants.%I.type.9ef]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.4: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.5e0)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.5e0) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.d99)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.fbf
// CHECK:STDOUT:     .F = imports.%Main.import_ref.c2c
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C as %I.type {
// CHECK:STDOUT:   %C.as.I.impl.F.decl: %C.as.I.impl.F.type = fn_decl @C.as.I.impl.F [concrete = constants.%C.as.I.impl.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7bf = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7bf = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C.1c7 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.I.impl.%C [concrete = constants.%C.1c7]
// CHECK:STDOUT:     %self: %C.1c7 = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%C.as.I.impl.F.decl), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl
// CHECK:STDOUT:   extend %I.type
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
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.998
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.bd9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.878b50.2: @I.%I.type (%I.type.5e0)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.5e0)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.5e0) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.d99)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.497)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F(%self.param: %C.1c7) {
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.d99) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self.d99) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.5e0
// CHECK:STDOUT:   %Self => constants.%Self.d99
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.497
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.9ef
// CHECK:STDOUT:   %Self => constants.%Self.a4e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%X, constants.%Self.d99) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.9ef
// CHECK:STDOUT:   %Self => constants.%Self.d99
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.739
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.427
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.24d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%X, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.9ef
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.95f
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.468
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.24d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%X, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.9ef
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type => constants.%C.1c7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7bf
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
// CHECK:STDOUT:   %C.6d9: type = class_type @C, @C(%InClassArgs) [concrete]
// CHECK:STDOUT:   %pattern_type.207: type = pattern_type %C.6d9 [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.571: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.571 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.2f8: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.c54: %I.type.2f8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.da6: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.cba: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self.c54) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.c14: %I.WithSelf.F.type.cba = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.c54 [symbolic]
// CHECK:STDOUT:   %pattern_type.77d: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %I.type.117: type = facet_type <@I, @I(%X)> [concrete]
// CHECK:STDOUT:   %Self.da3: %I.type.117 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.d5e: type = fn_type @I.WithSelf.F, @I.WithSelf(%X, %Self.c54) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.49f: %I.WithSelf.F.type.d5e = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.1e2: type = assoc_entity_type @I, @I(%X) [concrete]
// CHECK:STDOUT:   %assoc0.81d: %I.assoc_type.1e2 = assoc_entity element0, imports.%Main.import_ref.479 [concrete]
// CHECK:STDOUT:   %assoc0.085: %I.assoc_type.da6 = assoc_entity element0, imports.%Main.import_ref.a02 [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness imports.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.117 = facet_value %C.6d9, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.35f: type = fn_type @I.WithSelf.F, @I.WithSelf(%X, %I.facet) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.933: %I.WithSelf.F.type.35f = struct_value () [concrete]
// CHECK:STDOUT:   %.5cf: type = fn_type_with_self_type %I.WithSelf.F.type.35f, %I.facet [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type: type = fn_type @C.as.I.impl.F [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F: %C.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: %I.type.571 = import_ref Main//types, I, loaded [concrete = constants.%I.generic]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//types, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.X: type = import_ref Main//types, X, loaded [concrete = constants.%X]
// CHECK:STDOUT:   %Main.InClassArgs: type = import_ref Main//impl_in_class_args, InClassArgs, loaded [concrete = constants.%InClassArgs]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.998 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//types, loc5_10, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//impl_in_class_args, loc5_20, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.a93 = import_ref Main//impl_in_class_args, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c19: @I.WithSelf.%I.assoc_type (%I.assoc_type.da6) = import_ref Main//types, loc4_43, loaded [symbolic = @I.WithSelf.%assoc0 (constants.%assoc0.085)]
// CHECK:STDOUT:   %Main.F = import_ref Main//types, F, unloaded
// CHECK:STDOUT:   %Main.import_ref.479: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.cba) = import_ref Main//types, loc4_43, loaded [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.c14)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.63baa8.2: @I.%I.type (%I.type.2f8) = import_ref Main//types, loc4_23, loaded [symbolic = @I.%Self (constants.%Self.c54)]
// CHECK:STDOUT:   %Main.import_ref.fbf = import_ref Main//types, loc4_23, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.4: type = import_ref Main//types, loc4_14, loaded [symbolic = @I.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc7_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.bd9 = import_ref Main//types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.a02 = import_ref Main//types, loc4_43, unloaded
// CHECK:STDOUT:   %Main.import_ref.411: <witness> = import_ref Main//impl_in_class_args, loc7_29, loaded [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.eb5: type = import_ref Main//impl_in_class_args, loc7_19, loaded [concrete = constants.%C.6d9]
// CHECK:STDOUT:   %Main.import_ref.864: type = import_ref Main//impl_in_class_args, loc7_27, loaded [concrete = constants.%I.type.117]
// CHECK:STDOUT:   %Main.import_ref.27a: %C.as.I.impl.F.type = import_ref Main//impl_in_class_args, loc7_57, loaded [concrete = constants.%C.as.I.impl.F]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%Main.import_ref.27a), @C.as.I.impl [concrete]
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
// CHECK:STDOUT:     %c.param_patt: %pattern_type.207 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.207 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C.6d9 = value_param call_param0
// CHECK:STDOUT:     %.loc6_22: type = splice_block %C [concrete = constants.%C.6d9] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:       %InClassArgs.ref: type = name_ref InClassArgs, imports.%Main.InClassArgs [concrete = constants.%InClassArgs]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%InClassArgs) [concrete = constants.%C.6d9]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %C.6d9 = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(imports.%Main.import_ref.b3bc94.4: type) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2f8)]
// CHECK:STDOUT:   %Self: @I.%I.type (%I.type.2f8) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c54)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.fbf
// CHECK:STDOUT:     .F = imports.%Main.import_ref.c19
// CHECK:STDOUT:     witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: imports.%Main.import_ref.eb5 as imports.%Main.import_ref.864 [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.411
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
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.998
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @InClassArgs [from "impl_in_class_args.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.a93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.bd9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%c.param: %C.6d9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C.6d9 = name_ref c, %c
// CHECK:STDOUT:   %I.ref: %I.type.571 = name_ref I, imports.%Main.I [concrete = constants.%I.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, imports.%Main.X [concrete = constants.%X]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%X)> [concrete = constants.%I.type.117]
// CHECK:STDOUT:   %.loc6_34: %I.assoc_type.1e2 = specific_constant imports.%Main.import_ref.c19, @I.WithSelf(constants.%X, constants.%Self.c54) [concrete = constants.%assoc0.81d]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.1e2 = name_ref F, %.loc6_34 [concrete = constants.%assoc0.81d]
// CHECK:STDOUT:   %impl.elem0: %.5cf = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%C.as.I.impl.F]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %C.as.I.impl.F.call: init %empty_tuple.type = call %bound_method(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(imports.%Main.import_ref.b3bc94.3: type, imports.%Main.import_ref.63baa8.2: @I.%I.type (%I.type.2f8)) [from "types.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.2f8)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.2f8) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.c54)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.77d)]
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
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.c54) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self.c54) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.2f8
// CHECK:STDOUT:   %Self => constants.%Self.c54
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.77d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%X) {
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.117
// CHECK:STDOUT:   %Self => constants.%Self.da3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%X, constants.%Self.c54) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.117
// CHECK:STDOUT:   %Self => constants.%Self.c54
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.d5e
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.49f
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.1e2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.81d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%X, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.117
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.35f
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.933
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.1e2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.81d
// CHECK:STDOUT: }
// CHECK:STDOUT:

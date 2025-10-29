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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/index.carbon

// --- core_wrong_index_with.carbon

package Core library "[[@TEST_NAME]]";

class IndexWith {}

// --- fail_wrong_index_with.carbon

library "[[@TEST_NAME]]";

import Core library "core_wrong_index_with";

// CHECK:STDERR: fail_wrong_index_with.carbon:[[@LINE+4]]:10: error: value of type `type` is not callable [CallToNonCallable]
// CHECK:STDERR: fn F() { 0[1]; }
// CHECK:STDERR:          ^~~~
// CHECK:STDERR:
fn F() { 0[1]; }

// --- fail_missing_index_with.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_missing_index_with.carbon:[[@LINE+4]]:10: error: `Core.IndexWith` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: fn F() { 0[1]; }
// CHECK:STDERR:          ^~~~
// CHECK:STDERR:
fn F() { 0[1]; }

// --- core_wrong_arg_count.carbon

package Core library "[[@TEST_NAME]]";

interface IndexWith(SubscriptType:! type, ElementType:! type) {
  fn At[self: Self](subscript: SubscriptType) -> ElementType;
}


// --- fail_wrong_arg_count.carbon

library "[[@TEST_NAME]]";

import Core library "core_wrong_arg_count";

impl () as Core.IndexWith((), ()) {
  fn At[self: Self](subscript: ()) -> () {
    return ();
  }
}

// CHECK:STDERR: fail_wrong_arg_count.carbon:[[@LINE+8]]:10: error: 1 argument passed to generic interface expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: fn F() { ()[()]; }
// CHECK:STDERR:          ^~~~~~
// CHECK:STDERR: fail_wrong_arg_count.carbon:[[@LINE-11]]:1: in import [InImport]
// CHECK:STDERR: core_wrong_arg_count.carbon:4:1: note: calling generic interface declared here [InCallToEntity]
// CHECK:STDERR: interface IndexWith(SubscriptType:! type, ElementType:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() { ()[()]; }

// CHECK:STDOUT: --- core_wrong_index_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IndexWith: type = class_type @IndexWith [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IndexWith = %IndexWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.decl: type = class_decl @IndexWith [concrete = constants.%IndexWith] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @IndexWith {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%IndexWith
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_index_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %IndexWith: type = class_type @IndexWith [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//core_wrong_index_with
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: type = import_ref Core//core_wrong_index_with, IndexWith, loaded [concrete = constants.%IndexWith]
// CHECK:STDOUT:   %Core.import_ref.8f2: <witness> = import_ref Core//core_wrong_index_with, loc{{\d+_\d+}}, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Core.import_ref.4c7 = import_ref Core//core_wrong_index_with, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @IndexWith [from "core_wrong_index_with.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Core.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.4c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_index_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- core_wrong_arg_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubscriptType: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ElementType: type = symbolic_binding ElementType, 1 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.68b: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.68b = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.type.c35: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic]
// CHECK:STDOUT:   %Self: %IndexWith.type.c35 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 2, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.c2e: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %SubscriptType [symbolic]
// CHECK:STDOUT:   %pattern_type.984: type = pattern_type %ElementType [symbolic]
// CHECK:STDOUT:   %IndexWith.At.type: type = fn_type @IndexWith.At, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %IndexWith.At: %IndexWith.At.type = struct_value () [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %assoc0: %IndexWith.assoc_type = assoc_entity element0, @IndexWith.%IndexWith.At.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IndexWith = %IndexWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.decl: %IndexWith.type.68b = interface_decl @IndexWith [concrete = constants.%IndexWith.generic] {
// CHECK:STDOUT:     %SubscriptType.patt: %pattern_type.98f = symbolic_binding_pattern SubscriptType, 0 [concrete]
// CHECK:STDOUT:     %ElementType.patt: %pattern_type.98f = symbolic_binding_pattern ElementType, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %SubscriptType.loc4_21.2: type = symbolic_binding SubscriptType, 0 [symbolic = %SubscriptType.loc4_21.1 (constants.%SubscriptType)]
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %ElementType.loc4_43.2: type = symbolic_binding ElementType, 1 [symbolic = %ElementType.loc4_43.1 (constants.%ElementType)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(%SubscriptType.loc4_21.2: type, %ElementType.loc4_43.2: type) {
// CHECK:STDOUT:   %SubscriptType.loc4_21.1: type = symbolic_binding SubscriptType, 0 [symbolic = %SubscriptType.loc4_21.1 (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType.loc4_43.1: type = symbolic_binding ElementType, 1 [symbolic = %ElementType.loc4_43.1 (constants.%ElementType)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.loc4_21.1, %ElementType.loc4_43.1)> [symbolic = %IndexWith.type (constants.%IndexWith.type.c35)]
// CHECK:STDOUT:   %Self.2: @IndexWith.%IndexWith.type (%IndexWith.type.c35) = symbolic_binding Self, 2 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %IndexWith.At.type: type = fn_type @IndexWith.At, @IndexWith(%SubscriptType.loc4_21.1, %ElementType.loc4_43.1) [symbolic = %IndexWith.At.type (constants.%IndexWith.At.type)]
// CHECK:STDOUT:   %IndexWith.At: @IndexWith.%IndexWith.At.type (%IndexWith.At.type) = struct_value () [symbolic = %IndexWith.At (constants.%IndexWith.At)]
// CHECK:STDOUT:   %IndexWith.assoc_type: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.loc4_21.1, %ElementType.loc4_43.1) [symbolic = %IndexWith.assoc_type (constants.%IndexWith.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc5_61.2: @IndexWith.%IndexWith.assoc_type (%IndexWith.assoc_type) = assoc_entity element0, %IndexWith.At.decl [symbolic = %assoc0.loc5_61.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.c35) = symbolic_binding Self, 2 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %IndexWith.At.decl: @IndexWith.%IndexWith.At.type (%IndexWith.At.type) = fn_decl @IndexWith.At [symbolic = @IndexWith.%IndexWith.At (constants.%IndexWith.At)] {
// CHECK:STDOUT:       %self.patt: @IndexWith.At.%pattern_type.loc5_9 (%pattern_type.c2e) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IndexWith.At.%pattern_type.loc5_9 (%pattern_type.c2e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %subscript.patt: @IndexWith.At.%pattern_type.loc5_21 (%pattern_type.e68) = value_binding_pattern subscript [concrete]
// CHECK:STDOUT:       %subscript.param_patt: @IndexWith.At.%pattern_type.loc5_21 (%pattern_type.e68) = value_param_pattern %subscript.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IndexWith.At.%pattern_type.loc5_47 (%pattern_type.984) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IndexWith.At.%pattern_type.loc5_47 (%pattern_type.984) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %ElementType.ref: type = name_ref ElementType, @IndexWith.%ElementType.loc4_43.2 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:       %self.param: @IndexWith.At.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_15.1: type = splice_block %.loc5_15.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc5_15.2: @IndexWith.At.%IndexWith.type (%IndexWith.type.c35) = specific_constant @IndexWith.%Self.1, @IndexWith(constants.%SubscriptType, constants.%ElementType) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @IndexWith.At.%IndexWith.type (%IndexWith.type.c35) = name_ref Self, %.loc5_15.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc5_15.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IndexWith.At.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %subscript.param: @IndexWith.At.%SubscriptType (%SubscriptType) = value_param call_param1
// CHECK:STDOUT:       %SubscriptType.ref: type = name_ref SubscriptType, @IndexWith.%SubscriptType.loc4_21.2 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:       %subscript: @IndexWith.At.%SubscriptType (%SubscriptType) = value_binding subscript, %subscript.param
// CHECK:STDOUT:       %return.param: ref @IndexWith.At.%ElementType (%ElementType) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IndexWith.At.%ElementType (%ElementType) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_61.1: @IndexWith.%IndexWith.assoc_type (%IndexWith.assoc_type) = assoc_entity element0, %IndexWith.At.decl [symbolic = %assoc0.loc5_61.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .SubscriptType = <poisoned>
// CHECK:STDOUT:     .ElementType = <poisoned>
// CHECK:STDOUT:     .At = %assoc0.loc5_61.1
// CHECK:STDOUT:     witness = (%IndexWith.At.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IndexWith.At(@IndexWith.%SubscriptType.loc4_21.2: type, @IndexWith.%ElementType.loc4_43.2: type, @IndexWith.%Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.c35)) {
// CHECK:STDOUT:   %SubscriptType: type = symbolic_binding SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = symbolic_binding ElementType, 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.c35)]
// CHECK:STDOUT:   %Self: @IndexWith.At.%IndexWith.type (%IndexWith.type.c35) = symbolic_binding Self, 2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 2, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5_9: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc5_9 (constants.%pattern_type.c2e)]
// CHECK:STDOUT:   %pattern_type.loc5_21: type = pattern_type %SubscriptType [symbolic = %pattern_type.loc5_21 (constants.%pattern_type.e68)]
// CHECK:STDOUT:   %pattern_type.loc5_47: type = pattern_type %ElementType [symbolic = %pattern_type.loc5_47 (constants.%pattern_type.984)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IndexWith.At.%Self.binding.as_type (%Self.binding.as_type), %subscript.param: @IndexWith.At.%SubscriptType (%SubscriptType)) -> @IndexWith.At.%ElementType (%ElementType);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType, constants.%ElementType) {
// CHECK:STDOUT:   %SubscriptType.loc4_21.1 => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType.loc4_43.1 => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith.At(constants.%SubscriptType, constants.%ElementType, constants.%Self) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.c35
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc5_9 => constants.%pattern_type.c2e
// CHECK:STDOUT:   %pattern_type.loc5_21 => constants.%pattern_type.e68
// CHECK:STDOUT:   %pattern_type.loc5_47 => constants.%pattern_type.984
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_arg_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [concrete]
// CHECK:STDOUT:   %ElementType: type = symbolic_binding ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.b02: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic]
// CHECK:STDOUT:   %Self.a98: %IndexWith.type.b02 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %IndexWith.At.type.755: type = fn_type @IndexWith.At, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %IndexWith.At.f92: %IndexWith.At.type.755 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.984: type = pattern_type %ElementType [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 2, %Self.a98 [symbolic]
// CHECK:STDOUT:   %pattern_type.e2c: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %SubscriptType [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type.4e0: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %assoc0.845: %IndexWith.assoc_type.4e0 = assoc_entity element0, imports.%Core.import_ref.118 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.5ee: type = facet_type <@IndexWith, @IndexWith(%empty_tuple.type, %empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %Self.d39: %IndexWith.type.5ee = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %IndexWith.At.type.1a9: type = fn_type @IndexWith.At, @IndexWith(%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %IndexWith.At.721: %IndexWith.At.type.1a9 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.assoc_type.d10: type = assoc_entity_type @IndexWith, @IndexWith(%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %assoc0.819: %IndexWith.assoc_type.d10 = assoc_entity element0, imports.%Core.import_ref.118 [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness file.%IndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.IndexWith.impl.At.type: type = fn_type @empty_tuple.type.as.IndexWith.impl.At [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.IndexWith.impl.At: %empty_tuple.type.as.IndexWith.impl.At.type = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.5ee = facet_value %empty_tuple.type, (%IndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//core_wrong_arg_count
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.504 = import_ref Core//core_wrong_arg_count, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.efcd44.1: type = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.%SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %Core.import_ref.c5c78a.1: type = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.%ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %Core.import_ref.05e = import_ref Core//core_wrong_arg_count, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.1bb = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.At: @IndexWith.%IndexWith.At.type (%IndexWith.At.type.755) = import_ref Core//core_wrong_arg_count, At, loaded [symbolic = @IndexWith.%IndexWith.At (constants.%IndexWith.At.f92)]
// CHECK:STDOUT:   %Core.import_ref.efcd44.2: type = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.%SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %Core.import_ref.c5c78a.2: type = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.%ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %Core.import_ref.c11: @IndexWith.%IndexWith.type (%IndexWith.type.b02) = import_ref Core//core_wrong_arg_count, inst{{[0-9A-F]+}} [no loc], loaded [symbolic = @IndexWith.%Self (constants.%Self.a98)]
// CHECK:STDOUT:   %Core.import_ref.118 = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   impl_decl @empty_tuple.type.as.IndexWith.impl [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %.loc6_28: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_32: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_33.1: type = converted %.loc6_28, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc6_33.2: type = converted %.loc6_32, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%empty_tuple.type, constants.%empty_tuple.type)> [concrete = constants.%IndexWith.type.5ee]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (@empty_tuple.type.as.IndexWith.impl.%empty_tuple.type.as.IndexWith.impl.At.decl), @empty_tuple.type.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness %IndexWith.impl_witness_table [concrete = constants.%IndexWith.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(imports.%Core.import_ref.efcd44.1: type, imports.%Core.import_ref.c5c78a.1: type) [from "core_wrong_arg_count.carbon"] {
// CHECK:STDOUT:   %SubscriptType: type = symbolic_binding SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = symbolic_binding ElementType, 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.b02)]
// CHECK:STDOUT:   %Self: @IndexWith.%IndexWith.type (%IndexWith.type.b02) = symbolic_binding Self, 2 [symbolic = %Self (constants.%Self.a98)]
// CHECK:STDOUT:   %IndexWith.At.type: type = fn_type @IndexWith.At, @IndexWith(%SubscriptType, %ElementType) [symbolic = %IndexWith.At.type (constants.%IndexWith.At.type.755)]
// CHECK:STDOUT:   %IndexWith.At: @IndexWith.%IndexWith.At.type (%IndexWith.At.type.755) = struct_value () [symbolic = %IndexWith.At (constants.%IndexWith.At.f92)]
// CHECK:STDOUT:   %IndexWith.assoc_type: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic = %IndexWith.assoc_type (constants.%IndexWith.assoc_type.4e0)]
// CHECK:STDOUT:   %assoc0: @IndexWith.%IndexWith.assoc_type (%IndexWith.assoc_type.4e0) = assoc_entity element0, imports.%Core.import_ref.118 [symbolic = %assoc0 (constants.%assoc0.845)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.05e
// CHECK:STDOUT:     .At = imports.%Core.import_ref.1bb
// CHECK:STDOUT:     witness = (imports.%Core.At)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.IndexWith.impl: %.loc6_7.2 as %IndexWith.type {
// CHECK:STDOUT:   %empty_tuple.type.as.IndexWith.impl.At.decl: %empty_tuple.type.as.IndexWith.impl.At.type = fn_decl @empty_tuple.type.as.IndexWith.impl.At [concrete = constants.%empty_tuple.type.as.IndexWith.impl.At] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.cb1 = value_binding_pattern subscript [concrete]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.cb1 = value_param_pattern %subscript.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_40.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_40.2: type = converted %.loc7_40.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @empty_tuple.type.as.IndexWith.impl.%.loc6_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: %empty_tuple.type = value_param call_param1
// CHECK:STDOUT:     %.loc7_33.1: type = splice_block %.loc7_33.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc7_33.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc7_33.3: type = converted %.loc7_33.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %subscript: %empty_tuple.type = value_binding subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %empty_tuple.type.as.IndexWith.impl.At.decl
// CHECK:STDOUT:   witness = file.%IndexWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IndexWith.At(imports.%Core.import_ref.efcd44.2: type, imports.%Core.import_ref.c5c78a.2: type, imports.%Core.import_ref.c11: @IndexWith.%IndexWith.type (%IndexWith.type.b02)) [from "core_wrong_arg_count.carbon"] {
// CHECK:STDOUT:   %SubscriptType: type = symbolic_binding SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = symbolic_binding ElementType, 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.b02)]
// CHECK:STDOUT:   %Self: @IndexWith.At.%IndexWith.type (%IndexWith.type.b02) = symbolic_binding Self, 2 [symbolic = %Self (constants.%Self.a98)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 2, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.e2c)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %SubscriptType [symbolic = %pattern_type.2 (constants.%pattern_type.e68)]
// CHECK:STDOUT:   %pattern_type.3: type = pattern_type %ElementType [symbolic = %pattern_type.3 (constants.%pattern_type.984)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.IndexWith.impl.At(%self.param: %empty_tuple.type, %subscript.param: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_13.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc8_13.2: init %empty_tuple.type = tuple_init () to %return [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_14: init %empty_tuple.type = converted %.loc8_13.1, %.loc8_13.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc8_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc20_14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc20_11.2: %empty_tuple.type = converted %.loc20_11.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType, constants.%ElementType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith.At(constants.%SubscriptType, constants.%ElementType, constants.%Self.a98) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.b02
// CHECK:STDOUT:   %Self => constants.%Self.a98
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.e2c
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.e68
// CHECK:STDOUT:   %pattern_type.3 => constants.%pattern_type.984
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%empty_tuple.type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %SubscriptType => constants.%empty_tuple.type
// CHECK:STDOUT:   %ElementType => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.5ee
// CHECK:STDOUT:   %Self => constants.%Self.d39
// CHECK:STDOUT:   %IndexWith.At.type => constants.%IndexWith.At.type.1a9
// CHECK:STDOUT:   %IndexWith.At => constants.%IndexWith.At.721
// CHECK:STDOUT:   %IndexWith.assoc_type => constants.%IndexWith.assoc_type.d10
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.819
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith.At(constants.%empty_tuple.type, constants.%empty_tuple.type, constants.%IndexWith.facet) {
// CHECK:STDOUT:   %SubscriptType => constants.%empty_tuple.type
// CHECK:STDOUT:   %ElementType => constants.%empty_tuple.type
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.5ee
// CHECK:STDOUT:   %Self => constants.%IndexWith.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %pattern_type.3 => constants.%pattern_type.cb1
// CHECK:STDOUT: }
// CHECK:STDOUT:

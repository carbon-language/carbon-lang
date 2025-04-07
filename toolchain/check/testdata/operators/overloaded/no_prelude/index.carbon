// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/no_prelude/index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/no_prelude/index.carbon

// --- core_wrong_index_with.carbon

package Core library "[[@TEST_NAME]]";

class IndexWith {}

// --- fail_wrong_index_with.carbon

library "[[@TEST_NAME]]";

import Core library "core_wrong_index_with";

// CHECK:STDERR: fail_wrong_index_with.carbon:[[@LINE+4]]:10: error: type `Core.IntLiteral` does not support indexing [TypeNotIndexable]
// CHECK:STDERR: fn F() { 0[1]; }
// CHECK:STDERR:          ^~~~
// CHECK:STDERR:
fn F() { 0[1]; }

// --- fail_missing_index_with.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_missing_index_with.carbon:[[@LINE+8]]:10: error: `Core.IndexWith` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: fn F() { 0[1]; }
// CHECK:STDERR:          ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_index_with.carbon:[[@LINE+4]]:10: error: type `Core.IntLiteral` does not support indexing [TypeNotIndexable]
// CHECK:STDERR: fn F() { 0[1]; }
// CHECK:STDERR:          ^~~~
// CHECK:STDERR:
fn F() { 0[1]; }

// --- core_wrong_arg_count.carbon

package Core library "[[@TEST_NAME]]";

interface IndexWith(SubscriptType:! type) {
  fn At[self: Self](subscript: SubscriptType) -> ();
}

// --- wrong_arg_count.carbon

library "[[@TEST_NAME]]";

import Core library "core_wrong_arg_count";

impl () as Core.IndexWith(()) {
  fn At[self: Self](subscript: ()) -> () {
    return ();
  }
}

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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//core_wrong_index_with
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.8f2: <witness> = import_ref Core//core_wrong_index_with, loc4_18, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Core.import_ref.4c7 = import_ref Core//core_wrong_index_with, inst16 [no loc], unloaded
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
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.68b: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.68b = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.type.b94: type = facet_type <@IndexWith, @IndexWith(%SubscriptType)> [symbolic]
// CHECK:STDOUT:   %Self: %IndexWith.type.b94 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %At.type: type = fn_type @At, @IndexWith(%SubscriptType) [symbolic]
// CHECK:STDOUT:   %At: %At.type = struct_value () [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType) [symbolic]
// CHECK:STDOUT:   %assoc0: %IndexWith.assoc_type = assoc_entity element0, @IndexWith.%At.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IndexWith = %IndexWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.decl: %IndexWith.type.68b = interface_decl @IndexWith [concrete = constants.%IndexWith.generic] {
// CHECK:STDOUT:     %SubscriptType.patt.loc4_21.1: type = symbolic_binding_pattern SubscriptType, 0 [symbolic = %SubscriptType.patt.loc4_21.2 (constants.%SubscriptType.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SubscriptType.loc4_21.1: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType.loc4_21.2 (constants.%SubscriptType)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(%SubscriptType.loc4_21.1: type) {
// CHECK:STDOUT:   %SubscriptType.loc4_21.2: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType.loc4_21.2 (constants.%SubscriptType)]
// CHECK:STDOUT:   %SubscriptType.patt.loc4_21.2: type = symbolic_binding_pattern SubscriptType, 0 [symbolic = %SubscriptType.patt.loc4_21.2 (constants.%SubscriptType.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.loc4_21.2)> [symbolic = %IndexWith.type (constants.%IndexWith.type.b94)]
// CHECK:STDOUT:   %Self.2: @IndexWith.%IndexWith.type (%IndexWith.type.b94) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %At.type: type = fn_type @At, @IndexWith(%SubscriptType.loc4_21.2) [symbolic = %At.type (constants.%At.type)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type) = struct_value () [symbolic = %At (constants.%At)]
// CHECK:STDOUT:   %IndexWith.assoc_type: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.loc4_21.2) [symbolic = %IndexWith.assoc_type (constants.%IndexWith.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc5_52.2: @IndexWith.%IndexWith.assoc_type (%IndexWith.assoc_type) = assoc_entity element0, %At.decl [symbolic = %assoc0.loc5_52.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.b94) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %At.decl: @IndexWith.%At.type (%At.type) = fn_decl @At [symbolic = @IndexWith.%At (constants.%At)] {
// CHECK:STDOUT:       %self.patt: @At.%Self.as_type.loc5_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @At.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %subscript.patt: @At.%SubscriptType (%SubscriptType) = binding_pattern subscript
// CHECK:STDOUT:       %subscript.param_patt: @At.%SubscriptType (%SubscriptType) = value_param_pattern %subscript.patt, call_param1
// CHECK:STDOUT:       %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_51.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc5_51.2: type = converted %.loc5_51.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %self.param: @At.%Self.as_type.loc5_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_15.1: type = splice_block %.loc5_15.3 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_15.2: @At.%IndexWith.type (%IndexWith.type.b94) = specific_constant @IndexWith.%Self.1, @IndexWith(constants.%SubscriptType) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @At.%IndexWith.type (%IndexWith.type.b94) = name_ref Self, %.loc5_15.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc5_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_15.3: type = converted %Self.ref, %Self.as_type.loc5_15.2 [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @At.%Self.as_type.loc5_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %subscript.param: @At.%SubscriptType (%SubscriptType) = value_param call_param1
// CHECK:STDOUT:       %SubscriptType.ref: type = name_ref SubscriptType, @IndexWith.%SubscriptType.loc4_21.1 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:       %subscript: @At.%SubscriptType (%SubscriptType) = bind_name subscript, %subscript.param
// CHECK:STDOUT:       %return.param: ref %empty_tuple.type = out_param call_param2
// CHECK:STDOUT:       %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_52.1: @IndexWith.%IndexWith.assoc_type (%IndexWith.assoc_type) = assoc_entity element0, %At.decl [symbolic = %assoc0.loc5_52.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .SubscriptType = <poisoned>
// CHECK:STDOUT:     .At = %assoc0.loc5_52.1
// CHECK:STDOUT:     witness = (%At.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @At(@IndexWith.%SubscriptType.loc4_21.1: type, @IndexWith.%Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.b94)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.b94)]
// CHECK:STDOUT:   %Self: @At.%IndexWith.type (%IndexWith.type.b94) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @At.%Self.as_type.loc5_15.1 (%Self.as_type)](%subscript.param_patt: @At.%SubscriptType (%SubscriptType)) -> %empty_tuple.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType) {
// CHECK:STDOUT:   %SubscriptType.loc4_21.2 => constants.%SubscriptType
// CHECK:STDOUT:   %SubscriptType.patt.loc4_21.2 => constants.%SubscriptType.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At(constants.%SubscriptType, constants.%Self) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.b94
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@At.%SubscriptType) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(%SubscriptType.loc4_21.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- wrong_arg_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.bd2: type = facet_type <@IndexWith, @IndexWith(%SubscriptType)> [symbolic]
// CHECK:STDOUT:   %Self.30a: %IndexWith.type.bd2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %At.type.cf4: type = fn_type @At.1, @IndexWith(%SubscriptType) [symbolic]
// CHECK:STDOUT:   %At.281: %At.type.cf4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.30a [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type.290: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType) [symbolic]
// CHECK:STDOUT:   %assoc0.af8: %IndexWith.assoc_type.290 = assoc_entity element0, imports.%Core.import_ref.e99 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.a51: type = facet_type <@IndexWith, @IndexWith(%empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %Self.8f0: %IndexWith.type.a51 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %At.type.969: type = fn_type @At.1, @IndexWith(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %At.9b9: %At.type.969 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.assoc_type.3d4: type = assoc_entity_type @IndexWith, @IndexWith(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %assoc0.21b: %IndexWith.assoc_type.3d4 = assoc_entity element0, imports.%Core.import_ref.e99 [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%At.decl) [concrete]
// CHECK:STDOUT:   %At.type.486: type = fn_type @At.2 [concrete]
// CHECK:STDOUT:   %At.7c4: %At.type.486 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.a51 = facet_value %empty_tuple.type, (%impl_witness) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc0.e1e: %IndexWith.assoc_type.290 = assoc_entity element0, imports.%Core.import_ref.981 [symbolic]
// CHECK:STDOUT:   %.851: type = fn_type_with_self_type %At.type.969, %IndexWith.facet [concrete]
// CHECK:STDOUT:   %At.bound: <bound method> = bound_method %empty_tuple, %At.7c4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//core_wrong_arg_count
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.504 = import_ref Core//core_wrong_arg_count, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//core_wrong_arg_count, loc4_21, loaded [symbolic = @IndexWith.%SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %Core.import_ref.68a = import_ref Core//core_wrong_arg_count, inst26 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.613: @IndexWith.%IndexWith.assoc_type (%IndexWith.assoc_type.290) = import_ref Core//core_wrong_arg_count, loc5_52, loaded [symbolic = @IndexWith.%assoc0 (constants.%assoc0.e1e)]
// CHECK:STDOUT:   %Core.At: @IndexWith.%At.type (%At.type.cf4) = import_ref Core//core_wrong_arg_count, At, loaded [symbolic = @IndexWith.%At (constants.%At.281)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//core_wrong_arg_count, loc4_21, loaded [symbolic = @IndexWith.%SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %Core.import_ref.fb5: @IndexWith.%IndexWith.type (%IndexWith.type.bd2) = import_ref Core//core_wrong_arg_count, inst26 [no loc], loaded [symbolic = @IndexWith.%Self (constants.%Self.30a)]
// CHECK:STDOUT:   %Core.import_ref.e99: @IndexWith.%At.type (%At.type.cf4) = import_ref Core//core_wrong_arg_count, loc5_52, loaded [symbolic = @IndexWith.%At (constants.%At.281)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc6_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_7.2: type = converted %.loc6_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %.loc6_28: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_29: type = converted %.loc6_28, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%empty_tuple.type)> [concrete = constants.%IndexWith.type.a51]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%At.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(imports.%Core.import_ref.5ab3ec.1: type) [from "core_wrong_arg_count.carbon"] {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic = %SubscriptType.patt (constants.%SubscriptType.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.bd2)]
// CHECK:STDOUT:   %Self: @IndexWith.%IndexWith.type (%IndexWith.type.bd2) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.30a)]
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType) [symbolic = %At.type (constants.%At.type.cf4)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.cf4) = struct_value () [symbolic = %At (constants.%At.281)]
// CHECK:STDOUT:   %IndexWith.assoc_type: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType) [symbolic = %IndexWith.assoc_type (constants.%IndexWith.assoc_type.290)]
// CHECK:STDOUT:   %assoc0: @IndexWith.%IndexWith.assoc_type (%IndexWith.assoc_type.290) = assoc_entity element0, imports.%Core.import_ref.e99 [symbolic = %assoc0 (constants.%assoc0.af8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.68a
// CHECK:STDOUT:     .At = imports.%Core.import_ref.613
// CHECK:STDOUT:     witness = (imports.%Core.At)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc6_7.2 as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.486 = fn_decl @At.2 [concrete = constants.%At.7c4] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %subscript.patt: %empty_tuple.type = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %empty_tuple.type = value_param_pattern %subscript.patt, call_param1
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_40.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_40.2: type = converted %.loc7_40.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc6_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %empty_tuple.type = value_param call_param1
// CHECK:STDOUT:     %.loc7_33.1: type = splice_block %.loc7_33.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc7_33.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc7_33.3: type = converted %.loc7_33.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %subscript: %empty_tuple.type = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @At.1(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.fb5: @IndexWith.%IndexWith.type (%IndexWith.type.bd2)) [from "core_wrong_arg_count.carbon"] {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.bd2)]
// CHECK:STDOUT:   %Self: @At.1.%IndexWith.type (%IndexWith.type.bd2) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.30a)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @At.1.%Self.as_type (%Self.as_type)](%subscript.param_patt: @At.1.%SubscriptType (%SubscriptType)) -> %empty_tuple.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %empty_tuple.type](%subscript.param_patt: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_13: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_14: %empty_tuple.type = converted %.loc8_13, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc8_14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc12_14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple.loc12_11: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_11.2: %empty_tuple.type = converted %.loc12_11.1, %empty_tuple.loc12_11 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc12_14: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_15: %empty_tuple.type = converted %.loc12_14, %empty_tuple.loc12_14 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %impl.elem0: %.851 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%At.7c4]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12_11.2, %impl.elem0 [concrete = constants.%At.bound]
// CHECK:STDOUT:   %At.call: init %empty_tuple.type = call %bound_method(%.loc12_11.2, %.loc12_15)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(%SubscriptType) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@At.1.%SubscriptType) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType, constants.%Self.30a) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.bd2
// CHECK:STDOUT:   %Self => constants.%Self.30a
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %SubscriptType => constants.%empty_tuple.type
// CHECK:STDOUT:   %SubscriptType.patt => constants.%SubscriptType.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.a51
// CHECK:STDOUT:   %Self => constants.%Self.8f0
// CHECK:STDOUT:   %At.type => constants.%At.type.969
// CHECK:STDOUT:   %At => constants.%At.9b9
// CHECK:STDOUT:   %IndexWith.assoc_type => constants.%IndexWith.assoc_type.3d4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.21b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%empty_tuple.type, constants.%IndexWith.facet) {
// CHECK:STDOUT:   %SubscriptType => constants.%empty_tuple.type
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.a51
// CHECK:STDOUT:   %Self => constants.%IndexWith.facet
// CHECK:STDOUT:   %Self.as_type => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:

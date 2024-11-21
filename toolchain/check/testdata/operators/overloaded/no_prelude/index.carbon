// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/no_prelude/index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/no_prelude/index.carbon

// --- fail_wrong_index_with.carbon

library "[[@TEST_NAME]]";

namespace Core;
class Core.IndexWith {}
// CHECK:STDERR: fail_wrong_index_with.carbon:[[@LINE+4]]:10: error: type `Core.IntLiteral` does not support indexing [TypeNotIndexable]
// CHECK:STDERR: fn F() { 0[1]; }
// CHECK:STDERR:          ^~~~
// CHECK:STDERR:
fn F() { 0[1]; }

// --- fail_missing_index_with.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_missing_index_with.carbon:[[@LINE+7]]:10: error: package `Core` implicitly referenced here, but not found [CoreNotFound]
// CHECK:STDERR: fn F() { 0[1]; }
// CHECK:STDERR:          ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_index_with.carbon:[[@LINE+3]]:10: error: type `Core.IntLiteral` does not support indexing [TypeNotIndexable]
// CHECK:STDERR: fn F() { 0[1]; }
// CHECK:STDERR:          ^~~~
fn F() { 0[1]; }

// --- wrong_arg_count.carbon

library "[[@TEST_NAME]]";

namespace Core;
interface Core.IndexWith(SubscriptType:! type) {
  fn At[self: Self](subscript: SubscriptType) -> ();
}

impl () as Core.IndexWith(()) {
  fn At[self: Self](subscript: ()) -> () {
    return ();
  }
}

fn F() { ()[()]; }

// CHECK:STDOUT: --- fail_wrong_index_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IndexWith: type = class_type @IndexWith [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IndexWith = %IndexWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.decl: type = class_decl @IndexWith [template = constants.%IndexWith] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @IndexWith {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%IndexWith
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_10: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_12: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_index_with.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_10: Core.IntLiteral = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_12: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- wrong_arg_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %SubscriptType.patt: type = symbolic_binding_pattern SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.1: type = generic_interface_type @IndexWith [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %IndexWith: %IndexWith.type.1 = struct_value () [template]
// CHECK:STDOUT:   %IndexWith.type.2: type = facet_type <@IndexWith, @IndexWith(%SubscriptType)> [symbolic]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %At.type.1: type = fn_type @At.1, @IndexWith(%SubscriptType) [symbolic]
// CHECK:STDOUT:   %At.1: %At.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %IndexWith.type.2, %At.type.1 [symbolic]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @IndexWith.%At.decl [symbolic]
// CHECK:STDOUT:   %IndexWith.type.3: type = facet_type <@IndexWith, @IndexWith(%empty_tuple.type)> [template]
// CHECK:STDOUT:   %At.type.2: type = fn_type @At.2 [template]
// CHECK:STDOUT:   %At.2: %At.type.2 = struct_value () [template]
// CHECK:STDOUT:   %At.type.3: type = fn_type @At.1, @IndexWith(%empty_tuple.type) [template]
// CHECK:STDOUT:   %At.3: %At.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %IndexWith.type.3, %At.type.3 [template]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, @IndexWith.%At.decl [template]
// CHECK:STDOUT:   %.6: %IndexWith.type.2 = facet_value %empty_tuple.type, %empty_tuple.type [symbolic]
// CHECK:STDOUT:   %.7: <witness> = interface_witness (%At.2) [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.8: <bound method> = bound_method %empty_tuple, %At.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IndexWith = %IndexWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.decl: %IndexWith.type.1 = interface_decl @IndexWith [template = constants.%IndexWith] {
// CHECK:STDOUT:     %SubscriptType.patt.loc5_26.1: type = symbolic_binding_pattern SubscriptType, 0 [symbolic = %SubscriptType.patt.loc5_26.2 (constants.%SubscriptType.patt)]
// CHECK:STDOUT:     %SubscriptType.param_patt: type = value_param_pattern %SubscriptType.patt.loc5_26.1, runtime_param<invalid> [symbolic = %SubscriptType.patt.loc5_26.2 (constants.%SubscriptType.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SubscriptType.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %SubscriptType.loc5_26.1: type = bind_symbolic_name SubscriptType, 0, %SubscriptType.param [symbolic = %SubscriptType.loc5_26.2 (constants.%SubscriptType)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc9_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_7.2: type = converted %.loc9_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, file.%Core [template = file.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.1 = name_ref IndexWith, file.%IndexWith.decl [template = constants.%IndexWith]
// CHECK:STDOUT:     %.loc9_28: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_26: type = converted %.loc9_28, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%empty_tuple.type)> [template = constants.%IndexWith.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(%SubscriptType.loc5_26.1: type) {
// CHECK:STDOUT:   %SubscriptType.loc5_26.2: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType.loc5_26.2 (constants.%SubscriptType)]
// CHECK:STDOUT:   %SubscriptType.patt.loc5_26.2: type = symbolic_binding_pattern SubscriptType, 0 [symbolic = %SubscriptType.patt.loc5_26.2 (constants.%SubscriptType.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.loc5_26.2)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self.2: %IndexWith.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %At.type: type = fn_type @At.1, @IndexWith(%SubscriptType.loc5_26.2) [symbolic = %At.type (constants.%At.type.1)]
// CHECK:STDOUT:   %At: @IndexWith.%At.type (%At.type.1) = struct_value () [symbolic = %At (constants.%At.1)]
// CHECK:STDOUT:   %.loc6_52.2: type = assoc_entity_type @IndexWith.%IndexWith.type (%IndexWith.type.2), @IndexWith.%At.type (%At.type.1) [symbolic = %.loc6_52.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc6_52.3: @IndexWith.%.loc6_52.2 (%.2) = assoc_entity element0, %At.decl [symbolic = %.loc6_52.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %At.decl: @IndexWith.%At.type (%At.type.1) = fn_decl @At.1 [symbolic = @IndexWith.%At (constants.%At.1)] {
// CHECK:STDOUT:       %self.patt: @At.1.%.loc6_15.1 (%.1) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @At.1.%.loc6_15.1 (%.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %subscript.patt: @At.1.%SubscriptType (%SubscriptType) = binding_pattern subscript
// CHECK:STDOUT:       %subscript.param_patt: @At.1.%SubscriptType (%SubscriptType) = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:       %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc6_15.2: @At.1.%IndexWith.type (%IndexWith.type.2) = specific_constant @IndexWith.%Self.1, @IndexWith(constants.%SubscriptType) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @At.1.%IndexWith.type (%IndexWith.type.2) = name_ref Self, %.loc6_15.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc6_15.3: type = facet_access_type %Self.ref [symbolic = %.loc6_15.1 (constants.%.1)]
// CHECK:STDOUT:       %.loc6_15.4: type = converted %Self.ref, %.loc6_15.3 [symbolic = %.loc6_15.1 (constants.%.1)]
// CHECK:STDOUT:       %SubscriptType.ref: type = name_ref SubscriptType, @IndexWith.%SubscriptType.loc5_26.1 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:       %.loc6_51.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc6_51.2: type = converted %.loc6_51.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:       %self.param: @At.1.%.loc6_15.1 (%.1) = value_param runtime_param0
// CHECK:STDOUT:       %self: @At.1.%.loc6_15.1 (%.1) = bind_name self, %self.param
// CHECK:STDOUT:       %subscript.param: @At.1.%SubscriptType (%SubscriptType) = value_param runtime_param1
// CHECK:STDOUT:       %subscript: @At.1.%SubscriptType (%SubscriptType) = bind_name subscript, %subscript.param
// CHECK:STDOUT:       %return.param: ref %empty_tuple.type = out_param runtime_param2
// CHECK:STDOUT:       %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc6_52.1: @IndexWith.%.loc6_52.2 (%.2) = assoc_entity element0, %At.decl [symbolic = %.loc6_52.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .At = %.loc6_52.1
// CHECK:STDOUT:     witness = (%At.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc9_7.2 as %IndexWith.type {
// CHECK:STDOUT:   %At.decl: %At.type.2 = fn_decl @At.2 [template = constants.%At.2] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %subscript.patt: %empty_tuple.type = binding_pattern subscript
// CHECK:STDOUT:     %subscript.param_patt: %empty_tuple.type = value_param_pattern %subscript.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%.loc9_7.2 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_33.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_33.2: type = converted %.loc10_33.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_40.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_40.2: type = converted %.loc10_40.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %subscript.param: %empty_tuple.type = value_param runtime_param1
// CHECK:STDOUT:     %subscript: %empty_tuple.type = bind_name subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_31: <witness> = interface_witness (%At.decl) [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %At.decl
// CHECK:STDOUT:   witness = %.loc9_31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @At.1(@IndexWith.%SubscriptType.loc5_26.1: type, @IndexWith.%Self.1: @IndexWith.%IndexWith.type (%IndexWith.type.2)) {
// CHECK:STDOUT:   %SubscriptType: type = bind_symbolic_name SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2)]
// CHECK:STDOUT:   %Self: %IndexWith.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %.loc6_15.1: type = facet_access_type %Self [symbolic = %.loc6_15.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @At.1.%.loc6_15.1 (%.1)](%subscript.param_patt: @At.1.%SubscriptType (%SubscriptType)) -> %empty_tuple.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @At.2[%self.param_patt: %empty_tuple.type](%subscript.param_patt: %empty_tuple.type) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_13: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_14: %empty_tuple.type = converted %.loc11_13, %empty_tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc11_14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_11.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc15_14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple.loc15_11: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_11.2: %empty_tuple.type = converted %.loc15_11.1, %empty_tuple.loc15_11 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc15_14: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_15.1: %empty_tuple.type = converted %.loc15_14, %empty_tuple.loc15_14 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_15.2: %At.type.3 = interface_witness_access constants.%.7, element0 [template = constants.%At.2]
// CHECK:STDOUT:   %.loc15_15.3: <bound method> = bound_method %.loc15_11.2, %.loc15_15.2 [template = constants.%.8]
// CHECK:STDOUT:   %At.call: init %empty_tuple.type = call %.loc15_15.3(%.loc15_11.2, %.loc15_15.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType) {
// CHECK:STDOUT:   %SubscriptType.loc5_26.2 => constants.%SubscriptType
// CHECK:STDOUT:   %SubscriptType.patt.loc5_26.2 => constants.%SubscriptType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(@At.1.%SubscriptType) {
// CHECK:STDOUT:   %SubscriptType.loc5_26.2 => constants.%SubscriptType
// CHECK:STDOUT:   %SubscriptType.patt.loc5_26.2 => constants.%SubscriptType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%SubscriptType, constants.%Self) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %.loc6_15.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(%SubscriptType.loc5_26.2) {
// CHECK:STDOUT:   %SubscriptType.loc5_26.2 => constants.%SubscriptType
// CHECK:STDOUT:   %SubscriptType.patt.loc5_26.2 => constants.%SubscriptType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %SubscriptType.loc5_26.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %SubscriptType.patt.loc5_26.2 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %At.type => constants.%At.type.3
// CHECK:STDOUT:   %At => constants.%At.3
// CHECK:STDOUT:   %.loc6_52.2 => constants.%.4
// CHECK:STDOUT:   %.loc6_52.3 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @At.1(constants.%empty_tuple.type, constants.%.6) {
// CHECK:STDOUT:   %SubscriptType => constants.%empty_tuple.type
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.3
// CHECK:STDOUT:   %Self => constants.%.6
// CHECK:STDOUT:   %.loc6_15.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:

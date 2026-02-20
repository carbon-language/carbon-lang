// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
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

// CHECK:STDERR: fail_missing_index_with.carbon:[[@LINE+4]]:10: error: name `Core.IndexWith` implicitly referenced here, but not found [CoreNameNotFound]
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

class C {}

impl C as Core.IndexWith((), ()) {
  fn At[unused self: Self](unused subscript: ()) -> () {
    return ();
  }
}

fn F() {
  let c: C = {};
  // CHECK:STDERR: fail_wrong_arg_count.carbon:[[@LINE+7]]:3: error: 1 argument passed to generic interface expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   c[()];
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: core_wrong_arg_count.carbon:4:1: note: calling generic interface declared here [InCallToEntity]
  // CHECK:STDERR: interface IndexWith(SubscriptType:! type, ElementType:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  c[()];
}

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
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//core_wrong_index_with
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: type = import_ref Core//core_wrong_index_with, IndexWith, loaded [concrete = constants.%IndexWith]
// CHECK:STDOUT:   %Core.import_ref.8f2: <witness> = import_ref Core//core_wrong_index_with, loc{{\d+_\d+}}, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Core.import_ref.958 = import_ref Core//core_wrong_index_with, inst{{[0-9A-F]+}} [no loc], unloaded
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
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.958
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = <poisoned>
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %IndexWith.type.a3a: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.a3a = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.type.2c3: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic]
// CHECK:STDOUT:   %Self: %IndexWith.type.2c3 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 2, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.5ae: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %SubscriptType [symbolic]
// CHECK:STDOUT:   %.615: Core.Form = init_form %ElementType, call_param2 [symbolic]
// CHECK:STDOUT:   %pattern_type.946: type = pattern_type %ElementType [symbolic]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type: type = fn_type @IndexWith.WithSelf.At, @IndexWith(%SubscriptType, %ElementType, %Self) [symbolic]
// CHECK:STDOUT:   %IndexWith.WithSelf.At: %IndexWith.WithSelf.At.type = struct_value () [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType, %ElementType) [symbolic]
// CHECK:STDOUT:   %assoc0: %IndexWith.assoc_type = assoc_entity element0, @IndexWith.%IndexWith.WithSelf.At.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IndexWith = %IndexWith.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.decl: %IndexWith.type.a3a = interface_decl @IndexWith [concrete = constants.%IndexWith.generic] {
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
// CHECK:STDOUT:   %IndexWith.type.loc4: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.loc4_21.1, %ElementType.loc4_43.1)> [symbolic = %IndexWith.type.loc4 (constants.%IndexWith.type.2c3)]
// CHECK:STDOUT:   %Self.loc4_63.2: @IndexWith.%IndexWith.type.loc4 (%IndexWith.type.2c3) = symbolic_binding Self, 2 [symbolic = %Self.loc4_63.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_63.1: @IndexWith.%IndexWith.type.loc4 (%IndexWith.type.2c3) = symbolic_binding Self, 2 [symbolic = %Self.loc4_63.2 (constants.%Self)]
// CHECK:STDOUT:     interface_with_self_decl @IndexWith [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %IndexWith.WithSelf.At.decl: @IndexWith.%IndexWith.WithSelf.At.type (%IndexWith.WithSelf.At.type) = fn_decl @IndexWith.WithSelf.At [symbolic = @IndexWith.%IndexWith.WithSelf.At (constants.%IndexWith.WithSelf.At)] {
// CHECK:STDOUT:       %self.patt: @IndexWith.WithSelf.At.%pattern_type.loc5_9 (%pattern_type.5ae) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @IndexWith.WithSelf.At.%pattern_type.loc5_9 (%pattern_type.5ae) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %subscript.patt: @IndexWith.WithSelf.At.%pattern_type.loc5_21 (%pattern_type.51d) = value_binding_pattern subscript [concrete]
// CHECK:STDOUT:       %subscript.param_patt: @IndexWith.WithSelf.At.%pattern_type.loc5_21 (%pattern_type.51d) = value_param_pattern %subscript.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @IndexWith.WithSelf.At.%pattern_type.loc5_47 (%pattern_type.946) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @IndexWith.WithSelf.At.%pattern_type.loc5_47 (%pattern_type.946) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %ElementType.ref: type = name_ref ElementType, @IndexWith.%ElementType.loc4_43.2 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:       %.loc5_50.2: Core.Form = init_form %ElementType.ref, call_param2 [symbolic = %.loc5_50.1 (constants.%.615)]
// CHECK:STDOUT:       %self.param: @IndexWith.WithSelf.At.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_15.1: type = splice_block %.loc5_15.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc5_15.2: @IndexWith.WithSelf.At.%IndexWith.type (%IndexWith.type.2c3) = specific_constant @IndexWith.%Self.loc4_63.1, @IndexWith(constants.%SubscriptType, constants.%ElementType) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @IndexWith.WithSelf.At.%IndexWith.type (%IndexWith.type.2c3) = name_ref Self, %.loc5_15.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc5_15.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @IndexWith.WithSelf.At.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %subscript.param: @IndexWith.WithSelf.At.%SubscriptType (%SubscriptType) = value_param call_param1
// CHECK:STDOUT:       %SubscriptType.ref: type = name_ref SubscriptType, @IndexWith.%SubscriptType.loc4_21.2 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:       %subscript: @IndexWith.WithSelf.At.%SubscriptType (%SubscriptType) = value_binding subscript, %subscript.param
// CHECK:STDOUT:       %return.param: ref @IndexWith.WithSelf.At.%ElementType (%ElementType) = out_param call_param2
// CHECK:STDOUT:       %return: ref @IndexWith.WithSelf.At.%ElementType (%ElementType) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_61.1: @IndexWith.%IndexWith.assoc_type (%IndexWith.assoc_type) = assoc_entity element0, %IndexWith.WithSelf.At.decl [symbolic = %assoc0.loc5_61.2 (constants.%assoc0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_63.1
// CHECK:STDOUT:     .SubscriptType = <poisoned>
// CHECK:STDOUT:     .ElementType = <poisoned>
// CHECK:STDOUT:     .SubscriptType = <poisoned>
// CHECK:STDOUT:     .ElementType = <poisoned>
// CHECK:STDOUT:     .At = %assoc0.loc5_61.1
// CHECK:STDOUT:     witness = (%IndexWith.WithSelf.At.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IndexWith.WithSelf.At(@IndexWith.%SubscriptType.loc4_21.2: type, @IndexWith.%ElementType.loc4_43.2: type, @IndexWith.%Self.loc4_63.1: @IndexWith.%IndexWith.type.loc4 (%IndexWith.type.2c3)) {
// CHECK:STDOUT:   %SubscriptType: type = symbolic_binding SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = symbolic_binding ElementType, 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.2c3)]
// CHECK:STDOUT:   %Self: @IndexWith.WithSelf.At.%IndexWith.type (%IndexWith.type.2c3) = symbolic_binding Self, 2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 2, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5_9: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc5_9 (constants.%pattern_type.5ae)]
// CHECK:STDOUT:   %pattern_type.loc5_21: type = pattern_type %SubscriptType [symbolic = %pattern_type.loc5_21 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %.loc5_50.1: Core.Form = init_form %ElementType, call_param2 [symbolic = %.loc5_50.1 (constants.%.615)]
// CHECK:STDOUT:   %pattern_type.loc5_47: type = pattern_type %ElementType [symbolic = %pattern_type.loc5_47 (constants.%pattern_type.946)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @IndexWith.WithSelf.At.%Self.binding.as_type (%Self.binding.as_type), %subscript.param: @IndexWith.WithSelf.At.%SubscriptType (%SubscriptType)) -> out %return.param: @IndexWith.WithSelf.At.%ElementType (%ElementType);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType, constants.%ElementType) {
// CHECK:STDOUT:   %SubscriptType.loc4_21.1 => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType.loc4_43.1 => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType, constants.%ElementType, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith.WithSelf.At(constants.%SubscriptType, constants.%ElementType, constants.%Self) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.2c3
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc5_9 => constants.%pattern_type.5ae
// CHECK:STDOUT:   %pattern_type.loc5_21 => constants.%pattern_type.51d
// CHECK:STDOUT:   %.loc5_50.1 => constants.%.615
// CHECK:STDOUT:   %pattern_type.loc5_47 => constants.%pattern_type.946
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_arg_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [concrete]
// CHECK:STDOUT:   %ElementType: type = symbolic_binding ElementType, 1 [symbolic]
// CHECK:STDOUT:   %SubscriptType: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.1e9: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic]
// CHECK:STDOUT:   %Self.b10: %IndexWith.type.1e9 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.a9f: type = fn_type @IndexWith.WithSelf.At, @IndexWith(%SubscriptType, %ElementType, %Self.b10) [symbolic]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.a38: %IndexWith.WithSelf.At.type.a9f = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.946: type = pattern_type %ElementType [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.79f: type = symbolic_binding_type Self, 2, %Self.b10 [symbolic]
// CHECK:STDOUT:   %pattern_type.994: type = pattern_type %Self.binding.as_type.79f [symbolic]
// CHECK:STDOUT:   %.615: Core.Form = init_form %ElementType, call_param2 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %SubscriptType [symbolic]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %IndexWith.type.483: type = facet_type <@IndexWith, @IndexWith(%empty_tuple.type, %empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness @C.as.IndexWith.impl.%IndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.f44: %IndexWith.type.483 = symbolic_binding Self, 2 [symbolic]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.9fb: type = fn_type @IndexWith.WithSelf.At, @IndexWith(%empty_tuple.type, %empty_tuple.type, %Self.b10) [symbolic]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.0b7: %IndexWith.WithSelf.At.type.9fb = struct_value () [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type.d10: type = assoc_entity_type @IndexWith, @IndexWith(%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %assoc0.6a8: %IndexWith.assoc_type.d10 = assoc_entity element0, imports.%Core.import_ref.d74 [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %.067: Core.Form = init_form %empty_tuple.type, call_param2 [concrete]
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.type: type = fn_type @C.as.IndexWith.impl.At [concrete]
// CHECK:STDOUT:   %C.as.IndexWith.impl.At: %C.as.IndexWith.impl.At.type = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.483 = facet_value %C, (%IndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.10d: type = fn_type @IndexWith.WithSelf.At, @IndexWith(%empty_tuple.type, %empty_tuple.type, %IndexWith.facet) [concrete]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.6ef: %IndexWith.WithSelf.At.type.10d = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//core_wrong_arg_count
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.504 = import_ref Core//core_wrong_arg_count, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.1c4 = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.At: @IndexWith.%IndexWith.WithSelf.At.type (%IndexWith.WithSelf.At.type.a9f) = import_ref Core//core_wrong_arg_count, At, loaded [symbolic = @IndexWith.%IndexWith.WithSelf.At (constants.%IndexWith.WithSelf.At.a38)]
// CHECK:STDOUT:   %Core.import_ref.d74 = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.2: type = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.%SubscriptType.1 (constants.%SubscriptType)]
// CHECK:STDOUT:   %Core.import_ref.8e8b42.2: type = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.%ElementType.1 (constants.%ElementType)]
// CHECK:STDOUT:   %Core.import_ref.83171e.2: @IndexWith.%IndexWith.type.1 (%IndexWith.type.1e9) = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.%Self.1 (constants.%Self.b10)]
// CHECK:STDOUT:   %Core.import_ref.60d = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.3: type = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.%SubscriptType.1 (constants.%SubscriptType)]
// CHECK:STDOUT:   %Core.import_ref.8e8b42.3: type = import_ref Core//core_wrong_arg_count, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.%ElementType.1 (constants.%ElementType)]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.IndexWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %.loc8_27: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_31: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_32.1: type = converted %.loc8_27, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_32.2: type = converted %.loc8_31, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%empty_tuple.type, constants.%empty_tuple.type)> [concrete = constants.%IndexWith.type.483]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @IndexWith(imports.%Core.import_ref.b3bc94.3: type, imports.%Core.import_ref.8e8b42.3: type) [from "core_wrong_arg_count.carbon"] {
// CHECK:STDOUT:   %SubscriptType.1: type = symbolic_binding SubscriptType, 0 [symbolic = %SubscriptType.1 (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType.1: type = symbolic_binding ElementType, 1 [symbolic = %ElementType.1 (constants.%ElementType)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type.1: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.1, %ElementType.1)> [symbolic = %IndexWith.type.1 (constants.%IndexWith.type.1e9)]
// CHECK:STDOUT:   %Self.1: @IndexWith.%IndexWith.type.1 (%IndexWith.type.1e9) = symbolic_binding Self, 2 [symbolic = %Self.1 (constants.%Self.b10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.60d
// CHECK:STDOUT:     .At = imports.%Core.import_ref.1c4
// CHECK:STDOUT:     witness = (imports.%Core.At)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndexWith.impl: %C.ref as %IndexWith.type {
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.decl: %C.as.IndexWith.impl.At.type = fn_decl @C.as.IndexWith.impl.At [concrete = constants.%C.as.IndexWith.impl.At] {
// CHECK:STDOUT:     %self.patt: %pattern_type.7c7 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c7 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.cb1 = value_binding_pattern subscript [concrete]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.cb1 = value_param_pattern %subscript.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_54.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_54.2: type = converted %.loc9_54.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_54.3: Core.Form = init_form %.loc9_54.2, call_param2 [concrete = constants.%.067]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.IndexWith.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: %empty_tuple.type = value_param call_param1
// CHECK:STDOUT:     %.loc9_47.1: type = splice_block %.loc9_47.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc9_47.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc9_47.3: type = converted %.loc9_47.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %subscript: %empty_tuple.type = value_binding subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%C.as.IndexWith.impl.At.decl), @C.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness %IndexWith.impl_witness_table [concrete = constants.%IndexWith.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .At = %C.as.IndexWith.impl.At.decl
// CHECK:STDOUT:   witness = %IndexWith.impl_witness
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
// CHECK:STDOUT: generic fn @IndexWith.WithSelf.At(imports.%Core.import_ref.b3bc94.2: type, imports.%Core.import_ref.8e8b42.2: type, imports.%Core.import_ref.83171e.2: @IndexWith.%IndexWith.type.1 (%IndexWith.type.1e9)) [from "core_wrong_arg_count.carbon"] {
// CHECK:STDOUT:   %SubscriptType: type = symbolic_binding SubscriptType, 0 [symbolic = %SubscriptType (constants.%SubscriptType)]
// CHECK:STDOUT:   %ElementType: type = symbolic_binding ElementType, 1 [symbolic = %ElementType (constants.%ElementType)]
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType, %ElementType)> [symbolic = %IndexWith.type (constants.%IndexWith.type.1e9)]
// CHECK:STDOUT:   %Self: @IndexWith.WithSelf.At.%IndexWith.type (%IndexWith.type.1e9) = symbolic_binding Self, 2 [symbolic = %Self (constants.%Self.b10)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 2, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.79f)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.994)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %SubscriptType [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %ElementType, call_param2 [symbolic = %.1 (constants.%.615)]
// CHECK:STDOUT:   %pattern_type.3: type = pattern_type %ElementType [symbolic = %pattern_type.3 (constants.%pattern_type.946)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.IndexWith.impl.At(%self.param: %C, %subscript.param: %empty_tuple.type) -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_13.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_14: init %empty_tuple.type = converted %.loc10_13.1, %.loc10_13.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc10_14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_15.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc15_15.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_15.3: init %C to %.loc15_15.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc15_15.4: ref %C = temporary %.loc15_15.2, %.loc15_15.3
// CHECK:STDOUT:   %.loc15_15.5: ref %C = converted %.loc15_15.1, %.loc15_15.4
// CHECK:STDOUT:   %.loc15_15.6: %C = acquire_value %.loc15_15.5
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc15_15.6
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc23: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc15_15.4, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc15_15.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType, constants.%ElementType) {
// CHECK:STDOUT:   %SubscriptType.1 => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType.1 => constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%SubscriptType, constants.%ElementType, constants.%Self.b10) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith.WithSelf.At(constants.%SubscriptType, constants.%ElementType, constants.%Self.b10) {
// CHECK:STDOUT:   %SubscriptType => constants.%SubscriptType
// CHECK:STDOUT:   %ElementType => constants.%ElementType
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.1e9
// CHECK:STDOUT:   %Self => constants.%Self.b10
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.79f
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.994
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT:   %.1 => constants.%.615
// CHECK:STDOUT:   %pattern_type.3 => constants.%pattern_type.946
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%empty_tuple.type, constants.%empty_tuple.type) {
// CHECK:STDOUT:   %SubscriptType.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %ElementType.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %IndexWith.type.1 => constants.%IndexWith.type.483
// CHECK:STDOUT:   %Self.1 => constants.%Self.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%empty_tuple.type, constants.%empty_tuple.type, constants.%Self.b10) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %SubscriptType.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %ElementType.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %IndexWith.type.2 => constants.%IndexWith.type.483
// CHECK:STDOUT:   %Self.2 => constants.%Self.b10
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type => constants.%IndexWith.WithSelf.At.type.9fb
// CHECK:STDOUT:   %IndexWith.WithSelf.At => constants.%IndexWith.WithSelf.At.0b7
// CHECK:STDOUT:   %IndexWith.assoc_type => constants.%IndexWith.assoc_type.d10
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith(constants.%empty_tuple.type, constants.%empty_tuple.type, constants.%IndexWith.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %SubscriptType.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %ElementType.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %IndexWith.type.2 => constants.%IndexWith.type.483
// CHECK:STDOUT:   %Self.2 => constants.%IndexWith.facet
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type => constants.%IndexWith.WithSelf.At.type.10d
// CHECK:STDOUT:   %IndexWith.WithSelf.At => constants.%IndexWith.WithSelf.At.6ef
// CHECK:STDOUT:   %IndexWith.assoc_type => constants.%IndexWith.assoc_type.d10
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IndexWith.WithSelf.At(constants.%empty_tuple.type, constants.%empty_tuple.type, constants.%IndexWith.facet) {
// CHECK:STDOUT:   %SubscriptType => constants.%empty_tuple.type
// CHECK:STDOUT:   %ElementType => constants.%empty_tuple.type
// CHECK:STDOUT:   %IndexWith.type => constants.%IndexWith.type.483
// CHECK:STDOUT:   %Self => constants.%IndexWith.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.7c7
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.cb1
// CHECK:STDOUT:   %.1 => constants.%.067
// CHECK:STDOUT:   %pattern_type.3 => constants.%pattern_type.cb1
// CHECK:STDOUT: }
// CHECK:STDOUT:

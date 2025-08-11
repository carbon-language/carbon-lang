// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/init.carbon

// --- from_struct.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  var k: T;
}

fn InitFromStructGeneric(T:! type, x: T) -> T {
  var v: Class(T) = {.k = x};
  return v.k;
}

fn InitFromStructSpecific(x: i32) -> i32 {
  var v: Class(i32) = {.k = x};
  return v.k;
}

// --- adapt.carbon

library "[[@TEST_NAME]]";

class Adapt(T:! type) {
  adapt T;
}

fn InitFromAdaptedGeneric(T:! type, x: T) -> T {
  return (x as Adapt(T)) as T;
}

fn InitFromAdaptedSpecific(x: i32) -> i32 {
  return (x as Adapt(i32)) as i32;
}

// CHECK:STDOUT: --- from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem.e26: type = unbound_element_type %Class.fe1, %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.95a: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type.acd: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.5c0: %Class.as.Destroy.impl.Op.type.acd = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet.fdf: %Destroy.type = facet_value %Class.fe1, (%Destroy.impl_witness.95a) [symbolic]
// CHECK:STDOUT:   %struct_type.k.b21: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.b9e: <witness> = complete_type_witness %struct_type.k.b21 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %InitFromStructGeneric.type: type = fn_type @InitFromStructGeneric [concrete]
// CHECK:STDOUT:   %InitFromStructGeneric: %InitFromStructGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %pattern_type.3c1: type = pattern_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %.02c: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.fdf [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.specific_fn.841: <specific function> = specific_function %Class.as.Destroy.impl.Op.5c0, @Class.as.Destroy.impl.Op(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.2ae: <witness> = require_complete_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific.type: type = fn_type @InitFromStructSpecific [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific: %InitFromStructSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Class.247: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %Class.elem.2d8: type = unbound_element_type %Class.247, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.k.0bf: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.954: <witness> = complete_type_witness %struct_type.k.0bf [concrete]
// CHECK:STDOUT:   %pattern_type.0fa: type = pattern_type %Class.247 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.9fc: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type.9d9: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.e3d: %Class.as.Destroy.impl.Op.type.9d9 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f7c: type = ptr_type %Class.247 [concrete]
// CHECK:STDOUT:   %pattern_type.14a: type = pattern_type %ptr.f7c [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.specific_fn.19c: <specific function> = specific_function %Class.as.Destroy.impl.Op.e3d, @Class.as.Destroy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .InitFromStructGeneric = %InitFromStructGeneric.decl
// CHECK:STDOUT:     .InitFromStructSpecific = %InitFromStructSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructGeneric.decl: %InitFromStructGeneric.type = fn_decl @InitFromStructGeneric [concrete = constants.%InitFromStructGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromStructGeneric.%pattern_type.loc8 (%pattern_type.7dcd0a.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromStructGeneric.%pattern_type.loc8 (%pattern_type.7dcd0a.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromStructGeneric.%pattern_type.loc8 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromStructGeneric.%pattern_type.loc8 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_45: type = name_ref T, %T.loc8_26.2 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_26.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @InitFromStructGeneric.%T.loc8_26.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc8_39: type = name_ref T, %T.loc8_26.2 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:     %x: @InitFromStructGeneric.%T.loc8_26.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromStructGeneric.%T.loc8_26.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromStructGeneric.%T.loc8_26.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructSpecific.decl: %InitFromStructSpecific.type = fn_decl @InitFromStructSpecific [concrete = constants.%InitFromStructSpecific] {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_38: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_38: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %i32.loc13_30 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %i32 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Class.as.Destroy.impl(@Class.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type.acd)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type.acd) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op.5c0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Class.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type.acd) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op.5c0)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc4_23.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class.fe1)] {
// CHECK:STDOUT:         %.loc4_23.3: type = specific_constant constants.%Class.fe1, @Class(constants.%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_23.3 [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_13.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc4_13.1) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc4_13.1 [symbolic = %Class.elem (constants.%Class.elem.e26)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @Class.%T.loc4_13.1 (%T)} [symbolic = %struct_type.k (constants.%struct_type.k.b21)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.k [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.b9e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_13.2 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @Class.%Class.elem (%Class.elem.e26) = field_decl k, element0 [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class.fe1 [symbolic = @Class.as.Destroy.impl.%Class (constants.%Class.fe1)]
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%T) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.k.b21 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.b9e)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .k = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.9e0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc8_26.2: type) {
// CHECK:STDOUT:   %T.loc8_26.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %T.loc8_26.1 [symbolic = %pattern_type.loc8 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %T.loc8_26.1 [symbolic = %require_complete.loc8 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class.loc9_17.2: type = class_type @Class, @Class(%T.loc8_26.1) [symbolic = %Class.loc9_17.2 (constants.%Class.fe1)]
// CHECK:STDOUT:   %require_complete.loc9_17: <witness> = require_complete_type %Class.loc9_17.2 [symbolic = %require_complete.loc9_17 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %pattern_type.loc9: type = pattern_type %Class.loc9_17.2 [symbolic = %pattern_type.loc9 (constants.%pattern_type.3c1)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @InitFromStructGeneric.%T.loc8_26.1 (%T)} [symbolic = %struct_type.k (constants.%struct_type.k.b21)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc9_17.2, %T.loc8_26.1 [symbolic = %Class.elem (constants.%Class.elem.e26)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T.loc8_26.1) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class.loc9_17.2, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.fdf)]
// CHECK:STDOUT:   %.loc9_3.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc9_3.2 (constants.%.02c)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T.loc8_26.1) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type.acd)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @InitFromStructGeneric.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type.acd) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op.5c0)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Class.as.Destroy.impl.Op, @Class.as.Destroy.impl.Op(%T.loc8_26.1) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn.841)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class.loc9_17.2 [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %require_complete.loc9_3: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc9_3 (constants.%require_complete.2ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromStructGeneric.%T.loc8_26.1 (%T)) -> @InitFromStructGeneric.%T.loc8_26.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.3c1) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.3c1) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class.loc9_17.2 (%Class.fe1) = var %v.var_patt
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.loc8_26.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc9_28.1: @InitFromStructGeneric.%struct_type.k (%struct_type.k.b21) = struct_literal (%x.ref)
// CHECK:STDOUT:     %.loc9_28.2: ref @InitFromStructGeneric.%T.loc8_26.1 (%T) = class_element_access %v.var, element0
// CHECK:STDOUT:     %.loc9_28.3: init @InitFromStructGeneric.%T.loc8_26.1 (%T) = initialize_from %x.ref to %.loc9_28.2
// CHECK:STDOUT:     %.loc9_28.4: init @InitFromStructGeneric.%Class.loc9_17.2 (%Class.fe1) = class_init (%.loc9_28.3), %v.var
// CHECK:STDOUT:     %.loc9_3.1: init @InitFromStructGeneric.%Class.loc9_17.2 (%Class.fe1) = converted %.loc9_28.1, %.loc9_28.4
// CHECK:STDOUT:     assign %v.var, %.loc9_3.1
// CHECK:STDOUT:     %.loc9_17: type = splice_block %Class.loc9_17.1 [symbolic = %Class.loc9_17.2 (constants.%Class.fe1)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc9: type = name_ref T, %T.loc8_26.2 [symbolic = %T.loc8_26.1 (constants.%T)]
// CHECK:STDOUT:       %Class.loc9_17.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc9_17.2 (constants.%Class.fe1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class.loc9_17.2 (%Class.fe1) = bind_name v, %v.var
// CHECK:STDOUT:     %v.ref: ref @InitFromStructGeneric.%Class.loc9_17.2 (%Class.fe1) = name_ref v, %v
// CHECK:STDOUT:     %k.ref: @InitFromStructGeneric.%Class.elem (%Class.elem.e26) = name_ref k, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc10_11.1: ref @InitFromStructGeneric.%T.loc8_26.1 (%T) = class_element_access %v.ref, element0
// CHECK:STDOUT:     %.loc10_11.2: @InitFromStructGeneric.%T.loc8_26.1 (%T) = bind_value %.loc10_11.1
// CHECK:STDOUT:     %impl.elem0: @InitFromStructGeneric.%.loc9_3.2 (%.02c) = impl_witness_access constants.%Destroy.impl_witness.95a, element0 [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op.5c0)]
// CHECK:STDOUT:     %bound_method.loc9_3.1: <bound method> = bound_method %v.var, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Class.as.Destroy.impl.Op(constants.%T) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn.841)]
// CHECK:STDOUT:     %bound_method.loc9_3.2: <bound method> = bound_method %v.var, %specific_fn
// CHECK:STDOUT:     %addr: @InitFromStructGeneric.%ptr (%ptr.955) = addr_of %v.var
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_3.2(%addr)
// CHECK:STDOUT:     return %.loc10_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific(%x.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.0fa = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0fa = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Class.247 = var %v.var_patt
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc14_30.1: %struct_type.k.0bf = struct_literal (%x.ref)
// CHECK:STDOUT:   %.loc14_30.2: ref %i32 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc14_30.3: init %i32 = initialize_from %x.ref to %.loc14_30.2
// CHECK:STDOUT:   %.loc14_30.4: init %Class.247 = class_init (%.loc14_30.3), %v.var
// CHECK:STDOUT:   %.loc14_3: init %Class.247 = converted %.loc14_30.1, %.loc14_30.4
// CHECK:STDOUT:   assign %v.var, %.loc14_3
// CHECK:STDOUT:   %.loc14_19: type = splice_block %Class [concrete = constants.%Class.247] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.247]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Class.247 = bind_name v, %v.var
// CHECK:STDOUT:   %v.ref: ref %Class.247 = name_ref v, %v
// CHECK:STDOUT:   %k.ref: %Class.elem.2d8 = name_ref k, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:   %.loc15_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc15_11.2: %i32 = bind_value %.loc15_11.1
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.bound: <bound method> = bound_method %v.var, constants.%Class.as.Destroy.impl.Op.e3d
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%Class.as.Destroy.impl.Op.e3d, @Class.as.Destroy.impl.Op(constants.%i32) [concrete = constants.%Class.as.Destroy.impl.Op.specific_fn.19c]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %v.var, %Class.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f7c = addr_of %v.var
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %.loc15_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.e26
// CHECK:STDOUT:   %struct_type.k => constants.%struct_type.k.b21
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.b9e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.95a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type => constants.%Class.as.Destroy.impl.Op.type.acd
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op => constants.%Class.as.Destroy.impl.Op.5c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %ptr => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromStructGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_26.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc8 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2d8
// CHECK:STDOUT:   %struct_type.k => constants.%struct_type.k.0bf
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.954
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.9fc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type => constants.%Class.as.Destroy.impl.Op.type.9d9
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op => constants.%Class.as.Destroy.impl.Op.e3d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %ptr => constants.%ptr.f7c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.14a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Adapt.type: type = generic_class_type @Adapt [concrete]
// CHECK:STDOUT:   %Adapt.generic: %Adapt.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapt.2e4: type = class_type @Adapt, @Adapt(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Adapt.%Destroy.impl_witness_table, @Adapt.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.42a: type = ptr_type %Adapt.2e4 [symbolic]
// CHECK:STDOUT:   %pattern_type.97d: type = pattern_type %ptr.42a [symbolic]
// CHECK:STDOUT:   %Adapt.as.Destroy.impl.Op.type: type = fn_type @Adapt.as.Destroy.impl.Op, @Adapt.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Adapt.as.Destroy.impl.Op: %Adapt.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %complete_type.f87: <witness> = complete_type_witness %T [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %InitFromAdaptedGeneric.type: type = fn_type @InitFromAdaptedGeneric [concrete]
// CHECK:STDOUT:   %InitFromAdaptedGeneric: %InitFromAdaptedGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.26c: <witness> = require_complete_type %Adapt.2e4 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific.type: type = fn_type @InitFromAdaptedSpecific [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific: %InitFromAdaptedSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Adapt.526: type = class_type @Adapt, @Adapt(%i32) [concrete]
// CHECK:STDOUT:   %complete_type.1eb: <witness> = complete_type_witness %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Adapt = %Adapt.decl
// CHECK:STDOUT:     .InitFromAdaptedGeneric = %InitFromAdaptedGeneric.decl
// CHECK:STDOUT:     .InitFromAdaptedSpecific = %InitFromAdaptedSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Adapt.decl: %Adapt.type = class_decl @Adapt [concrete = constants.%Adapt.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedGeneric.decl: %InitFromAdaptedGeneric.type = fn_decl @InitFromAdaptedGeneric [concrete = constants.%InitFromAdaptedGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.7dc) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.7dc) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_46: type = name_ref T, %T.loc8_27.2 [symbolic = %T.loc8_27.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_27.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_27.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @InitFromAdaptedGeneric.%T.loc8_27.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc8_40: type = name_ref T, %T.loc8_27.2 [symbolic = %T.loc8_27.1 (constants.%T)]
// CHECK:STDOUT:     %x: @InitFromAdaptedGeneric.%T.loc8_27.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromAdaptedGeneric.%T.loc8_27.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromAdaptedGeneric.%T.loc8_27.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedSpecific.decl: %InitFromAdaptedSpecific.type = fn_decl @InitFromAdaptedSpecific [concrete = constants.%InitFromAdaptedSpecific] {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_39: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_39: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %i32.loc12_31 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc12_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc12_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %i32 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Adapt.as.Destroy.impl(@Adapt.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(%T) [symbolic = %Adapt (constants.%Adapt.2e4)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Adapt.%Destroy.impl_witness_table, @Adapt.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Adapt.as.Destroy.impl.Op.type: type = fn_type @Adapt.as.Destroy.impl.Op, @Adapt.as.Destroy.impl(%T) [symbolic = %Adapt.as.Destroy.impl.Op.type (constants.%Adapt.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Adapt.as.Destroy.impl.Op: @Adapt.as.Destroy.impl.%Adapt.as.Destroy.impl.Op.type (%Adapt.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Adapt.as.Destroy.impl.Op (constants.%Adapt.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Adapt.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Adapt.as.Destroy.impl.Op.decl: @Adapt.as.Destroy.impl.%Adapt.as.Destroy.impl.Op.type (%Adapt.as.Destroy.impl.Op.type) = fn_decl @Adapt.as.Destroy.impl.Op [symbolic = @Adapt.as.Destroy.impl.%Adapt.as.Destroy.impl.Op (constants.%Adapt.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Adapt.as.Destroy.impl.Op.%pattern_type (%pattern_type.97d) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Adapt.as.Destroy.impl.Op.%pattern_type (%pattern_type.97d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Adapt.as.Destroy.impl.Op.%ptr (%ptr.42a) = value_param call_param0
// CHECK:STDOUT:       %.loc4_23.2: type = splice_block %Self.ref [symbolic = %Adapt (constants.%Adapt.2e4)] {
// CHECK:STDOUT:         %.loc4_23.3: type = specific_constant constants.%Adapt.2e4, @Adapt(constants.%T) [symbolic = %Adapt (constants.%Adapt.2e4)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_23.3 [symbolic = %Adapt (constants.%Adapt.2e4)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Adapt.as.Destroy.impl.Op.%ptr (%ptr.42a) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Adapt.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Adapt.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Adapt(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_13.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %T.loc4_13.1 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.f87)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_13.2 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     adapt_decl %T.ref [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Adapt.2e4 [symbolic = @Adapt.as.Destroy.impl.%Adapt (constants.%Adapt.2e4)]
// CHECK:STDOUT:     impl_decl @Adapt.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Adapt.as.Destroy.impl.%Adapt.as.Destroy.impl.Op.decl), @Adapt.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Adapt.as.Destroy.impl(constants.%T) [symbolic = @Adapt.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%T [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.f87)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Adapt.2e4
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Adapt.as.Destroy.impl.Op(@Adapt.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(%T) [symbolic = %Adapt (constants.%Adapt.2e4)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Adapt [symbolic = %ptr (constants.%ptr.42a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.97d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Adapt.as.Destroy.impl.Op.%ptr (%ptr.42a)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromAdaptedGeneric(%T.loc8_27.2: type) {
// CHECK:STDOUT:   %T.loc8_27.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_27.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc8_27.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %T.loc8_27.1 [symbolic = %require_complete.loc8 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Adapt.loc9_23.2: type = class_type @Adapt, @Adapt(%T.loc8_27.1) [symbolic = %Adapt.loc9_23.2 (constants.%Adapt.2e4)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %Adapt.loc9_23.2 [symbolic = %require_complete.loc9 (constants.%require_complete.26c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromAdaptedGeneric.%T.loc8_27.1 (%T)) -> @InitFromAdaptedGeneric.%T.loc8_27.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @InitFromAdaptedGeneric.%T.loc8_27.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:     %T.ref.loc9_22: type = name_ref T, %T.loc8_27.2 [symbolic = %T.loc8_27.1 (constants.%T)]
// CHECK:STDOUT:     %Adapt.loc9_23.1: type = class_type @Adapt, @Adapt(constants.%T) [symbolic = %Adapt.loc9_23.2 (constants.%Adapt.2e4)]
// CHECK:STDOUT:     %.loc9_13.1: @InitFromAdaptedGeneric.%Adapt.loc9_23.2 (%Adapt.2e4) = as_compatible %x.ref
// CHECK:STDOUT:     %.loc9_13.2: @InitFromAdaptedGeneric.%Adapt.loc9_23.2 (%Adapt.2e4) = converted %x.ref, %.loc9_13.1
// CHECK:STDOUT:     %T.ref.loc9_29: type = name_ref T, %T.loc8_27.2 [symbolic = %T.loc8_27.1 (constants.%T)]
// CHECK:STDOUT:     %.loc9_26.1: @InitFromAdaptedGeneric.%T.loc8_27.1 (%T) = as_compatible %.loc9_13.2
// CHECK:STDOUT:     %.loc9_26.2: @InitFromAdaptedGeneric.%T.loc8_27.1 (%T) = converted %.loc9_13.2, %.loc9_26.1
// CHECK:STDOUT:     return %.loc9_26.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromAdaptedSpecific(%x.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:   %int_32.loc13_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(constants.%i32) [concrete = constants.%Adapt.526]
// CHECK:STDOUT:   %.loc13_13.1: %Adapt.526 = as_compatible %x.ref
// CHECK:STDOUT:   %.loc13_13.2: %Adapt.526 = converted %x.ref, %.loc13_13.1
// CHECK:STDOUT:   %int_32.loc13_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc13_28.1: %i32 = as_compatible %.loc13_13.2
// CHECK:STDOUT:   %.loc13_28.2: %i32 = converted %.loc13_13.2, %.loc13_28.1
// CHECK:STDOUT:   return %.loc13_28.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapt(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.f87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapt.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Adapt => constants.%Adapt.2e4
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapt.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Adapt => constants.%Adapt.2e4
// CHECK:STDOUT:   %ptr => constants.%ptr.42a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.97d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromAdaptedGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc8_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Adapt(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.1eb
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_lookup.carbon

// --- member_access.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  var b: T;
}

class Derived(T:! type) {
  extend base: Base(T);
  var d: T;
}

fn AccessDerived[T:! type](x: Derived(T)) -> T {
  return x.d;
}

fn AccessBase[T:! type](x: Derived(T)) -> T {
  return x.b;
}

fn AccessConcrete(x: Derived(i32)) -> i32 {
  return x.b;
}

// --- fail_no_member.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  var b: T;
}

class Derived(T:! type) {
  extend base: Base(T);
  var d: T;
}

fn AccessMissingBase[T:! type](x: Base(T)) -> T {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+4]]:10: error: member name `nonesuch` not found in `Base(T)` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   return x.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return x.nonesuch;
}

fn AccessMissingDerived[T:! type](x: Derived(T)) -> T {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+4]]:10: error: member name `nonesuch` not found in `Derived(T)` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   return x.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return x.nonesuch;
}

fn AccessMissingConcrete(x: Derived(i32)) -> i32 {
  // CHECK:STDERR: fail_no_member.carbon:[[@LINE+4]]:10: error: member name `nonesuch` not found in `Derived(i32)` [MemberNameNotFoundInSpecificScope]
  // CHECK:STDERR:   return x.nonesuch;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return x.nonesuch;
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Base.elem.9af: type = unbound_element_type %Base.370, %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.9f8: <witness> = impl_witness @Base.%Destroy.impl_witness_table, @Base.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.b7c: type = ptr_type %Base.370 [symbolic]
// CHECK:STDOUT:   %pattern_type.8d4: type = pattern_type %ptr.b7c [symbolic]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.type: type = fn_type @Base.as.Destroy.impl.Op, @Base.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op: %Base.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.b.f69: type = struct_type {.b: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.eaf: <witness> = complete_type_witness %struct_type.b.f69 [symbolic]
// CHECK:STDOUT:   %Derived.type: type = generic_class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.generic: %Derived.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.85c: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.97d: <witness> = require_complete_type %Base.370 [symbolic]
// CHECK:STDOUT:   %Derived.elem.8b3: type = unbound_element_type %Derived.85c, %Base.370 [symbolic]
// CHECK:STDOUT:   %Derived.elem.6d2: type = unbound_element_type %Derived.85c, %T [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.fa7: <witness> = impl_witness @Derived.%Destroy.impl_witness_table, @Derived.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.178: type = ptr_type %Derived.85c [symbolic]
// CHECK:STDOUT:   %pattern_type.520: type = pattern_type %ptr.178 [symbolic]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.type: type = fn_type @Derived.as.Destroy.impl.Op, @Derived.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op: %Derived.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.base.d.37c: type = struct_type {.base: %Base.370, .d: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.8ad: <witness> = complete_type_witness %struct_type.base.d.37c [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %Derived.85c [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %AccessDerived.type: type = fn_type @AccessDerived [concrete]
// CHECK:STDOUT:   %AccessDerived: %AccessDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.5f4: <witness> = require_complete_type %Derived.85c [symbolic]
// CHECK:STDOUT:   %AccessBase.type: type = fn_type @AccessBase [concrete]
// CHECK:STDOUT:   %AccessBase: %AccessBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Derived.115: type = class_type @Derived, @Derived(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.9c5: type = pattern_type %Derived.115 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %AccessConcrete.type: type = fn_type @AccessConcrete [concrete]
// CHECK:STDOUT:   %AccessConcrete: %AccessConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Base.10a: type = class_type @Base, @Base(%i32) [concrete]
// CHECK:STDOUT:   %Base.elem.a98: type = unbound_element_type %Base.10a, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.b.0a3: type = struct_type {.b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ba8: <witness> = complete_type_witness %struct_type.b.0a3 [concrete]
// CHECK:STDOUT:   %Derived.elem.0c4: type = unbound_element_type %Derived.115, %Base.10a [concrete]
// CHECK:STDOUT:   %Derived.elem.b58: type = unbound_element_type %Derived.115, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.base.d.ffa: type = struct_type {.base: %Base.10a, .d: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.544: <witness> = complete_type_witness %struct_type.base.d.ffa [concrete]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .AccessDerived = %AccessDerived.decl
// CHECK:STDOUT:     .AccessBase = %AccessBase.decl
// CHECK:STDOUT:     .AccessConcrete = %AccessConcrete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: %Derived.type = class_decl @Derived [concrete = constants.%Derived.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessDerived.decl: %AccessDerived.type = fn_decl @AccessDerived [concrete = constants.%AccessDerived] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @AccessDerived.%pattern_type.loc13_28 (%pattern_type.423) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @AccessDerived.%pattern_type.loc13_28 (%pattern_type.423) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @AccessDerived.%pattern_type.loc13_43 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @AccessDerived.%pattern_type.loc13_43 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_46: type = name_ref T, %T.loc13_18.2 [symbolic = %T.loc13_18.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc13_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_18.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessDerived.%Derived.loc13_40.1 (%Derived.85c) = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %Derived.loc13_40.2 [symbolic = %Derived.loc13_40.1 (constants.%Derived.85c)] {
// CHECK:STDOUT:       %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived.generic]
// CHECK:STDOUT:       %T.ref.loc13_39: type = name_ref T, %T.loc13_18.2 [symbolic = %T.loc13_18.1 (constants.%T)]
// CHECK:STDOUT:       %Derived.loc13_40.2: type = class_type @Derived, @Derived(constants.%T) [symbolic = %Derived.loc13_40.1 (constants.%Derived.85c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @AccessDerived.%Derived.loc13_40.1 (%Derived.85c) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessDerived.%T.loc13_18.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @AccessDerived.%T.loc13_18.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBase.decl: %AccessBase.type = fn_decl @AccessBase [concrete = constants.%AccessBase] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @AccessBase.%pattern_type.loc17_25 (%pattern_type.423) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @AccessBase.%pattern_type.loc17_25 (%pattern_type.423) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @AccessBase.%pattern_type.loc17_40 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @AccessBase.%pattern_type.loc17_40 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc17_43: type = name_ref T, %T.loc17_15.2 [symbolic = %T.loc17_15.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc17_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_15.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessBase.%Derived.loc17_37.1 (%Derived.85c) = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %Derived.loc17_37.2 [symbolic = %Derived.loc17_37.1 (constants.%Derived.85c)] {
// CHECK:STDOUT:       %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived.generic]
// CHECK:STDOUT:       %T.ref.loc17_36: type = name_ref T, %T.loc17_15.2 [symbolic = %T.loc17_15.1 (constants.%T)]
// CHECK:STDOUT:       %Derived.loc17_37.2: type = class_type @Derived, @Derived(constants.%T) [symbolic = %Derived.loc17_37.1 (constants.%Derived.85c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @AccessBase.%Derived.loc17_37.1 (%Derived.85c) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessBase.%T.loc17_15.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @AccessBase.%T.loc17_15.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessConcrete.decl: %AccessConcrete.type = fn_decl @AccessConcrete [concrete = constants.%AccessConcrete] {
// CHECK:STDOUT:     %x.patt: %pattern_type.9c5 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.9c5 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc21_39: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21_39: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %x.param: %Derived.115 = value_param call_param0
// CHECK:STDOUT:     %.loc21: type = splice_block %Derived [concrete = constants.%Derived.115] {
// CHECK:STDOUT:       %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived.generic]
// CHECK:STDOUT:       %int_32.loc21_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc21_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Derived: type = class_type @Derived, @Derived(constants.%i32) [concrete = constants.%Derived.115]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %Derived.115 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Base.as.Destroy.impl(@Base.%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Base.%Destroy.impl_witness_table, @Base.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.9f8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.type: type = fn_type @Base.as.Destroy.impl.Op, @Base.as.Destroy.impl(%T) [symbolic = %Base.as.Destroy.impl.Op.type (constants.%Base.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op: @Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.type (%Base.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Base.as.Destroy.impl.Op (constants.%Base.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%Base.370 as constants.%Destroy.type {
// CHECK:STDOUT:     %Base.as.Destroy.impl.Op.decl: @Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.type (%Base.as.Destroy.impl.Op.type) = fn_decl @Base.as.Destroy.impl.Op [symbolic = @Base.as.Destroy.impl.%Base.as.Destroy.impl.Op (constants.%Base.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Base.as.Destroy.impl.Op.%pattern_type (%pattern_type.8d4) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Base.as.Destroy.impl.Op.%pattern_type (%pattern_type.8d4) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_27.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Base.as.Destroy.impl.Op.%ptr (%ptr.b7c) = value_param call_param0
// CHECK:STDOUT:       %.loc4_27.2: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.370)] {
// CHECK:STDOUT:         %.loc4_27.3: type = specific_constant constants.%Base.370, @Base(constants.%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_27.3 [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Base.as.Destroy.impl.Op.%ptr (%ptr.b7c) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Base.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Base.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Derived.as.Destroy.impl(@Derived.%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Derived.%Destroy.impl_witness_table, @Derived.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.fa7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.type: type = fn_type @Derived.as.Destroy.impl.Op, @Derived.as.Destroy.impl(%T) [symbolic = %Derived.as.Destroy.impl.Op.type (constants.%Derived.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op: @Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op.type (%Derived.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Derived.as.Destroy.impl.Op (constants.%Derived.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%Derived.85c as constants.%Destroy.type {
// CHECK:STDOUT:     %Derived.as.Destroy.impl.Op.decl: @Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op.type (%Derived.as.Destroy.impl.Op.type) = fn_decl @Derived.as.Destroy.impl.Op [symbolic = @Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op (constants.%Derived.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Derived.as.Destroy.impl.Op.%pattern_type (%pattern_type.520) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Derived.as.Destroy.impl.Op.%pattern_type (%pattern_type.520) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc8_25.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Derived.as.Destroy.impl.Op.%ptr (%ptr.178) = value_param call_param0
// CHECK:STDOUT:       %.loc8_25.2: type = splice_block %Self.ref [symbolic = %Derived (constants.%Derived.85c)] {
// CHECK:STDOUT:         %.loc8_25.3: type = specific_constant constants.%Derived.85c, @Derived(constants.%T) [symbolic = %Derived (constants.%Derived.85c)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc8_25.3 [symbolic = %Derived (constants.%Derived.85c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Derived.as.Destroy.impl.Op.%ptr (%ptr.178) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Derived.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Derived.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T.loc4_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_17.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc4_17.1) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.loc4_17.1 [symbolic = %Base.elem (constants.%Base.elem.9af)]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: @Base.%T.loc4_17.1 (%T)} [symbolic = %struct_type.b (constants.%struct_type.b.f69)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.b [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.eaf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_17.2 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @Base.%Base.elem (%Base.elem.9af) = field_decl b, element0 [concrete]
// CHECK:STDOUT:     impl_decl @Base.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.decl), @Base.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Base.as.Destroy.impl(constants.%T) [symbolic = @Base.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.9f8)]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.b.f69 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.eaf)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.370
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .b = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.loc8_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2: type = class_type @Base, @Base(%T.loc8_15.1) [symbolic = %Base.loc9_22.2 (constants.%Base.370)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %Base.loc9_22.2 [symbolic = %require_complete.loc9 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T.loc8_15.1) [symbolic = %Derived (constants.%Derived.85c)]
// CHECK:STDOUT:   %Derived.elem.loc9: type = unbound_element_type %Derived, %Base.loc9_22.2 [symbolic = %Derived.elem.loc9 (constants.%Derived.elem.8b3)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %T.loc8_15.1 [symbolic = %require_complete.loc10 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Derived.elem.loc10: type = unbound_element_type %Derived, %T.loc8_15.1 [symbolic = %Derived.elem.loc10 (constants.%Derived.elem.6d2)]
// CHECK:STDOUT:   %struct_type.base.d: type = struct_type {.base: @Derived.%Base.loc9_22.2 (%Base.370), .d: @Derived.%T.loc8_15.1 (%T)} [symbolic = %struct_type.base.d (constants.%struct_type.base.d.37c)]
// CHECK:STDOUT:   %complete_type.loc11_1.2: <witness> = complete_type_witness %struct_type.base.d [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.8ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %Base.loc9_22.1: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc9_22.2 (constants.%Base.370)]
// CHECK:STDOUT:     %.loc9: @Derived.%Derived.elem.loc9 (%Derived.elem.8b3) = base_decl %Base.loc9_22.1, element0 [concrete]
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %.loc10: @Derived.%Derived.elem.loc10 (%Derived.elem.6d2) = field_decl d, element1 [concrete]
// CHECK:STDOUT:     impl_decl @Derived.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op.decl), @Derived.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Derived.as.Destroy.impl(constants.%T) [symbolic = @Derived.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.fa7)]
// CHECK:STDOUT:     %complete_type.loc11_1.1: <witness> = complete_type_witness constants.%struct_type.base.d.37c [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.8ad)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc11_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived.85c
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     .d = %.loc10
// CHECK:STDOUT:     .b = <poisoned>
// CHECK:STDOUT:     extend %Base.loc9_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Base.as.Destroy.impl.Op(@Base.%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Base [symbolic = %ptr (constants.%ptr.b7c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.8d4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Base.as.Destroy.impl.Op.%ptr (%ptr.b7c)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Derived.as.Destroy.impl.Op(@Derived.%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic = %Derived (constants.%Derived.85c)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Derived [symbolic = %ptr (constants.%ptr.178)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.520)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Derived.as.Destroy.impl.Op.%ptr (%ptr.178)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessDerived(%T.loc13_18.2: type) {
// CHECK:STDOUT:   %T.loc13_18.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_18.1 (constants.%T)]
// CHECK:STDOUT:   %Derived.loc13_40.1: type = class_type @Derived, @Derived(%T.loc13_18.1) [symbolic = %Derived.loc13_40.1 (constants.%Derived.85c)]
// CHECK:STDOUT:   %pattern_type.loc13_28: type = pattern_type %Derived.loc13_40.1 [symbolic = %pattern_type.loc13_28 (constants.%pattern_type.423)]
// CHECK:STDOUT:   %pattern_type.loc13_43: type = pattern_type %T.loc13_18.1 [symbolic = %pattern_type.loc13_43 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %Derived.loc13_40.1 [symbolic = %require_complete.loc13 (constants.%require_complete.5f4)]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived.loc13_40.1, %T.loc13_18.1 [symbolic = %Derived.elem (constants.%Derived.elem.6d2)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.loc13_18.1 [symbolic = %require_complete.loc14 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AccessDerived.%Derived.loc13_40.1 (%Derived.85c)) -> @AccessDerived.%T.loc13_18.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessDerived.%Derived.loc13_40.1 (%Derived.85c) = name_ref x, %x
// CHECK:STDOUT:     %d.ref: @AccessDerived.%Derived.elem (%Derived.elem.6d2) = name_ref d, @Derived.%.loc10 [concrete = @Derived.%.loc10]
// CHECK:STDOUT:     %.loc14_11.1: ref @AccessDerived.%T.loc13_18.1 (%T) = class_element_access %x.ref, element1
// CHECK:STDOUT:     %.loc14_11.2: @AccessDerived.%T.loc13_18.1 (%T) = bind_value %.loc14_11.1
// CHECK:STDOUT:     return %.loc14_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessBase(%T.loc17_15.2: type) {
// CHECK:STDOUT:   %T.loc17_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_15.1 (constants.%T)]
// CHECK:STDOUT:   %Derived.loc17_37.1: type = class_type @Derived, @Derived(%T.loc17_15.1) [symbolic = %Derived.loc17_37.1 (constants.%Derived.85c)]
// CHECK:STDOUT:   %pattern_type.loc17_25: type = pattern_type %Derived.loc17_37.1 [symbolic = %pattern_type.loc17_25 (constants.%pattern_type.423)]
// CHECK:STDOUT:   %pattern_type.loc17_40: type = pattern_type %T.loc17_15.1 [symbolic = %pattern_type.loc17_40 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17: <witness> = require_complete_type %Derived.loc17_37.1 [symbolic = %require_complete.loc17 (constants.%require_complete.5f4)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc17_15.1) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %require_complete.loc18_11: <witness> = require_complete_type %Base [symbolic = %require_complete.loc18_11 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.loc17_15.1 [symbolic = %Base.elem (constants.%Base.elem.9af)]
// CHECK:STDOUT:   %require_complete.loc18_13: <witness> = require_complete_type %T.loc17_15.1 [symbolic = %require_complete.loc18_13 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AccessBase.%Derived.loc17_37.1 (%Derived.85c)) -> @AccessBase.%T.loc17_15.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessBase.%Derived.loc17_37.1 (%Derived.85c) = name_ref x, %x
// CHECK:STDOUT:     %b.ref: @AccessBase.%Base.elem (%Base.elem.9af) = name_ref b, @Base.%.loc5 [concrete = @Base.%.loc5]
// CHECK:STDOUT:     %.loc18_11.1: ref @AccessBase.%Base (%Base.370) = class_element_access %x.ref, element0
// CHECK:STDOUT:     %.loc18_11.2: ref @AccessBase.%Base (%Base.370) = converted %x.ref, %.loc18_11.1
// CHECK:STDOUT:     %.loc18_11.3: ref @AccessBase.%T.loc17_15.1 (%T) = class_element_access %.loc18_11.2, element0
// CHECK:STDOUT:     %.loc18_11.4: @AccessBase.%T.loc17_15.1 (%T) = bind_value %.loc18_11.3
// CHECK:STDOUT:     return %.loc18_11.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessConcrete(%x.param: %Derived.115) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Derived.115 = name_ref x, %x
// CHECK:STDOUT:   %b.ref: %Base.elem.a98 = name_ref b, @Base.%.loc5 [concrete = @Base.%.loc5]
// CHECK:STDOUT:   %.loc22_11.1: ref %Base.10a = class_element_access %x.ref, element0
// CHECK:STDOUT:   %.loc22_11.2: ref %Base.10a = converted %x.ref, %.loc22_11.1
// CHECK:STDOUT:   %.loc22_11.3: ref %i32 = class_element_access %.loc22_11.2, element0
// CHECK:STDOUT:   %.loc22_11.4: %i32 = bind_value %.loc22_11.3
// CHECK:STDOUT:   return %.loc22_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.9af
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.f69
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.eaf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.9f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %ptr => constants.%ptr.b7c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2 => constants.%Base.370
// CHECK:STDOUT:   %require_complete.loc9 => constants.%require_complete.97d
// CHECK:STDOUT:   %Derived => constants.%Derived.85c
// CHECK:STDOUT:   %Derived.elem.loc9 => constants.%Derived.elem.8b3
// CHECK:STDOUT:   %require_complete.loc10 => constants.%require_complete.4ae
// CHECK:STDOUT:   %Derived.elem.loc10 => constants.%Derived.elem.6d2
// CHECK:STDOUT:   %struct_type.base.d => constants.%struct_type.base.d.37c
// CHECK:STDOUT:   %complete_type.loc11_1.2 => constants.%complete_type.8ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.fa7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Derived => constants.%Derived.85c
// CHECK:STDOUT:   %ptr => constants.%ptr.178
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.520
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessDerived(constants.%T) {
// CHECK:STDOUT:   %T.loc13_18.1 => constants.%T
// CHECK:STDOUT:   %Derived.loc13_40.1 => constants.%Derived.85c
// CHECK:STDOUT:   %pattern_type.loc13_28 => constants.%pattern_type.423
// CHECK:STDOUT:   %pattern_type.loc13_43 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessBase(constants.%T) {
// CHECK:STDOUT:   %T.loc17_15.1 => constants.%T
// CHECK:STDOUT:   %Derived.loc17_37.1 => constants.%Derived.85c
// CHECK:STDOUT:   %pattern_type.loc17_25 => constants.%pattern_type.423
// CHECK:STDOUT:   %pattern_type.loc17_40 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%i32) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2 => constants.%Base.10a
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.ba8
// CHECK:STDOUT:   %Derived => constants.%Derived.115
// CHECK:STDOUT:   %Derived.elem.loc9 => constants.%Derived.elem.0c4
// CHECK:STDOUT:   %require_complete.loc10 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Derived.elem.loc10 => constants.%Derived.elem.b58
// CHECK:STDOUT:   %struct_type.base.d => constants.%struct_type.base.d.ffa
// CHECK:STDOUT:   %complete_type.loc11_1.2 => constants.%complete_type.544
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Base => constants.%Base.10a
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.a98
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.0a3
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.ba8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Base.elem.9af: type = unbound_element_type %Base.370, %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.9f8: <witness> = impl_witness @Base.%Destroy.impl_witness_table, @Base.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.b7c: type = ptr_type %Base.370 [symbolic]
// CHECK:STDOUT:   %pattern_type.8d4: type = pattern_type %ptr.b7c [symbolic]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.type: type = fn_type @Base.as.Destroy.impl.Op, @Base.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op: %Base.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.b.f69: type = struct_type {.b: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.eaf: <witness> = complete_type_witness %struct_type.b.f69 [symbolic]
// CHECK:STDOUT:   %Derived.type: type = generic_class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.generic: %Derived.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.85c: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.97d: <witness> = require_complete_type %Base.370 [symbolic]
// CHECK:STDOUT:   %Derived.elem.8b3: type = unbound_element_type %Derived.85c, %Base.370 [symbolic]
// CHECK:STDOUT:   %Derived.elem.6d2: type = unbound_element_type %Derived.85c, %T [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.fa7: <witness> = impl_witness @Derived.%Destroy.impl_witness_table, @Derived.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.178: type = ptr_type %Derived.85c [symbolic]
// CHECK:STDOUT:   %pattern_type.520: type = pattern_type %ptr.178 [symbolic]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.type: type = fn_type @Derived.as.Destroy.impl.Op, @Derived.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op: %Derived.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.base.d.37c: type = struct_type {.base: %Base.370, .d: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.8ad: <witness> = complete_type_witness %struct_type.base.d.37c [symbolic]
// CHECK:STDOUT:   %pattern_type.9f7: type = pattern_type %Base.370 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %AccessMissingBase.type: type = fn_type @AccessMissingBase [concrete]
// CHECK:STDOUT:   %AccessMissingBase: %AccessMissingBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %Derived.85c [symbolic]
// CHECK:STDOUT:   %AccessMissingDerived.type: type = fn_type @AccessMissingDerived [concrete]
// CHECK:STDOUT:   %AccessMissingDerived: %AccessMissingDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.5f4: <witness> = require_complete_type %Derived.85c [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Derived.115: type = class_type @Derived, @Derived(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.9c5: type = pattern_type %Derived.115 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete.type: type = fn_type @AccessMissingConcrete [concrete]
// CHECK:STDOUT:   %AccessMissingConcrete: %AccessMissingConcrete.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Base.10a: type = class_type @Base, @Base(%i32) [concrete]
// CHECK:STDOUT:   %Base.elem.a98: type = unbound_element_type %Base.10a, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.b.0a3: type = struct_type {.b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ba8: <witness> = complete_type_witness %struct_type.b.0a3 [concrete]
// CHECK:STDOUT:   %Derived.elem.0c4: type = unbound_element_type %Derived.115, %Base.10a [concrete]
// CHECK:STDOUT:   %Derived.elem.b58: type = unbound_element_type %Derived.115, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.base.d.ffa: type = struct_type {.base: %Base.10a, .d: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.544: <witness> = complete_type_witness %struct_type.base.d.ffa [concrete]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .AccessMissingBase = %AccessMissingBase.decl
// CHECK:STDOUT:     .AccessMissingDerived = %AccessMissingDerived.decl
// CHECK:STDOUT:     .AccessMissingConcrete = %AccessMissingConcrete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: %Derived.type = class_decl @Derived [concrete = constants.%Derived.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingBase.decl: %AccessMissingBase.type = fn_decl @AccessMissingBase [concrete = constants.%AccessMissingBase] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @AccessMissingBase.%pattern_type.loc13_32 (%pattern_type.9f7) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @AccessMissingBase.%pattern_type.loc13_32 (%pattern_type.9f7) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @AccessMissingBase.%pattern_type.loc13_44 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @AccessMissingBase.%pattern_type.loc13_44 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_47: type = name_ref T, %T.loc13_22.2 [symbolic = %T.loc13_22.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc13_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_22.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessMissingBase.%Base.loc13_41.1 (%Base.370) = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %Base.loc13_41.2 [symbolic = %Base.loc13_41.1 (constants.%Base.370)] {
// CHECK:STDOUT:       %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:       %T.ref.loc13_40: type = name_ref T, %T.loc13_22.2 [symbolic = %T.loc13_22.1 (constants.%T)]
// CHECK:STDOUT:       %Base.loc13_41.2: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc13_41.1 (constants.%Base.370)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @AccessMissingBase.%Base.loc13_41.1 (%Base.370) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessMissingBase.%T.loc13_22.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @AccessMissingBase.%T.loc13_22.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingDerived.decl: %AccessMissingDerived.type = fn_decl @AccessMissingDerived [concrete = constants.%AccessMissingDerived] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @AccessMissingDerived.%pattern_type.loc21_35 (%pattern_type.423) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @AccessMissingDerived.%pattern_type.loc21_35 (%pattern_type.423) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @AccessMissingDerived.%pattern_type.loc21_50 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @AccessMissingDerived.%pattern_type.loc21_50 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc21_53: type = name_ref T, %T.loc21_25.2 [symbolic = %T.loc21_25.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc21_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_25.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessMissingDerived.%Derived.loc21_47.1 (%Derived.85c) = value_param call_param0
// CHECK:STDOUT:     %.loc21: type = splice_block %Derived.loc21_47.2 [symbolic = %Derived.loc21_47.1 (constants.%Derived.85c)] {
// CHECK:STDOUT:       %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived.generic]
// CHECK:STDOUT:       %T.ref.loc21_46: type = name_ref T, %T.loc21_25.2 [symbolic = %T.loc21_25.1 (constants.%T)]
// CHECK:STDOUT:       %Derived.loc21_47.2: type = class_type @Derived, @Derived(constants.%T) [symbolic = %Derived.loc21_47.1 (constants.%Derived.85c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @AccessMissingDerived.%Derived.loc21_47.1 (%Derived.85c) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessMissingDerived.%T.loc21_25.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @AccessMissingDerived.%T.loc21_25.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingConcrete.decl: %AccessMissingConcrete.type = fn_decl @AccessMissingConcrete [concrete = constants.%AccessMissingConcrete] {
// CHECK:STDOUT:     %x.patt: %pattern_type.9c5 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.9c5 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc29_46: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc29_46: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %x.param: %Derived.115 = value_param call_param0
// CHECK:STDOUT:     %.loc29: type = splice_block %Derived [concrete = constants.%Derived.115] {
// CHECK:STDOUT:       %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived.generic]
// CHECK:STDOUT:       %int_32.loc29_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc29_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Derived: type = class_type @Derived, @Derived(constants.%i32) [concrete = constants.%Derived.115]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %Derived.115 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Base.as.Destroy.impl(@Base.%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Base.%Destroy.impl_witness_table, @Base.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.9f8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.type: type = fn_type @Base.as.Destroy.impl.Op, @Base.as.Destroy.impl(%T) [symbolic = %Base.as.Destroy.impl.Op.type (constants.%Base.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op: @Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.type (%Base.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Base.as.Destroy.impl.Op (constants.%Base.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%Base.370 as constants.%Destroy.type {
// CHECK:STDOUT:     %Base.as.Destroy.impl.Op.decl: @Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.type (%Base.as.Destroy.impl.Op.type) = fn_decl @Base.as.Destroy.impl.Op [symbolic = @Base.as.Destroy.impl.%Base.as.Destroy.impl.Op (constants.%Base.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Base.as.Destroy.impl.Op.%pattern_type (%pattern_type.8d4) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Base.as.Destroy.impl.Op.%pattern_type (%pattern_type.8d4) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_27.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Base.as.Destroy.impl.Op.%ptr (%ptr.b7c) = value_param call_param0
// CHECK:STDOUT:       %.loc4_27.2: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.370)] {
// CHECK:STDOUT:         %.loc4_27.3: type = specific_constant constants.%Base.370, @Base(constants.%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_27.3 [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Base.as.Destroy.impl.Op.%ptr (%ptr.b7c) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Base.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Base.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Derived.as.Destroy.impl(@Derived.%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Derived.%Destroy.impl_witness_table, @Derived.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.fa7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.type: type = fn_type @Derived.as.Destroy.impl.Op, @Derived.as.Destroy.impl(%T) [symbolic = %Derived.as.Destroy.impl.Op.type (constants.%Derived.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op: @Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op.type (%Derived.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Derived.as.Destroy.impl.Op (constants.%Derived.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%Derived.85c as constants.%Destroy.type {
// CHECK:STDOUT:     %Derived.as.Destroy.impl.Op.decl: @Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op.type (%Derived.as.Destroy.impl.Op.type) = fn_decl @Derived.as.Destroy.impl.Op [symbolic = @Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op (constants.%Derived.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Derived.as.Destroy.impl.Op.%pattern_type (%pattern_type.520) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Derived.as.Destroy.impl.Op.%pattern_type (%pattern_type.520) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc8_25.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Derived.as.Destroy.impl.Op.%ptr (%ptr.178) = value_param call_param0
// CHECK:STDOUT:       %.loc8_25.2: type = splice_block %Self.ref [symbolic = %Derived (constants.%Derived.85c)] {
// CHECK:STDOUT:         %.loc8_25.3: type = specific_constant constants.%Derived.85c, @Derived(constants.%T) [symbolic = %Derived (constants.%Derived.85c)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc8_25.3 [symbolic = %Derived (constants.%Derived.85c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Derived.as.Destroy.impl.Op.%ptr (%ptr.178) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Derived.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Derived.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T.loc4_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_17.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc4_17.1) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.loc4_17.1 [symbolic = %Base.elem (constants.%Base.elem.9af)]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: @Base.%T.loc4_17.1 (%T)} [symbolic = %struct_type.b (constants.%struct_type.b.f69)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.b [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.eaf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_17.2 [symbolic = %T.loc4_17.1 (constants.%T)]
// CHECK:STDOUT:     %.loc5: @Base.%Base.elem (%Base.elem.9af) = field_decl b, element0 [concrete]
// CHECK:STDOUT:     impl_decl @Base.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.decl), @Base.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Base.as.Destroy.impl(constants.%T) [symbolic = @Base.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.9f8)]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.b.f69 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.eaf)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.370
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .b = %.loc5
// CHECK:STDOUT:     .nonesuch = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T.loc8_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2: type = class_type @Base, @Base(%T.loc8_15.1) [symbolic = %Base.loc9_22.2 (constants.%Base.370)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %Base.loc9_22.2 [symbolic = %require_complete.loc9 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T.loc8_15.1) [symbolic = %Derived (constants.%Derived.85c)]
// CHECK:STDOUT:   %Derived.elem.loc9: type = unbound_element_type %Derived, %Base.loc9_22.2 [symbolic = %Derived.elem.loc9 (constants.%Derived.elem.8b3)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %T.loc8_15.1 [symbolic = %require_complete.loc10 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Derived.elem.loc10: type = unbound_element_type %Derived, %T.loc8_15.1 [symbolic = %Derived.elem.loc10 (constants.%Derived.elem.6d2)]
// CHECK:STDOUT:   %struct_type.base.d: type = struct_type {.base: @Derived.%Base.loc9_22.2 (%Base.370), .d: @Derived.%T.loc8_15.1 (%T)} [symbolic = %struct_type.base.d (constants.%struct_type.base.d.37c)]
// CHECK:STDOUT:   %complete_type.loc11_1.2: <witness> = complete_type_witness %struct_type.base.d [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.8ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %Base.loc9_22.1: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc9_22.2 (constants.%Base.370)]
// CHECK:STDOUT:     %.loc9: @Derived.%Derived.elem.loc9 (%Derived.elem.8b3) = base_decl %Base.loc9_22.1, element0 [concrete]
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc8_15.2 [symbolic = %T.loc8_15.1 (constants.%T)]
// CHECK:STDOUT:     %.loc10: @Derived.%Derived.elem.loc10 (%Derived.elem.6d2) = field_decl d, element1 [concrete]
// CHECK:STDOUT:     impl_decl @Derived.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op.decl), @Derived.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Derived.as.Destroy.impl(constants.%T) [symbolic = @Derived.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.fa7)]
// CHECK:STDOUT:     %complete_type.loc11_1.1: <witness> = complete_type_witness constants.%struct_type.base.d.37c [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.8ad)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc11_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived.85c
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     .d = %.loc10
// CHECK:STDOUT:     .nonesuch = <poisoned>
// CHECK:STDOUT:     extend %Base.loc9_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Base.as.Destroy.impl.Op(@Base.%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Base [symbolic = %ptr (constants.%ptr.b7c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.8d4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Base.as.Destroy.impl.Op.%ptr (%ptr.b7c)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Derived.as.Destroy.impl.Op(@Derived.%T.loc8_15.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic = %Derived (constants.%Derived.85c)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Derived [symbolic = %ptr (constants.%ptr.178)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.520)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Derived.as.Destroy.impl.Op.%ptr (%ptr.178)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingBase(%T.loc13_22.2: type) {
// CHECK:STDOUT:   %T.loc13_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_22.1 (constants.%T)]
// CHECK:STDOUT:   %Base.loc13_41.1: type = class_type @Base, @Base(%T.loc13_22.1) [symbolic = %Base.loc13_41.1 (constants.%Base.370)]
// CHECK:STDOUT:   %pattern_type.loc13_32: type = pattern_type %Base.loc13_41.1 [symbolic = %pattern_type.loc13_32 (constants.%pattern_type.9f7)]
// CHECK:STDOUT:   %pattern_type.loc13_44: type = pattern_type %T.loc13_22.1 [symbolic = %pattern_type.loc13_44 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base.loc13_41.1 [symbolic = %require_complete (constants.%require_complete.97d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AccessMissingBase.%Base.loc13_41.1 (%Base.370)) -> @AccessMissingBase.%T.loc13_22.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessMissingBase.%Base.loc13_41.1 (%Base.370) = name_ref x, %x
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingDerived(%T.loc21_25.2: type) {
// CHECK:STDOUT:   %T.loc21_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_25.1 (constants.%T)]
// CHECK:STDOUT:   %Derived.loc21_47.1: type = class_type @Derived, @Derived(%T.loc21_25.1) [symbolic = %Derived.loc21_47.1 (constants.%Derived.85c)]
// CHECK:STDOUT:   %pattern_type.loc21_35: type = pattern_type %Derived.loc21_47.1 [symbolic = %pattern_type.loc21_35 (constants.%pattern_type.423)]
// CHECK:STDOUT:   %pattern_type.loc21_50: type = pattern_type %T.loc21_25.1 [symbolic = %pattern_type.loc21_50 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21: <witness> = require_complete_type %Derived.loc21_47.1 [symbolic = %require_complete.loc21 (constants.%require_complete.5f4)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc21_25.1) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %require_complete.loc26: <witness> = require_complete_type %Base [symbolic = %require_complete.loc26 (constants.%require_complete.97d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AccessMissingDerived.%Derived.loc21_47.1 (%Derived.85c)) -> @AccessMissingDerived.%T.loc21_25.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessMissingDerived.%Derived.loc21_47.1 (%Derived.85c) = name_ref x, %x
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMissingConcrete(%x.param: %Derived.115) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Derived.115 = name_ref x, %x
// CHECK:STDOUT:   %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.9af
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.f69
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.eaf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.9f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %ptr => constants.%ptr.b7c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2 => constants.%Base.370
// CHECK:STDOUT:   %require_complete.loc9 => constants.%require_complete.97d
// CHECK:STDOUT:   %Derived => constants.%Derived.85c
// CHECK:STDOUT:   %Derived.elem.loc9 => constants.%Derived.elem.8b3
// CHECK:STDOUT:   %require_complete.loc10 => constants.%require_complete.4ae
// CHECK:STDOUT:   %Derived.elem.loc10 => constants.%Derived.elem.6d2
// CHECK:STDOUT:   %struct_type.base.d => constants.%struct_type.base.d.37c
// CHECK:STDOUT:   %complete_type.loc11_1.2 => constants.%complete_type.8ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.fa7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Derived => constants.%Derived.85c
// CHECK:STDOUT:   %ptr => constants.%ptr.178
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.520
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingBase(constants.%T) {
// CHECK:STDOUT:   %T.loc13_22.1 => constants.%T
// CHECK:STDOUT:   %Base.loc13_41.1 => constants.%Base.370
// CHECK:STDOUT:   %pattern_type.loc13_32 => constants.%pattern_type.9f7
// CHECK:STDOUT:   %pattern_type.loc13_44 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingDerived(constants.%T) {
// CHECK:STDOUT:   %T.loc21_25.1 => constants.%T
// CHECK:STDOUT:   %Derived.loc21_47.1 => constants.%Derived.85c
// CHECK:STDOUT:   %pattern_type.loc21_35 => constants.%pattern_type.423
// CHECK:STDOUT:   %pattern_type.loc21_50 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%i32) {
// CHECK:STDOUT:   %T.loc8_15.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2 => constants.%Base.10a
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.ba8
// CHECK:STDOUT:   %Derived => constants.%Derived.115
// CHECK:STDOUT:   %Derived.elem.loc9 => constants.%Derived.elem.0c4
// CHECK:STDOUT:   %require_complete.loc10 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Derived.elem.loc10 => constants.%Derived.elem.b58
// CHECK:STDOUT:   %struct_type.base.d => constants.%struct_type.base.d.ffa
// CHECK:STDOUT:   %complete_type.loc11_1.2 => constants.%complete_type.544
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Base => constants.%Base.10a
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.a98
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.0a3
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.ba8
// CHECK:STDOUT: }
// CHECK:STDOUT:

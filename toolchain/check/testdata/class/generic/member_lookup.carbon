// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Base.elem.9af: type = unbound_element_type %Base.370, %T [symbolic]
// CHECK:STDOUT:   %struct_type.b.f69: type = struct_type {.b: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.eaf: <witness> = complete_type_witness %struct_type.b.f69 [symbolic]
// CHECK:STDOUT:   %Derived.type: type = generic_class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.generic: %Derived.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.85c: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.97d: <witness> = require_complete_type %Base.370 [symbolic]
// CHECK:STDOUT:   %Derived.elem.8b3: type = unbound_element_type %Derived.85c, %Base.370 [symbolic]
// CHECK:STDOUT:   %Derived.elem.6d2: type = unbound_element_type %Derived.85c, %T [symbolic]
// CHECK:STDOUT:   %struct_type.base.d.37c: type = struct_type {.base: %Base.370, .d: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.8ad: <witness> = complete_type_witness %struct_type.base.d.37c [symbolic]
// CHECK:STDOUT:   %AccessDerived.type: type = fn_type @AccessDerived [concrete]
// CHECK:STDOUT:   %AccessDerived: %AccessDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.5f4: <witness> = require_complete_type %Derived.85c [symbolic]
// CHECK:STDOUT:   %AccessBase.type: type = fn_type @AccessBase [concrete]
// CHECK:STDOUT:   %AccessBase: %AccessBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Derived.115: type = class_type @Derived, @Derived(%i32) [concrete]
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
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc4_17.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: %Derived.type = class_decl @Derived [concrete = constants.%Derived.generic] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessDerived.decl: %AccessDerived.type = fn_decl @AccessDerived [concrete = constants.%AccessDerived] {
// CHECK:STDOUT:     %T.patt.loc13_18.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_18.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @AccessDerived.%Derived.loc13_40.2 (%Derived.85c) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @AccessDerived.%Derived.loc13_40.2 (%Derived.85c) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @AccessDerived.%T.loc13_18.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessDerived.%T.loc13_18.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_46: type = name_ref T, %T.loc13_18.1 [symbolic = %T.loc13_18.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc13_18.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_18.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessDerived.%Derived.loc13_40.2 (%Derived.85c) = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %Derived.loc13_40.1 [symbolic = %Derived.loc13_40.2 (constants.%Derived.85c)] {
// CHECK:STDOUT:       %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived.generic]
// CHECK:STDOUT:       %T.ref.loc13_39: type = name_ref T, %T.loc13_18.1 [symbolic = %T.loc13_18.2 (constants.%T)]
// CHECK:STDOUT:       %Derived.loc13_40.1: type = class_type @Derived, @Derived(constants.%T) [symbolic = %Derived.loc13_40.2 (constants.%Derived.85c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @AccessDerived.%Derived.loc13_40.2 (%Derived.85c) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessDerived.%T.loc13_18.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @AccessDerived.%T.loc13_18.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBase.decl: %AccessBase.type = fn_decl @AccessBase [concrete = constants.%AccessBase] {
// CHECK:STDOUT:     %T.patt.loc17_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @AccessBase.%Derived.loc17_37.2 (%Derived.85c) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @AccessBase.%Derived.loc17_37.2 (%Derived.85c) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @AccessBase.%T.loc17_15.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessBase.%T.loc17_15.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc17_43: type = name_ref T, %T.loc17_15.1 [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc17_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessBase.%Derived.loc17_37.2 (%Derived.85c) = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %Derived.loc17_37.1 [symbolic = %Derived.loc17_37.2 (constants.%Derived.85c)] {
// CHECK:STDOUT:       %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived.generic]
// CHECK:STDOUT:       %T.ref.loc17_36: type = name_ref T, %T.loc17_15.1 [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:       %Derived.loc17_37.1: type = class_type @Derived, @Derived(constants.%T) [symbolic = %Derived.loc17_37.2 (constants.%Derived.85c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @AccessBase.%Derived.loc17_37.2 (%Derived.85c) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessBase.%T.loc17_15.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @AccessBase.%T.loc17_15.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessConcrete.decl: %AccessConcrete.type = fn_decl @AccessConcrete [concrete = constants.%AccessConcrete] {
// CHECK:STDOUT:     %x.patt: %Derived.115 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %Derived.115 = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT: generic class @Base(%T.loc4_17.1: type) {
// CHECK:STDOUT:   %T.loc4_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_17.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Base.%T.loc4_17.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc4_17.2) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.loc4_17.2 [symbolic = %Base.elem (constants.%Base.elem.9af)]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: @Base.%T.loc4_17.2 (%T)} [symbolic = %struct_type.b (constants.%struct_type.b.f69)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Base.%struct_type.b (%struct_type.b.f69) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.eaf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5_8: @Base.%Base.elem (%Base.elem.9af) = field_decl b, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc5_3: @Base.%Base.elem (%Base.elem.9af) = var_pattern %.loc5_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @Base.%Base.elem (%Base.elem.9af) = var <none>
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.b.f69 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.eaf)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.370
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .b = %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc8_15.1: type) {
// CHECK:STDOUT:   %T.loc8_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2: type = class_type @Base, @Base(%T.loc8_15.2) [symbolic = %Base.loc9_22.2 (constants.%Base.370)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type @Derived.%Base.loc9_22.2 (%Base.370) [symbolic = %require_complete.loc9 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T.loc8_15.2) [symbolic = %Derived (constants.%Derived.85c)]
// CHECK:STDOUT:   %Derived.elem.loc9: type = unbound_element_type %Derived, %Base.loc9_22.2 [symbolic = %Derived.elem.loc9 (constants.%Derived.elem.8b3)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type @Derived.%T.loc8_15.2 (%T) [symbolic = %require_complete.loc10 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Derived.elem.loc10: type = unbound_element_type %Derived, %T.loc8_15.2 [symbolic = %Derived.elem.loc10 (constants.%Derived.elem.6d2)]
// CHECK:STDOUT:   %struct_type.base.d: type = struct_type {.base: @Derived.%Base.loc9_22.2 (%Base.370), .d: @Derived.%T.loc8_15.2 (%T)} [symbolic = %struct_type.base.d (constants.%struct_type.base.d.37c)]
// CHECK:STDOUT:   %complete_type.loc11_1.2: <witness> = complete_type_witness @Derived.%struct_type.base.d (%struct_type.base.d.37c) [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.8ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %Base.loc9_22.1: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc9_22.2 (constants.%Base.370)]
// CHECK:STDOUT:     %.loc9: @Derived.%Derived.elem.loc9 (%Derived.elem.8b3) = base_decl %Base.loc9_22.1, element0 [concrete]
// CHECK:STDOUT:     %.loc10_8: @Derived.%Derived.elem.loc10 (%Derived.elem.6d2) = field_decl d, element1 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc10_3: @Derived.%Derived.elem.loc10 (%Derived.elem.6d2) = var_pattern %.loc10_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @Derived.%Derived.elem.loc10 (%Derived.elem.6d2) = var <none>
// CHECK:STDOUT:     %complete_type.loc11_1.1: <witness> = complete_type_witness %struct_type.base.d.37c [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.8ad)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc11_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived.85c
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     .d = %.loc10_8
// CHECK:STDOUT:     .b = <poisoned>
// CHECK:STDOUT:     extend %Base.loc9_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessDerived(%T.loc13_18.1: type) {
// CHECK:STDOUT:   %T.loc13_18.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_18.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_18.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Derived.loc13_40.2: type = class_type @Derived, @Derived(%T.loc13_18.2) [symbolic = %Derived.loc13_40.2 (constants.%Derived.85c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type @AccessDerived.%Derived.loc13_40.2 (%Derived.85c) [symbolic = %require_complete.loc13 (constants.%require_complete.5f4)]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived.loc13_40.2, %T.loc13_18.2 [symbolic = %Derived.elem (constants.%Derived.elem.6d2)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type @AccessDerived.%T.loc13_18.2 (%T) [symbolic = %require_complete.loc14 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc13_18.1: type](%x.param_patt: @AccessDerived.%Derived.loc13_40.2 (%Derived.85c)) -> @AccessDerived.%T.loc13_18.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessDerived.%Derived.loc13_40.2 (%Derived.85c) = name_ref x, %x
// CHECK:STDOUT:     %d.ref: @AccessDerived.%Derived.elem (%Derived.elem.6d2) = name_ref d, @Derived.%.loc10_8 [concrete = @Derived.%.loc10_8]
// CHECK:STDOUT:     %.loc14_11.1: ref @AccessDerived.%T.loc13_18.2 (%T) = class_element_access %x.ref, element1
// CHECK:STDOUT:     %.loc14_11.2: @AccessDerived.%T.loc13_18.2 (%T) = bind_value %.loc14_11.1
// CHECK:STDOUT:     return %.loc14_11.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessBase(%T.loc17_15.1: type) {
// CHECK:STDOUT:   %T.loc17_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc17_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc17_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Derived.loc17_37.2: type = class_type @Derived, @Derived(%T.loc17_15.2) [symbolic = %Derived.loc17_37.2 (constants.%Derived.85c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17: <witness> = require_complete_type @AccessBase.%Derived.loc17_37.2 (%Derived.85c) [symbolic = %require_complete.loc17 (constants.%require_complete.5f4)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc17_15.2) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %require_complete.loc18_11: <witness> = require_complete_type @AccessBase.%Base (%Base.370) [symbolic = %require_complete.loc18_11 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.loc17_15.2 [symbolic = %Base.elem (constants.%Base.elem.9af)]
// CHECK:STDOUT:   %require_complete.loc18_13: <witness> = require_complete_type @AccessBase.%T.loc17_15.2 (%T) [symbolic = %require_complete.loc18_13 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc17_15.1: type](%x.param_patt: @AccessBase.%Derived.loc17_37.2 (%Derived.85c)) -> @AccessBase.%T.loc17_15.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessBase.%Derived.loc17_37.2 (%Derived.85c) = name_ref x, %x
// CHECK:STDOUT:     %b.ref: @AccessBase.%Base.elem (%Base.elem.9af) = name_ref b, @Base.%.loc5_8 [concrete = @Base.%.loc5_8]
// CHECK:STDOUT:     %.loc18_11.1: ref @AccessBase.%Base (%Base.370) = class_element_access %x.ref, element0
// CHECK:STDOUT:     %.loc18_11.2: ref @AccessBase.%Base (%Base.370) = converted %x.ref, %.loc18_11.1
// CHECK:STDOUT:     %.loc18_11.3: ref @AccessBase.%T.loc17_15.2 (%T) = class_element_access %.loc18_11.2, element0
// CHECK:STDOUT:     %.loc18_11.4: @AccessBase.%T.loc17_15.2 (%T) = bind_value %.loc18_11.3
// CHECK:STDOUT:     return %.loc18_11.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessConcrete(%x.param_patt: %Derived.115) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Derived.115 = name_ref x, %x
// CHECK:STDOUT:   %b.ref: %Base.elem.a98 = name_ref b, @Base.%.loc5_8 [concrete = @Base.%.loc5_8]
// CHECK:STDOUT:   %.loc22_11.1: ref %Base.10a = class_element_access %x.ref, element0
// CHECK:STDOUT:   %.loc22_11.2: ref %Base.10a = converted %x.ref, %.loc22_11.1
// CHECK:STDOUT:   %.loc22_11.3: ref %i32 = class_element_access %.loc22_11.2, element0
// CHECK:STDOUT:   %.loc22_11.4: %i32 = bind_value %.loc22_11.3
// CHECK:STDOUT:   return %.loc22_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.9af
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.f69
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.eaf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(%T.loc4_17.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
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
// CHECK:STDOUT: specific @Base(@Derived.%T.loc8_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(%T.loc8_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessDerived(constants.%T) {
// CHECK:STDOUT:   %T.loc13_18.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_18.2 => constants.%T.patt
// CHECK:STDOUT:   %Derived.loc13_40.2 => constants.%Derived.85c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(@AccessDerived.%T.loc13_18.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessBase(constants.%T) {
// CHECK:STDOUT:   %T.loc17_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc17_15.2 => constants.%T.patt
// CHECK:STDOUT:   %Derived.loc17_37.2 => constants.%Derived.85c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(@AccessBase.%T.loc17_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(@AccessBase.%T.loc17_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%i32) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
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
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T.patt
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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Base.elem.9af: type = unbound_element_type %Base.370, %T [symbolic]
// CHECK:STDOUT:   %struct_type.b.f69: type = struct_type {.b: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.eaf: <witness> = complete_type_witness %struct_type.b.f69 [symbolic]
// CHECK:STDOUT:   %Derived.type: type = generic_class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.generic: %Derived.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.85c: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.97d: <witness> = require_complete_type %Base.370 [symbolic]
// CHECK:STDOUT:   %Derived.elem.8b3: type = unbound_element_type %Derived.85c, %Base.370 [symbolic]
// CHECK:STDOUT:   %Derived.elem.6d2: type = unbound_element_type %Derived.85c, %T [symbolic]
// CHECK:STDOUT:   %struct_type.base.d.37c: type = struct_type {.base: %Base.370, .d: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.8ad: <witness> = complete_type_witness %struct_type.base.d.37c [symbolic]
// CHECK:STDOUT:   %AccessMissingBase.type: type = fn_type @AccessMissingBase [concrete]
// CHECK:STDOUT:   %AccessMissingBase: %AccessMissingBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %AccessMissingDerived.type: type = fn_type @AccessMissingDerived [concrete]
// CHECK:STDOUT:   %AccessMissingDerived: %AccessMissingDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.5f4: <witness> = require_complete_type %Derived.85c [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Derived.115: type = class_type @Derived, @Derived(%i32) [concrete]
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
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc4_17.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: %Derived.type = class_decl @Derived [concrete = constants.%Derived.generic] {
// CHECK:STDOUT:     %T.patt.loc8_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingBase.decl: %AccessMissingBase.type = fn_decl @AccessMissingBase [concrete = constants.%AccessMissingBase] {
// CHECK:STDOUT:     %T.patt.loc13_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @AccessMissingBase.%Base.loc13_41.2 (%Base.370) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @AccessMissingBase.%Base.loc13_41.2 (%Base.370) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @AccessMissingBase.%T.loc13_22.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessMissingBase.%T.loc13_22.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_47: type = name_ref T, %T.loc13_22.1 [symbolic = %T.loc13_22.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc13_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_22.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessMissingBase.%Base.loc13_41.2 (%Base.370) = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %Base.loc13_41.1 [symbolic = %Base.loc13_41.2 (constants.%Base.370)] {
// CHECK:STDOUT:       %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:       %T.ref.loc13_40: type = name_ref T, %T.loc13_22.1 [symbolic = %T.loc13_22.2 (constants.%T)]
// CHECK:STDOUT:       %Base.loc13_41.1: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc13_41.2 (constants.%Base.370)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @AccessMissingBase.%Base.loc13_41.2 (%Base.370) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessMissingBase.%T.loc13_22.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @AccessMissingBase.%T.loc13_22.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingDerived.decl: %AccessMissingDerived.type = fn_decl @AccessMissingDerived [concrete = constants.%AccessMissingDerived] {
// CHECK:STDOUT:     %T.patt.loc21_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.85c) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.85c) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @AccessMissingDerived.%T.loc21_25.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @AccessMissingDerived.%T.loc21_25.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc21_53: type = name_ref T, %T.loc21_25.1 [symbolic = %T.loc21_25.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc21_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_25.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.85c) = value_param call_param0
// CHECK:STDOUT:     %.loc21: type = splice_block %Derived.loc21_47.1 [symbolic = %Derived.loc21_47.2 (constants.%Derived.85c)] {
// CHECK:STDOUT:       %Derived.ref: %Derived.type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived.generic]
// CHECK:STDOUT:       %T.ref.loc21_46: type = name_ref T, %T.loc21_25.1 [symbolic = %T.loc21_25.2 (constants.%T)]
// CHECK:STDOUT:       %Derived.loc21_47.1: type = class_type @Derived, @Derived(constants.%T) [symbolic = %Derived.loc21_47.2 (constants.%Derived.85c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.85c) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @AccessMissingDerived.%T.loc21_25.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @AccessMissingDerived.%T.loc21_25.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessMissingConcrete.decl: %AccessMissingConcrete.type = fn_decl @AccessMissingConcrete [concrete = constants.%AccessMissingConcrete] {
// CHECK:STDOUT:     %x.patt: %Derived.115 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %Derived.115 = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT: generic class @Base(%T.loc4_17.1: type) {
// CHECK:STDOUT:   %T.loc4_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_17.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Base.%T.loc4_17.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc4_17.2) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.loc4_17.2 [symbolic = %Base.elem (constants.%Base.elem.9af)]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: @Base.%T.loc4_17.2 (%T)} [symbolic = %struct_type.b (constants.%struct_type.b.f69)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Base.%struct_type.b (%struct_type.b.f69) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.eaf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5_8: @Base.%Base.elem (%Base.elem.9af) = field_decl b, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc5_3: @Base.%Base.elem (%Base.elem.9af) = var_pattern %.loc5_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @Base.%Base.elem (%Base.elem.9af) = var <none>
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.b.f69 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.eaf)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.370
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .b = %.loc5_8
// CHECK:STDOUT:     .nonesuch = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc8_15.1: type) {
// CHECK:STDOUT:   %T.loc8_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc9_22.2: type = class_type @Base, @Base(%T.loc8_15.2) [symbolic = %Base.loc9_22.2 (constants.%Base.370)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type @Derived.%Base.loc9_22.2 (%Base.370) [symbolic = %require_complete.loc9 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T.loc8_15.2) [symbolic = %Derived (constants.%Derived.85c)]
// CHECK:STDOUT:   %Derived.elem.loc9: type = unbound_element_type %Derived, %Base.loc9_22.2 [symbolic = %Derived.elem.loc9 (constants.%Derived.elem.8b3)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type @Derived.%T.loc8_15.2 (%T) [symbolic = %require_complete.loc10 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Derived.elem.loc10: type = unbound_element_type %Derived, %T.loc8_15.2 [symbolic = %Derived.elem.loc10 (constants.%Derived.elem.6d2)]
// CHECK:STDOUT:   %struct_type.base.d: type = struct_type {.base: @Derived.%Base.loc9_22.2 (%Base.370), .d: @Derived.%T.loc8_15.2 (%T)} [symbolic = %struct_type.base.d (constants.%struct_type.base.d.37c)]
// CHECK:STDOUT:   %complete_type.loc11_1.2: <witness> = complete_type_witness @Derived.%struct_type.base.d (%struct_type.base.d.37c) [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.8ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_15.1 [symbolic = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:     %Base.loc9_22.1: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc9_22.2 (constants.%Base.370)]
// CHECK:STDOUT:     %.loc9: @Derived.%Derived.elem.loc9 (%Derived.elem.8b3) = base_decl %Base.loc9_22.1, element0 [concrete]
// CHECK:STDOUT:     %.loc10_8: @Derived.%Derived.elem.loc10 (%Derived.elem.6d2) = field_decl d, element1 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc10_3: @Derived.%Derived.elem.loc10 (%Derived.elem.6d2) = var_pattern %.loc10_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @Derived.%Derived.elem.loc10 (%Derived.elem.6d2) = var <none>
// CHECK:STDOUT:     %complete_type.loc11_1.1: <witness> = complete_type_witness %struct_type.base.d.37c [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.8ad)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc11_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived.85c
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     .d = %.loc10_8
// CHECK:STDOUT:     .nonesuch = <poisoned>
// CHECK:STDOUT:     extend %Base.loc9_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingBase(%T.loc13_22.1: type) {
// CHECK:STDOUT:   %T.loc13_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Base.loc13_41.2: type = class_type @Base, @Base(%T.loc13_22.2) [symbolic = %Base.loc13_41.2 (constants.%Base.370)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @AccessMissingBase.%Base.loc13_41.2 (%Base.370) [symbolic = %require_complete (constants.%require_complete.97d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc13_22.1: type](%x.param_patt: @AccessMissingBase.%Base.loc13_41.2 (%Base.370)) -> @AccessMissingBase.%T.loc13_22.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessMissingBase.%Base.loc13_41.2 (%Base.370) = name_ref x, %x
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AccessMissingDerived(%T.loc21_25.1: type) {
// CHECK:STDOUT:   %T.loc21_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc21_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc21_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Derived.loc21_47.2: type = class_type @Derived, @Derived(%T.loc21_25.2) [symbolic = %Derived.loc21_47.2 (constants.%Derived.85c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21: <witness> = require_complete_type @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.85c) [symbolic = %require_complete.loc21 (constants.%require_complete.5f4)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc21_25.2) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %require_complete.loc26: <witness> = require_complete_type @AccessMissingDerived.%Base (%Base.370) [symbolic = %require_complete.loc26 (constants.%require_complete.97d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc21_25.1: type](%x.param_patt: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.85c)) -> @AccessMissingDerived.%T.loc21_25.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AccessMissingDerived.%Derived.loc21_47.2 (%Derived.85c) = name_ref x, %x
// CHECK:STDOUT:     %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessMissingConcrete(%x.param_patt: %Derived.115) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Derived.115 = name_ref x, %x
// CHECK:STDOUT:   %nonesuch.ref: <error> = name_ref nonesuch, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.9af
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.f69
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.eaf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(%T.loc4_17.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
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
// CHECK:STDOUT: specific @Base(@Derived.%T.loc8_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(%T.loc8_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingBase(constants.%T) {
// CHECK:STDOUT:   %T.loc13_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_22.2 => constants.%T.patt
// CHECK:STDOUT:   %Base.loc13_41.2 => constants.%Base.370
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(@AccessMissingBase.%T.loc13_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AccessMissingDerived(constants.%T) {
// CHECK:STDOUT:   %T.loc21_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc21_25.2 => constants.%T.patt
// CHECK:STDOUT:   %Derived.loc21_47.2 => constants.%Derived.85c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(@AccessMissingDerived.%T.loc21_25.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(@AccessMissingDerived.%T.loc21_25.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%i32) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc8_15.2 => constants.%T.patt
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
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Base => constants.%Base.10a
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.a98
// CHECK:STDOUT:   %struct_type.b => constants.%struct_type.b.0a3
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.ba8
// CHECK:STDOUT: }
// CHECK:STDOUT:

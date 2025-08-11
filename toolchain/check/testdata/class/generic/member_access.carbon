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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_access.carbon

// --- member_access.carbon

class Class(T:! type) {
  var x: T;

  fn Get[self: Self]() -> T { return self.x; }

  fn GetAddr[addr self: Self*]() -> T* { return &self->x; }
}

fn DirectFieldAccess(x: Class(i32)) -> i32 {
  return x.x;
}

fn MethodCall(x: Class(i32)) -> i32 {
  return x.Get();
}

fn AddrMethodCall(p: Class(i32)*) -> i32 {
  return *p->GetAddr();
}

// --- static_member_fn_call.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  fn Make() -> Class(T) { return {}; }
}

fn StaticMemberFunctionCall(T:! type) -> Class(T) {
  return Class(T).Make();
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem.e26: type = unbound_element_type %Class.fe1, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.3c1: type = pattern_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Class.Get.type.fd9: type = fn_type @Class.Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.Get.cf9: %Class.Get.type.fd9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.type.402: type = fn_type @Class.GetAddr, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.GetAddr.102: %Class.GetAddr.type.402 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x.2ac [symbolic]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.2ae: <witness> = require_complete_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.247: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.0fa: type = pattern_type %Class.247 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %DirectFieldAccess.type: type = fn_type @DirectFieldAccess [concrete]
// CHECK:STDOUT:   %DirectFieldAccess: %DirectFieldAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Class.elem.2d8: type = unbound_element_type %Class.247, %i32 [concrete]
// CHECK:STDOUT:   %Class.Get.type.59d: type = fn_type @Class.Get, @Class(%i32) [concrete]
// CHECK:STDOUT:   %Class.Get.a40: %Class.Get.type.59d = struct_value () [concrete]
// CHECK:STDOUT:   %Class.GetAddr.type.be7: type = fn_type @Class.GetAddr, @Class(%i32) [concrete]
// CHECK:STDOUT:   %Class.GetAddr.909: %Class.GetAddr.type.be7 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %MethodCall.type: type = fn_type @MethodCall [concrete]
// CHECK:STDOUT:   %MethodCall: %MethodCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Class.Get.a40, @Class.Get(%i32) [concrete]
// CHECK:STDOUT:   %ptr.f7c: type = ptr_type %Class.247 [concrete]
// CHECK:STDOUT:   %pattern_type.14a: type = pattern_type %ptr.f7c [concrete]
// CHECK:STDOUT:   %AddrMethodCall.type: type = fn_type @AddrMethodCall [concrete]
// CHECK:STDOUT:   %AddrMethodCall: %AddrMethodCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %Class.GetAddr.909, @Class.GetAddr(%i32) [concrete]
// CHECK:STDOUT:   %complete_type.3d0: <witness> = complete_type_witness %ptr.235 [concrete]
// CHECK:STDOUT:   %complete_type.6ee: <witness> = complete_type_witness %ptr.f7c [concrete]
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
// CHECK:STDOUT:     .DirectFieldAccess = %DirectFieldAccess.decl
// CHECK:STDOUT:     .MethodCall = %MethodCall.decl
// CHECK:STDOUT:     .AddrMethodCall = %AddrMethodCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc2_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DirectFieldAccess.decl: %DirectFieldAccess.type = fn_decl @DirectFieldAccess [concrete = constants.%DirectFieldAccess] {
// CHECK:STDOUT:     %x.patt: %pattern_type.0fa = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.0fa = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc10_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %x.param: %Class.247 = value_param call_param0
// CHECK:STDOUT:     %.loc10: type = splice_block %Class [concrete = constants.%Class.247] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %int_32.loc10_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc10_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.247]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %Class.247 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MethodCall.decl: %MethodCall.type = fn_decl @MethodCall [concrete = constants.%MethodCall] {
// CHECK:STDOUT:     %x.patt: %pattern_type.0fa = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.0fa = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc14_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %x.param: %Class.247 = value_param call_param0
// CHECK:STDOUT:     %.loc14: type = splice_block %Class [concrete = constants.%Class.247] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %int_32.loc14_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.247]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %Class.247 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrMethodCall.decl: %AddrMethodCall.type = fn_decl @AddrMethodCall [concrete = constants.%AddrMethodCall] {
// CHECK:STDOUT:     %p.patt: %pattern_type.14a = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.14a = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc18_38: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_38: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.f7c = value_param call_param0
// CHECK:STDOUT:     %.loc18: type = splice_block %ptr [concrete = constants.%ptr.f7c] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %int_32.loc18_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.247]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class [concrete = constants.%ptr.f7c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.f7c = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Class.as.Destroy.impl(@Class.%T.loc2_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Class.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc2_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc2_23.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class.fe1)] {
// CHECK:STDOUT:         %.loc2_23.3: type = specific_constant constants.%Class.fe1, @Class(constants.%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc2_23.3 [symbolic = %Class (constants.%Class.fe1)]
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
// CHECK:STDOUT: generic class @Class(%T.loc2_13.2: type) {
// CHECK:STDOUT:   %T.loc2_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc2_13.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc2_13.1) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc2_13.1 [symbolic = %Class.elem (constants.%Class.elem.e26)]
// CHECK:STDOUT:   %Class.Get.type: type = fn_type @Class.Get, @Class(%T.loc2_13.1) [symbolic = %Class.Get.type (constants.%Class.Get.type.fd9)]
// CHECK:STDOUT:   %Class.Get: @Class.%Class.Get.type (%Class.Get.type.fd9) = struct_value () [symbolic = %Class.Get (constants.%Class.Get.cf9)]
// CHECK:STDOUT:   %Class.GetAddr.type: type = fn_type @Class.GetAddr, @Class(%T.loc2_13.1) [symbolic = %Class.GetAddr.type (constants.%Class.GetAddr.type.402)]
// CHECK:STDOUT:   %Class.GetAddr: @Class.%Class.GetAddr.type (%Class.GetAddr.type.402) = struct_value () [symbolic = %Class.GetAddr (constants.%Class.GetAddr.102)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.loc2_13.1 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac)]
// CHECK:STDOUT:   %complete_type.loc8_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc8_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc2_13.2 [symbolic = %T.loc2_13.1 (constants.%T)]
// CHECK:STDOUT:     %.loc3: @Class.%Class.elem (%Class.elem.e26) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %Class.Get.decl: @Class.%Class.Get.type (%Class.Get.type.fd9) = fn_decl @Class.Get [symbolic = @Class.%Class.Get (constants.%Class.Get.cf9)] {
// CHECK:STDOUT:       %self.patt: @Class.Get.%pattern_type.loc5_10 (%pattern_type.3c1) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.Get.%pattern_type.loc5_10 (%pattern_type.3c1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.Get.%pattern_type.loc5_24 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.Get.%pattern_type.loc5_24 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc2_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Class.Get.%Class (%Class.fe1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_16.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class.fe1)] {
// CHECK:STDOUT:         %.loc5_16.2: type = specific_constant constants.%Class.fe1, @Class(constants.%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_16.2 [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.Get.%Class (%Class.fe1) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.Get.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.Get.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.GetAddr.decl: @Class.%Class.GetAddr.type (%Class.GetAddr.type.402) = fn_decl @Class.GetAddr [symbolic = @Class.%Class.GetAddr (constants.%Class.GetAddr.102)] {
// CHECK:STDOUT:       %self.patt: @Class.GetAddr.%pattern_type.loc7_19 (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.GetAddr.%pattern_type.loc7_19 (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc7_14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:       %return.patt: @Class.GetAddr.%pattern_type.loc7_34 (%pattern_type.afe) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.GetAddr.%pattern_type.loc7_34 (%pattern_type.afe) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc2_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %ptr.loc7_38.2: type = ptr_type %T.ref [symbolic = %ptr.loc7_38.1 (constants.%ptr.79f)]
// CHECK:STDOUT:       %self.param: @Class.GetAddr.%ptr.loc7_29.1 (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc7_29: type = splice_block %ptr.loc7_29.2 [symbolic = %ptr.loc7_29.1 (constants.%ptr.955)] {
// CHECK:STDOUT:         %.loc7_25: type = specific_constant constants.%Class.fe1, @Class(constants.%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc7_25 [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:         %ptr.loc7_29.2: type = ptr_type %Self.ref [symbolic = %ptr.loc7_29.1 (constants.%ptr.955)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.GetAddr.%ptr.loc7_29.1 (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.GetAddr.%ptr.loc7_38.1 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.GetAddr.%ptr.loc7_38.1 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class.fe1 [symbolic = @Class.as.Destroy.impl.%Class (constants.%Class.fe1)]
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%T) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type.loc8_1.1: <witness> = complete_type_witness constants.%struct_type.x.2ac [symbolic = %complete_type.loc8_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc8_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc3
// CHECK:STDOUT:     .Get = %Class.Get.decl
// CHECK:STDOUT:     .GetAddr = %Class.GetAddr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.Get(@Class.%T.loc2_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %pattern_type.loc5_10: type = pattern_type %Class [symbolic = %pattern_type.loc5_10 (constants.%pattern_type.3c1)]
// CHECK:STDOUT:   %pattern_type.loc5_24: type = pattern_type %T [symbolic = %pattern_type.loc5_24 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_14: <witness> = require_complete_type %Class [symbolic = %require_complete.loc5_14 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic = %Class.elem (constants.%Class.elem.e26)]
// CHECK:STDOUT:   %require_complete.loc5_42: <witness> = require_complete_type %T [symbolic = %require_complete.loc5_42 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.Get.%Class (%Class.fe1)) -> @Class.Get.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.Get.%Class (%Class.fe1) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Class.Get.%Class.elem (%Class.elem.e26) = name_ref x, @Class.%.loc3 [concrete = @Class.%.loc3]
// CHECK:STDOUT:     %.loc5_42.1: ref @Class.Get.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc5_42.2: @Class.Get.%T (%T) = bind_value %.loc5_42.1
// CHECK:STDOUT:     return %.loc5_42.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.GetAddr(@Class.%T.loc2_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %ptr.loc7_29.1: type = ptr_type %Class [symbolic = %ptr.loc7_29.1 (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type.loc7_19: type = pattern_type %ptr.loc7_29.1 [symbolic = %pattern_type.loc7_19 (constants.%pattern_type.9e0)]
// CHECK:STDOUT:   %ptr.loc7_38.1: type = ptr_type %T [symbolic = %ptr.loc7_38.1 (constants.%ptr.79f)]
// CHECK:STDOUT:   %pattern_type.loc7_34: type = pattern_type %ptr.loc7_38.1 [symbolic = %pattern_type.loc7_34 (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_34: <witness> = require_complete_type %ptr.loc7_38.1 [symbolic = %require_complete.loc7_34 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc7_23: <witness> = require_complete_type %ptr.loc7_29.1 [symbolic = %require_complete.loc7_23 (constants.%require_complete.2ae)]
// CHECK:STDOUT:   %require_complete.loc7_54: <witness> = require_complete_type %Class [symbolic = %require_complete.loc7_54 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic = %Class.elem (constants.%Class.elem.e26)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.GetAddr.%ptr.loc7_29.1 (%ptr.955)) -> @Class.GetAddr.%ptr.loc7_38.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.GetAddr.%ptr.loc7_29.1 (%ptr.955) = name_ref self, %self
// CHECK:STDOUT:     %.loc7_54.1: ref @Class.GetAddr.%Class (%Class.fe1) = deref %self.ref
// CHECK:STDOUT:     %x.ref: @Class.GetAddr.%Class.elem (%Class.elem.e26) = name_ref x, @Class.%.loc3 [concrete = @Class.%.loc3]
// CHECK:STDOUT:     %.loc7_54.2: ref @Class.GetAddr.%T (%T) = class_element_access %.loc7_54.1, element0
// CHECK:STDOUT:     %addr: @Class.GetAddr.%ptr.loc7_38.1 (%ptr.79f) = addr_of %.loc7_54.2
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.%T.loc2_13.2: type) {
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
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param: %Class.247) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc11_10: %Class.247 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc11_11: %Class.elem.2d8 = name_ref x, @Class.%.loc3 [concrete = @Class.%.loc3]
// CHECK:STDOUT:   %.loc11_11.1: ref %i32 = class_element_access %x.ref.loc11_10, element0
// CHECK:STDOUT:   %.loc11_11.2: %i32 = bind_value %.loc11_11.1
// CHECK:STDOUT:   return %.loc11_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param: %Class.247) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.247 = name_ref x, %x
// CHECK:STDOUT:   %.loc15_11: %Class.Get.type.59d = specific_constant @Class.%Class.Get.decl, @Class(constants.%i32) [concrete = constants.%Class.Get.a40]
// CHECK:STDOUT:   %Get.ref: %Class.Get.type.59d = name_ref Get, %.loc15_11 [concrete = constants.%Class.Get.a40]
// CHECK:STDOUT:   %Class.Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Class.Get.specific_fn: <specific function> = specific_function %Get.ref, @Class.Get(constants.%i32) [concrete = constants.%Class.Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Class.Get.specific_fn
// CHECK:STDOUT:   %Class.Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   %.loc15_17.1: %i32 = value_of_initializer %Class.Get.call
// CHECK:STDOUT:   %.loc15_17.2: %i32 = converted %Class.Get.call, %.loc15_17.1
// CHECK:STDOUT:   return %.loc15_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param: %ptr.f7c) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.f7c = name_ref p, %p
// CHECK:STDOUT:   %.loc19_12.1: ref %Class.247 = deref %p.ref
// CHECK:STDOUT:   %.loc19_12.2: %Class.GetAddr.type.be7 = specific_constant @Class.%Class.GetAddr.decl, @Class(constants.%i32) [concrete = constants.%Class.GetAddr.909]
// CHECK:STDOUT:   %GetAddr.ref: %Class.GetAddr.type.be7 = name_ref GetAddr, %.loc19_12.2 [concrete = constants.%Class.GetAddr.909]
// CHECK:STDOUT:   %Class.GetAddr.bound: <bound method> = bound_method %.loc19_12.1, %GetAddr.ref
// CHECK:STDOUT:   %Class.GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @Class.GetAddr(constants.%i32) [concrete = constants.%Class.GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc19_12.1, %Class.GetAddr.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f7c = addr_of %.loc19_12.1
// CHECK:STDOUT:   %Class.GetAddr.call: init %ptr.235 = call %bound_method(%addr)
// CHECK:STDOUT:   %.loc19_22.1: %ptr.235 = value_of_initializer %Class.GetAddr.call
// CHECK:STDOUT:   %.loc19_22.2: %ptr.235 = converted %Class.GetAddr.call, %.loc19_22.1
// CHECK:STDOUT:   %.loc19_10.1: ref %i32 = deref %.loc19_22.2
// CHECK:STDOUT:   %.loc19_10.2: %i32 = bind_value %.loc19_10.1
// CHECK:STDOUT:   return %.loc19_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc2_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.e26
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.fd9
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.cf9
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.402
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.102
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2ac
// CHECK:STDOUT:   %complete_type.loc8_1.2 => constants.%complete_type.433
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %pattern_type.loc5_10 => constants.%pattern_type.3c1
// CHECK:STDOUT:   %pattern_type.loc5_24 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %ptr.loc7_29.1 => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type.loc7_19 => constants.%pattern_type.9e0
// CHECK:STDOUT:   %ptr.loc7_38.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type.loc7_34 => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %ptr => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc2_13.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2d8
// CHECK:STDOUT:   %Class.Get.type => constants.%Class.Get.type.59d
// CHECK:STDOUT:   %Class.Get => constants.%Class.Get.a40
// CHECK:STDOUT:   %Class.GetAddr.type => constants.%Class.GetAddr.type.be7
// CHECK:STDOUT:   %Class.GetAddr => constants.%Class.GetAddr.909
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc8_1.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Get(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %pattern_type.loc5_10 => constants.%pattern_type.0fa
// CHECK:STDOUT:   %pattern_type.loc5_24 => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_14 => constants.%complete_type.1ec
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2d8
// CHECK:STDOUT:   %require_complete.loc5_42 => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.GetAddr(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %ptr.loc7_29.1 => constants.%ptr.f7c
// CHECK:STDOUT:   %pattern_type.loc7_19 => constants.%pattern_type.14a
// CHECK:STDOUT:   %ptr.loc7_38.1 => constants.%ptr.235
// CHECK:STDOUT:   %pattern_type.loc7_34 => constants.%pattern_type.fe8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_34 => constants.%complete_type.3d0
// CHECK:STDOUT:   %require_complete.loc7_23 => constants.%complete_type.6ee
// CHECK:STDOUT:   %require_complete.loc7_54 => constants.%complete_type.1ec
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2d8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- static_member_fn_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.3c1: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %Class.Make.type: type = fn_type @Class.Make, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.Make: %Class.Make.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.95a: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class, (%Destroy.impl_witness.95a) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Class.val: %Class = struct_value () [symbolic]
// CHECK:STDOUT:   %StaticMemberFunctionCall.type: type = fn_type @StaticMemberFunctionCall [concrete]
// CHECK:STDOUT:   %StaticMemberFunctionCall: %StaticMemberFunctionCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.Make.specific_fn: <specific function> = specific_function %Class.Make, @Class.Make(%T) [symbolic]
// CHECK:STDOUT:   %.02c: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Class.as.Destroy.impl.Op, @Class.as.Destroy.impl.Op(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.2ae: <witness> = require_complete_type %ptr.955 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .StaticMemberFunctionCall = %StaticMemberFunctionCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMemberFunctionCall.decl: %StaticMemberFunctionCall.type = fn_decl @StaticMemberFunctionCall [concrete = constants.%StaticMemberFunctionCall] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.patt: @StaticMemberFunctionCall.%pattern_type (%pattern_type.3c1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @StaticMemberFunctionCall.%pattern_type (%pattern_type.3c1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc8: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, %T.loc8_29.2 [symbolic = %T.loc8_29.1 (constants.%T)]
// CHECK:STDOUT:     %Class.loc8_49.2: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc8_49.1 (constants.%Class)]
// CHECK:STDOUT:     %T.loc8_29.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_29.1 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @StaticMemberFunctionCall.%Class.loc8_49.1 (%Class) = out_param call_param0
// CHECK:STDOUT:     %return: ref @StaticMemberFunctionCall.%Class.loc8_49.1 (%Class) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Class.as.Destroy.impl(@Class.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Class.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc4_23.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc4_23.3: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_23.3 [symbolic = %Class (constants.%Class)]
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
// CHECK:STDOUT:   %Class.Make.type: type = fn_type @Class.Make, @Class(%T.loc4_13.1) [symbolic = %Class.Make.type (constants.%Class.Make.type)]
// CHECK:STDOUT:   %Class.Make: @Class.%Class.Make.type (%Class.Make.type) = struct_value () [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.Make.decl: @Class.%Class.Make.type (%Class.Make.type) = fn_decl @Class.Make [symbolic = @Class.%Class.Make (constants.%Class.Make)] {
// CHECK:STDOUT:       %return.patt: @Class.Make.%pattern_type (%pattern_type.3c1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.Make.%pattern_type (%pattern_type.3c1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc5_23.2: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc5_23.1 (constants.%Class)]
// CHECK:STDOUT:       %return.param: ref @Class.Make.%Class.loc5_23.1 (%Class) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Class.Make.%Class.loc5_23.1 (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [symbolic = @Class.as.Destroy.impl.%Class (constants.%Class)]
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%T) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .Class = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = %Class.Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.Make(@Class.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc5_23.1: type = class_type @Class, @Class(%T) [symbolic = %Class.loc5_23.1 (constants.%Class)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class.loc5_23.1 [symbolic = %pattern_type (constants.%pattern_type.3c1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc5_23.1 [symbolic = %require_complete (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.val: @Class.Make.%Class.loc5_23.1 (%Class) = struct_value () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @Class.Make.%Class.loc5_23.1 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_35.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_35.2: init @Class.Make.%Class.loc5_23.1 (%Class) = class_init (), %return [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     %.loc5_36: init @Class.Make.%Class.loc5_23.1 (%Class) = converted %.loc5_35.1, %.loc5_35.2 [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     return %.loc5_36 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.9e0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StaticMemberFunctionCall(%T.loc8_29.2: type) {
// CHECK:STDOUT:   %T.loc8_29.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_29.1 (constants.%T)]
// CHECK:STDOUT:   %Class.loc8_49.1: type = class_type @Class, @Class(%T.loc8_29.1) [symbolic = %Class.loc8_49.1 (constants.%Class)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class.loc8_49.1 [symbolic = %pattern_type (constants.%pattern_type.3c1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %Class.loc8_49.1 [symbolic = %require_complete.loc9 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.Make.type: type = fn_type @Class.Make, @Class(%T.loc8_29.1) [symbolic = %Class.Make.type (constants.%Class.Make.type)]
// CHECK:STDOUT:   %Class.Make: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = struct_value () [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:   %Class.Make.specific_fn.loc9_18.2: <specific function> = specific_function %Class.Make, @Class.Make(%T.loc8_29.1) [symbolic = %Class.Make.specific_fn.loc9_18.2 (constants.%Class.Make.specific_fn)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T.loc8_29.1) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class.loc8_49.1, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc8_39.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc8_39.2 (constants.%.02c)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T.loc8_29.1) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @StaticMemberFunctionCall.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Class.as.Destroy.impl.Op, @Class.as.Destroy.impl.Op(%T.loc8_29.1) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class.loc8_49.1 [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc8 (constants.%require_complete.2ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @StaticMemberFunctionCall.%Class.loc8_49.1 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref.loc9: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc9: type = name_ref T, %T.loc8_29.2 [symbolic = %T.loc8_29.1 (constants.%T)]
// CHECK:STDOUT:     %Class.loc9: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc8_49.1 (constants.%Class)]
// CHECK:STDOUT:     %.loc9: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = specific_constant @Class.%Class.Make.decl, @Class(constants.%T) [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:     %Make.ref: @StaticMemberFunctionCall.%Class.Make.type (%Class.Make.type) = name_ref Make, %.loc9 [symbolic = %Class.Make (constants.%Class.Make)]
// CHECK:STDOUT:     %Class.Make.specific_fn.loc9_18.1: <specific function> = specific_function %Make.ref, @Class.Make(constants.%T) [symbolic = %Class.Make.specific_fn.loc9_18.2 (constants.%Class.Make.specific_fn)]
// CHECK:STDOUT:     %.loc8_39.1: ref @StaticMemberFunctionCall.%Class.loc8_49.1 (%Class) = splice_block %return {}
// CHECK:STDOUT:     %Class.Make.call: init @StaticMemberFunctionCall.%Class.loc8_49.1 (%Class) = call %Class.Make.specific_fn.loc9_18.1() to %.loc8_39.1
// CHECK:STDOUT:     %impl.elem0: @StaticMemberFunctionCall.%.loc8_39.2 (%.02c) = impl_witness_access constants.%Destroy.impl_witness.95a, element0 [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:     %bound_method.loc8_39.1: <bound method> = bound_method %.loc8_39.1, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Class.as.Destroy.impl.Op(constants.%T) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc8_39.2: <bound method> = bound_method %.loc8_39.1, %specific_fn
// CHECK:STDOUT:     %addr: @StaticMemberFunctionCall.%ptr (%ptr.955) = addr_of %.loc8_39.1
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_39.2(%addr)
// CHECK:STDOUT:     return %Class.Make.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.Make.type => constants.%Class.Make.type
// CHECK:STDOUT:   %Class.Make => constants.%Class.Make
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.Make(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class.loc5_23.1 => constants.%Class
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3c1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4f8
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.95a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type => constants.%Class.as.Destroy.impl.Op.type
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op => constants.%Class.as.Destroy.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StaticMemberFunctionCall(constants.%T) {
// CHECK:STDOUT:   %T.loc8_29.1 => constants.%T
// CHECK:STDOUT:   %Class.loc8_49.1 => constants.%Class
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3c1
// CHECK:STDOUT: }
// CHECK:STDOUT:

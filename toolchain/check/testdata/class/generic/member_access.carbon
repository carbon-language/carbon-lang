// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// --- fail_todo_static_member_fn_call.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  fn Make() -> Class(T) { return {}; }
}

fn StaticMemberFunctionCall(T:! type) -> Class(T) {
  // CHECK:STDERR: fail_todo_static_member_fn_call.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `Class(T)` to `Class(T)` [ConversionFailure]
  // CHECK:STDERR:   return Class(T).Make();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_static_member_fn_call.carbon:[[@LINE+4]]:3: note: type `Class(T)` does not implement interface `Core.ImplicitAs(Class(T))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return Class(T).Make();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return Class(T).Make();
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.fe1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem.e26: type = unbound_element_type %Class.fe1, %T [symbolic]
// CHECK:STDOUT:   %Get.type.fd9: type = fn_type @Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Get.cf9: %Get.type.fd9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %GetAddr.type.402: type = fn_type @GetAddr, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetAddr.102: %GetAddr.type.402 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x.2ac [symbolic]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class.fe1 [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.2ae: <witness> = require_complete_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.247: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %DirectFieldAccess.type: type = fn_type @DirectFieldAccess [concrete]
// CHECK:STDOUT:   %DirectFieldAccess: %DirectFieldAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Class.elem.2d8: type = unbound_element_type %Class.247, %i32 [concrete]
// CHECK:STDOUT:   %Get.type.59d: type = fn_type @Get, @Class(%i32) [concrete]
// CHECK:STDOUT:   %Get.a40: %Get.type.59d = struct_value () [concrete]
// CHECK:STDOUT:   %GetAddr.type.be7: type = fn_type @GetAddr, @Class(%i32) [concrete]
// CHECK:STDOUT:   %GetAddr.909: %GetAddr.type.be7 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %MethodCall.type: type = fn_type @MethodCall [concrete]
// CHECK:STDOUT:   %MethodCall: %MethodCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.a40, @Get(%i32) [concrete]
// CHECK:STDOUT:   %ptr.f7c: type = ptr_type %Class.247 [concrete]
// CHECK:STDOUT:   %AddrMethodCall.type: type = fn_type @AddrMethodCall [concrete]
// CHECK:STDOUT:   %AddrMethodCall: %AddrMethodCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %GetAddr.specific_fn: <specific function> = specific_function %GetAddr.909, @GetAddr(%i32) [concrete]
// CHECK:STDOUT:   %complete_type.3d0: <witness> = complete_type_witness %ptr.235 [concrete]
// CHECK:STDOUT:   %complete_type.6ee: <witness> = complete_type_witness %ptr.f7c [concrete]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .DirectFieldAccess = %DirectFieldAccess.decl
// CHECK:STDOUT:     .MethodCall = %MethodCall.decl
// CHECK:STDOUT:     .AddrMethodCall = %AddrMethodCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc2_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc2_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DirectFieldAccess.decl: %DirectFieldAccess.type = fn_decl @DirectFieldAccess [concrete = constants.%DirectFieldAccess] {
// CHECK:STDOUT:     %x.patt: %Class.247 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %Class.247 = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:     %x.patt: %Class.247 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %Class.247 = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
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
// CHECK:STDOUT:     %p.patt: %ptr.f7c = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.f7c = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc18_38: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_38: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.f7c = value_param call_param0
// CHECK:STDOUT:     %.loc18: type = splice_block %ptr [concrete = constants.%ptr.f7c] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %int_32.loc18_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.247]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.247 [concrete = constants.%ptr.f7c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.f7c = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc2_13.1: type) {
// CHECK:STDOUT:   %T.loc2_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc2_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Class.%T.loc2_13.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc2_13.2) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Class.%Class (%Class.fe1), @Class.%T.loc2_13.2 (%T) [symbolic = %Class.elem (constants.%Class.elem.e26)]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T.loc2_13.2) [symbolic = %Get.type (constants.%Get.type.fd9)]
// CHECK:STDOUT:   %Get: @Class.%Get.type (%Get.type.fd9) = struct_value () [symbolic = %Get (constants.%Get.cf9)]
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T.loc2_13.2) [symbolic = %GetAddr.type (constants.%GetAddr.type.402)]
// CHECK:STDOUT:   %GetAddr: @Class.%GetAddr.type (%GetAddr.type.402) = struct_value () [symbolic = %GetAddr (constants.%GetAddr.102)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.loc2_13.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac)]
// CHECK:STDOUT:   %complete_type.loc8_1.2: <witness> = complete_type_witness @Class.%struct_type.x (%struct_type.x.2ac) [symbolic = %complete_type.loc8_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc3_8: @Class.%Class.elem (%Class.elem.e26) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc3_3: @Class.%Class.elem (%Class.elem.e26) = var_pattern %.loc3_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @Class.%Class.elem (%Class.elem.e26) = var <none>
// CHECK:STDOUT:     %Get.decl: @Class.%Get.type (%Get.type.fd9) = fn_decl @Get [symbolic = @Class.%Get (constants.%Get.cf9)] {
// CHECK:STDOUT:       %self.patt: @Get.%Class (%Class.fe1) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Get.%Class (%Class.fe1) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @Get.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Get.%T (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc2_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Get.%Class (%Class.fe1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_16.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class.fe1)] {
// CHECK:STDOUT:         %.loc5_16.2: type = specific_constant constants.%Class.fe1, @Class(constants.%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_16.2 [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Get.%Class (%Class.fe1) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Get.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Get.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetAddr.decl: @Class.%GetAddr.type (%GetAddr.type.402) = fn_decl @GetAddr [symbolic = @Class.%GetAddr (constants.%GetAddr.102)] {
// CHECK:STDOUT:       %self.patt: @GetAddr.%ptr.loc7_29.1 (%ptr.955) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @GetAddr.%ptr.loc7_29.1 (%ptr.955) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %.loc7_14: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:       %return.patt: @GetAddr.%ptr.loc7_38.1 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @GetAddr.%ptr.loc7_38.1 (%ptr.79f) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc2_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %ptr.loc7_38.2: type = ptr_type %T [symbolic = %ptr.loc7_38.1 (constants.%ptr.79f)]
// CHECK:STDOUT:       %self.param: @GetAddr.%ptr.loc7_29.1 (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc7_29: type = splice_block %ptr.loc7_29.2 [symbolic = %ptr.loc7_29.1 (constants.%ptr.955)] {
// CHECK:STDOUT:         %.loc7_25: type = specific_constant constants.%Class.fe1, @Class(constants.%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc7_25 [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:         %ptr.loc7_29.2: type = ptr_type %Class.fe1 [symbolic = %ptr.loc7_29.1 (constants.%ptr.955)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GetAddr.%ptr.loc7_29.1 (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @GetAddr.%ptr.loc7_38.1 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @GetAddr.%ptr.loc7_38.1 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc8_1.1: <witness> = complete_type_witness %struct_type.x.2ac [symbolic = %complete_type.loc8_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc8_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.fe1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc3_8
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:     .GetAddr = %GetAddr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Get(@Class.%T.loc2_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_14: <witness> = require_complete_type @Get.%Class (%Class.fe1) [symbolic = %require_complete.loc5_14 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Get.%Class (%Class.fe1), @Get.%T (%T) [symbolic = %Class.elem (constants.%Class.elem.e26)]
// CHECK:STDOUT:   %require_complete.loc5_42: <witness> = require_complete_type @Get.%T (%T) [symbolic = %require_complete.loc5_42 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Get.%Class (%Class.fe1)]() -> @Get.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Get.%Class (%Class.fe1) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Get.%Class.elem (%Class.elem.e26) = name_ref x, @Class.%.loc3_8 [concrete = @Class.%.loc3_8]
// CHECK:STDOUT:     %.loc5_42.1: ref @Get.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc5_42.2: @Get.%T (%T) = bind_value %.loc5_42.1
// CHECK:STDOUT:     return %.loc5_42.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetAddr(@Class.%T.loc2_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.fe1)]
// CHECK:STDOUT:   %ptr.loc7_29.1: type = ptr_type @GetAddr.%Class (%Class.fe1) [symbolic = %ptr.loc7_29.1 (constants.%ptr.955)]
// CHECK:STDOUT:   %ptr.loc7_38.1: type = ptr_type @GetAddr.%T (%T) [symbolic = %ptr.loc7_38.1 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_34: <witness> = require_complete_type @GetAddr.%ptr.loc7_38.1 (%ptr.79f) [symbolic = %require_complete.loc7_34 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc7_23: <witness> = require_complete_type @GetAddr.%ptr.loc7_29.1 (%ptr.955) [symbolic = %require_complete.loc7_23 (constants.%require_complete.2ae)]
// CHECK:STDOUT:   %require_complete.loc7_54: <witness> = require_complete_type @GetAddr.%Class (%Class.fe1) [symbolic = %require_complete.loc7_54 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @GetAddr.%Class (%Class.fe1), @GetAddr.%T (%T) [symbolic = %Class.elem (constants.%Class.elem.e26)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self.param_patt: @GetAddr.%ptr.loc7_29.1 (%ptr.955)]() -> @GetAddr.%ptr.loc7_38.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetAddr.%ptr.loc7_29.1 (%ptr.955) = name_ref self, %self
// CHECK:STDOUT:     %.loc7_54.1: ref @GetAddr.%Class (%Class.fe1) = deref %self.ref
// CHECK:STDOUT:     %x.ref: @GetAddr.%Class.elem (%Class.elem.e26) = name_ref x, @Class.%.loc3_8 [concrete = @Class.%.loc3_8]
// CHECK:STDOUT:     %.loc7_54.2: ref @GetAddr.%T (%T) = class_element_access %.loc7_54.1, element0
// CHECK:STDOUT:     %addr: @GetAddr.%ptr.loc7_38.1 (%ptr.79f) = addr_of %.loc7_54.2
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param_patt: %Class.247) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc11_10: %Class.247 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc11_11: %Class.elem.2d8 = name_ref x, @Class.%.loc3_8 [concrete = @Class.%.loc3_8]
// CHECK:STDOUT:   %.loc11_11.1: ref %i32 = class_element_access %x.ref.loc11_10, element0
// CHECK:STDOUT:   %.loc11_11.2: %i32 = bind_value %.loc11_11.1
// CHECK:STDOUT:   return %.loc11_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param_patt: %Class.247) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.247 = name_ref x, %x
// CHECK:STDOUT:   %.loc15_11: %Get.type.59d = specific_constant @Class.%Get.decl, @Class(constants.%i32) [concrete = constants.%Get.a40]
// CHECK:STDOUT:   %Get.ref: %Get.type.59d = name_ref Get, %.loc15_11 [concrete = constants.%Get.a40]
// CHECK:STDOUT:   %Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.ref, @Get(constants.%i32) [concrete = constants.%Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Get.specific_fn
// CHECK:STDOUT:   %Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   %.loc15_17.1: %i32 = value_of_initializer %Get.call
// CHECK:STDOUT:   %.loc15_17.2: %i32 = converted %Get.call, %.loc15_17.1
// CHECK:STDOUT:   return %.loc15_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param_patt: %ptr.f7c) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.f7c = name_ref p, %p
// CHECK:STDOUT:   %.loc19_12.1: ref %Class.247 = deref %p.ref
// CHECK:STDOUT:   %.loc19_12.2: %GetAddr.type.be7 = specific_constant @Class.%GetAddr.decl, @Class(constants.%i32) [concrete = constants.%GetAddr.909]
// CHECK:STDOUT:   %GetAddr.ref: %GetAddr.type.be7 = name_ref GetAddr, %.loc19_12.2 [concrete = constants.%GetAddr.909]
// CHECK:STDOUT:   %GetAddr.bound: <bound method> = bound_method %.loc19_12.1, %GetAddr.ref
// CHECK:STDOUT:   %GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @GetAddr(constants.%i32) [concrete = constants.%GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc19_12.1, %GetAddr.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f7c = addr_of %.loc19_12.1
// CHECK:STDOUT:   %GetAddr.call: init %ptr.235 = call %bound_method(%addr)
// CHECK:STDOUT:   %.loc19_22.1: %ptr.235 = value_of_initializer %GetAddr.call
// CHECK:STDOUT:   %.loc19_22.2: %ptr.235 = converted %GetAddr.call, %.loc19_22.1
// CHECK:STDOUT:   %.loc19_10.1: ref %i32 = deref %.loc19_22.2
// CHECK:STDOUT:   %.loc19_10.2: %i32 = bind_value %.loc19_10.1
// CHECK:STDOUT:   return %.loc19_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.e26
// CHECK:STDOUT:   %Get.type => constants.%Get.type.fd9
// CHECK:STDOUT:   %Get => constants.%Get.cf9
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type.402
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr.102
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2ac
// CHECK:STDOUT:   %complete_type.loc8_1.2 => constants.%complete_type.433
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Get.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.fe1
// CHECK:STDOUT:   %ptr.loc7_29.1 => constants.%ptr.955
// CHECK:STDOUT:   %ptr.loc7_38.1 => constants.%ptr.79f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetAddr.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc2_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2d8
// CHECK:STDOUT:   %Get.type => constants.%Get.type.59d
// CHECK:STDOUT:   %Get => constants.%Get.a40
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type.be7
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr.909
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc8_1.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_14 => constants.%complete_type.1ec
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2d8
// CHECK:STDOUT:   %require_complete.loc5_42 => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Class => constants.%Class.247
// CHECK:STDOUT:   %ptr.loc7_29.1 => constants.%ptr.f7c
// CHECK:STDOUT:   %ptr.loc7_38.1 => constants.%ptr.235
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_34 => constants.%complete_type.3d0
// CHECK:STDOUT:   %require_complete.loc7_23 => constants.%complete_type.6ee
// CHECK:STDOUT:   %require_complete.loc7_54 => constants.%complete_type.1ec
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2d8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_static_member_fn_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Class.val: %Class = struct_value () [symbolic]
// CHECK:STDOUT:   %StaticMemberFunctionCall.type: type = fn_type @StaticMemberFunctionCall [concrete]
// CHECK:STDOUT:   %StaticMemberFunctionCall: %StaticMemberFunctionCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.53b: type = facet_type <@ImplicitAs, @ImplicitAs(%Class)> [symbolic]
// CHECK:STDOUT:   %require_complete.d81: <witness> = require_complete_type %ImplicitAs.type.53b [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMemberFunctionCall.decl: %StaticMemberFunctionCall.type = fn_decl @StaticMemberFunctionCall [concrete = constants.%StaticMemberFunctionCall] {
// CHECK:STDOUT:     %T.patt.loc8_29.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_29.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc8: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, %T.loc8_29.1 [symbolic = %T.loc8_29.2 (constants.%T)]
// CHECK:STDOUT:     %Class.loc8_49.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc8_49.2 (constants.%Class)]
// CHECK:STDOUT:     %T.loc8_29.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_29.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = out_param call_param0
// CHECK:STDOUT:     %return: ref @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Class(%T.loc4_13.2) [symbolic = %Make.type (constants.%Make.type)]
// CHECK:STDOUT:   %Make: @Class.%Make.type (%Make.type) = struct_value () [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Make.decl: @Class.%Make.type (%Make.type) = fn_decl @Make [symbolic = @Class.%Make (constants.%Make)] {
// CHECK:STDOUT:       %return.patt: @Make.%Class.loc5_23.1 (%Class) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Make.%Class.loc5_23.1 (%Class) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc5_23.2: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc5_23.1 (constants.%Class)]
// CHECK:STDOUT:       %return.param: ref @Make.%Class.loc5_23.1 (%Class) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Make.%Class.loc5_23.1 (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .Class = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(@Class.%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc5_23.1: type = class_type @Class, @Class(%T) [symbolic = %Class.loc5_23.1 (constants.%Class)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Make.%Class.loc5_23.1 (%Class) [symbolic = %require_complete (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.val: @Make.%Class.loc5_23.1 (%Class) = struct_value () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param_patt: @Make.%Class.loc5_23.1 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_35.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_35.2: init @Make.%Class.loc5_23.1 (%Class) = class_init (), %return [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     %.loc5_36: init @Make.%Class.loc5_23.1 (%Class) = converted %.loc5_35.1, %.loc5_35.2 [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     return %.loc5_36 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StaticMemberFunctionCall(%T.loc8_29.1: type) {
// CHECK:STDOUT:   %T.loc8_29.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_29.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_29.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_29.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Class.loc8_49.2: type = class_type @Class, @Class(%T.loc8_29.2) [symbolic = %Class.loc8_49.2 (constants.%Class)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_18: <witness> = require_complete_type @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) [symbolic = %require_complete.loc16_18 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Class(%T.loc8_29.2) [symbolic = %Make.type (constants.%Make.type)]
// CHECK:STDOUT:   %Make: @StaticMemberFunctionCall.%Make.type (%Make.type) = struct_value () [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:   %Make.specific_fn.loc16_18.2: <specific function> = specific_function %Make, @Make(%T.loc8_29.2) [symbolic = %Make.specific_fn.loc16_18.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Class.loc8_49.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.53b)]
// CHECK:STDOUT:   %require_complete.loc16_25: <witness> = require_complete_type @StaticMemberFunctionCall.%ImplicitAs.type (%ImplicitAs.type.53b) [symbolic = %require_complete.loc16_25 (constants.%require_complete.d81)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc8_29.1: type) -> %return.param_patt: @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref.loc16: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc16: type = name_ref T, %T.loc8_29.1 [symbolic = %T.loc8_29.2 (constants.%T)]
// CHECK:STDOUT:     %Class.loc16: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc8_49.2 (constants.%Class)]
// CHECK:STDOUT:     %.loc16_18: @StaticMemberFunctionCall.%Make.type (%Make.type) = specific_constant @Class.%Make.decl, @Class(constants.%T) [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:     %Make.ref: @StaticMemberFunctionCall.%Make.type (%Make.type) = name_ref Make, %.loc16_18 [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:     %Make.specific_fn.loc16_18.1: <specific function> = specific_function %Make.ref, @Make(constants.%T) [symbolic = %Make.specific_fn.loc16_18.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:     %.loc16_24: ref @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = temporary_storage
// CHECK:STDOUT:     %Make.call: init @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = call %Make.specific_fn.loc16_18.1() to %.loc16_24
// CHECK:STDOUT:     %.loc16_25: @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = converted %Make.call, <error> [concrete = <error>]
// CHECK:STDOUT:     return <error> to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type
// CHECK:STDOUT:   %Make => constants.%Make
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class.loc5_23.1 => constants.%Class
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4f8
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Make.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc4_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StaticMemberFunctionCall(constants.%T) {
// CHECK:STDOUT:   %T.loc8_29.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_29.2 => constants.%T
// CHECK:STDOUT:   %Class.loc8_49.2 => constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@StaticMemberFunctionCall.%T.loc8_29.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(@StaticMemberFunctionCall.%T.loc8_29.2) {}
// CHECK:STDOUT:

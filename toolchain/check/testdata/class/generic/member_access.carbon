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
  // CHECK:STDERR: fail_todo_static_member_fn_call.carbon:[[@LINE+6]]:3: error: cannot implicitly convert from `Class(T)` to `Class(T)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return Class(T).Make();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_static_member_fn_call.carbon:[[@LINE+3]]:3: note: type `Class(T)` does not implement interface `ImplicitAs(Class(T))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return Class(T).Make();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  return Class(T).Make();
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.1: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.elem.1: type = unbound_element_type %Class.1, %T [symbolic]
// CHECK:STDOUT:   %Get.type.1: type = fn_type @Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Get.1: %Get.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %Class.1 [symbolic]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %GetAddr.type.1: type = fn_type @GetAddr, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetAddr.1: %GetAddr.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%i32) [template]
// CHECK:STDOUT:   %DirectFieldAccess.type: type = fn_type @DirectFieldAccess [template]
// CHECK:STDOUT:   %DirectFieldAccess: %DirectFieldAccess.type = struct_value () [template]
// CHECK:STDOUT:   %Class.elem.2: type = unbound_element_type %Class.2, %i32 [template]
// CHECK:STDOUT:   %Get.type.2: type = fn_type @Get, @Class(%i32) [template]
// CHECK:STDOUT:   %Get.2: %Get.type.2 = struct_value () [template]
// CHECK:STDOUT:   %GetAddr.type.2: type = fn_type @GetAddr, @Class(%i32) [template]
// CHECK:STDOUT:   %GetAddr.2: %GetAddr.type.2 = struct_value () [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.x.2 [template]
// CHECK:STDOUT:   %MethodCall.type: type = fn_type @MethodCall [template]
// CHECK:STDOUT:   %MethodCall: %MethodCall.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.5: type = ptr_type %Class.2 [template]
// CHECK:STDOUT:   %AddrMethodCall.type: type = fn_type @AddrMethodCall [template]
// CHECK:STDOUT:   %AddrMethodCall: %AddrMethodCall.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.6: type = ptr_type %i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .DirectFieldAccess = %DirectFieldAccess.decl
// CHECK:STDOUT:     .MethodCall = %MethodCall.decl
// CHECK:STDOUT:     .AddrMethodCall = %AddrMethodCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc2_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc2_13.1, runtime_param<invalid> [symbolic = %T.patt.loc2_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc2_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DirectFieldAccess.decl: %DirectFieldAccess.type = fn_decl @DirectFieldAccess [template = constants.%DirectFieldAccess] {
// CHECK:STDOUT:     %x.patt: %Class.2 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %Class.2 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.generic]
// CHECK:STDOUT:     %int_32.loc10_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_31: init type = call constants.%Int(%int_32.loc10_31) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_34.1: type = value_of_initializer %int.make_type_signed.loc10_31 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_34.2: type = converted %int.make_type_signed.loc10_31, %.loc10_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%i32) [template = constants.%Class.2]
// CHECK:STDOUT:     %int_32.loc10_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_40: init type = call constants.%Int(%int_32.loc10_40) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_40.1: type = value_of_initializer %int.make_type_signed.loc10_40 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_40.2: type = converted %int.make_type_signed.loc10_40, %.loc10_40.1 [template = constants.%i32]
// CHECK:STDOUT:     %x.param: %Class.2 = value_param runtime_param0
// CHECK:STDOUT:     %x: %Class.2 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MethodCall.decl: %MethodCall.type = fn_decl @MethodCall [template = constants.%MethodCall] {
// CHECK:STDOUT:     %x.patt: %Class.2 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %Class.2 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.generic]
// CHECK:STDOUT:     %int_32.loc14_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc14_24: init type = call constants.%Int(%int_32.loc14_24) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_27.1: type = value_of_initializer %int.make_type_signed.loc14_24 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_27.2: type = converted %int.make_type_signed.loc14_24, %.loc14_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%i32) [template = constants.%Class.2]
// CHECK:STDOUT:     %int_32.loc14_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc14_33: init type = call constants.%Int(%int_32.loc14_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_33.1: type = value_of_initializer %int.make_type_signed.loc14_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_33.2: type = converted %int.make_type_signed.loc14_33, %.loc14_33.1 [template = constants.%i32]
// CHECK:STDOUT:     %x.param: %Class.2 = value_param runtime_param0
// CHECK:STDOUT:     %x: %Class.2 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrMethodCall.decl: %AddrMethodCall.type = fn_decl @AddrMethodCall [template = constants.%AddrMethodCall] {
// CHECK:STDOUT:     %p.patt: %ptr.5 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.5 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.generic]
// CHECK:STDOUT:     %int_32.loc18_28: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc18_28: init type = call constants.%Int(%int_32.loc18_28) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_31.1: type = value_of_initializer %int.make_type_signed.loc18_28 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_31.2: type = converted %int.make_type_signed.loc18_28, %.loc18_31.1 [template = constants.%i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%i32) [template = constants.%Class.2]
// CHECK:STDOUT:     %ptr: type = ptr_type %Class.2 [template = constants.%ptr.5]
// CHECK:STDOUT:     %int_32.loc18_38: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc18_38: init type = call constants.%Int(%int_32.loc18_38) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_38.1: type = value_of_initializer %int.make_type_signed.loc18_38 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_38.2: type = converted %int.make_type_signed.loc18_38, %.loc18_38.1 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.5 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.5 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc2_13.1: type) {
// CHECK:STDOUT:   %T.loc2_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc2_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc2_13.2) [symbolic = %Class (constants.%Class.1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Class.%Class (%Class.1), @Class.%T.loc2_13.2 (%T) [symbolic = %Class.elem (constants.%Class.elem.1)]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T.loc2_13.2) [symbolic = %Get.type (constants.%Get.type.1)]
// CHECK:STDOUT:   %Get: @Class.%Get.type (%Get.type.1) = struct_value () [symbolic = %Get (constants.%Get.1)]
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T.loc2_13.2) [symbolic = %GetAddr.type (constants.%GetAddr.type.1)]
// CHECK:STDOUT:   %GetAddr: @Class.%GetAddr.type (%GetAddr.type.1) = struct_value () [symbolic = %GetAddr (constants.%GetAddr.1)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.loc2_13.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.1)]
// CHECK:STDOUT:   %complete_type.loc8_1.2: <witness> = complete_type_witness @Class.%struct_type.x (%struct_type.x.1) [symbolic = %complete_type.loc8_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc2_13.1 [symbolic = %T.loc2_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc3: @Class.%Class.elem (%Class.elem.1) = field_decl x, element0 [template]
// CHECK:STDOUT:     %Get.decl: @Class.%Get.type (%Get.type.1) = fn_decl @Get [symbolic = @Class.%Get (constants.%Get.1)] {
// CHECK:STDOUT:       %self.patt: @Get.%Class (%Class.1) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Get.%Class (%Class.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Get.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Get.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_16: type = specific_constant constants.%Class.1, @Class(constants.%T) [symbolic = %Class (constants.%Class.1)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc5_16 [symbolic = %Class (constants.%Class.1)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc2_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Get.%Class (%Class.1) = value_param runtime_param0
// CHECK:STDOUT:       %self: @Get.%Class (%Class.1) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Get.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Get.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetAddr.decl: @Class.%GetAddr.type (%GetAddr.type.1) = fn_decl @GetAddr [symbolic = @Class.%GetAddr (constants.%GetAddr.1)] {
// CHECK:STDOUT:       %self.patt: @GetAddr.%ptr.loc7_29.1 (%ptr.1) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @GetAddr.%ptr.loc7_29.1 (%ptr.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %.loc7_14: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:       %return.patt: @GetAddr.%ptr.loc7_38.1 (%ptr.2) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @GetAddr.%ptr.loc7_38.1 (%ptr.2) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc7_25: type = specific_constant constants.%Class.1, @Class(constants.%T) [symbolic = %Class (constants.%Class.1)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc7_25 [symbolic = %Class (constants.%Class.1)]
// CHECK:STDOUT:       %ptr.loc7_29.2: type = ptr_type %Class.1 [symbolic = %ptr.loc7_29.1 (constants.%ptr.1)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc2_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %ptr.loc7_38.2: type = ptr_type %T [symbolic = %ptr.loc7_38.1 (constants.%ptr.2)]
// CHECK:STDOUT:       %self.param: @GetAddr.%ptr.loc7_29.1 (%ptr.1) = value_param runtime_param0
// CHECK:STDOUT:       %self: @GetAddr.%ptr.loc7_29.1 (%ptr.1) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @GetAddr.%ptr.loc7_38.1 (%ptr.2) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @GetAddr.%ptr.loc7_38.1 (%ptr.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc8_1.1: <witness> = complete_type_witness %struct_type.x.1 [symbolic = %complete_type.loc8_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.1
// CHECK:STDOUT:     .x = %.loc3
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:     .GetAddr = %GetAddr.decl
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc8_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Get(@Class.%T.loc2_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Get.%Class (%Class.1), @Get.%T (%T) [symbolic = %Class.elem (constants.%Class.elem.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Get.%Class (%Class.1)]() -> @Get.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Get.%Class (%Class.1) = name_ref self, %self
// CHECK:STDOUT:     %x.ref: @Get.%Class.elem (%Class.elem.1) = name_ref x, @Class.%.loc3 [template = @Class.%.loc3]
// CHECK:STDOUT:     %.loc5_42.1: ref @Get.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc5_42.2: @Get.%T (%T) = bind_value %.loc5_42.1
// CHECK:STDOUT:     return %.loc5_42.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetAddr(@Class.%T.loc2_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.1)]
// CHECK:STDOUT:   %ptr.loc7_29.1: type = ptr_type @GetAddr.%Class (%Class.1) [symbolic = %ptr.loc7_29.1 (constants.%ptr.1)]
// CHECK:STDOUT:   %ptr.loc7_38.1: type = ptr_type @GetAddr.%T (%T) [symbolic = %ptr.loc7_38.1 (constants.%ptr.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @GetAddr.%Class (%Class.1), @GetAddr.%T (%T) [symbolic = %Class.elem (constants.%Class.elem.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self.param_patt: @GetAddr.%ptr.loc7_29.1 (%ptr.1)]() -> @GetAddr.%ptr.loc7_38.1 (%ptr.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetAddr.%ptr.loc7_29.1 (%ptr.1) = name_ref self, %self
// CHECK:STDOUT:     %.loc7_54.1: ref @GetAddr.%Class (%Class.1) = deref %self.ref
// CHECK:STDOUT:     %x.ref: @GetAddr.%Class.elem (%Class.elem.1) = name_ref x, @Class.%.loc3 [template = @Class.%.loc3]
// CHECK:STDOUT:     %.loc7_54.2: ref @GetAddr.%T (%T) = class_element_access %.loc7_54.1, element0
// CHECK:STDOUT:     %addr: @GetAddr.%ptr.loc7_38.1 (%ptr.2) = addr_of %.loc7_54.2
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param_patt: %Class.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc11_10: %Class.2 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc11_11: %Class.elem.2 = name_ref x, @Class.%.loc3 [template = @Class.%.loc3]
// CHECK:STDOUT:   %.loc11_11.1: ref %i32 = class_element_access %x.ref.loc11_10, element0
// CHECK:STDOUT:   %.loc11_11.2: %i32 = bind_value %.loc11_11.1
// CHECK:STDOUT:   return %.loc11_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param_patt: %Class.2) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.2 = name_ref x, %x
// CHECK:STDOUT:   %.loc15_11: %Get.type.2 = specific_constant @Class.%Get.decl, @Class(constants.%i32) [template = constants.%Get.2]
// CHECK:STDOUT:   %Get.ref: %Get.type.2 = name_ref Get, %.loc15_11 [template = constants.%Get.2]
// CHECK:STDOUT:   %Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.bound, @Get(constants.%i32)
// CHECK:STDOUT:   %Get.call: init %i32 = call %Get.specific_fn(%x.ref)
// CHECK:STDOUT:   %.loc15_17.1: %i32 = value_of_initializer %Get.call
// CHECK:STDOUT:   %.loc15_17.2: %i32 = converted %Get.call, %.loc15_17.1
// CHECK:STDOUT:   return %.loc15_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param_patt: %ptr.5) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.5 = name_ref p, %p
// CHECK:STDOUT:   %.loc19_12.1: ref %Class.2 = deref %p.ref
// CHECK:STDOUT:   %.loc19_12.2: %GetAddr.type.2 = specific_constant @Class.%GetAddr.decl, @Class(constants.%i32) [template = constants.%GetAddr.2]
// CHECK:STDOUT:   %GetAddr.ref: %GetAddr.type.2 = name_ref GetAddr, %.loc19_12.2 [template = constants.%GetAddr.2]
// CHECK:STDOUT:   %GetAddr.bound: <bound method> = bound_method %.loc19_12.1, %GetAddr.ref
// CHECK:STDOUT:   %GetAddr.specific_fn: <specific function> = specific_function %GetAddr.bound, @GetAddr(constants.%i32)
// CHECK:STDOUT:   %addr: %ptr.5 = addr_of %.loc19_12.1
// CHECK:STDOUT:   %GetAddr.call: init %ptr.6 = call %GetAddr.specific_fn(%addr)
// CHECK:STDOUT:   %.loc19_22.1: %ptr.6 = value_of_initializer %GetAddr.call
// CHECK:STDOUT:   %.loc19_22.2: %ptr.6 = converted %GetAddr.call, %.loc19_22.1
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
// CHECK:STDOUT:   %Class => constants.%Class.1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.1
// CHECK:STDOUT:   %Get.type => constants.%Get.type.1
// CHECK:STDOUT:   %Get => constants.%Get.1
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type.1
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr.1
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.1
// CHECK:STDOUT:   %complete_type.loc8_1.2 => constants.%complete_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Get.%T) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetAddr.%T) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.1
// CHECK:STDOUT:   %ptr.loc7_29.1 => constants.%ptr.1
// CHECK:STDOUT:   %ptr.loc7_38.1 => constants.%ptr.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc2_13.2) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc2_13.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc2_13.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2
// CHECK:STDOUT:   %Get.type => constants.%Get.type.2
// CHECK:STDOUT:   %Get => constants.%Get.2
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type.2
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.2
// CHECK:STDOUT:   %complete_type.loc8_1.2 => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %ptr.loc7_29.1 => constants.%ptr.5
// CHECK:STDOUT:   %ptr.loc7_38.1 => constants.%ptr.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_static_member_fn_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Class.val: %Class = struct_value () [symbolic]
// CHECK:STDOUT:   %StaticMemberFunctionCall.type: type = fn_type @StaticMemberFunctionCall [template]
// CHECK:STDOUT:   %StaticMemberFunctionCall: %StaticMemberFunctionCall.type = struct_value () [template]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .StaticMemberFunctionCall = %StaticMemberFunctionCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMemberFunctionCall.decl: %StaticMemberFunctionCall.type = fn_decl @StaticMemberFunctionCall [template = constants.%StaticMemberFunctionCall] {
// CHECK:STDOUT:     %T.patt.loc8_29.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_29.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_29.1, runtime_param<invalid> [symbolic = %T.patt.loc8_29.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc8: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, %T.loc8_29.1 [symbolic = %T.loc8_29.2 (constants.%T)]
// CHECK:STDOUT:     %Class.loc8_49.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc8_49.2 (constants.%Class)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_29.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_29.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = out_param runtime_param0
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
// CHECK:STDOUT:       %return.param_patt: @Make.%Class.loc5_23.1 (%Class) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc5_23.2: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc5_23.1 (constants.%Class)]
// CHECK:STDOUT:       %return.param: ref @Make.%Class.loc5_23.1 (%Class) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @Make.%Class.loc5_23.1 (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(@Class.%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc5_23.1: type = class_type @Class, @Class(%T) [symbolic = %Class.loc5_23.1 (constants.%Class)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
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
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Class(%T.loc8_29.2) [symbolic = %Make.type (constants.%Make.type)]
// CHECK:STDOUT:   %Make: @StaticMemberFunctionCall.%Make.type (%Make.type) = struct_value () [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:   %Make.specific_fn.loc15_18.2: <specific function> = specific_function %Make, @Make(%T.loc8_29.2) [symbolic = %Make.specific_fn.loc15_18.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) -> %return.param_patt: @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref.loc15: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc15: type = name_ref T, %T.loc8_29.1 [symbolic = %T.loc8_29.2 (constants.%T)]
// CHECK:STDOUT:     %Class.loc15: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc8_49.2 (constants.%Class)]
// CHECK:STDOUT:     %.loc15_18: @StaticMemberFunctionCall.%Make.type (%Make.type) = specific_constant @Class.%Make.decl, @Class(constants.%T) [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:     %Make.ref: @StaticMemberFunctionCall.%Make.type (%Make.type) = name_ref Make, %.loc15_18 [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:     %Make.specific_fn.loc15_18.1: <specific function> = specific_function %Make.ref, @Make(constants.%T) [symbolic = %Make.specific_fn.loc15_18.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:     %.loc15_24: ref @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = temporary_storage
// CHECK:STDOUT:     %Make.call: init @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = call %Make.specific_fn.loc15_18.1() to %.loc15_24
// CHECK:STDOUT:     %.loc15_25: @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = converted %Make.call, <error> [template = <error>]
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
// CHECK:STDOUT: specific @Class(@Make.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class.loc5_23.1 => constants.%Class
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc4_13.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@StaticMemberFunctionCall.%T.loc8_29.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StaticMemberFunctionCall(constants.%T) {
// CHECK:STDOUT:   %T.loc8_29.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_29.2 => constants.%T
// CHECK:STDOUT:   %Class.loc8_49.2 => constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(@StaticMemberFunctionCall.%T.loc8_29.2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class.loc5_23.1 => constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:

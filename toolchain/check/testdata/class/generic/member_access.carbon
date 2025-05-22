// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
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
  //@dump-sem-ir-begin
  return x.x;
  //@dump-sem-ir-end
}

fn MethodCall(x: Class(i32)) -> i32 {
  //@dump-sem-ir-begin
  return x.Get();
  //@dump-sem-ir-end
}

fn AddrMethodCall(p: Class(i32)*) -> i32 {
  //@dump-sem-ir-begin
  return *p->GetAddr();
  //@dump-sem-ir-end
}

// --- static_member_fn_call.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  fn Make() -> Class(T) { return {}; }
}

fn StaticMemberFunctionCall(T:! type) -> Class(T) {
  //@dump-sem-ir-begin
  return Class(T).Make();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.247: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %Class.elem.2d8: type = unbound_element_type %Class.247, %i32 [concrete]
// CHECK:STDOUT:   %Get.type.59d: type = fn_type @Get, @Class(%i32) [concrete]
// CHECK:STDOUT:   %Get.a40: %Get.type.59d = struct_value () [concrete]
// CHECK:STDOUT:   %GetAddr.type.be7: type = fn_type @GetAddr, @Class(%i32) [concrete]
// CHECK:STDOUT:   %GetAddr.909: %GetAddr.type.be7 = struct_value () [concrete]
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.a40, @Get(%i32) [concrete]
// CHECK:STDOUT:   %ptr.f7c: type = ptr_type %Class.247 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %GetAddr.specific_fn: <specific function> = specific_function %GetAddr.909, @GetAddr(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x.param: %Class.247) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc12_10: %Class.247 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc12_11: %Class.elem.2d8 = name_ref x, @Class.%.loc3 [concrete = @Class.%.loc3]
// CHECK:STDOUT:   %.loc12_11.1: ref %i32 = class_element_access %x.ref.loc12_10, element0
// CHECK:STDOUT:   %.loc12_11.2: %i32 = bind_value %.loc12_11.1
// CHECK:STDOUT:   return %.loc12_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x.param: %Class.247) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.247 = name_ref x, %x
// CHECK:STDOUT:   %.loc18_11: %Get.type.59d = specific_constant @Class.%Get.decl, @Class(constants.%i32) [concrete = constants.%Get.a40]
// CHECK:STDOUT:   %Get.ref: %Get.type.59d = name_ref Get, %.loc18_11 [concrete = constants.%Get.a40]
// CHECK:STDOUT:   %Get.bound: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Get.specific_fn: <specific function> = specific_function %Get.ref, @Get(constants.%i32) [concrete = constants.%Get.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.ref, %Get.specific_fn
// CHECK:STDOUT:   %Get.call: init %i32 = call %bound_method(%x.ref)
// CHECK:STDOUT:   %.loc18_17.1: %i32 = value_of_initializer %Get.call
// CHECK:STDOUT:   %.loc18_17.2: %i32 = converted %Get.call, %.loc18_17.1
// CHECK:STDOUT:   return %.loc18_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p.param: %ptr.f7c) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.f7c = name_ref p, %p
// CHECK:STDOUT:   %.loc24_12.1: ref %Class.247 = deref %p.ref
// CHECK:STDOUT:   %.loc24_12.2: %GetAddr.type.be7 = specific_constant @Class.%GetAddr.decl, @Class(constants.%i32) [concrete = constants.%GetAddr.909]
// CHECK:STDOUT:   %GetAddr.ref: %GetAddr.type.be7 = name_ref GetAddr, %.loc24_12.2 [concrete = constants.%GetAddr.909]
// CHECK:STDOUT:   %GetAddr.bound: <bound method> = bound_method %.loc24_12.1, %GetAddr.ref
// CHECK:STDOUT:   %GetAddr.specific_fn: <specific function> = specific_function %GetAddr.ref, @GetAddr(constants.%i32) [concrete = constants.%GetAddr.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc24_12.1, %GetAddr.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f7c = addr_of %.loc24_12.1
// CHECK:STDOUT:   %GetAddr.call: init %ptr.235 = call %bound_method(%addr)
// CHECK:STDOUT:   %.loc24_22.1: %ptr.235 = value_of_initializer %GetAddr.call
// CHECK:STDOUT:   %.loc24_22.2: %ptr.235 = converted %GetAddr.call, %.loc24_22.1
// CHECK:STDOUT:   %.loc24_10.1: ref %i32 = deref %.loc24_22.2
// CHECK:STDOUT:   %.loc24_10.2: %i32 = bind_value %.loc24_10.1
// CHECK:STDOUT:   return %.loc24_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- static_member_fn_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.3c1: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StaticMemberFunctionCall(%T.loc8_29.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc8_49.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Class(%T.loc8_29.2) [symbolic = %Make.type (constants.%Make.type)]
// CHECK:STDOUT:   %Make: @StaticMemberFunctionCall.%Make.type (%Make.type) = struct_value () [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:   %Make.specific_fn.loc10_18.2: <specific function> = specific_function %Make, @Make(%T.loc8_29.2) [symbolic = %Make.specific_fn.loc10_18.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref.loc10: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc8_29.1 [symbolic = %T.loc8_29.2 (constants.%T)]
// CHECK:STDOUT:     %Class.loc10: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc8_49.2 (constants.%Class)]
// CHECK:STDOUT:     %.loc10: @StaticMemberFunctionCall.%Make.type (%Make.type) = specific_constant @Class.%Make.decl, @Class(constants.%T) [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:     %Make.ref: @StaticMemberFunctionCall.%Make.type (%Make.type) = name_ref Make, %.loc10 [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:     %Make.specific_fn.loc10_18.1: <specific function> = specific_function %Make.ref, @Make(constants.%T) [symbolic = %Make.specific_fn.loc10_18.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Make.call: init @StaticMemberFunctionCall.%Class.loc8_49.2 (%Class) = call %Make.specific_fn.loc10_18.1() to %.loc8
// CHECK:STDOUT:     return %Make.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StaticMemberFunctionCall(constants.%T) {
// CHECK:STDOUT:   %T.loc8_29.2 => constants.%T
// CHECK:STDOUT:   %Class.loc8_49.2 => constants.%Class
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3c1
// CHECK:STDOUT: }
// CHECK:STDOUT:

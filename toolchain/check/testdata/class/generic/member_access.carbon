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
  // CHECK:STDERR: fail_todo_static_member_fn_call.carbon:[[@LINE+6]]:3: ERROR: Cannot implicitly convert from `Class` to `Class`.
  // CHECK:STDERR:   return Class(T).Make();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_static_member_fn_call.carbon:[[@LINE+3]]:3: Type `Class` does not implement interface `ImplicitAs`.
  // CHECK:STDERR:   return Class(T).Make();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  return Class(T).Make();
}

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %Get.type.1: type = fn_type @Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Get.1: %Get.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type %Class.2 [symbolic]
// CHECK:STDOUT:   %.4: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %GetAddr.type.1: type = fn_type @GetAddr, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetAddr.1: %GetAddr.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(i32) [template]
// CHECK:STDOUT:   %DirectFieldAccess.type: type = fn_type @DirectFieldAccess [template]
// CHECK:STDOUT:   %DirectFieldAccess: %DirectFieldAccess.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Class.3, i32 [template]
// CHECK:STDOUT:   %Get.type.2: type = fn_type @Get, @Class(i32) [template]
// CHECK:STDOUT:   %Get.2: %Get.type.2 = struct_value () [template]
// CHECK:STDOUT:   %GetAddr.type.2: type = fn_type @GetAddr, @Class(i32) [template]
// CHECK:STDOUT:   %GetAddr.2: %GetAddr.type.2 = struct_value () [template]
// CHECK:STDOUT:   %MethodCall.type: type = fn_type @MethodCall [template]
// CHECK:STDOUT:   %MethodCall: %MethodCall.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = ptr_type %Class.3 [template]
// CHECK:STDOUT:   %AddrMethodCall.type: type = fn_type @AddrMethodCall [template]
// CHECK:STDOUT:   %AddrMethodCall: %AddrMethodCall.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc2_13.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc2_13.2: type = bind_symbolic_name T 0, %T.loc2_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DirectFieldAccess.decl: %DirectFieldAccess.type = fn_decl @DirectFieldAccess [template = constants.%DirectFieldAccess] {
// CHECK:STDOUT:     %Class.ref.loc10: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %int.make_type_32.loc10_31: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc10_30.1: type = value_of_initializer %int.make_type_32.loc10_31 [template = i32]
// CHECK:STDOUT:     %.loc10_30.2: type = converted %int.make_type_32.loc10_31, %.loc10_30.1 [template = i32]
// CHECK:STDOUT:     %Class.loc10: type = class_type @Class, @Class(i32) [template = constants.%Class.3]
// CHECK:STDOUT:     %x.loc10_22.1: %Class.3 = param x, runtime_param0
// CHECK:STDOUT:     @DirectFieldAccess.%x: %Class.3 = bind_name x, %x.loc10_22.1
// CHECK:STDOUT:     %int.make_type_32.loc10_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc10_40.1: type = value_of_initializer %int.make_type_32.loc10_40 [template = i32]
// CHECK:STDOUT:     %.loc10_40.2: type = converted %int.make_type_32.loc10_40, %.loc10_40.1 [template = i32]
// CHECK:STDOUT:     @DirectFieldAccess.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MethodCall.decl: %MethodCall.type = fn_decl @MethodCall [template = constants.%MethodCall] {
// CHECK:STDOUT:     %Class.ref.loc14: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %int.make_type_32.loc14_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_23.1: type = value_of_initializer %int.make_type_32.loc14_24 [template = i32]
// CHECK:STDOUT:     %.loc14_23.2: type = converted %int.make_type_32.loc14_24, %.loc14_23.1 [template = i32]
// CHECK:STDOUT:     %Class.loc14: type = class_type @Class, @Class(i32) [template = constants.%Class.3]
// CHECK:STDOUT:     %x.loc14_15.1: %Class.3 = param x, runtime_param0
// CHECK:STDOUT:     @MethodCall.%x: %Class.3 = bind_name x, %x.loc14_15.1
// CHECK:STDOUT:     %int.make_type_32.loc14_33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_33.1: type = value_of_initializer %int.make_type_32.loc14_33 [template = i32]
// CHECK:STDOUT:     %.loc14_33.2: type = converted %int.make_type_32.loc14_33, %.loc14_33.1 [template = i32]
// CHECK:STDOUT:     @MethodCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrMethodCall.decl: %AddrMethodCall.type = fn_decl @AddrMethodCall [template = constants.%AddrMethodCall] {
// CHECK:STDOUT:     %Class.ref.loc18: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %int.make_type_32.loc18_28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_27.1: type = value_of_initializer %int.make_type_32.loc18_28 [template = i32]
// CHECK:STDOUT:     %.loc18_27.2: type = converted %int.make_type_32.loc18_28, %.loc18_27.1 [template = i32]
// CHECK:STDOUT:     %Class.loc18: type = class_type @Class, @Class(i32) [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc18_32: type = ptr_type %Class.3 [template = constants.%.8]
// CHECK:STDOUT:     %p.loc18_19.1: %.8 = param p, runtime_param0
// CHECK:STDOUT:     @AddrMethodCall.%p: %.8 = bind_name p, %p.loc18_19.1
// CHECK:STDOUT:     %int.make_type_32.loc18_38: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_38.1: type = value_of_initializer %int.make_type_32.loc18_38 [template = i32]
// CHECK:STDOUT:     %.loc18_38.2: type = converted %int.make_type_32.loc18_38, %.loc18_38.1 [template = i32]
// CHECK:STDOUT:     @AddrMethodCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc2_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T) [symbolic = %Get.type (constants.%Get.type.1)]
// CHECK:STDOUT:   %Get: @Class.%Get.type (%Get.type.1) = struct_value () [symbolic = %Get (constants.%Get.1)]
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T) [symbolic = %GetAddr.type (constants.%GetAddr.type.1)]
// CHECK:STDOUT:   %GetAddr: @Class.%GetAddr.type (%GetAddr.type.1) = struct_value () [symbolic = %GetAddr (constants.%GetAddr.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref.loc3: type = name_ref T, file.%T.loc2_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc3: @Class.%.1 (%.2) = field_decl x, element0 [template]
// CHECK:STDOUT:     %Get.decl: @Class.%Get.type (%Get.type.1) = fn_decl @Get [symbolic = %Get (constants.%Get.1)] {
// CHECK:STDOUT:       %.loc5: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = @Get.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref.loc5: type = name_ref Self, %.loc5 [symbolic = @Get.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %self.loc5_10.1: @Get.%Class (%Class.2) = param self, runtime_param0
// CHECK:STDOUT:       %self.loc5_10.2: @Get.%Class (%Class.2) = bind_name self, %self.loc5_10.1
// CHECK:STDOUT:       %T.ref.loc5: type = name_ref T, file.%T.loc2_13.2 [symbolic = @Get.%T (constants.%T)]
// CHECK:STDOUT:       %return.var.loc5: ref @Get.%T (%T) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetAddr.decl: @Class.%GetAddr.type (%GetAddr.type.1) = fn_decl @GetAddr [symbolic = %GetAddr (constants.%GetAddr.1)] {
// CHECK:STDOUT:       %.loc7_25: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = @GetAddr.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref.loc7: type = name_ref Self, %.loc7_25 [symbolic = @GetAddr.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %.loc7_29: type = ptr_type %Class.2 [symbolic = @GetAddr.%.1 (constants.%.3)]
// CHECK:STDOUT:       %self.loc7_19.1: @GetAddr.%.1 (%.3) = param self, runtime_param0
// CHECK:STDOUT:       %self.loc7_19.3: @GetAddr.%.1 (%.3) = bind_name self, %self.loc7_19.1
// CHECK:STDOUT:       %.loc7_14: @GetAddr.%.1 (%.3) = addr_pattern %self.loc7_19.3
// CHECK:STDOUT:       %T.ref.loc7: type = name_ref T, file.%T.loc2_13.2 [symbolic = @GetAddr.%T (constants.%T)]
// CHECK:STDOUT:       %.loc7_38: type = ptr_type %T [symbolic = @GetAddr.%.2 (constants.%.4)]
// CHECK:STDOUT:       %return.var.loc7: ref @GetAddr.%.2 (%.4) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .x = %.loc3
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:     .GetAddr = %GetAddr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Get(file.%T.loc2_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = unbound_element_type @Get.%Class (%Class.2), @Get.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[@Class.%self.loc5_10.2: @Get.%Class (%Class.2)]() -> @Get.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Get.%Class (%Class.2) = name_ref self, @Class.%self.loc5_10.2
// CHECK:STDOUT:     %x.ref: @Get.%.1 (%.2) = name_ref x, @Class.%.loc3 [template = @Class.%.loc3]
// CHECK:STDOUT:     %.loc5_42.1: ref @Get.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc5_42.2: @Get.%T (%T) = bind_value %.loc5_42.1
// CHECK:STDOUT:     return %.loc5_42.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetAddr(file.%T.loc2_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = ptr_type @GetAddr.%Class (%Class.2) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: type = ptr_type @GetAddr.%T (%T) [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.3: type = unbound_element_type @GetAddr.%Class (%Class.2), @GetAddr.%T (%T) [symbolic = %.3 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr @Class.%self.loc7_19.3: @GetAddr.%.1 (%.3)]() -> @GetAddr.%.2 (%.4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetAddr.%.1 (%.3) = name_ref self, @Class.%self.loc7_19.3
// CHECK:STDOUT:     %.loc7_54.1: ref @GetAddr.%Class (%Class.2) = deref %self.ref
// CHECK:STDOUT:     %x.ref: @GetAddr.%.3 (%.2) = name_ref x, @Class.%.loc3 [template = @Class.%.loc3]
// CHECK:STDOUT:     %.loc7_54.2: ref @GetAddr.%T (%T) = class_element_access %.loc7_54.1, element0
// CHECK:STDOUT:     %.loc7_49: @GetAddr.%.2 (%.4) = addr_of %.loc7_54.2
// CHECK:STDOUT:     return %.loc7_49
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DirectFieldAccess(%x: %Class.3) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc11_10: %Class.3 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc11_11: %.7 = name_ref x, @Class.%.loc3 [template = @Class.%.loc3]
// CHECK:STDOUT:   %.loc11_11.1: ref i32 = class_element_access %x.ref.loc11_10, element0
// CHECK:STDOUT:   %.loc11_11.2: i32 = bind_value %.loc11_11.1
// CHECK:STDOUT:   return %.loc11_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MethodCall(%x: %Class.3) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.3 = name_ref x, %x
// CHECK:STDOUT:   %.loc15_11.1: %Get.type.2 = specific_constant @Class.%Get.decl, @Class(i32) [template = constants.%Get.2]
// CHECK:STDOUT:   %Get.ref: %Get.type.2 = name_ref Get, %.loc15_11.1 [template = constants.%Get.2]
// CHECK:STDOUT:   %.loc15_11.2: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Get.call: init i32 = call %.loc15_11.2(%x.ref)
// CHECK:STDOUT:   %.loc15_17.1: i32 = value_of_initializer %Get.call
// CHECK:STDOUT:   %.loc15_17.2: i32 = converted %Get.call, %.loc15_17.1
// CHECK:STDOUT:   return %.loc15_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrMethodCall(%p: %.8) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.8 = name_ref p, %p
// CHECK:STDOUT:   %.loc19_12.1: ref %Class.3 = deref %p.ref
// CHECK:STDOUT:   %.loc19_12.2: %GetAddr.type.2 = specific_constant @Class.%GetAddr.decl, @Class(i32) [template = constants.%GetAddr.2]
// CHECK:STDOUT:   %GetAddr.ref: %GetAddr.type.2 = name_ref GetAddr, %.loc19_12.2 [template = constants.%GetAddr.2]
// CHECK:STDOUT:   %.loc19_12.3: <bound method> = bound_method %.loc19_12.1, %GetAddr.ref
// CHECK:STDOUT:   %.loc19_12.4: %.8 = addr_of %.loc19_12.1
// CHECK:STDOUT:   %GetAddr.call: init %.9 = call %.loc19_12.3(%.loc19_12.4)
// CHECK:STDOUT:   %.loc19_21.1: %.9 = value_of_initializer %GetAddr.call
// CHECK:STDOUT:   %.loc19_21.2: %.9 = converted %GetAddr.call, %.loc19_21.1
// CHECK:STDOUT:   %.loc19_10.1: ref i32 = deref %.loc19_21.2
// CHECK:STDOUT:   %.loc19_10.2: i32 = bind_value %.loc19_10.1
// CHECK:STDOUT:   return %.loc19_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %Get.type => constants.%Get.type.1
// CHECK:STDOUT:   %Get => constants.%Get.1
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type.1
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Get.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetAddr.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT:   %.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %Get.type => constants.%Get.type.2
// CHECK:STDOUT:   %Get => constants.%Get.2
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type.2
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_static_member_fn_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %struct: %Class.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %StaticMemberFunctionCall.type: type = fn_type @StaticMemberFunctionCall [template]
// CHECK:STDOUT:   %StaticMemberFunctionCall: %StaticMemberFunctionCall.type = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic]
// CHECK:STDOUT:   %.4: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%.1 (%.4) = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %.4 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.4, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %.7: type = interface_type @ImplicitAs, @ImplicitAs(%Class.2) [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%Class.2) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %.7, %Convert.type.2 [symbolic]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %.10: %.5 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %ImplicitAs.type = import_ref Core//prelude/operators/as, inst+37, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/as, inst+42, unloaded
// CHECK:STDOUT:   %import_ref.3: @ImplicitAs.%.2 (%.5) = import_ref Core//prelude/operators/as, inst+59, loaded [symbolic = @ImplicitAs.%.3 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+52, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .StaticMemberFunctionCall = %StaticMemberFunctionCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %StaticMemberFunctionCall.decl: %StaticMemberFunctionCall.type = fn_decl @StaticMemberFunctionCall [template = constants.%StaticMemberFunctionCall] {
// CHECK:STDOUT:     %T.loc8_29.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @StaticMemberFunctionCall.%T.loc8: type = bind_symbolic_name T 0, %T.loc8_29.1 [symbolic = @StaticMemberFunctionCall.%T.1 (constants.%T)]
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, @StaticMemberFunctionCall.%T.loc8 [symbolic = @StaticMemberFunctionCall.%T.1 (constants.%T)]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%T) [symbolic = @StaticMemberFunctionCall.%Class.1 (constants.%Class.2)]
// CHECK:STDOUT:     @StaticMemberFunctionCall.%return: ref @StaticMemberFunctionCall.%Class.1 (%Class.2) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @ImplicitAs.%.1 (%.4), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: @ImplicitAs.%.2 (%.5) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Class(%T) [symbolic = %Make.type (constants.%Make.type)]
// CHECK:STDOUT:   %Make: @Class.%Make.type (%Make.type) = struct_value () [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Make.decl: @Class.%Make.type (%Make.type) = fn_decl @Make [symbolic = %Make (constants.%Make)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:       %T.ref: type = name_ref T, file.%T.loc4_13.2 [symbolic = @Make.%T (constants.%T)]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%T) [symbolic = @Make.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %return.var: ref @Make.%Class (%Class.2) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(file.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @Make.%Class (%Class.2) = struct_value () [symbolic = %struct (constants.%struct)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Class.%return.var: @Make.%Class (%Class.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_35.1: %.2 = struct_literal ()
// CHECK:STDOUT:     %.loc5_35.2: init @Make.%Class (%Class.2) = class_init (), @Class.%return.var [symbolic = %struct (constants.%struct)]
// CHECK:STDOUT:     %.loc5_36: init @Make.%Class (%Class.2) = converted %.loc5_35.1, %.loc5_35.2 [symbolic = %struct (constants.%struct)]
// CHECK:STDOUT:     return %.loc5_36 to @Class.%return.var
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StaticMemberFunctionCall(%T.loc8: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %Class.1: type = class_type @Class, @Class(%T.1) [symbolic = %Class.1 (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type: type = fn_type @Make, @Class(%T.1) [symbolic = %Make.type (constants.%Make.type)]
// CHECK:STDOUT:   %Make: @StaticMemberFunctionCall.%Make.type (%Make.type) = struct_value () [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Class.1) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Class.1) [symbolic = %Convert.type (constants.%Convert.type.2)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @StaticMemberFunctionCall.%.1 (%.7), @StaticMemberFunctionCall.%Convert.type (%Convert.type.2) [symbolic = %.2 (constants.%.8)]
// CHECK:STDOUT:   %.3: @StaticMemberFunctionCall.%.2 (%.8) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.3 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc8: type) -> %return: @StaticMemberFunctionCall.%Class.1 (%Class.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %Class.loc15: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.1 (constants.%Class.2)]
// CHECK:STDOUT:     %.loc15_18: @StaticMemberFunctionCall.%Make.type (%Make.type) = specific_constant @Class.%Make.decl, @Class(constants.%T) [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:     %Make.ref: @StaticMemberFunctionCall.%Make.type (%Make.type) = name_ref Make, %.loc15_18 [symbolic = %Make (constants.%Make)]
// CHECK:STDOUT:     %.loc15_23.1: ref @StaticMemberFunctionCall.%Class.1 (%Class.2) = temporary_storage
// CHECK:STDOUT:     %Make.call: init @StaticMemberFunctionCall.%Class.1 (%Class.2) = call %Make.ref() to %.loc15_23.1
// CHECK:STDOUT:     %.loc15_25.1: type = interface_type @ImplicitAs, @ImplicitAs(constants.%Class.2) [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:     %.loc15_25.2: @StaticMemberFunctionCall.%.2 (%.8) = specific_constant imports.%import_ref.3, @ImplicitAs(constants.%Class.2) [symbolic = %.3 (constants.%.9)]
// CHECK:STDOUT:     %Convert.ref: @StaticMemberFunctionCall.%.2 (%.8) = name_ref Convert, %.loc15_25.2 [symbolic = %.3 (constants.%.9)]
// CHECK:STDOUT:     %.loc15_23.2: ref @StaticMemberFunctionCall.%Class.1 (%Class.2) = temporary %.loc15_23.1, %Make.call
// CHECK:STDOUT:     %.loc15_25.3: @StaticMemberFunctionCall.%Class.1 (%Class.2) = converted %Make.call, <error> [template = <error>]
// CHECK:STDOUT:     return <error> to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%.1 (%.4)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %.1: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %Self: %.4 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.type => constants.%Make.type
// CHECK:STDOUT:   %Make => constants.%Make
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Make.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@StaticMemberFunctionCall.%T.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StaticMemberFunctionCall(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %Class.1 => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Class.2) {
// CHECK:STDOUT:   %Dest => constants.%Class.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT:   %.3 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@StaticMemberFunctionCall.%Class.1) {
// CHECK:STDOUT:   %Dest => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class(T:! type) {
  fn Get[addr self: Self*]() -> T* {
    return &self->k;
  }

  var k: T;
}

// TODO: The following should work.
fn Run() -> i32 {
  // CHECK:STDERR: fail_todo_use.carbon:[[@LINE+4]]:23: ERROR: Cannot implicitly convert from `i32` to `T`.
  // CHECK:STDERR:   var v: Class(i32) = {.k = 0};
  // CHECK:STDERR:                       ^~~~~~~~
  // CHECK:STDERR:
  var v: Class(i32) = {.k = 0};
  // CHECK:STDERR: fail_todo_use.carbon:[[@LINE+10]]:3: ERROR: Cannot implicitly convert from `T*` to `i32`.
  // CHECK:STDERR:   return v.Get();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use.carbon:[[@LINE+6]]:10: ERROR: Cannot implicitly convert from `Class*` to `Class*`.
  // CHECK:STDERR:   return v.Get();
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_todo_use.carbon:[[@LINE-21]]:15: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn Get[addr self: Self*]() -> T* {
  // CHECK:STDERR:               ^~~~
  return v.Get();
}

// CHECK:STDOUT: --- fail_todo_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.1: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Class = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class [template]
// CHECK:STDOUT:   %.2: type = ptr_type Class [template]
// CHECK:STDOUT:   %.3: type = ptr_type T [symbolic]
// CHECK:STDOUT:   %Get: type = fn_type @Get [template]
// CHECK:STDOUT:   %struct.2: Get = struct_value () [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Class, T [symbolic]
// CHECK:STDOUT:   %.5: type = struct_type {.k: T} [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type {.k: T} [symbolic]
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %struct.3: Run = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, (i32) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type Class [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: Class = class_decl @Class [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc7_13.1: type = param T
// CHECK:STDOUT:     %T.loc7_13.2: type = bind_symbolic_name T 0, %T.loc7_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct.3] {
// CHECK:STDOUT:     @Run.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Get.decl: Get = fn_decl @Get [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class.2 [template = constants.%Class.2]
// CHECK:STDOUT:     %.loc8_25: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:     %self.loc8_15.1: Class* = param self
// CHECK:STDOUT:     %self.loc8_15.3: Class* = bind_name self, %self.loc8_15.1
// CHECK:STDOUT:     %.loc8_10: Class* = addr_pattern %self.loc8_15.3
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, file.%T.loc7_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc8_34: type = ptr_type T [symbolic = constants.%.3]
// CHECK:STDOUT:     %return.var: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.ref.loc12: type = name_ref T, file.%T.loc7_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc12: <unbound element of class Class> = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .Get = %Get.decl
// CHECK:STDOUT:   .k = %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Get[addr @Class.%self.loc8_15.3: Class*]() -> T* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class* = name_ref self, @Class.%self.loc8_15.3
// CHECK:STDOUT:   %.loc9_17.1: ref Class = deref %self.ref
// CHECK:STDOUT:   %k.ref: <unbound element of class Class> = name_ref k, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:   %.loc9_17.2: ref T = class_element_access %.loc9_17.1, element0
// CHECK:STDOUT:   %.loc9_12: T* = addr_of %.loc9_17.2
// CHECK:STDOUT:   return %.loc9_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: Class = name_ref Class, file.%Class.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %Class: type = class_type @Class, (i32) [template = constants.%Class.3]
// CHECK:STDOUT:   %v.var: ref Class = var v
// CHECK:STDOUT:   %v: ref Class = bind_name v, %v.var
// CHECK:STDOUT:   %.loc21_29: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_30: {.k: i32} = struct_literal (%.loc21_29)
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   %v.ref: ref Class = name_ref v, %v
// CHECK:STDOUT:   %Get.ref: Get = name_ref Get, @Class.%Get.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc32_11: <bound method> = bound_method %v.ref, %Get.ref
// CHECK:STDOUT:   %.loc32_10: Class* = addr_of %v.ref
// CHECK:STDOUT:   %Get.call: init T* = call %.loc32_11(<invalid>) [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

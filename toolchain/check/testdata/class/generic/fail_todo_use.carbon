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
  // CHECK:STDERR: fail_todo_use.carbon:[[@LINE+3]]:10: ERROR: Value of type `type` is not callable.
  // CHECK:STDERR:   var v: Class(i32) = {.k = 0};
  // CHECK:STDERR:          ^~~~~~
  var v: Class(i32) = {.k = 0};
  return v.Get();
}

// CHECK:STDOUT: --- fail_todo_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT:   %.2: type = ptr_type T [symbolic]
// CHECK:STDOUT:   %.3: type = unbound_element_type Class, T [symbolic]
// CHECK:STDOUT:   %.4: type = struct_type {.k: T} [symbolic]
// CHECK:STDOUT:   %.5: type = ptr_type {.k: T} [symbolic]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.k: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Run = %Run
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {
// CHECK:STDOUT:     %T.loc7_13.1: type = param T
// CHECK:STDOUT:     %T.loc7_13.2: type = bind_symbolic_name T 0, %T.loc7_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template] {
// CHECK:STDOUT:     @Run.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Get: <function> = fn_decl @Get [template] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc8_25: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %self.loc8_15.1: Class* = param self
// CHECK:STDOUT:     %self.loc8_15.3: Class* = bind_name self, %self.loc8_15.1
// CHECK:STDOUT:     %.loc8_10: Class* = addr_pattern %self.loc8_15.3
// CHECK:STDOUT:     %T.ref.loc8: type = name_ref T, file.%T.loc7_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc8_34: type = ptr_type T [symbolic = constants.%.2]
// CHECK:STDOUT:     %return.var: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.ref.loc12: type = name_ref T, file.%T.loc7_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc12: <unbound element of class Class> = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .Get = %Get
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
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT:   %.loc20_29: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc20_30: {.k: i32} = struct_literal (%.loc20_29)
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   %v.ref: ref <error> = name_ref v, %v
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

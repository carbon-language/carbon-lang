// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[self: Class]() -> i32;
  fn G[addr self: Class*]() -> i32;

  var k: i32;
}

fn Class.F[self: Class]() -> i32 {
  return self.k;
}

fn Call(c: Class) -> i32 {
  // TODO: The sem-ir for this call doesn't distinguish the `self` argument from
  // the explicit arguments.
  return c.F();
}

fn CallWithAddr() -> i32 {
  var c: Class;
  return c.G();
}

fn CallFThroughPointer(p: Class*) -> i32 {
  return (*p).F();
}

fn CallGThroughPointer(p: Class*) -> i32 {
  return (*p).G();
}

fn Make() -> Class;

fn CallFOnInitializingExpr() -> i32 {
  return Make().F();
}

fn CallGOnInitializingExpr() -> i32 {
  return Make().G();
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc12: type = struct_type {.k: i32}
// CHECK:STDOUT:   %.loc7: type = ptr_type {.k: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "method.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %Call: <function> = fn_decl @Call
// CHECK:STDOUT:   %CallWithAddr: <function> = fn_decl @CallWithAddr
// CHECK:STDOUT:   %CallFThroughPointer: <function> = fn_decl @CallFThroughPointer
// CHECK:STDOUT:   %CallGThroughPointer: <function> = fn_decl @CallGThroughPointer
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make
// CHECK:STDOUT:   %CallFOnInitializingExpr: <function> = fn_decl @CallFOnInitializingExpr
// CHECK:STDOUT:   %CallGOnInitializingExpr: <function> = fn_decl @CallGOnInitializingExpr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %.loc11_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc11_8.2: <unbound field of class Class> = field k, member0
// CHECK:STDOUT:   %k: <unbound field of class Class> = bind_name k, %.loc11_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .k = %k
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class = name_ref self, %self
// CHECK:STDOUT:   %.loc15_14.1: ref i32 = class_field_access %self.ref, member0
// CHECK:STDOUT:   %.loc15_14.2: i32 = bind_value %.loc15_14.1
// CHECK:STDOUT:   return %.loc15_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.addr: Class*]() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%c: Class) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc21_11: <bound method> = bound_method %c.ref, @Class.%F
// CHECK:STDOUT:   %.loc21_13: init i32 = call %.loc21_11(%c.ref)
// CHECK:STDOUT:   %.loc21_15.1: i32 = value_of_initializer %.loc21_13
// CHECK:STDOUT:   %.loc21_15.2: i32 = converted %.loc21_13, %.loc21_15.1
// CHECK:STDOUT:   return %.loc21_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithAddr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc26_11: <bound method> = bound_method %c.ref, @Class.%G
// CHECK:STDOUT:   %.loc26_10: Class* = address_of %c.ref
// CHECK:STDOUT:   %.loc26_13: init i32 = call %.loc26_11(%.loc26_10)
// CHECK:STDOUT:   %.loc26_15.1: i32 = value_of_initializer %.loc26_13
// CHECK:STDOUT:   %.loc26_15.2: i32 = converted %.loc26_13, %.loc26_15.1
// CHECK:STDOUT:   return %.loc26_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFThroughPointer(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc30_11.1: ref Class = deref %p.ref
// CHECK:STDOUT:   %.loc30_14: <bound method> = bound_method %.loc30_11.1, @Class.%F
// CHECK:STDOUT:   %.loc30_11.2: Class = bind_value %.loc30_11.1
// CHECK:STDOUT:   %.loc30_16: init i32 = call %.loc30_14(%.loc30_11.2)
// CHECK:STDOUT:   %.loc30_18.1: i32 = value_of_initializer %.loc30_16
// CHECK:STDOUT:   %.loc30_18.2: i32 = converted %.loc30_16, %.loc30_18.1
// CHECK:STDOUT:   return %.loc30_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGThroughPointer(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc34_11.1: ref Class = deref %p.ref
// CHECK:STDOUT:   %.loc34_14: <bound method> = bound_method %.loc34_11.1, @Class.%G
// CHECK:STDOUT:   %.loc34_11.2: Class* = address_of %.loc34_11.1
// CHECK:STDOUT:   %.loc34_16: init i32 = call %.loc34_14(%.loc34_11.2)
// CHECK:STDOUT:   %.loc34_18.1: i32 = value_of_initializer %.loc34_16
// CHECK:STDOUT:   %.loc34_18.2: i32 = converted %.loc34_16, %.loc34_18.1
// CHECK:STDOUT:   return %.loc34_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFOnInitializingExpr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: <function> = name_ref Make, file.%Make
// CHECK:STDOUT:   %.loc40_14.1: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc40_14.2: init Class = call %Make.ref() to %.loc40_14.1
// CHECK:STDOUT:   %.loc40_14.3: ref Class = temporary %.loc40_14.1, %.loc40_14.2
// CHECK:STDOUT:   %.loc40_16: <bound method> = bound_method %.loc40_14.3, @Class.%F
// CHECK:STDOUT:   %.loc40_14.4: Class = bind_value %.loc40_14.3
// CHECK:STDOUT:   %.loc40_18: init i32 = call %.loc40_16(%.loc40_14.4)
// CHECK:STDOUT:   %.loc40_20.1: i32 = value_of_initializer %.loc40_18
// CHECK:STDOUT:   %.loc40_20.2: i32 = converted %.loc40_18, %.loc40_20.1
// CHECK:STDOUT:   return %.loc40_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGOnInitializingExpr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: <function> = name_ref Make, file.%Make
// CHECK:STDOUT:   %.loc44_14.1: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc44_14.2: init Class = call %Make.ref() to %.loc44_14.1
// CHECK:STDOUT:   %.loc44_14.3: ref Class = temporary %.loc44_14.1, %.loc44_14.2
// CHECK:STDOUT:   %.loc44_16: <bound method> = bound_method %.loc44_14.3, @Class.%G
// CHECK:STDOUT:   %.loc44_14.4: Class* = address_of %.loc44_14.3
// CHECK:STDOUT:   %.loc44_18: init i32 = call %.loc44_16(%.loc44_14.4)
// CHECK:STDOUT:   %.loc44_20.1: i32 = value_of_initializer %.loc44_18
// CHECK:STDOUT:   %.loc44_20.2: i32 = converted %.loc44_18, %.loc44_20.1
// CHECK:STDOUT:   return %.loc44_20.2
// CHECK:STDOUT: }

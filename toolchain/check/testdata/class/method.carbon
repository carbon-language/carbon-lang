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

fn CallOnConstBoundMethod() -> i32 {
  return ({.k = 1} as Class).F();
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

// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.k: i32} [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: Class = struct_value (%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl, .Call = %Call, .CallOnConstBoundMethod = %CallOnConstBoundMethod, .CallWithAddr = %CallWithAddr, .CallFThroughPointer = %CallFThroughPointer, .CallGThroughPointer = %CallGThroughPointer, .Make = %Make, .CallFOnInitializingExpr = %CallFOnInitializingExpr, .CallGOnInitializingExpr = %CallGOnInitializingExpr} [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %Call: <function> = fn_decl @Call [template]
// CHECK:STDOUT:   %CallOnConstBoundMethod: <function> = fn_decl @CallOnConstBoundMethod [template]
// CHECK:STDOUT:   %CallWithAddr: <function> = fn_decl @CallWithAddr [template]
// CHECK:STDOUT:   %CallFThroughPointer: <function> = fn_decl @CallFThroughPointer [template]
// CHECK:STDOUT:   %CallGThroughPointer: <function> = fn_decl @CallGThroughPointer [template]
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make [template]
// CHECK:STDOUT:   %CallFOnInitializingExpr: <function> = fn_decl @CallFOnInitializingExpr [template]
// CHECK:STDOUT:   %CallGOnInitializingExpr: <function> = fn_decl @CallGOnInitializingExpr [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %.loc11: <unbound element of class Class> = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .k = %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class = name_ref self, %self
// CHECK:STDOUT:   %.loc15_14.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc15_14.2: i32 = bind_value %.loc15_14.1
// CHECK:STDOUT:   return %.loc15_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr %self: Class*]() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%c: Class) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: Class = name_ref c, %c
// CHECK:STDOUT:   %.loc21_11: <bound method> = bound_method %c.ref, @Class.%F
// CHECK:STDOUT:   %.loc21_13.1: init i32 = call %.loc21_11(%c.ref)
// CHECK:STDOUT:   %.loc21_15: i32 = value_of_initializer %.loc21_13.1
// CHECK:STDOUT:   %.loc21_13.2: i32 = converted %.loc21_13.1, %.loc21_15
// CHECK:STDOUT:   return %.loc21_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallOnConstBoundMethod() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25_17: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc25_18.1: {.k: i32} = struct_literal (%.loc25_17)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %.loc25_18.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc25_18.3: ref i32 = class_element_access %.loc25_18.2, element0
// CHECK:STDOUT:   %.loc25_18.4: init i32 = initialize_from %.loc25_17 to %.loc25_18.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc25_18.5: init Class = class_init (%.loc25_18.4), %.loc25_18.2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc25_18.6: ref Class = temporary %.loc25_18.2, %.loc25_18.5
// CHECK:STDOUT:   %.loc25_18.7: ref Class = converted %.loc25_18.1, %.loc25_18.6
// CHECK:STDOUT:   %.loc25_29: <bound method> = bound_method %.loc25_18.7, @Class.%F
// CHECK:STDOUT:   %.loc25_18.8: Class = bind_value %.loc25_18.7
// CHECK:STDOUT:   %.loc25_31.1: init i32 = call %.loc25_29(%.loc25_18.8)
// CHECK:STDOUT:   %.loc25_33: i32 = value_of_initializer %.loc25_31.1
// CHECK:STDOUT:   %.loc25_31.2: i32 = converted %.loc25_31.1, %.loc25_33
// CHECK:STDOUT:   return %.loc25_31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithAddr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref Class = name_ref c, %c
// CHECK:STDOUT:   %.loc30_11: <bound method> = bound_method %c.ref, @Class.%G
// CHECK:STDOUT:   %.loc30_10: Class* = addr_of %c.ref
// CHECK:STDOUT:   %.loc30_13.1: init i32 = call %.loc30_11(%.loc30_10)
// CHECK:STDOUT:   %.loc30_15: i32 = value_of_initializer %.loc30_13.1
// CHECK:STDOUT:   %.loc30_13.2: i32 = converted %.loc30_13.1, %.loc30_15
// CHECK:STDOUT:   return %.loc30_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFThroughPointer(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc34_11.1: ref Class = deref %p.ref
// CHECK:STDOUT:   %.loc34_14: <bound method> = bound_method %.loc34_11.1, @Class.%F
// CHECK:STDOUT:   %.loc34_11.2: Class = bind_value %.loc34_11.1
// CHECK:STDOUT:   %.loc34_16.1: init i32 = call %.loc34_14(%.loc34_11.2)
// CHECK:STDOUT:   %.loc34_18: i32 = value_of_initializer %.loc34_16.1
// CHECK:STDOUT:   %.loc34_16.2: i32 = converted %.loc34_16.1, %.loc34_18
// CHECK:STDOUT:   return %.loc34_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGThroughPointer(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc38_11.1: ref Class = deref %p.ref
// CHECK:STDOUT:   %.loc38_14: <bound method> = bound_method %.loc38_11.1, @Class.%G
// CHECK:STDOUT:   %.loc38_11.2: Class* = addr_of %.loc38_11.1
// CHECK:STDOUT:   %.loc38_16.1: init i32 = call %.loc38_14(%.loc38_11.2)
// CHECK:STDOUT:   %.loc38_18: i32 = value_of_initializer %.loc38_16.1
// CHECK:STDOUT:   %.loc38_16.2: i32 = converted %.loc38_16.1, %.loc38_18
// CHECK:STDOUT:   return %.loc38_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFOnInitializingExpr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: <function> = name_ref Make, file.%Make [template = file.%Make]
// CHECK:STDOUT:   %.loc44_14.1: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc44_14.2: init Class = call %Make.ref() to %.loc44_14.1
// CHECK:STDOUT:   %.loc44_14.3: ref Class = temporary %.loc44_14.1, %.loc44_14.2
// CHECK:STDOUT:   %.loc44_16: <bound method> = bound_method %.loc44_14.3, @Class.%F
// CHECK:STDOUT:   %.loc44_14.4: Class = bind_value %.loc44_14.3
// CHECK:STDOUT:   %.loc44_18.1: init i32 = call %.loc44_16(%.loc44_14.4)
// CHECK:STDOUT:   %.loc44_20: i32 = value_of_initializer %.loc44_18.1
// CHECK:STDOUT:   %.loc44_18.2: i32 = converted %.loc44_18.1, %.loc44_20
// CHECK:STDOUT:   return %.loc44_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGOnInitializingExpr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: <function> = name_ref Make, file.%Make [template = file.%Make]
// CHECK:STDOUT:   %.loc48_14.1: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc48_14.2: init Class = call %Make.ref() to %.loc48_14.1
// CHECK:STDOUT:   %.loc48_14.3: ref Class = temporary %.loc48_14.1, %.loc48_14.2
// CHECK:STDOUT:   %.loc48_16: <bound method> = bound_method %.loc48_14.3, @Class.%G
// CHECK:STDOUT:   %.loc48_14.4: Class* = addr_of %.loc48_14.3
// CHECK:STDOUT:   %.loc48_18.1: init i32 = call %.loc48_16(%.loc48_14.4)
// CHECK:STDOUT:   %.loc48_20: i32 = value_of_initializer %.loc48_18.1
// CHECK:STDOUT:   %.loc48_18.2: i32 = converted %.loc48_18.1, %.loc48_20
// CHECK:STDOUT:   return %.loc48_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

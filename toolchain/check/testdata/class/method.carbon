// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[self: Self]() -> i32;
  fn G[addr self: Self*]() -> i32;

  alias A = F;

  var k: i32;
}

fn Class.F[self: Self]() -> i32 {
  return self.k;
}

fn Call(c: Class) -> i32 {
  // TODO: The sem-ir for this call doesn't distinguish the `self` argument from
  // the explicit arguments.
  return c.F();
}

fn CallAlias(c: Class) -> i32 {
  return c.A();
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
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type Class [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.k: i32} [template]
// CHECK:STDOUT:   %Call: type = fn_type @Call [template]
// CHECK:STDOUT:   %struct.3: Call = struct_value () [template]
// CHECK:STDOUT:   %CallAlias: type = fn_type @CallAlias [template]
// CHECK:STDOUT:   %struct.4: CallAlias = struct_value () [template]
// CHECK:STDOUT:   %CallOnConstBoundMethod: type = fn_type @CallOnConstBoundMethod [template]
// CHECK:STDOUT:   %struct.5: CallOnConstBoundMethod = struct_value () [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %struct.6: Class = struct_value (%.6) [template]
// CHECK:STDOUT:   %CallWithAddr: type = fn_type @CallWithAddr [template]
// CHECK:STDOUT:   %struct.7: CallWithAddr = struct_value () [template]
// CHECK:STDOUT:   %CallFThroughPointer: type = fn_type @CallFThroughPointer [template]
// CHECK:STDOUT:   %struct.8: CallFThroughPointer = struct_value () [template]
// CHECK:STDOUT:   %CallGThroughPointer: type = fn_type @CallGThroughPointer [template]
// CHECK:STDOUT:   %struct.9: CallGThroughPointer = struct_value () [template]
// CHECK:STDOUT:   %Make: type = fn_type @Make [template]
// CHECK:STDOUT:   %struct.10: Make = struct_value () [template]
// CHECK:STDOUT:   %CallFOnInitializingExpr: type = fn_type @CallFOnInitializingExpr [template]
// CHECK:STDOUT:   %struct.11: CallFOnInitializingExpr = struct_value () [template]
// CHECK:STDOUT:   %CallGOnInitializingExpr: type = fn_type @CallGOnInitializingExpr [template]
// CHECK:STDOUT:   %struct.12: CallGOnInitializingExpr = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallAlias = %CallAlias.decl
// CHECK:STDOUT:     .CallOnConstBoundMethod = %CallOnConstBoundMethod.decl
// CHECK:STDOUT:     .CallWithAddr = %CallWithAddr.decl
// CHECK:STDOUT:     .CallFThroughPointer = %CallFThroughPointer.decl
// CHECK:STDOUT:     .CallGThroughPointer = %CallGThroughPointer.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .CallFOnInitializingExpr = %CallFOnInitializingExpr.decl
// CHECK:STDOUT:     .CallGOnInitializingExpr = %CallGOnInitializingExpr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc16_12.1: Class = param self
// CHECK:STDOUT:     @F.%self: Class = bind_name self, %self.loc16_12.1
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: Call = fn_decl @Call [template = constants.%struct.3] {
// CHECK:STDOUT:     %Class.ref.loc20: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc20_9.1: Class = param c
// CHECK:STDOUT:     @Call.%c: Class = bind_name c, %c.loc20_9.1
// CHECK:STDOUT:     @Call.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallAlias.decl: CallAlias = fn_decl @CallAlias [template = constants.%struct.4] {
// CHECK:STDOUT:     %Class.ref.loc26: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc26_14.1: Class = param c
// CHECK:STDOUT:     @CallAlias.%c: Class = bind_name c, %c.loc26_14.1
// CHECK:STDOUT:     @CallAlias.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallOnConstBoundMethod.decl: CallOnConstBoundMethod = fn_decl @CallOnConstBoundMethod [template = constants.%struct.5] {
// CHECK:STDOUT:     @CallOnConstBoundMethod.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWithAddr.decl: CallWithAddr = fn_decl @CallWithAddr [template = constants.%struct.7] {
// CHECK:STDOUT:     @CallWithAddr.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFThroughPointer.decl: CallFThroughPointer = fn_decl @CallFThroughPointer [template = constants.%struct.8] {
// CHECK:STDOUT:     %Class.ref.loc39: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc39: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:     %p.loc39_24.1: Class* = param p
// CHECK:STDOUT:     @CallFThroughPointer.%p: Class* = bind_name p, %p.loc39_24.1
// CHECK:STDOUT:     @CallFThroughPointer.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGThroughPointer.decl: CallGThroughPointer = fn_decl @CallGThroughPointer [template = constants.%struct.9] {
// CHECK:STDOUT:     %Class.ref.loc43: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc43: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:     %p.loc43_24.1: Class* = param p
// CHECK:STDOUT:     @CallGThroughPointer.%p: Class* = bind_name p, %p.loc43_24.1
// CHECK:STDOUT:     @CallGThroughPointer.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: Make = fn_decl @Make [template = constants.%struct.10] {
// CHECK:STDOUT:     %Class.ref.loc47: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     @Make.%return: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFOnInitializingExpr.decl: CallFOnInitializingExpr = fn_decl @CallFOnInitializingExpr [template = constants.%struct.11] {
// CHECK:STDOUT:     @CallFOnInitializingExpr.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGOnInitializingExpr.decl: CallGOnInitializingExpr = fn_decl @CallGOnInitializingExpr [template = constants.%struct.12] {
// CHECK:STDOUT:     @CallGOnInitializingExpr.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref.loc8: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc8_8.1: Class = param self
// CHECK:STDOUT:     %self.loc8_8.2: Class = bind_name self, %self.loc8_8.1
// CHECK:STDOUT:     %return.var.loc8: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc9: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc9_23: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:     %self.loc9_13.1: Class* = param self
// CHECK:STDOUT:     %self.loc9_13.3: Class* = bind_name self, %self.loc9_13.1
// CHECK:STDOUT:     %.loc9_8: Class* = addr_pattern %self.loc9_13.3
// CHECK:STDOUT:     %return.var.loc9: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: F = name_ref F, %F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %A: F = bind_alias A, %F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13: <unbound element of class Class> = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .A = %A
// CHECK:STDOUT:   .k = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class = name_ref self, %self
// CHECK:STDOUT:   %k.ref: <unbound element of class Class> = name_ref k, @Class.%.loc13 [template = @Class.%.loc13]
// CHECK:STDOUT:   %.loc17_14.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc17_14.2: i32 = bind_value %.loc17_14.1
// CHECK:STDOUT:   return %.loc17_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr @Class.%self.loc9_13.3: Class*]() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%c: Class) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref: F = name_ref F, @Class.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc23_11: <bound method> = bound_method %c.ref, %F.ref
// CHECK:STDOUT:   %F.call: init i32 = call %.loc23_11(%c.ref)
// CHECK:STDOUT:   %.loc23_15.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc23_15.2: i32 = converted %F.call, %.loc23_15.1
// CHECK:STDOUT:   return %.loc23_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAlias(%c: Class) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: Class = name_ref c, %c
// CHECK:STDOUT:   %A.ref: F = name_ref A, @Class.%A [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc27_11: <bound method> = bound_method %c.ref, %A.ref
// CHECK:STDOUT:   %F.call: init i32 = call %.loc27_11(%c.ref)
// CHECK:STDOUT:   %.loc27_15.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc27_15.2: i32 = converted %F.call, %.loc27_15.1
// CHECK:STDOUT:   return %.loc27_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallOnConstBoundMethod() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_17: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc31_18.1: {.k: i32} = struct_literal (%.loc31_17)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc31_18.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc31_18.3: ref i32 = class_element_access %.loc31_18.2, element0
// CHECK:STDOUT:   %.loc31_18.4: init i32 = initialize_from %.loc31_17 to %.loc31_18.3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc31_18.5: init Class = class_init (%.loc31_18.4), %.loc31_18.2 [template = constants.%struct.6]
// CHECK:STDOUT:   %.loc31_18.6: ref Class = temporary %.loc31_18.2, %.loc31_18.5
// CHECK:STDOUT:   %.loc31_20.1: ref Class = converted %.loc31_18.1, %.loc31_18.6
// CHECK:STDOUT:   %F.ref: F = name_ref F, @Class.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc31_29: <bound method> = bound_method %.loc31_20.1, %F.ref
// CHECK:STDOUT:   %.loc31_20.2: Class = bind_value %.loc31_20.1
// CHECK:STDOUT:   %F.call: init i32 = call %.loc31_29(%.loc31_20.2)
// CHECK:STDOUT:   %.loc31_33.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc31_33.2: i32 = converted %F.call, %.loc31_33.1
// CHECK:STDOUT:   return %.loc31_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithAddr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref: G = name_ref G, @Class.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc36_11: <bound method> = bound_method %c.ref, %G.ref
// CHECK:STDOUT:   %.loc36_10: Class* = addr_of %c.ref
// CHECK:STDOUT:   %G.call: init i32 = call %.loc36_11(%.loc36_10)
// CHECK:STDOUT:   %.loc36_15.1: i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc36_15.2: i32 = converted %G.call, %.loc36_15.1
// CHECK:STDOUT:   return %.loc36_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFThroughPointer(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc40_11.1: ref Class = deref %p.ref
// CHECK:STDOUT:   %F.ref: F = name_ref F, @Class.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc40_14: <bound method> = bound_method %.loc40_11.1, %F.ref
// CHECK:STDOUT:   %.loc40_11.2: Class = bind_value %.loc40_11.1
// CHECK:STDOUT:   %F.call: init i32 = call %.loc40_14(%.loc40_11.2)
// CHECK:STDOUT:   %.loc40_18.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc40_18.2: i32 = converted %F.call, %.loc40_18.1
// CHECK:STDOUT:   return %.loc40_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGThroughPointer(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc44_11.1: ref Class = deref %p.ref
// CHECK:STDOUT:   %G.ref: G = name_ref G, @Class.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc44_14: <bound method> = bound_method %.loc44_11.1, %G.ref
// CHECK:STDOUT:   %.loc44_11.2: Class* = addr_of %.loc44_11.1
// CHECK:STDOUT:   %G.call: init i32 = call %.loc44_14(%.loc44_11.2)
// CHECK:STDOUT:   %.loc44_18.1: i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc44_18.2: i32 = converted %G.call, %.loc44_18.1
// CHECK:STDOUT:   return %.loc44_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFOnInitializingExpr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: Make = name_ref Make, file.%Make.decl [template = constants.%struct.10]
// CHECK:STDOUT:   %.loc50_14.1: ref Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init Class = call %Make.ref() to %.loc50_14.1
// CHECK:STDOUT:   %.loc50_14.2: ref Class = temporary %.loc50_14.1, %Make.call
// CHECK:STDOUT:   %F.ref: F = name_ref F, @Class.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc50_16: <bound method> = bound_method %.loc50_14.2, %F.ref
// CHECK:STDOUT:   %.loc50_14.3: Class = bind_value %.loc50_14.2
// CHECK:STDOUT:   %F.call: init i32 = call %.loc50_16(%.loc50_14.3)
// CHECK:STDOUT:   %.loc50_20.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc50_20.2: i32 = converted %F.call, %.loc50_20.1
// CHECK:STDOUT:   return %.loc50_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGOnInitializingExpr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: Make = name_ref Make, file.%Make.decl [template = constants.%struct.10]
// CHECK:STDOUT:   %.loc54_14.1: ref Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init Class = call %Make.ref() to %.loc54_14.1
// CHECK:STDOUT:   %.loc54_14.2: ref Class = temporary %.loc54_14.1, %Make.call
// CHECK:STDOUT:   %G.ref: G = name_ref G, @Class.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc54_16: <bound method> = bound_method %.loc54_14.2, %G.ref
// CHECK:STDOUT:   %.loc54_14.3: Class* = addr_of %.loc54_14.2
// CHECK:STDOUT:   %G.call: init i32 = call %.loc54_16(%.loc54_14.3)
// CHECK:STDOUT:   %.loc54_20.1: i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc54_20.2: i32 = converted %G.call, %.loc54_20.1
// CHECK:STDOUT:   return %.loc54_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

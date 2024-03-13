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
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.k: i32} [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: Class = struct_value (%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Call = %Call
// CHECK:STDOUT:     .CallAlias = %CallAlias
// CHECK:STDOUT:     .CallOnConstBoundMethod = %CallOnConstBoundMethod
// CHECK:STDOUT:     .CallWithAddr = %CallWithAddr
// CHECK:STDOUT:     .CallFThroughPointer = %CallFThroughPointer
// CHECK:STDOUT:     .CallGThroughPointer = %CallGThroughPointer
// CHECK:STDOUT:     .Make = %Make
// CHECK:STDOUT:     .CallFOnInitializingExpr = %CallFOnInitializingExpr
// CHECK:STDOUT:     .CallGOnInitializingExpr = %CallGOnInitializingExpr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc16_12.1: Class = param self
// CHECK:STDOUT:     @F.%self: Class = bind_name self, %self.loc16_12.1
// CHECK:STDOUT:     %return.var.loc16: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call: <function> = fn_decl @Call [template] {
// CHECK:STDOUT:     %Class.ref.loc20: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc20_9.1: Class = param c
// CHECK:STDOUT:     @Call.%c: Class = bind_name c, %c.loc20_9.1
// CHECK:STDOUT:     %return.var.loc20: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallAlias: <function> = fn_decl @CallAlias [template] {
// CHECK:STDOUT:     %Class.ref.loc26: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc26_14.1: Class = param c
// CHECK:STDOUT:     @CallAlias.%c: Class = bind_name c, %c.loc26_14.1
// CHECK:STDOUT:     %return.var.loc26: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallOnConstBoundMethod: <function> = fn_decl @CallOnConstBoundMethod [template] {
// CHECK:STDOUT:     %return.var.loc30: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWithAddr: <function> = fn_decl @CallWithAddr [template] {
// CHECK:STDOUT:     %return.var.loc34: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFThroughPointer: <function> = fn_decl @CallFThroughPointer [template] {
// CHECK:STDOUT:     %Class.ref.loc39: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc39: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %p.loc39_24.1: Class* = param p
// CHECK:STDOUT:     @CallFThroughPointer.%p: Class* = bind_name p, %p.loc39_24.1
// CHECK:STDOUT:     %return.var.loc39: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGThroughPointer: <function> = fn_decl @CallGThroughPointer [template] {
// CHECK:STDOUT:     %Class.ref.loc43: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc43: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %p.loc43_24.1: Class* = param p
// CHECK:STDOUT:     @CallGThroughPointer.%p: Class* = bind_name p, %p.loc43_24.1
// CHECK:STDOUT:     %return.var.loc43: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make [template] {
// CHECK:STDOUT:     %Class.ref.loc47: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     @Make.%return: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFOnInitializingExpr: <function> = fn_decl @CallFOnInitializingExpr [template] {
// CHECK:STDOUT:     %return.var.loc49: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGOnInitializingExpr: <function> = fn_decl @CallGOnInitializingExpr [template] {
// CHECK:STDOUT:     %return.var.loc53: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Self.ref.loc8: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc8_8.1: Class = param self
// CHECK:STDOUT:     %self.loc8_8.2: Class = bind_name self, %self.loc8_8.1
// CHECK:STDOUT:     %return.var.loc8: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %Self.ref.loc9: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc9_23: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %self.loc9_13.1: Class* = param self
// CHECK:STDOUT:     %self.loc9_13.3: Class* = bind_name self, %self.loc9_13.1
// CHECK:STDOUT:     %.loc9_8: Class* = addr_pattern %self.loc9_13.3
// CHECK:STDOUT:     %return.var.loc9: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, %F [template = %F]
// CHECK:STDOUT:   %A: <function> = bind_alias A, %F [template = %F]
// CHECK:STDOUT:   %.loc13: <unbound element of class Class> = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
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
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Class.%F [template = @Class.%F]
// CHECK:STDOUT:   %.loc23_11: <bound method> = bound_method %c.ref, %F.ref
// CHECK:STDOUT:   %.loc23_13.1: init i32 = call %.loc23_11(%c.ref)
// CHECK:STDOUT:   %.loc23_15: i32 = value_of_initializer %.loc23_13.1
// CHECK:STDOUT:   %.loc23_13.2: i32 = converted %.loc23_13.1, %.loc23_15
// CHECK:STDOUT:   return %.loc23_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAlias(%c: Class) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: Class = name_ref c, %c
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, @Class.%A [template = @Class.%F]
// CHECK:STDOUT:   %.loc27_11: <bound method> = bound_method %c.ref, %A.ref
// CHECK:STDOUT:   %.loc27_13.1: init i32 = call %.loc27_11(%c.ref)
// CHECK:STDOUT:   %.loc27_15: i32 = value_of_initializer %.loc27_13.1
// CHECK:STDOUT:   %.loc27_13.2: i32 = converted %.loc27_13.1, %.loc27_15
// CHECK:STDOUT:   return %.loc27_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallOnConstBoundMethod() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_17: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc31_18.1: {.k: i32} = struct_literal (%.loc31_17)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc31_18.2: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc31_18.3: ref i32 = class_element_access %.loc31_18.2, element0
// CHECK:STDOUT:   %.loc31_18.4: init i32 = initialize_from %.loc31_17 to %.loc31_18.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc31_18.5: init Class = class_init (%.loc31_18.4), %.loc31_18.2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc31_18.6: ref Class = temporary %.loc31_18.2, %.loc31_18.5
// CHECK:STDOUT:   %.loc31_18.7: ref Class = converted %.loc31_18.1, %.loc31_18.6
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Class.%F [template = @Class.%F]
// CHECK:STDOUT:   %.loc31_29: <bound method> = bound_method %.loc31_18.7, %F.ref
// CHECK:STDOUT:   %.loc31_18.8: Class = bind_value %.loc31_18.7
// CHECK:STDOUT:   %.loc31_31.1: init i32 = call %.loc31_29(%.loc31_18.8)
// CHECK:STDOUT:   %.loc31_33: i32 = value_of_initializer %.loc31_31.1
// CHECK:STDOUT:   %.loc31_31.2: i32 = converted %.loc31_31.1, %.loc31_33
// CHECK:STDOUT:   return %.loc31_31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithAddr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, @Class.%G [template = @Class.%G]
// CHECK:STDOUT:   %.loc36_11: <bound method> = bound_method %c.ref, %G.ref
// CHECK:STDOUT:   %.loc36_10: Class* = addr_of %c.ref
// CHECK:STDOUT:   %.loc36_13.1: init i32 = call %.loc36_11(%.loc36_10)
// CHECK:STDOUT:   %.loc36_15: i32 = value_of_initializer %.loc36_13.1
// CHECK:STDOUT:   %.loc36_13.2: i32 = converted %.loc36_13.1, %.loc36_15
// CHECK:STDOUT:   return %.loc36_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFThroughPointer(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc40_11.1: ref Class = deref %p.ref
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Class.%F [template = @Class.%F]
// CHECK:STDOUT:   %.loc40_14: <bound method> = bound_method %.loc40_11.1, %F.ref
// CHECK:STDOUT:   %.loc40_11.2: Class = bind_value %.loc40_11.1
// CHECK:STDOUT:   %.loc40_16.1: init i32 = call %.loc40_14(%.loc40_11.2)
// CHECK:STDOUT:   %.loc40_18: i32 = value_of_initializer %.loc40_16.1
// CHECK:STDOUT:   %.loc40_16.2: i32 = converted %.loc40_16.1, %.loc40_18
// CHECK:STDOUT:   return %.loc40_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGThroughPointer(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_ref p, %p
// CHECK:STDOUT:   %.loc44_11.1: ref Class = deref %p.ref
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, @Class.%G [template = @Class.%G]
// CHECK:STDOUT:   %.loc44_14: <bound method> = bound_method %.loc44_11.1, %G.ref
// CHECK:STDOUT:   %.loc44_11.2: Class* = addr_of %.loc44_11.1
// CHECK:STDOUT:   %.loc44_16.1: init i32 = call %.loc44_14(%.loc44_11.2)
// CHECK:STDOUT:   %.loc44_18: i32 = value_of_initializer %.loc44_16.1
// CHECK:STDOUT:   %.loc44_16.2: i32 = converted %.loc44_16.1, %.loc44_18
// CHECK:STDOUT:   return %.loc44_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFOnInitializingExpr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: <function> = name_ref Make, file.%Make [template = file.%Make]
// CHECK:STDOUT:   %.loc50_14.1: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc50_14.2: init Class = call %Make.ref() to %.loc50_14.1
// CHECK:STDOUT:   %.loc50_14.3: ref Class = temporary %.loc50_14.1, %.loc50_14.2
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Class.%F [template = @Class.%F]
// CHECK:STDOUT:   %.loc50_16: <bound method> = bound_method %.loc50_14.3, %F.ref
// CHECK:STDOUT:   %.loc50_14.4: Class = bind_value %.loc50_14.3
// CHECK:STDOUT:   %.loc50_18.1: init i32 = call %.loc50_16(%.loc50_14.4)
// CHECK:STDOUT:   %.loc50_20: i32 = value_of_initializer %.loc50_18.1
// CHECK:STDOUT:   %.loc50_18.2: i32 = converted %.loc50_18.1, %.loc50_20
// CHECK:STDOUT:   return %.loc50_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGOnInitializingExpr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: <function> = name_ref Make, file.%Make [template = file.%Make]
// CHECK:STDOUT:   %.loc54_14.1: ref Class = temporary_storage
// CHECK:STDOUT:   %.loc54_14.2: init Class = call %Make.ref() to %.loc54_14.1
// CHECK:STDOUT:   %.loc54_14.3: ref Class = temporary %.loc54_14.1, %.loc54_14.2
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, @Class.%G [template = @Class.%G]
// CHECK:STDOUT:   %.loc54_16: <bound method> = bound_method %.loc54_14.3, %G.ref
// CHECK:STDOUT:   %.loc54_14.4: Class* = addr_of %.loc54_14.3
// CHECK:STDOUT:   %.loc54_18.1: init i32 = call %.loc54_16(%.loc54_14.4)
// CHECK:STDOUT:   %.loc54_20: i32 = value_of_initializer %.loc54_18.1
// CHECK:STDOUT:   %.loc54_18.2: i32 = converted %.loc54_18.1, %.loc54_20
// CHECK:STDOUT:   return %.loc54_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

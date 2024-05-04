// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() -> i32 { return 1; }
fn B() -> i32 { return 2; }

fn F() -> i32 {
  return if true then A() else B();
}

fn G() -> i32 {
  return if false then A() else B();
}

fn Constant() -> i32 {
  var v: if true then i32 else i32* = 1;
  var w: if false then i32 else i32* = &v;
  return *w;
}

fn PartiallyConstant(t: type) -> i32 {
  var v: if true then i32 else t = 1;
  var w: if false then t else i32* = &v;
  return *w;
}

// CHECK:STDOUT: --- constant_condition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %.4: bool = bool_literal true [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.4: G = struct_value () [template]
// CHECK:STDOUT:   %.5: bool = bool_literal false [template]
// CHECK:STDOUT:   %Constant: type = fn_type @Constant [template]
// CHECK:STDOUT:   %struct.5: Constant = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type i32 [template]
// CHECK:STDOUT:   %PartiallyConstant: type = fn_type @PartiallyConstant [template]
// CHECK:STDOUT:   %struct.6: PartiallyConstant = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartiallyConstant = %PartiallyConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {
// CHECK:STDOUT:     @A.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.3] {
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.4] {
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: Constant = fn_decl @Constant [template = constants.%struct.5] {
// CHECK:STDOUT:     @Constant.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PartiallyConstant.decl: PartiallyConstant = fn_decl @PartiallyConstant [template = constants.%struct.6] {
// CHECK:STDOUT:     %t.loc24_22.1: type = param t
// CHECK:STDOUT:     @PartiallyConstant.%t: type = bind_name t, %t.loc24_22.1
// CHECK:STDOUT:     @PartiallyConstant.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_13: bool = bool_literal true [template = constants.%.4]
// CHECK:STDOUT:   if %.loc11_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc11_24.1: i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc11_24.2: i32 = converted %A.call, %.loc11_24.1
// CHECK:STDOUT:   br !if.expr.result(%.loc11_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%B.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref()
// CHECK:STDOUT:   %.loc11_27.1: i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc11_27.2: i32 = converted %B.call, %.loc11_27.1
// CHECK:STDOUT:   br !if.expr.result(%.loc11_27.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc11_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc15_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc15_25.1: i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc15_25.2: i32 = converted %A.call, %.loc15_25.1
// CHECK:STDOUT:   br !if.expr.result(%.loc15_25.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%B.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref()
// CHECK:STDOUT:   %.loc15_28.1: i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc15_28.2: i32 = converted %B.call, %.loc15_28.1
// CHECK:STDOUT:   br !if.expr.result(%.loc15_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc15_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc15_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_13: bool = bool_literal true [template = constants.%.4]
// CHECK:STDOUT:   if %.loc19_13 br !if.expr.then.loc19 else br !if.expr.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc19:
// CHECK:STDOUT:   br !if.expr.result.loc19(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc19:
// CHECK:STDOUT:   %.loc19_35: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   br !if.expr.result.loc19(%.loc19_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc19:
// CHECK:STDOUT:   %.loc19_10: type = block_arg !if.expr.result.loc19 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc19_39: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, %.loc19_39
// CHECK:STDOUT:   %.loc20_13: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc20_13 br !if.expr.then.loc20 else br !if.expr.else.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc20:
// CHECK:STDOUT:   br !if.expr.result.loc20(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc20:
// CHECK:STDOUT:   %.loc20_36: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   br !if.expr.result.loc20(%.loc20_36)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc20:
// CHECK:STDOUT:   %.loc20_10: type = block_arg !if.expr.result.loc20 [template = constants.%.6]
// CHECK:STDOUT:   %w.var: ref i32* = var w
// CHECK:STDOUT:   %w: ref i32* = bind_name w, %w.var
// CHECK:STDOUT:   %v.ref: ref i32 = name_ref v, %v
// CHECK:STDOUT:   %.loc20_40: i32* = addr_of %v.ref
// CHECK:STDOUT:   assign %w.var, %.loc20_40
// CHECK:STDOUT:   %w.ref: ref i32* = name_ref w, %w
// CHECK:STDOUT:   %.loc21_11: i32* = bind_value %w.ref
// CHECK:STDOUT:   %.loc21_10.1: ref i32 = deref %.loc21_11
// CHECK:STDOUT:   %.loc21_10.2: i32 = bind_value %.loc21_10.1
// CHECK:STDOUT:   return %.loc21_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartiallyConstant(%t: type) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25_13: bool = bool_literal true [template = constants.%.4]
// CHECK:STDOUT:   if %.loc25_13 br !if.expr.then.loc25 else br !if.expr.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc25:
// CHECK:STDOUT:   br !if.expr.result.loc25(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc25:
// CHECK:STDOUT:   %t.ref.loc25: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc25(%t.ref.loc25)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc25:
// CHECK:STDOUT:   %.loc25_10: type = block_arg !if.expr.result.loc25 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc25_36: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, %.loc25_36
// CHECK:STDOUT:   %.loc26_13: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc26_13 br !if.expr.then.loc26 else br !if.expr.else.loc26
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc26:
// CHECK:STDOUT:   %t.ref.loc26: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc26(%t.ref.loc26)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc26:
// CHECK:STDOUT:   %.loc26_34: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   br !if.expr.result.loc26(%.loc26_34)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc26:
// CHECK:STDOUT:   %.loc26_10: type = block_arg !if.expr.result.loc26 [template = constants.%.6]
// CHECK:STDOUT:   %w.var: ref i32* = var w
// CHECK:STDOUT:   %w: ref i32* = bind_name w, %w.var
// CHECK:STDOUT:   %v.ref: ref i32 = name_ref v, %v
// CHECK:STDOUT:   %.loc26_38: i32* = addr_of %v.ref
// CHECK:STDOUT:   assign %w.var, %.loc26_38
// CHECK:STDOUT:   %w.ref: ref i32* = name_ref w, %w
// CHECK:STDOUT:   %.loc27_11: i32* = bind_value %w.ref
// CHECK:STDOUT:   %.loc27_10.1: ref i32 = deref %.loc27_11
// CHECK:STDOUT:   %.loc27_10.2: i32 = bind_value %.loc27_10.1
// CHECK:STDOUT:   return %.loc27_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

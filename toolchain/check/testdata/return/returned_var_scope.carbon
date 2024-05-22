// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn UnrelatedScopes() -> i32 {
  if (true) {
    returned var v: i32 = 0;
  }
  if (true) {
    returned var w: i32 = 1;
  }
  return 0;
}

fn EnclosingButAfter(b: bool) -> i32 {
  if (b) {
    returned var v: i32 = 0;
    return var;
  }
  returned var w: i32 = 1;
  return var;
}

// CHECK:STDOUT: --- returned_var_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %UnrelatedScopes: type = fn_type @UnrelatedScopes [template]
// CHECK:STDOUT:   %struct.2: UnrelatedScopes = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %Bool: type = fn_type @Bool [template]
// CHECK:STDOUT:   %struct.3: Bool = struct_value () [template]
// CHECK:STDOUT:   %EnclosingButAfter: type = fn_type @EnclosingButAfter [template]
// CHECK:STDOUT:   %struct.4: EnclosingButAfter = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .UnrelatedScopes = %UnrelatedScopes.decl
// CHECK:STDOUT:     .EnclosingButAfter = %EnclosingButAfter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %UnrelatedScopes.decl: UnrelatedScopes = fn_decl @UnrelatedScopes [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc7: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_25.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:     %.loc7_25.2: type = converted %int.make_type_32.loc7, %.loc7_25.1 [template = i32]
// CHECK:STDOUT:     @UnrelatedScopes.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Bool = import_ref ir2, inst+2, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %EnclosingButAfter.decl: EnclosingButAfter = fn_decl @EnclosingButAfter [template = constants.%struct.4] {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%struct.3() [template = bool]
// CHECK:STDOUT:     %.loc17_25.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc17_25.2: type = converted %bool.make_type, %.loc17_25.1 [template = bool]
// CHECK:STDOUT:     %b.loc17_22.1: bool = param b
// CHECK:STDOUT:     @EnclosingButAfter.%b: bool = bind_name b, %b.loc17_22.1
// CHECK:STDOUT:     %int.make_type_32.loc17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc17_34.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:     %.loc17_34.2: type = converted %int.make_type_32.loc17, %.loc17_34.1 [template = i32]
// CHECK:STDOUT:     @EnclosingButAfter.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UnrelatedScopes() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc8 br !if.then.loc8 else br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc8:
// CHECK:STDOUT:   %int.make_type_32.loc9: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc9_21.1: type = value_of_initializer %int.make_type_32.loc9 [template = i32]
// CHECK:STDOUT:   %.loc9_21.2: type = converted %int.make_type_32.loc9, %.loc9_21.1 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc9_27: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign %v.var, %.loc9_27
// CHECK:STDOUT:   br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc8:
// CHECK:STDOUT:   %.loc11: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc11 br !if.then.loc11 else br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11:
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc12_21.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_21.2: type = converted %int.make_type_32.loc12, %.loc12_21.1 [template = i32]
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc12_27: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   assign %w.var, %.loc12_27
// CHECK:STDOUT:   br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11:
// CHECK:STDOUT:   %.loc14: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EnclosingButAfter(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int.make_type_32.loc19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc19_21.1: type = value_of_initializer %int.make_type_32.loc19 [template = i32]
// CHECK:STDOUT:   %.loc19_21.2: type = converted %int.make_type_32.loc19, %.loc19_21.1 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc19_27: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign %v.var, %.loc19_27
// CHECK:STDOUT:   %.loc19_18: i32 = bind_value %v
// CHECK:STDOUT:   return %.loc19_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc22_19.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_19.2: type = converted %int.make_type_32.loc22, %.loc22_19.1 [template = i32]
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc22_25: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   assign %w.var, %.loc22_25
// CHECK:STDOUT:   %.loc22_16: i32 = bind_value %w
// CHECK:STDOUT:   return %.loc22_16
// CHECK:STDOUT: }
// CHECK:STDOUT:

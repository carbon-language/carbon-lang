// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_complement.carbon

fn Complement(a: i32) -> i32 = "int.complement";
fn And(a: i32, b: i32) -> i32 = "int.and";

var arr: [i32; And(Complement(0x123456), 0xFFFFFF)];
let arr_p: [i32; 0xEDCBA9]* = &arr;

fn RuntimeCall(a: i32) -> i32 {
  return Complement(a);
}

// CHECK:STDOUT: --- int_complement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %Complement: type = fn_type @Complement [template]
// CHECK:STDOUT:   %struct.2: Complement = struct_value () [template]
// CHECK:STDOUT:   %And: type = fn_type @And [template]
// CHECK:STDOUT:   %struct.3: And = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1193046 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -1193047 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 16777215 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 15584169 [template]
// CHECK:STDOUT:   %.6: type = array_type %.5, i32 [template]
// CHECK:STDOUT:   %.7: type = ptr_type [i32; 15584169] [template]
// CHECK:STDOUT:   %RuntimeCall: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %struct.4: RuntimeCall = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Complement = %Complement.decl
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Complement.decl: Complement = fn_decl @Complement [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc2_18: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc2_18.1: type = value_of_initializer %int.make_type_32.loc2_18 [template = i32]
// CHECK:STDOUT:     %.loc2_18.2: type = converted %int.make_type_32.loc2_18, %.loc2_18.1 [template = i32]
// CHECK:STDOUT:     %a.loc2_15.1: i32 = param a
// CHECK:STDOUT:     @Complement.%a: i32 = bind_name a, %a.loc2_15.1
// CHECK:STDOUT:     %int.make_type_32.loc2_26: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc2_26.1: type = value_of_initializer %int.make_type_32.loc2_26 [template = i32]
// CHECK:STDOUT:     %.loc2_26.2: type = converted %int.make_type_32.loc2_26, %.loc2_26.1 [template = i32]
// CHECK:STDOUT:     @Complement.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %And.decl: And = fn_decl @And [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc3_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc3_11.1: type = value_of_initializer %int.make_type_32.loc3_11 [template = i32]
// CHECK:STDOUT:     %.loc3_11.2: type = converted %int.make_type_32.loc3_11, %.loc3_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc3_8.1: i32 = param a
// CHECK:STDOUT:     @And.%a: i32 = bind_name a, %a.loc3_8.1
// CHECK:STDOUT:     %int.make_type_32.loc3_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc3_19.1: type = value_of_initializer %int.make_type_32.loc3_19 [template = i32]
// CHECK:STDOUT:     %.loc3_19.2: type = converted %int.make_type_32.loc3_19, %.loc3_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc3_16.1: i32 = param b
// CHECK:STDOUT:     @And.%b: i32 = bind_name b, %b.loc3_16.1
// CHECK:STDOUT:     %int.make_type_32.loc3_27: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc3_27.1: type = value_of_initializer %int.make_type_32.loc3_27 [template = i32]
// CHECK:STDOUT:     %.loc3_27.2: type = converted %int.make_type_32.loc3_27, %.loc3_27.1 [template = i32]
// CHECK:STDOUT:     @And.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %And.ref: And = name_ref And, %And.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %Complement.ref: Complement = name_ref Complement, %Complement.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc5_31: i32 = int_literal 1193046 [template = constants.%.2]
// CHECK:STDOUT:   %int.complement: init i32 = call %Complement.ref(%.loc5_31) [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_42: i32 = int_literal 16777215 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_19.1: i32 = value_of_initializer %int.complement [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_19.2: i32 = converted %int.complement, %.loc5_19.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.and: init i32 = call %And.ref(%.loc5_19.2, %.loc5_42) [template = constants.%.5]
// CHECK:STDOUT:   %.loc5_11.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_11.2: type = converted %int.make_type_32.loc5, %.loc5_11.1 [template = i32]
// CHECK:STDOUT:   %.loc5_51: type = array_type %int.and, i32 [template = constants.%.6]
// CHECK:STDOUT:   %arr.var: ref [i32; 15584169] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 15584169] = bind_name arr, %arr.var
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc6_18: i32 = int_literal 15584169 [template = constants.%.5]
// CHECK:STDOUT:   %.loc6_13.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_13.2: type = converted %int.make_type_32.loc6, %.loc6_13.1 [template = i32]
// CHECK:STDOUT:   %.loc6_26: type = array_type %.loc6_18, i32 [template = constants.%.6]
// CHECK:STDOUT:   %.loc6_27: type = ptr_type [i32; 15584169] [template = constants.%.7]
// CHECK:STDOUT:   %arr.ref: ref [i32; 15584169] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc6_31: [i32; 15584169]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 15584169]* = bind_name arr_p, %.loc6_31
// CHECK:STDOUT:   %import_ref.8: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.9: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %RuntimeCall.decl: RuntimeCall = fn_decl @RuntimeCall [template = constants.%struct.4] {
// CHECK:STDOUT:     %int.make_type_32.loc8_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc8_19.1: type = value_of_initializer %int.make_type_32.loc8_19 [template = i32]
// CHECK:STDOUT:     %.loc8_19.2: type = converted %int.make_type_32.loc8_19, %.loc8_19.1 [template = i32]
// CHECK:STDOUT:     %a.loc8_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc8_16.1
// CHECK:STDOUT:     %int.make_type_32.loc8_27: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc8_27.1: type = value_of_initializer %int.make_type_32.loc8_27 [template = i32]
// CHECK:STDOUT:     %.loc8_27.2: type = converted %int.make_type_32.loc8_27, %.loc8_27.1 [template = i32]
// CHECK:STDOUT:     @RuntimeCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Complement(%a: i32) -> i32 = "int.complement";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And(%a: i32, %b: i32) -> i32 = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Complement.ref: Complement = name_ref Complement, file.%Complement.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %int.complement: init i32 = call %Complement.ref(%a.ref)
// CHECK:STDOUT:   %.loc9_23.1: i32 = value_of_initializer %int.complement
// CHECK:STDOUT:   %.loc9_23.2: i32 = converted %int.complement, %.loc9_23.1
// CHECK:STDOUT:   return %.loc9_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

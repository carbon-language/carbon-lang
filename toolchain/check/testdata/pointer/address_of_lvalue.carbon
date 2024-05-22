// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};

  var p: {.a: i32, .b: i32}* = &s;
  var q: i32* = &s.a;
  var r: i32* = &s.b;

  var t: (i32, i32) = (1, 2);
  var t0: i32* = &t[0];
  var t1: i32* = &t[1];
}

// CHECK:STDOUT: --- address_of_lvalue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %struct.2: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct.3: {.a: i32, .b: i32} = struct_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.6: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.8: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.9: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %tuple: (i32, i32) = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.8: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.9: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.10: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc8_15: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc8_15.1: type = value_of_initializer %int.make_type_32.loc8_15 [template = i32]
// CHECK:STDOUT:   %.loc8_15.2: type = converted %int.make_type_32.loc8_15, %.loc8_15.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc8_24: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc8_24.1: type = value_of_initializer %int.make_type_32.loc8_24 [template = i32]
// CHECK:STDOUT:   %.loc8_24.2: type = converted %int.make_type_32.loc8_24, %.loc8_24.1 [template = i32]
// CHECK:STDOUT:   %.loc8_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   %s.var: ref {.a: i32, .b: i32} = var s
// CHECK:STDOUT:   %s: ref {.a: i32, .b: i32} = bind_name s, %s.var
// CHECK:STDOUT:   %.loc8_37: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_45: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_46.1: {.a: i32, .b: i32} = struct_literal (%.loc8_37, %.loc8_45)
// CHECK:STDOUT:   %.loc8_46.2: ref i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc8_46.3: init i32 = initialize_from %.loc8_37 to %.loc8_46.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_46.4: ref i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc8_46.5: init i32 = initialize_from %.loc8_45 to %.loc8_46.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_46.6: init {.a: i32, .b: i32} = struct_init (%.loc8_46.3, %.loc8_46.5) to %s.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc8_47: init {.a: i32, .b: i32} = converted %.loc8_46.1, %.loc8_46.6 [template = constants.%struct.3]
// CHECK:STDOUT:   assign %s.var, %.loc8_47
// CHECK:STDOUT:   %int.make_type_32.loc10_15: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc10_15.1: type = value_of_initializer %int.make_type_32.loc10_15 [template = i32]
// CHECK:STDOUT:   %.loc10_15.2: type = converted %int.make_type_32.loc10_15, %.loc10_15.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_24: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc10_24.1: type = value_of_initializer %int.make_type_32.loc10_24 [template = i32]
// CHECK:STDOUT:   %.loc10_24.2: type = converted %int.make_type_32.loc10_24, %.loc10_24.1 [template = i32]
// CHECK:STDOUT:   %.loc10_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_28: type = ptr_type {.a: i32, .b: i32} [template = constants.%.3]
// CHECK:STDOUT:   %p.var: ref {.a: i32, .b: i32}* = var p
// CHECK:STDOUT:   %p: ref {.a: i32, .b: i32}* = bind_name p, %p.var
// CHECK:STDOUT:   %s.ref.loc10: ref {.a: i32, .b: i32} = name_ref s, %s
// CHECK:STDOUT:   %.loc10_32: {.a: i32, .b: i32}* = addr_of %s.ref.loc10
// CHECK:STDOUT:   assign %p.var, %.loc10_32
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc11_13.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_13.2: type = converted %int.make_type_32.loc11, %.loc11_13.1 [template = i32]
// CHECK:STDOUT:   %.loc11_13.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   %q.var: ref i32* = var q
// CHECK:STDOUT:   %q: ref i32* = bind_name q, %q.var
// CHECK:STDOUT:   %s.ref.loc11: ref {.a: i32, .b: i32} = name_ref s, %s
// CHECK:STDOUT:   %.loc11_19: ref i32 = struct_access %s.ref.loc11, element0
// CHECK:STDOUT:   %.loc11_17: i32* = addr_of %.loc11_19
// CHECK:STDOUT:   assign %q.var, %.loc11_17
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc12_13.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_13.2: type = converted %int.make_type_32.loc12, %.loc12_13.1 [template = i32]
// CHECK:STDOUT:   %.loc12_13.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   %r.var: ref i32* = var r
// CHECK:STDOUT:   %r: ref i32* = bind_name r, %r.var
// CHECK:STDOUT:   %s.ref.loc12: ref {.a: i32, .b: i32} = name_ref s, %s
// CHECK:STDOUT:   %.loc12_19: ref i32 = struct_access %s.ref.loc12, element1
// CHECK:STDOUT:   %.loc12_17: i32* = addr_of %.loc12_19
// CHECK:STDOUT:   assign %r.var, %.loc12_17
// CHECK:STDOUT:   %int.make_type_32.loc14_11: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_16: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc14_19.1: (type, type) = tuple_literal (%int.make_type_32.loc14_11, %int.make_type_32.loc14_16)
// CHECK:STDOUT:   %.loc14_19.2: type = value_of_initializer %int.make_type_32.loc14_11 [template = i32]
// CHECK:STDOUT:   %.loc14_19.3: type = converted %int.make_type_32.loc14_11, %.loc14_19.2 [template = i32]
// CHECK:STDOUT:   %.loc14_19.4: type = value_of_initializer %int.make_type_32.loc14_16 [template = i32]
// CHECK:STDOUT:   %.loc14_19.5: type = converted %int.make_type_32.loc14_16, %.loc14_19.4 [template = i32]
// CHECK:STDOUT:   %.loc14_19.6: type = converted %.loc14_19.1, constants.%.8 [template = constants.%.8]
// CHECK:STDOUT:   %t.var: ref (i32, i32) = var t
// CHECK:STDOUT:   %t: ref (i32, i32) = bind_name t, %t.var
// CHECK:STDOUT:   %.loc14_24: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_27: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_28.1: (i32, i32) = tuple_literal (%.loc14_24, %.loc14_27)
// CHECK:STDOUT:   %.loc14_28.2: ref i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc14_28.3: init i32 = initialize_from %.loc14_24 to %.loc14_28.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_28.4: ref i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc14_28.5: init i32 = initialize_from %.loc14_27 to %.loc14_28.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_28.6: init (i32, i32) = tuple_init (%.loc14_28.3, %.loc14_28.5) to %t.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_29: init (i32, i32) = converted %.loc14_28.1, %.loc14_28.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign %t.var, %.loc14_29
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc15_14.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_14.2: type = converted %int.make_type_32.loc15, %.loc15_14.1 [template = i32]
// CHECK:STDOUT:   %.loc15_14.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   %t0.var: ref i32* = var t0
// CHECK:STDOUT:   %t0: ref i32* = bind_name t0, %t0.var
// CHECK:STDOUT:   %t.ref.loc15: ref (i32, i32) = name_ref t, %t
// CHECK:STDOUT:   %.loc15_21: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc15_22: ref i32 = tuple_index %t.ref.loc15, %.loc15_21
// CHECK:STDOUT:   %.loc15_18: i32* = addr_of %.loc15_22
// CHECK:STDOUT:   assign %t0.var, %.loc15_18
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc16_14.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_14.2: type = converted %int.make_type_32.loc16, %.loc16_14.1 [template = i32]
// CHECK:STDOUT:   %.loc16_14.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   %t1.var: ref i32* = var t1
// CHECK:STDOUT:   %t1: ref i32* = bind_name t1, %t1.var
// CHECK:STDOUT:   %t.ref.loc16: ref (i32, i32) = name_ref t, %t
// CHECK:STDOUT:   %.loc16_21: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc16_22: ref i32 = tuple_index %t.ref.loc16, %.loc16_21
// CHECK:STDOUT:   %.loc16_18: i32* = addr_of %.loc16_22
// CHECK:STDOUT:   assign %t1.var, %.loc16_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:

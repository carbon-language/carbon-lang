// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit;

var a_ref: {.a: i32} = {.a = 0};
var b_ref: {.a: {.b: i32, .c: (i32,)}, .d: i32} =
    {.a = {.b = 0, .c = (0,)}, .d = 0};

// --- implicit.impl.carbon

impl package Implicit;

var a: {.a: i32} = a_ref;
var b: {.a: {.b: i32, .c: (i32,)}, .d: i32} = b_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %struct.2: {.a: i32} = struct_value (%.3) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.6: type = struct_type {.b: i32, .c: (i32,)} [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: {.b: i32, .c: (i32,)}, .d: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type {.b: i32, .c: (i32,)} [template]
// CHECK:STDOUT:   %.9: type = struct_type {.a: {.b: i32, .c: (i32,)}*, .d: i32} [template]
// CHECK:STDOUT:   %.10: type = ptr_type {.a: {.b: i32, .c: (i32,)}*, .d: i32} [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.3) [template]
// CHECK:STDOUT:   %struct.3: {.b: i32, .c: (i32,)} = struct_value (%.3, %tuple) [template]
// CHECK:STDOUT:   %struct.4: {.a: {.b: i32, .c: (i32,)}, .d: i32} = struct_value (%struct.3, %.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_ref = %b_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc4_17.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_17.2: type = converted %int.make_type_32.loc4, %.loc4_17.1 [template = i32]
// CHECK:STDOUT:   %.loc4_20: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:   %a_ref.var: ref {.a: i32} = var a_ref
// CHECK:STDOUT:   %a_ref: ref {.a: i32} = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc5_22: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc5_22.1: type = value_of_initializer %int.make_type_32.loc5_22 [template = i32]
// CHECK:STDOUT:   %.loc5_22.2: type = converted %int.make_type_32.loc5_22, %.loc5_22.1 [template = i32]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc5_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc5_36.1: (type,) = tuple_literal (%int.make_type_32.loc5_32)
// CHECK:STDOUT:   %.loc5_36.2: type = value_of_initializer %int.make_type_32.loc5_32 [template = i32]
// CHECK:STDOUT:   %.loc5_36.3: type = converted %int.make_type_32.loc5_32, %.loc5_36.2 [template = i32]
// CHECK:STDOUT:   %.loc5_36.4: type = converted %.loc5_36.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc5_37: type = struct_type {.b: i32, .c: (i32,)} [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc5_44: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc5_44.1: type = value_of_initializer %int.make_type_32.loc5_44 [template = i32]
// CHECK:STDOUT:   %.loc5_44.2: type = converted %int.make_type_32.loc5_44, %.loc5_44.1 [template = i32]
// CHECK:STDOUT:   %.loc5_47: type = struct_type {.a: {.b: i32, .c: (i32,)}, .d: i32} [template = constants.%.7]
// CHECK:STDOUT:   %b_ref.var: ref {.a: {.b: i32, .c: (i32,)}, .d: i32} = var b_ref
// CHECK:STDOUT:   %b_ref: ref {.a: {.b: i32, .c: (i32,)}, .d: i32} = bind_name b_ref, %b_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_30: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_31.1: {.a: i32} = struct_literal (%.loc4_30)
// CHECK:STDOUT:   %.loc4_31.2: init {.a: i32} = struct_init (%.loc4_30) to file.%a_ref.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc4_32: init {.a: i32} = converted %.loc4_31.1, %.loc4_31.2 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_32
// CHECK:STDOUT:   %.loc6_17: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_26: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_28.1: (i32,) = tuple_literal (%.loc6_26)
// CHECK:STDOUT:   %.loc6_29.1: {.b: i32, .c: (i32,)} = struct_literal (%.loc6_17, %.loc6_28.1)
// CHECK:STDOUT:   %.loc6_37: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_38.1: {.a: {.b: i32, .c: (i32,)}, .d: i32} = struct_literal (%.loc6_29.1, %.loc6_37)
// CHECK:STDOUT:   %.loc6_38.2: ref {.b: i32, .c: (i32,)} = struct_access file.%b_ref.var, element0
// CHECK:STDOUT:   %.loc6_29.2: ref i32 = struct_access %.loc6_38.2, element0
// CHECK:STDOUT:   %.loc6_29.3: init i32 = initialize_from %.loc6_17 to %.loc6_29.2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_29.4: ref (i32,) = struct_access %.loc6_38.2, element1
// CHECK:STDOUT:   %.loc6_28.2: init (i32,) = tuple_init (%.loc6_26) to %.loc6_29.4 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.5: init (i32,) = converted %.loc6_28.1, %.loc6_28.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.6: init (i32,) = initialize_from %.loc6_29.5 to %.loc6_29.4 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc6_29.7: init {.b: i32, .c: (i32,)} = struct_init (%.loc6_29.3, %.loc6_29.6) to %.loc6_38.2 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc6_38.3: init {.b: i32, .c: (i32,)} = converted %.loc6_29.1, %.loc6_29.7 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc6_38.4: ref i32 = struct_access file.%b_ref.var, element1
// CHECK:STDOUT:   %.loc6_38.5: init i32 = initialize_from %.loc6_37 to %.loc6_38.4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_38.6: init {.a: {.b: i32, .c: (i32,)}, .d: i32} = struct_init (%.loc6_38.3, %.loc6_38.5) to file.%b_ref.var [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc6_39: init {.a: {.b: i32, .c: (i32,)}, .d: i32} = converted %.loc6_38.1, %.loc6_38.6 [template = constants.%struct.4]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc6_39
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.3: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.5: type = struct_type {.b: i32, .c: (i32,)} [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: {.b: i32, .c: (i32,)}, .d: i32} [template]
// CHECK:STDOUT:   %.7: type = ptr_type {.b: i32, .c: (i32,)} [template]
// CHECK:STDOUT:   %.8: type = struct_type {.a: {.b: i32, .c: (i32,)}*, .d: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type {.a: {.b: i32, .c: (i32,)}*, .d: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = %import_ref.1
// CHECK:STDOUT:     .b_ref = %import_ref.2
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: ref {.a: i32} = import_ref ir0, inst+16, loaded
// CHECK:STDOUT:   %import_ref.2: ref {.a: {.b: i32, .c: (i32,)}, .d: i32} = import_ref ir0, inst+59, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc4_13.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_13.2: type = converted %int.make_type_32.loc4, %.loc4_13.1 [template = i32]
// CHECK:STDOUT:   %.loc4_16: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref {.a: i32} = var a
// CHECK:STDOUT:   %a: ref {.a: i32} = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc5_18: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc5_18.1: type = value_of_initializer %int.make_type_32.loc5_18 [template = i32]
// CHECK:STDOUT:   %.loc5_18.2: type = converted %int.make_type_32.loc5_18, %.loc5_18.1 [template = i32]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc5_28: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc5_32.1: (type,) = tuple_literal (%int.make_type_32.loc5_28)
// CHECK:STDOUT:   %.loc5_32.2: type = value_of_initializer %int.make_type_32.loc5_28 [template = i32]
// CHECK:STDOUT:   %.loc5_32.3: type = converted %int.make_type_32.loc5_28, %.loc5_32.2 [template = i32]
// CHECK:STDOUT:   %.loc5_32.4: type = converted %.loc5_32.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_33: type = struct_type {.b: i32, .c: (i32,)} [template = constants.%.5]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc5_40: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc5_40.1: type = value_of_initializer %int.make_type_32.loc5_40 [template = i32]
// CHECK:STDOUT:   %.loc5_40.2: type = converted %int.make_type_32.loc5_40, %.loc5_40.1 [template = i32]
// CHECK:STDOUT:   %.loc5_43: type = struct_type {.a: {.b: i32, .c: (i32,)}, .d: i32} [template = constants.%.6]
// CHECK:STDOUT:   %b.var: ref {.a: {.b: i32, .c: (i32,)}, .d: i32} = var b
// CHECK:STDOUT:   %b: ref {.a: {.b: i32, .c: (i32,)}, .d: i32} = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref {.a: i32} = name_ref a_ref, file.%import_ref.1
// CHECK:STDOUT:   %.loc4_20.1: ref i32 = struct_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc4_20.2: i32 = bind_value %.loc4_20.1
// CHECK:STDOUT:   %.loc4_20.3: init {.a: i32} = struct_init (%.loc4_20.2) to file.%a.var
// CHECK:STDOUT:   %.loc4_25: init {.a: i32} = converted %a_ref.ref, %.loc4_20.3
// CHECK:STDOUT:   assign file.%a.var, %.loc4_25
// CHECK:STDOUT:   %b_ref.ref: ref {.a: {.b: i32, .c: (i32,)}, .d: i32} = name_ref b_ref, file.%import_ref.2
// CHECK:STDOUT:   %.loc5_47.1: ref {.b: i32, .c: (i32,)} = struct_access %b_ref.ref, element0
// CHECK:STDOUT:   %.loc5_47.2: ref i32 = struct_access %.loc5_47.1, element0
// CHECK:STDOUT:   %.loc5_47.3: i32 = bind_value %.loc5_47.2
// CHECK:STDOUT:   %.loc5_47.4: ref {.b: i32, .c: (i32,)} = struct_access file.%b.var, element0
// CHECK:STDOUT:   %.loc5_47.5: ref i32 = struct_access %.loc5_47.4, element0
// CHECK:STDOUT:   %.loc5_47.6: init i32 = initialize_from %.loc5_47.3 to %.loc5_47.5
// CHECK:STDOUT:   %.loc5_47.7: ref (i32,) = struct_access %.loc5_47.1, element1
// CHECK:STDOUT:   %.loc5_47.8: ref i32 = tuple_access %.loc5_47.7, element0
// CHECK:STDOUT:   %.loc5_47.9: i32 = bind_value %.loc5_47.8
// CHECK:STDOUT:   %.loc5_47.10: ref (i32,) = struct_access %.loc5_47.4, element1
// CHECK:STDOUT:   %.loc5_47.11: init (i32,) = tuple_init (%.loc5_47.9) to %.loc5_47.10
// CHECK:STDOUT:   %.loc5_47.12: init (i32,) = converted %.loc5_47.7, %.loc5_47.11
// CHECK:STDOUT:   %.loc5_47.13: init (i32,) = initialize_from %.loc5_47.12 to %.loc5_47.10
// CHECK:STDOUT:   %.loc5_47.14: init {.b: i32, .c: (i32,)} = struct_init (%.loc5_47.6, %.loc5_47.13) to %.loc5_47.4
// CHECK:STDOUT:   %.loc5_47.15: init {.b: i32, .c: (i32,)} = converted %.loc5_47.1, %.loc5_47.14
// CHECK:STDOUT:   %.loc5_47.16: ref i32 = struct_access %b_ref.ref, element1
// CHECK:STDOUT:   %.loc5_47.17: i32 = bind_value %.loc5_47.16
// CHECK:STDOUT:   %.loc5_47.18: ref i32 = struct_access file.%b.var, element1
// CHECK:STDOUT:   %.loc5_47.19: init i32 = initialize_from %.loc5_47.17 to %.loc5_47.18
// CHECK:STDOUT:   %.loc5_47.20: init {.a: {.b: i32, .c: (i32,)}, .d: i32} = struct_init (%.loc5_47.15, %.loc5_47.19) to file.%b.var
// CHECK:STDOUT:   %.loc5_52: init {.a: {.b: i32, .c: (i32,)}, .d: i32} = converted %b_ref.ref, %.loc5_47.20
// CHECK:STDOUT:   assign file.%b.var, %.loc5_52
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_type_mismatch.carbon

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: ERROR: Cannot implicitly convert from `String` to `i32`.
// CHECK:STDERR: var a: [i32; 3] = (1, "Hello", "World");
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: [i32; 3] = (1, "Hello", "World");

var t1: (i32, String, String);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: ERROR: Cannot implicitly convert from `String` to `i32`.
// CHECK:STDERR: var b: [i32; 3] = t1;
// CHECK:STDERR:                   ^~
// CHECK:STDERR:
var b: [i32; 3] = t1;

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:19: ERROR: Cannot initialize array of 3 element(s) from 2 initializer(s).
// CHECK:STDERR: var c: [i32; 3] = (1, 2);
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
var c: [i32; 3] = (1, 2);

var t2: (i32, i32);
// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:19: ERROR: Cannot initialize array of 3 element(s) from tuple with 2 element(s).
// CHECK:STDERR: var d: [i32; 3] = t2;
// CHECK:STDERR:                   ^~
var d: [i32; 3] = t2;

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: type = ptr_type String [template]
// CHECK:STDOUT:   %.7: String = string_literal "Hello" [template]
// CHECK:STDOUT:   %.8: String = string_literal "World" [template]
// CHECK:STDOUT:   %.9: type = tuple_type (i32, String, String) [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.11: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.12: type = tuple_type (i32, String*, String*) [template]
// CHECK:STDOUT:   %.13: type = ptr_type (i32, String*, String*) [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.15: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.16: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.17: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .t1 = %t1
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .t2 = %t2
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_14: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_9.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_9.2: type = converted %int.make_type_32.loc15, %.loc15_9.1 [template = i32]
// CHECK:STDOUT:   %.loc15_15: type = array_type %.loc15_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref [i32; 3] = var a
// CHECK:STDOUT:   %a: ref [i32; 3] = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.2: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_29.1: (type, type, type) = tuple_literal (%int.make_type_32.loc17, String, String)
// CHECK:STDOUT:   %.loc17_29.2: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_29.3: type = converted %int.make_type_32.loc17, %.loc17_29.2 [template = i32]
// CHECK:STDOUT:   %.loc17_29.4: type = converted %.loc17_29.1, constants.%.9 [template = constants.%.9]
// CHECK:STDOUT:   %t1.var: ref (i32, String, String) = var t1
// CHECK:STDOUT:   %t1: ref (i32, String, String) = bind_name t1, %t1.var
// CHECK:STDOUT:   %import_ref.3: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_14: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc22_9.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_9.2: type = converted %int.make_type_32.loc22, %.loc22_9.1 [template = i32]
// CHECK:STDOUT:   %.loc22_15: type = array_type %.loc22_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %b.var: ref [i32; 3] = var b
// CHECK:STDOUT:   %b: ref [i32; 3] = bind_name b, %b.var
// CHECK:STDOUT:   %import_ref.4: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc28_14: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc28_9.1: type = value_of_initializer %int.make_type_32.loc28 [template = i32]
// CHECK:STDOUT:   %.loc28_9.2: type = converted %int.make_type_32.loc28, %.loc28_9.1 [template = i32]
// CHECK:STDOUT:   %.loc28_15: type = array_type %.loc28_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref [i32; 3] = var c
// CHECK:STDOUT:   %c: ref [i32; 3] = bind_name c, %c.var
// CHECK:STDOUT:   %import_ref.5: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc30_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %import_ref.6: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc30_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc30_18.1: (type, type) = tuple_literal (%int.make_type_32.loc30_10, %int.make_type_32.loc30_15)
// CHECK:STDOUT:   %.loc30_18.2: type = value_of_initializer %int.make_type_32.loc30_10 [template = i32]
// CHECK:STDOUT:   %.loc30_18.3: type = converted %int.make_type_32.loc30_10, %.loc30_18.2 [template = i32]
// CHECK:STDOUT:   %.loc30_18.4: type = value_of_initializer %int.make_type_32.loc30_15 [template = i32]
// CHECK:STDOUT:   %.loc30_18.5: type = converted %int.make_type_32.loc30_15, %.loc30_18.4 [template = i32]
// CHECK:STDOUT:   %.loc30_18.6: type = converted %.loc30_18.1, constants.%.15 [template = constants.%.15]
// CHECK:STDOUT:   %t2.var: ref (i32, i32) = var t2
// CHECK:STDOUT:   %t2: ref (i32, i32) = bind_name t2, %t2.var
// CHECK:STDOUT:   %import_ref.7: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc34_14: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc34_9.1: type = value_of_initializer %int.make_type_32.loc34 [template = i32]
// CHECK:STDOUT:   %.loc34_9.2: type = converted %int.make_type_32.loc34, %.loc34_9.1 [template = i32]
// CHECK:STDOUT:   %.loc34_15: type = array_type %.loc34_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %d.var: ref [i32; 3] = var d
// CHECK:STDOUT:   %d: ref [i32; 3] = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_20: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc15_23: String = string_literal "Hello" [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_32: String = string_literal "World" [template = constants.%.8]
// CHECK:STDOUT:   %.loc15_39.1: (i32, String, String) = tuple_literal (%.loc15_20, %.loc15_23, %.loc15_32)
// CHECK:STDOUT:   %.loc15_39.2: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc15_39.3: ref i32 = array_index file.%a.var, %.loc15_39.2
// CHECK:STDOUT:   %.loc15_39.4: init i32 = initialize_from %.loc15_20 to %.loc15_39.3 [template = constants.%.5]
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   %t1.ref: ref (i32, String, String) = name_ref t1, file.%t1
// CHECK:STDOUT:   %.loc22_19.1: ref i32 = tuple_access %t1.ref, element0
// CHECK:STDOUT:   %.loc22_19.2: i32 = bind_value %.loc22_19.1
// CHECK:STDOUT:   %.loc22_19.3: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc22_19.4: ref i32 = array_index file.%b.var, %.loc22_19.3
// CHECK:STDOUT:   %.loc22_19.5: init i32 = initialize_from %.loc22_19.2 to %.loc22_19.4
// CHECK:STDOUT:   %.loc22_19.6: ref String = tuple_access %t1.ref, element1
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   %.loc28_20: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc28_23: i32 = int_literal 2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc28_24: (i32, i32) = tuple_literal (%.loc28_20, %.loc28_23)
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %t2.ref: ref (i32, i32) = name_ref t2, file.%t2
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

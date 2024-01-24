// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: ((i32, i32), i32) = ((12, 76), 6);

// CHECK:STDOUT: --- nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type ((type, type), type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = tuple_type ((i32, i32), i32) [template]
// CHECK:STDOUT:   %.5: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.6: type = tuple_type ((i32, i32)*, i32) [template]
// CHECK:STDOUT:   %.7: type = ptr_type ((i32, i32)*, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 76 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.11: (i32, i32) = tuple_value (%.8, %.9) [template]
// CHECK:STDOUT:   %.12: ((i32, i32), i32) = tuple_value (%.11, %.10) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %.loc7_18.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_24.1: ((type, type), type) = tuple_literal (%.loc7_18.1, i32)
// CHECK:STDOUT:   %.loc7_18.2: type = converted %.loc7_18.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_24.2: type = converted %.loc7_24.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %x.var: ref ((i32, i32), i32) = var x
// CHECK:STDOUT:   %x: ref ((i32, i32), i32) = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7_30: i32 = int_literal 12 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_34: i32 = int_literal 76 [template = constants.%.9]
// CHECK:STDOUT:   %.loc7_36.1: (i32, i32) = tuple_literal (%.loc7_30, %.loc7_34)
// CHECK:STDOUT:   %.loc7_39: i32 = int_literal 6 [template = constants.%.10]
// CHECK:STDOUT:   %.loc7_40.1: ((i32, i32), i32) = tuple_literal (%.loc7_36.1, %.loc7_39)
// CHECK:STDOUT:   %.loc7_40.2: ref (i32, i32) = tuple_access %x.var, element0
// CHECK:STDOUT:   %.loc7_36.2: ref i32 = tuple_access %.loc7_40.2, element0
// CHECK:STDOUT:   %.loc7_36.3: init i32 = initialize_from %.loc7_30 to %.loc7_36.2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_36.4: ref i32 = tuple_access %.loc7_40.2, element1
// CHECK:STDOUT:   %.loc7_36.5: init i32 = initialize_from %.loc7_34 to %.loc7_36.4 [template = constants.%.9]
// CHECK:STDOUT:   %.loc7_36.6: init (i32, i32) = tuple_init (%.loc7_36.3, %.loc7_36.5) to %.loc7_40.2 [template = constants.%.11]
// CHECK:STDOUT:   %.loc7_36.7: init (i32, i32) = converted %.loc7_36.1, %.loc7_36.6 [template = constants.%.11]
// CHECK:STDOUT:   %.loc7_40.3: ref i32 = tuple_access %x.var, element1
// CHECK:STDOUT:   %.loc7_40.4: init i32 = initialize_from %.loc7_39 to %.loc7_40.3 [template = constants.%.10]
// CHECK:STDOUT:   %.loc7_40.5: init ((i32, i32), i32) = tuple_init (%.loc7_36.7, %.loc7_40.4) to %x.var [template = constants.%.12]
// CHECK:STDOUT:   %.loc7_40.6: init ((i32, i32), i32) = converted %.loc7_40.1, %.loc7_40.5 [template = constants.%.12]
// CHECK:STDOUT:   assign %x.var, %.loc7_40.6
// CHECK:STDOUT: }
// CHECK:STDOUT:

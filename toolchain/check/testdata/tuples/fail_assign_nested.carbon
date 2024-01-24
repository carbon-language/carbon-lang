// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:36: ERROR: Cannot initialize tuple of 2 element(s) from tuple with 3 element(s).
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                    ^~~~~~~~~
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

// CHECK:STDOUT: --- fail_assign_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type ((type, type), (type, type)) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = tuple_type ((i32, i32), (i32, i32)) [template]
// CHECK:STDOUT:   %.5: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.6: type = tuple_type ((i32, i32)*, (i32, i32)*) [template]
// CHECK:STDOUT:   %.7: type = ptr_type ((i32, i32)*, (i32, i32)*) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.11: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.15: type = tuple_type ((i32, i32, i32), (i32, i32, i32)) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %.loc10_18.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc10_30.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc10_31.1: ((type, type), (type, type)) = tuple_literal (%.loc10_18.1, %.loc10_30.1)
// CHECK:STDOUT:   %.loc10_18.2: type = converted %.loc10_18.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_30.2: type = converted %.loc10_30.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_31.2: type = converted %.loc10_31.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %x.var: ref ((i32, i32), (i32, i32)) = var x
// CHECK:STDOUT:   %x: ref ((i32, i32), (i32, i32)) = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_37: i32 = int_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc10_40: i32 = int_literal 2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc10_43: i32 = int_literal 3 [template = constants.%.10]
// CHECK:STDOUT:   %.loc10_44: (i32, i32, i32) = tuple_literal (%.loc10_37, %.loc10_40, %.loc10_43)
// CHECK:STDOUT:   %.loc10_48: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:   %.loc10_51: i32 = int_literal 5 [template = constants.%.13]
// CHECK:STDOUT:   %.loc10_54: i32 = int_literal 6 [template = constants.%.14]
// CHECK:STDOUT:   %.loc10_55: (i32, i32, i32) = tuple_literal (%.loc10_48, %.loc10_51, %.loc10_54)
// CHECK:STDOUT:   %.loc10_56: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%.loc10_44, %.loc10_55)
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

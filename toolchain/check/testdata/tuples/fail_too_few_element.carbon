// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_too_few_element.carbon:[[@LINE+3]]:21: ERROR: Cannot initialize tuple of 2 element(s) from tuple with 1 element(s).
// CHECK:STDERR: var x: (i32, i32) = (2, );
// CHECK:STDERR:                     ^~~~~
var x: (i32, i32) = (2, );

// CHECK:STDOUT: --- fail_too_few_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc10_9: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc10_14: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc10_17.1: (type, type) = tuple_literal (%int.make_type_32.loc10_9, %int.make_type_32.loc10_14)
// CHECK:STDOUT:   %.loc10_17.2: type = value_of_initializer %int.make_type_32.loc10_9 [template = i32]
// CHECK:STDOUT:   %.loc10_17.3: type = converted %int.make_type_32.loc10_9, %.loc10_17.2 [template = i32]
// CHECK:STDOUT:   %.loc10_17.4: type = value_of_initializer %int.make_type_32.loc10_14 [template = i32]
// CHECK:STDOUT:   %.loc10_17.5: type = converted %int.make_type_32.loc10_14, %.loc10_17.4 [template = i32]
// CHECK:STDOUT:   %.loc10_17.6: type = converted %.loc10_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref (i32, i32) = var x
// CHECK:STDOUT:   %x: ref (i32, i32) = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_22: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_25: (i32,) = tuple_literal (%.loc10_22)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

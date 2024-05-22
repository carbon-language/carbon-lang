// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: This creates an array temporary, which we don't yet support evaluating.

// CHECK:STDERR: fail_aggregate.carbon:[[@LINE+3]]:43: ERROR: Array bound is not a constant.
// CHECK:STDERR: var array_index: [i32; 1] = (0,) as [i32; ((5, 7, 1, 9) as [i32; 4])[2]];
// CHECK:STDERR:                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var array_index: [i32; 1] = (0,) as [i32; ((5, 7, 1, 9) as [i32; 4])[2]];

// CHECK:STDOUT: --- fail_aggregate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.10: type = tuple_type (i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.12: type = array_type %.11, i32 [template]
// CHECK:STDOUT:   %.13: type = ptr_type [i32; 4] [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %array: [i32; 4] = tuple_value (%.7, %.8, %.2, %.9) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .array_index = %array_index
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc12_24: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_19.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_19.2: type = converted %int.make_type_32, %.loc12_19.1 [template = i32]
// CHECK:STDOUT:   %.loc12_25: type = array_type %.loc12_24, i32 [template = constants.%.3]
// CHECK:STDOUT:   %array_index.var: ref [i32; 1] = var array_index
// CHECK:STDOUT:   %array_index: ref [i32; 1] = bind_name array_index, %array_index.var
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_30: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_32: (i32,) = tuple_literal (%.loc12_30)
// CHECK:STDOUT:   %int.make_type_32.loc12_38: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc12_45: i32 = int_literal 5 [template = constants.%.7]
// CHECK:STDOUT:   %.loc12_48: i32 = int_literal 7 [template = constants.%.8]
// CHECK:STDOUT:   %.loc12_51: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_54: i32 = int_literal 9 [template = constants.%.9]
// CHECK:STDOUT:   %.loc12_55.1: (i32, i32, i32, i32) = tuple_literal (%.loc12_45, %.loc12_48, %.loc12_51, %.loc12_54)
// CHECK:STDOUT:   %int.make_type_32.loc12_61: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc12_66: i32 = int_literal 4 [template = constants.%.11]
// CHECK:STDOUT:   %.loc12_61.1: type = value_of_initializer %int.make_type_32.loc12_61 [template = i32]
// CHECK:STDOUT:   %.loc12_61.2: type = converted %int.make_type_32.loc12_61, %.loc12_61.1 [template = i32]
// CHECK:STDOUT:   %.loc12_67: type = array_type %.loc12_66, i32 [template = constants.%.12]
// CHECK:STDOUT:   %.loc12_55.2: ref [i32; 4] = temporary_storage
// CHECK:STDOUT:   %.loc12_55.3: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_55.4: ref i32 = array_index %.loc12_55.2, %.loc12_55.3
// CHECK:STDOUT:   %.loc12_55.5: init i32 = initialize_from %.loc12_45 to %.loc12_55.4 [template = constants.%.7]
// CHECK:STDOUT:   %.loc12_55.6: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_55.7: ref i32 = array_index %.loc12_55.2, %.loc12_55.6
// CHECK:STDOUT:   %.loc12_55.8: init i32 = initialize_from %.loc12_48 to %.loc12_55.7 [template = constants.%.8]
// CHECK:STDOUT:   %.loc12_55.9: i32 = int_literal 2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc12_55.10: ref i32 = array_index %.loc12_55.2, %.loc12_55.9
// CHECK:STDOUT:   %.loc12_55.11: init i32 = initialize_from %.loc12_51 to %.loc12_55.10 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_55.12: i32 = int_literal 3 [template = constants.%.15]
// CHECK:STDOUT:   %.loc12_55.13: ref i32 = array_index %.loc12_55.2, %.loc12_55.12
// CHECK:STDOUT:   %.loc12_55.14: init i32 = initialize_from %.loc12_54 to %.loc12_55.13 [template = constants.%.9]
// CHECK:STDOUT:   %.loc12_55.15: init [i32; 4] = array_init (%.loc12_55.5, %.loc12_55.8, %.loc12_55.11, %.loc12_55.14) to %.loc12_55.2 [template = constants.%array]
// CHECK:STDOUT:   %.loc12_57.1: init [i32; 4] = converted %.loc12_55.1, %.loc12_55.15 [template = constants.%array]
// CHECK:STDOUT:   %.loc12_70: i32 = int_literal 2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc12_57.2: ref [i32; 4] = temporary %.loc12_55.2, %.loc12_57.1
// CHECK:STDOUT:   %.loc12_71.1: ref i32 = array_index %.loc12_57.2, %.loc12_70
// CHECK:STDOUT:   %.loc12_71.2: i32 = bind_value %.loc12_71.1
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

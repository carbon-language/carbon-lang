// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/two_entries.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/two_entries.carbon

let v: {.a: i32, .b: i32} = {.a = 1, .b = 2};
let w: {.a: i32, .b: i32} = v;

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
var y: {.a: i32, .b: i32} = x;

// CHECK:STDOUT: --- two_entries.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct: %.2 = struct_value (%.4, %.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .v = @__global_init.%v
// CHECK:STDOUT:     .w = @__global_init.%w
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11_13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_13.1: type = value_of_initializer %int.make_type_32.loc11_13 [template = i32]
// CHECK:STDOUT:   %.loc11_13.2: type = converted %int.make_type_32.loc11_13, %.loc11_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_22.1: type = value_of_initializer %int.make_type_32.loc11_22 [template = i32]
// CHECK:STDOUT:   %.loc11_22.2: type = converted %int.make_type_32.loc11_22, %.loc11_22.1 [template = i32]
// CHECK:STDOUT:   %.loc11_25: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_32.loc12_13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_13.1: type = value_of_initializer %int.make_type_32.loc12_13 [template = i32]
// CHECK:STDOUT:   %.loc12_13.2: type = converted %int.make_type_32.loc12_13, %.loc12_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc12_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_22.1: type = value_of_initializer %int.make_type_32.loc12_22 [template = i32]
// CHECK:STDOUT:   %.loc12_22.2: type = converted %int.make_type_32.loc12_22, %.loc12_22.1 [template = i32]
// CHECK:STDOUT:   %.loc12_25: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_32.loc14_13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_13.1: type = value_of_initializer %int.make_type_32.loc14_13 [template = i32]
// CHECK:STDOUT:   %.loc14_13.2: type = converted %int.make_type_32.loc14_13, %.loc14_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_22.1: type = value_of_initializer %int.make_type_32.loc14_22 [template = i32]
// CHECK:STDOUT:   %.loc14_22.2: type = converted %int.make_type_32.loc14_22, %.loc14_22.1 [template = i32]
// CHECK:STDOUT:   %.loc14_25: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref %.2 = var x
// CHECK:STDOUT:   %x: ref %.2 = bind_name x, %x.var
// CHECK:STDOUT:   %int.make_type_32.loc15_13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_13.1: type = value_of_initializer %int.make_type_32.loc15_13 [template = i32]
// CHECK:STDOUT:   %.loc15_13.2: type = converted %int.make_type_32.loc15_13, %.loc15_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc15_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_22.1: type = value_of_initializer %int.make_type_32.loc15_22 [template = i32]
// CHECK:STDOUT:   %.loc15_22.2: type = converted %int.make_type_32.loc15_22, %.loc15_22.1 [template = i32]
// CHECK:STDOUT:   %.loc15_25: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   %y.var: ref %.2 = var y
// CHECK:STDOUT:   %y: ref %.2 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_35: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_43: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_44: %.2 = struct_literal (%.loc11_35, %.loc11_43)
// CHECK:STDOUT:   %struct: %.2 = struct_value (%.loc11_35, %.loc11_43) [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_45: %.2 = converted %.loc11_44, %struct [template = constants.%struct]
// CHECK:STDOUT:   %v: %.2 = bind_name v, %.loc11_45
// CHECK:STDOUT:   %v.ref: %.2 = name_ref v, %v
// CHECK:STDOUT:   %w: %.2 = bind_name w, %v.ref
// CHECK:STDOUT:   %.loc14_35: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_43: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_44.1: %.2 = struct_literal (%.loc14_35, %.loc14_43)
// CHECK:STDOUT:   %.loc14_44.2: ref i32 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_44.3: init i32 = initialize_from %.loc14_35 to %.loc14_44.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_44.4: ref i32 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_44.5: init i32 = initialize_from %.loc14_43 to %.loc14_44.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_44.6: init %.2 = struct_init (%.loc14_44.3, %.loc14_44.5) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc14_45: init %.2 = converted %.loc14_44.1, %.loc14_44.6 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_45
// CHECK:STDOUT:   %x.ref: ref %.2 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc15_29.1: ref i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc15_29.2: i32 = bind_value %.loc15_29.1
// CHECK:STDOUT:   %.loc15_29.3: ref i32 = struct_access file.%y.var, element0
// CHECK:STDOUT:   %.loc15_29.4: init i32 = initialize_from %.loc15_29.2 to %.loc15_29.3
// CHECK:STDOUT:   %.loc15_29.5: ref i32 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc15_29.6: i32 = bind_value %.loc15_29.5
// CHECK:STDOUT:   %.loc15_29.7: ref i32 = struct_access file.%y.var, element1
// CHECK:STDOUT:   %.loc15_29.8: init i32 = initialize_from %.loc15_29.6 to %.loc15_29.7
// CHECK:STDOUT:   %.loc15_29.9: init %.2 = struct_init (%.loc15_29.4, %.loc15_29.8) to file.%y.var
// CHECK:STDOUT:   %.loc15_30: init %.2 = converted %x.ref, %.loc15_29.9
// CHECK:STDOUT:   assign file.%y.var, %.loc15_30
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

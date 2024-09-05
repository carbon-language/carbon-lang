// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/two_elements.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/two_elements.carbon

let v: (i32, i32) = (4, 102);
let w: (i32, i32) = v;

var x: (i32, i32) = (4, 102);
var y: (i32, i32) = x;

// CHECK:STDOUT: --- two_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 102 [template]
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.5, %.6) [template]
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
// CHECK:STDOUT:   %int.make_type_32.loc11_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_17.1: %.2 = tuple_literal (%int.make_type_32.loc11_9, %int.make_type_32.loc11_14)
// CHECK:STDOUT:   %.loc11_17.2: type = value_of_initializer %int.make_type_32.loc11_9 [template = i32]
// CHECK:STDOUT:   %.loc11_17.3: type = converted %int.make_type_32.loc11_9, %.loc11_17.2 [template = i32]
// CHECK:STDOUT:   %.loc11_17.4: type = value_of_initializer %int.make_type_32.loc11_14 [template = i32]
// CHECK:STDOUT:   %.loc11_17.5: type = converted %int.make_type_32.loc11_14, %.loc11_17.4 [template = i32]
// CHECK:STDOUT:   %.loc11_17.6: type = converted %.loc11_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc12_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc12_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_17.1: %.2 = tuple_literal (%int.make_type_32.loc12_9, %int.make_type_32.loc12_14)
// CHECK:STDOUT:   %.loc12_17.2: type = value_of_initializer %int.make_type_32.loc12_9 [template = i32]
// CHECK:STDOUT:   %.loc12_17.3: type = converted %int.make_type_32.loc12_9, %.loc12_17.2 [template = i32]
// CHECK:STDOUT:   %.loc12_17.4: type = value_of_initializer %int.make_type_32.loc12_14 [template = i32]
// CHECK:STDOUT:   %.loc12_17.5: type = converted %int.make_type_32.loc12_14, %.loc12_17.4 [template = i32]
// CHECK:STDOUT:   %.loc12_17.6: type = converted %.loc12_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc14_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_17.1: %.2 = tuple_literal (%int.make_type_32.loc14_9, %int.make_type_32.loc14_14)
// CHECK:STDOUT:   %.loc14_17.2: type = value_of_initializer %int.make_type_32.loc14_9 [template = i32]
// CHECK:STDOUT:   %.loc14_17.3: type = converted %int.make_type_32.loc14_9, %.loc14_17.2 [template = i32]
// CHECK:STDOUT:   %.loc14_17.4: type = value_of_initializer %int.make_type_32.loc14_14 [template = i32]
// CHECK:STDOUT:   %.loc14_17.5: type = converted %int.make_type_32.loc14_14, %.loc14_17.4 [template = i32]
// CHECK:STDOUT:   %.loc14_17.6: type = converted %.loc14_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref %.3 = var x
// CHECK:STDOUT:   %x: ref %.3 = bind_name x, %x.var
// CHECK:STDOUT:   %int.make_type_32.loc15_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc15_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_17.1: %.2 = tuple_literal (%int.make_type_32.loc15_9, %int.make_type_32.loc15_14)
// CHECK:STDOUT:   %.loc15_17.2: type = value_of_initializer %int.make_type_32.loc15_9 [template = i32]
// CHECK:STDOUT:   %.loc15_17.3: type = converted %int.make_type_32.loc15_9, %.loc15_17.2 [template = i32]
// CHECK:STDOUT:   %.loc15_17.4: type = value_of_initializer %int.make_type_32.loc15_14 [template = i32]
// CHECK:STDOUT:   %.loc15_17.5: type = converted %int.make_type_32.loc15_14, %.loc15_17.4 [template = i32]
// CHECK:STDOUT:   %.loc15_17.6: type = converted %.loc15_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %y.var: ref %.3 = var y
// CHECK:STDOUT:   %y: ref %.3 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_22: i32 = int_literal 4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_25: i32 = int_literal 102 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_28: %.3 = tuple_literal (%.loc11_22, %.loc11_25)
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.loc11_22, %.loc11_25) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_29: %.3 = converted %.loc11_28, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %v: %.3 = bind_name v, %.loc11_29
// CHECK:STDOUT:   %v.ref: %.3 = name_ref v, %v
// CHECK:STDOUT:   %w: %.3 = bind_name w, %v.ref
// CHECK:STDOUT:   %.loc14_22: i32 = int_literal 4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25: i32 = int_literal 102 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_28.1: %.3 = tuple_literal (%.loc14_22, %.loc14_25)
// CHECK:STDOUT:   %.loc14_28.2: ref i32 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_28.3: init i32 = initialize_from %.loc14_22 to %.loc14_28.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_28.4: ref i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_28.5: init i32 = initialize_from %.loc14_25 to %.loc14_28.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_28.6: init %.3 = tuple_init (%.loc14_28.3, %.loc14_28.5) to file.%x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_29: init %.3 = converted %.loc14_28.1, %.loc14_28.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_29
// CHECK:STDOUT:   %x.ref: ref %.3 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc15_21.1: ref i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %.loc15_21.2: i32 = bind_value %.loc15_21.1
// CHECK:STDOUT:   %.loc15_21.3: ref i32 = tuple_access file.%y.var, element0
// CHECK:STDOUT:   %.loc15_21.4: init i32 = initialize_from %.loc15_21.2 to %.loc15_21.3
// CHECK:STDOUT:   %.loc15_21.5: ref i32 = tuple_access %x.ref, element1
// CHECK:STDOUT:   %.loc15_21.6: i32 = bind_value %.loc15_21.5
// CHECK:STDOUT:   %.loc15_21.7: ref i32 = tuple_access file.%y.var, element1
// CHECK:STDOUT:   %.loc15_21.8: init i32 = initialize_from %.loc15_21.6 to %.loc15_21.7
// CHECK:STDOUT:   %.loc15_21.9: init %.3 = tuple_init (%.loc15_21.4, %.loc15_21.8) to file.%y.var
// CHECK:STDOUT:   %.loc15_22: init %.3 = converted %x.ref, %.loc15_21.9
// CHECK:STDOUT:   assign file.%y.var, %.loc15_22
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

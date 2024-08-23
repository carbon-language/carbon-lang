// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/nested_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/nested_tuple.carbon

var x: ((i32, i32), i32) = ((12, 76), 6);

// CHECK:STDOUT: --- nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (%.2, type) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (%.4, i32) [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (%.6, i32) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 76 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %tuple.1: %.4 = tuple_value (%.9, %.10) [template]
// CHECK:STDOUT:   %tuple.2: %.5 = tuple_value (%tuple.1, %.11) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_18: %.2 = tuple_literal (%int.make_type_32.loc11_10, %int.make_type_32.loc11_15)
// CHECK:STDOUT:   %int.make_type_32.loc11_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_24.1: %.3 = tuple_literal (%.loc11_18, %int.make_type_32.loc11_21)
// CHECK:STDOUT:   %.loc11_24.2: type = value_of_initializer %int.make_type_32.loc11_10 [template = i32]
// CHECK:STDOUT:   %.loc11_24.3: type = converted %int.make_type_32.loc11_10, %.loc11_24.2 [template = i32]
// CHECK:STDOUT:   %.loc11_24.4: type = value_of_initializer %int.make_type_32.loc11_15 [template = i32]
// CHECK:STDOUT:   %.loc11_24.5: type = converted %int.make_type_32.loc11_15, %.loc11_24.4 [template = i32]
// CHECK:STDOUT:   %.loc11_24.6: type = converted %.loc11_18, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_24.7: type = value_of_initializer %int.make_type_32.loc11_21 [template = i32]
// CHECK:STDOUT:   %.loc11_24.8: type = converted %int.make_type_32.loc11_21, %.loc11_24.7 [template = i32]
// CHECK:STDOUT:   %.loc11_24.9: type = converted %.loc11_24.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %x.var: ref %.5 = var x
// CHECK:STDOUT:   %x: ref %.5 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_30: i32 = int_literal 12 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_34: i32 = int_literal 76 [template = constants.%.10]
// CHECK:STDOUT:   %.loc11_36.1: %.4 = tuple_literal (%.loc11_30, %.loc11_34)
// CHECK:STDOUT:   %.loc11_39: i32 = int_literal 6 [template = constants.%.11]
// CHECK:STDOUT:   %.loc11_40.1: %.5 = tuple_literal (%.loc11_36.1, %.loc11_39)
// CHECK:STDOUT:   %.loc11_40.2: ref %.4 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc11_36.2: ref i32 = tuple_access %.loc11_40.2, element0
// CHECK:STDOUT:   %.loc11_36.3: init i32 = initialize_from %.loc11_30 to %.loc11_36.2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_36.4: ref i32 = tuple_access %.loc11_40.2, element1
// CHECK:STDOUT:   %.loc11_36.5: init i32 = initialize_from %.loc11_34 to %.loc11_36.4 [template = constants.%.10]
// CHECK:STDOUT:   %.loc11_36.6: init %.4 = tuple_init (%.loc11_36.3, %.loc11_36.5) to %.loc11_40.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_40.3: init %.4 = converted %.loc11_36.1, %.loc11_36.6 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_40.4: ref i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_40.5: init i32 = initialize_from %.loc11_39 to %.loc11_40.4 [template = constants.%.11]
// CHECK:STDOUT:   %.loc11_40.6: init %.5 = tuple_init (%.loc11_40.3, %.loc11_40.5) to file.%x.var [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc11_41: init %.5 = converted %.loc11_40.1, %.loc11_40.6 [template = constants.%tuple.2]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_41
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

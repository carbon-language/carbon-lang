// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/array_element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/array_element_access.carbon

var a: [i32; 2] = (12, 24);
var b: i32 = 1;
var c: i32 = a[0];
var d: i32 = a[b];

// CHECK:STDOUT: --- array_element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 24 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5, %.6) [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32.loc11, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_32.loc12, %.loc12_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %int.make_type_32.loc13, %.loc13_8.1 [template = i32]
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_8.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_8.2: type = converted %int.make_type_32.loc14, %.loc14_8.1 [template = i32]
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: i32 = int_literal 12 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_24: i32 = int_literal 24 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_26.1: %.7 = tuple_literal (%.loc11_20, %.loc11_24)
// CHECK:STDOUT:   %.loc11_26.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_26.3: ref i32 = array_index file.%a.var, %.loc11_26.2
// CHECK:STDOUT:   %.loc11_26.4: init i32 = initialize_from %.loc11_20 to %.loc11_26.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_26.5: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_26.6: ref i32 = array_index file.%a.var, %.loc11_26.5
// CHECK:STDOUT:   %.loc11_26.7: init i32 = initialize_from %.loc11_24 to %.loc11_26.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_26.8: init %.3 = array_init (%.loc11_26.4, %.loc11_26.7) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_27: init %.3 = converted %.loc11_26.1, %.loc11_26.8 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_27
// CHECK:STDOUT:   %.loc12: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   assign file.%b.var, %.loc12
// CHECK:STDOUT:   %a.ref.loc13: ref %.3 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc13_16: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_17.1: ref i32 = array_index %a.ref.loc13, %.loc13_16
// CHECK:STDOUT:   %.loc13_17.2: i32 = bind_value %.loc13_17.1
// CHECK:STDOUT:   assign file.%c.var, %.loc13_17.2
// CHECK:STDOUT:   %a.ref.loc14: ref %.3 = name_ref a, file.%a
// CHECK:STDOUT:   %b.ref: ref i32 = name_ref b, file.%b
// CHECK:STDOUT:   %.loc14_16: i32 = bind_value %b.ref
// CHECK:STDOUT:   %.loc14_17.1: ref i32 = array_index %a.ref.loc14, %.loc14_16
// CHECK:STDOUT:   %.loc14_17.2: i32 = bind_value %.loc14_17.1
// CHECK:STDOUT:   assign file.%d.var, %.loc14_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

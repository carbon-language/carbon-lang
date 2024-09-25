// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_array_large_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_array_large_index.carbon

var a: [i32; 1] = (12,);

// CHECK:STDERR: fail_array_large_index.carbon:[[@LINE+4]]:16: error: array index `1` is past the end of type `[i32; 1]`
// CHECK:STDERR: var b: i32 = a[1];
// CHECK:STDERR:                ^
// CHECK:STDERR:
var b: i32 = a[1];

// CHECK:STDERR: fail_array_large_index.carbon:[[@LINE+3]]:16: error: array index `2147483647` is past the end of type `[i32; 1]`
// CHECK:STDERR: var c: i32 = a[0x7FFF_FFFF];
// CHECK:STDERR:                ^~~~~~~~~~~
var c: i32 = a[0x7FFF_FFFF];

// CHECK:STDOUT: --- fail_array_large_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 2147483647 [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32.loc11, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_8.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_8.2: type = converted %int.make_type_32.loc17, %.loc17_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_8.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_8.2: type = converted %int.make_type_32.loc22, %.loc22_8.1 [template = i32]
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: i32 = int_literal 12 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_23.1: %.6 = tuple_literal (%.loc11_20)
// CHECK:STDOUT:   %.loc11_23.2: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_23.3: ref i32 = array_index file.%a.var, %.loc11_23.2
// CHECK:STDOUT:   %.loc11_23.4: init i32 = initialize_from %.loc11_20 to %.loc11_23.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_23.5: init %.3 = array_init (%.loc11_23.4) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_24: init %.3 = converted %.loc11_23.1, %.loc11_23.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_24
// CHECK:STDOUT:   %a.ref.loc17: ref %.3 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc17_16: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc17_17.1: ref i32 = array_index %a.ref.loc17, %.loc17_16 [template = <error>]
// CHECK:STDOUT:   %.loc17_17.2: i32 = bind_value %.loc17_17.1
// CHECK:STDOUT:   assign file.%b.var, %.loc17_17.2
// CHECK:STDOUT:   %a.ref.loc22: ref %.3 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc22_16: i32 = int_literal 2147483647 [template = constants.%.8]
// CHECK:STDOUT:   %.loc22_27.1: ref i32 = array_index %a.ref.loc22, %.loc22_16 [template = <error>]
// CHECK:STDOUT:   %.loc22_27.2: i32 = bind_value %.loc22_27.1
// CHECK:STDOUT:   assign file.%c.var, %.loc22_27.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

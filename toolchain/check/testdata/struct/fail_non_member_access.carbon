// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_non_member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_non_member_access.carbon

var x: {.a: i32} = {.a = 4};
// CHECK:STDERR: fail_non_member_access.carbon:[[@LINE+3]]:14: ERROR: Type `{.a: i32}` does not have a member `b`.
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:              ^~~
var y: i32 = x.b;

// CHECK:STDOUT: --- fail_non_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %struct: %.2 = struct_value (%.3) [template]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_13.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_13.2: type = converted %int.make_type_32.loc11, %.loc11_13.1 [template = i32]
// CHECK:STDOUT:   %.loc11_16: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref %.2 = var x
// CHECK:STDOUT:   %x: ref %.2 = bind_name x, %x.var
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %y.var: ref i32 = var y
// CHECK:STDOUT:   %y: ref i32 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_26: i32 = int_literal 4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_27.1: %.2 = struct_literal (%.loc11_26)
// CHECK:STDOUT:   %.loc11_27.2: init %.2 = struct_init (%.loc11_26) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_28: init %.2 = converted %.loc11_27.1, %.loc11_27.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_28
// CHECK:STDOUT:   %x.ref: ref %.2 = name_ref x, file.%x
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

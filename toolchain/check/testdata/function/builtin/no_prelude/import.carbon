// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/no_prelude/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/no_prelude/import.carbon

// --- core.carbon

package Core library "[[@TEST_NAME]]";

fn Int32() -> type = "int.make_type_32";
fn TestAdd(a: i32, b: i32) -> i32 = "int.sadd";

// --- use.carbon

// CHECK:STDERR: use.carbon:[[@LINE+4]]:1: ERROR: Imported API 'Core//test' not found.
// CHECK:STDERR: import Core library "test";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Core library "test";

// CHECK:STDERR: use.carbon:[[@LINE+3]]:11: ERROR: Name `Core.Int32` implicitly referenced here, but not found.
// CHECK:STDERR: var arr: [i32; Core.TestAdd(1, 2)] = (1, 2, 3);
// CHECK:STDERR:           ^~~
var arr: [i32; Core.TestAdd(1, 2)] = (1, 2, 3);

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %TestAdd.type: type = fn_type @TestAdd [template]
// CHECK:STDOUT:   %TestAdd: %TestAdd.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int32 = %Int32.decl
// CHECK:STDOUT:     .TestAdd = %TestAdd.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32.decl: %Int32.type = fn_decl @Int32 [template = constants.%Int32] {
// CHECK:STDOUT:     @Int32.%return: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAdd.decl: %TestAdd.type = fn_decl @TestAdd [template = constants.%TestAdd] {
// CHECK:STDOUT:     %int.make_type_32.loc5_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_15.1: type = value_of_initializer %int.make_type_32.loc5_15 [template = i32]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %int.make_type_32.loc5_15, %.loc5_15.1 [template = i32]
// CHECK:STDOUT:     %a.loc5_12.1: i32 = param a
// CHECK:STDOUT:     @TestAdd.%a: i32 = bind_name a, %a.loc5_12.1
// CHECK:STDOUT:     %int.make_type_32.loc5_23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_23.1: type = value_of_initializer %int.make_type_32.loc5_23 [template = i32]
// CHECK:STDOUT:     %.loc5_23.2: type = converted %int.make_type_32.loc5_23, %.loc5_23.1 [template = i32]
// CHECK:STDOUT:     %b.loc5_20.1: i32 = param b
// CHECK:STDOUT:     @TestAdd.%b: i32 = bind_name b, %b.loc5_20.1
// CHECK:STDOUT:     %int.make_type_32.loc5_31: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_31.1: type = value_of_initializer %int.make_type_32.loc5_31 [template = i32]
// CHECK:STDOUT:     %.loc5_31.2: type = converted %int.make_type_32.loc5_31, %.loc5_31.1 [template = i32]
// CHECK:STDOUT:     @TestAdd.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdd(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %TestAdd.ref: <error> = name_ref TestAdd, <error> [template = <error>]
// CHECK:STDOUT:   %.loc11_29: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_32: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_34: type = array_type <error>, <error> [template = <error>]
// CHECK:STDOUT:   %arr.var: ref <error> = var arr
// CHECK:STDOUT:   %arr: ref <error> = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_39: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_42: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_45: i32 = int_literal 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_46: %.4 = tuple_literal (%.loc11_39, %.loc11_42, %.loc11_45)
// CHECK:STDOUT:   assign file.%arr.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

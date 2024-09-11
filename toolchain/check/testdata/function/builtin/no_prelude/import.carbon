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

import Core library "core";

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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %TestAdd.type: type = fn_type @TestAdd [template]
// CHECK:STDOUT:   %TestAdd: %TestAdd.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: %.5 = tuple_value (%.2, %.3, %.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .TestAdd = %import_ref.2
// CHECK:STDOUT:     import Core//core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//core, inst+2, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %TestAdd.type = import_ref Core//core, inst+20, loaded [template = constants.%TestAdd]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %TestAdd.ref: %TestAdd.type = name_ref TestAdd, imports.%import_ref.2 [template = constants.%TestAdd]
// CHECK:STDOUT:   %.loc4_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_32: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %int.sadd: init i32 = call %TestAdd.ref(%.loc4_29, %.loc4_32) [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_34: type = array_type %int.sadd, i32 [template = constants.%.5]
// CHECK:STDOUT:   %arr.var: ref %.5 = var arr
// CHECK:STDOUT:   %arr: ref %.5 = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdd(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_39: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_42: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_45: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_46.1: %.7 = tuple_literal (%.loc4_39, %.loc4_42, %.loc4_45)
// CHECK:STDOUT:   %.loc4_46.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc4_46.3: ref i32 = array_index file.%arr.var, %.loc4_46.2
// CHECK:STDOUT:   %.loc4_46.4: init i32 = initialize_from %.loc4_39 to %.loc4_46.3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_46.5: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_46.6: ref i32 = array_index file.%arr.var, %.loc4_46.5
// CHECK:STDOUT:   %.loc4_46.7: init i32 = initialize_from %.loc4_42 to %.loc4_46.6 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_46.8: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_46.9: ref i32 = array_index file.%arr.var, %.loc4_46.8
// CHECK:STDOUT:   %.loc4_46.10: init i32 = initialize_from %.loc4_45 to %.loc4_46.9 [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_46.11: init %.5 = array_init (%.loc4_46.4, %.loc4_46.7, %.loc4_46.10) to file.%arr.var [template = constants.%array]
// CHECK:STDOUT:   %.loc4_47: init %.5 = converted %.loc4_46.1, %.loc4_46.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_47
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

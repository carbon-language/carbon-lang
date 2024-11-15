// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/fail_access_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/fail_access_error.carbon

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_access_error.carbon:[[@LINE+3]]:17: error: name `oops` not found [NameNotFound]
// CHECK:STDERR: var b: i32 = a.(oops);
// CHECK:STDERR:                 ^~~~
var b: i32 = a.(oops);

// CHECK:STDOUT: --- fail_access_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.15 [template]
// CHECK:STDOUT:   %.29: i32 = int_value 12 [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.3, %Convert.15 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 6 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.29, %.31) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_17.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc11_9, %int.make_type_32.loc11_14)
// CHECK:STDOUT:   %.loc11_17.2: type = value_of_initializer %int.make_type_32.loc11_9 [template = i32]
// CHECK:STDOUT:   %.loc11_17.3: type = converted %int.make_type_32.loc11_9, %.loc11_17.2 [template = i32]
// CHECK:STDOUT:   %.loc11_17.4: type = value_of_initializer %int.make_type_32.loc11_14 [template = i32]
// CHECK:STDOUT:   %.loc11_17.5: type = converted %int.make_type_32.loc11_14, %.loc11_17.4 [template = i32]
// CHECK:STDOUT:   %.loc11_17.6: type = converted %.loc11_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_22: Core.IntLiteral = int_value 12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_26: Core.IntLiteral = int_value 6 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type.3 = tuple_literal (%.loc11_22, %.loc11_26)
// CHECK:STDOUT:   %.loc11_27.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_27.3: <bound method> = bound_method %.loc11_22, %.loc11_27.2 [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.1: init i32 = call %.loc11_27.3(%.loc11_22) [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_27.4: init i32 = converted %.loc11_22, %int.convert_checked.loc11_27.1 [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_27.5: ref i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_27.6: init i32 = initialize_from %.loc11_27.4 to %.loc11_27.5 [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_27.7: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_27.8: <bound method> = bound_method %.loc11_26, %.loc11_27.7 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.2: init i32 = call %.loc11_27.8(%.loc11_26) [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_27.9: init i32 = converted %.loc11_26, %int.convert_checked.loc11_27.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_27.10: ref i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_27.11: init i32 = initialize_from %.loc11_27.9 to %.loc11_27.10 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_27.12: init %tuple.type.2 = tuple_init (%.loc11_27.6, %.loc11_27.11) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_28: init %tuple.type.2 = converted %.loc11_27.1, %.loc11_27.12 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %oops.ref: <error> = name_ref oops, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

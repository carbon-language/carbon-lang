// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/assign_return_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/assign_return_value.carbon

fn F() -> (i32,) { return (0,); }

fn Run() {
  var t: [i32; 1] = F();
}

// CHECK:STDOUT: --- assign_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.4) [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: <type of Run> = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: type = array_type %.5, i32 [template]
// CHECK:STDOUT:   %.7: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_16.1: (type,) = tuple_literal (%int.make_type_32)
// CHECK:STDOUT:     %.loc11_16.2: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_16.3: type = converted %int.make_type_32, %.loc11_16.2 [template = i32]
// CHECK:STDOUT:     %.loc11_16.4: type = converted %.loc11_16.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref (i32,) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: <type of Run> = fn_decl @Run [template = constants.%Run] {}
// CHECK:STDOUT:   %import_ref.2: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32,) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_28: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_30: (i32,) = tuple_literal (%.loc11_28)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc11_28) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_31: (i32,) = converted %.loc11_30, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc11_31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_16: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc14_11.2: type = converted %int.make_type_32, %.loc14_11.1 [template = i32]
// CHECK:STDOUT:   %.loc14_17: type = array_type %.loc14_16, i32 [template = constants.%.6]
// CHECK:STDOUT:   %t.var: ref [i32; 1] = var t
// CHECK:STDOUT:   %t: ref [i32; 1] = bind_name t, %t.var
// CHECK:STDOUT:   %F.ref: <type of F> = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init (i32,) = call %F.ref()
// CHECK:STDOUT:   %.loc14_22.1: ref (i32,) = temporary_storage
// CHECK:STDOUT:   %.loc14_22.2: ref (i32,) = temporary %.loc14_22.1, %F.call
// CHECK:STDOUT:   %.loc14_22.3: ref i32 = tuple_access %.loc14_22.2, element0
// CHECK:STDOUT:   %.loc14_22.4: i32 = bind_value %.loc14_22.3
// CHECK:STDOUT:   %.loc14_22.5: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_22.6: ref i32 = array_index %t.var, %.loc14_22.5
// CHECK:STDOUT:   %.loc14_22.7: init i32 = initialize_from %.loc14_22.4 to %.loc14_22.6
// CHECK:STDOUT:   %.loc14_22.8: init [i32; 1] = array_init (%.loc14_22.7) to %t.var
// CHECK:STDOUT:   %.loc14_24: init [i32; 1] = converted %F.call, %.loc14_22.8
// CHECK:STDOUT:   assign %t.var, %.loc14_24
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

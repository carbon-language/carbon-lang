// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/array_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/array_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var v: [(i32, i32, i32); 2] = (F(), F());
}

// CHECK:STDOUT: --- array_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.3: G = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, (i32, i32, i32) [template]
// CHECK:STDOUT:   %.6: type = ptr_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.7: type = ptr_type [(i32, i32, i32); 2] [template]
// CHECK:STDOUT:   %.8: type = tuple_type ((i32, i32, i32), (i32, i32, i32)) [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc11_12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_22: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_25.1: (type, type, type) = tuple_literal (%int.make_type_32.loc11_12, %int.make_type_32.loc11_17, %int.make_type_32.loc11_22)
// CHECK:STDOUT:     %.loc11_25.2: type = value_of_initializer %int.make_type_32.loc11_12 [template = i32]
// CHECK:STDOUT:     %.loc11_25.3: type = converted %int.make_type_32.loc11_12, %.loc11_25.2 [template = i32]
// CHECK:STDOUT:     %.loc11_25.4: type = value_of_initializer %int.make_type_32.loc11_17 [template = i32]
// CHECK:STDOUT:     %.loc11_25.5: type = converted %int.make_type_32.loc11_17, %.loc11_25.4 [template = i32]
// CHECK:STDOUT:     %.loc11_25.6: type = value_of_initializer %int.make_type_32.loc11_22 [template = i32]
// CHECK:STDOUT:     %.loc11_25.7: type = converted %int.make_type_32.loc11_22, %.loc11_25.6 [template = i32]
// CHECK:STDOUT:     %.loc11_25.8: type = converted %.loc11_25.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref (i32, i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {}
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32, i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc14_12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_22: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc14_25.1: (type, type, type) = tuple_literal (%int.make_type_32.loc14_12, %int.make_type_32.loc14_17, %int.make_type_32.loc14_22)
// CHECK:STDOUT:   %.loc14_28: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_25.2: type = value_of_initializer %int.make_type_32.loc14_12 [template = i32]
// CHECK:STDOUT:   %.loc14_25.3: type = converted %int.make_type_32.loc14_12, %.loc14_25.2 [template = i32]
// CHECK:STDOUT:   %.loc14_25.4: type = value_of_initializer %int.make_type_32.loc14_17 [template = i32]
// CHECK:STDOUT:   %.loc14_25.5: type = converted %int.make_type_32.loc14_17, %.loc14_25.4 [template = i32]
// CHECK:STDOUT:   %.loc14_25.6: type = value_of_initializer %int.make_type_32.loc14_22 [template = i32]
// CHECK:STDOUT:   %.loc14_25.7: type = converted %int.make_type_32.loc14_22, %.loc14_25.6 [template = i32]
// CHECK:STDOUT:   %.loc14_25.8: type = converted %.loc14_25.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_29: type = array_type %.loc14_28, (i32, i32, i32) [template = constants.%.5]
// CHECK:STDOUT:   %v.var: ref [(i32, i32, i32); 2] = var v
// CHECK:STDOUT:   %v: ref [(i32, i32, i32); 2] = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc14_34: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc14_42.3: ref (i32, i32, i32) = splice_block %.loc14_42.2 {
// CHECK:STDOUT:     %.loc14_42.1: i32 = int_literal 0 [template = constants.%.9]
// CHECK:STDOUT:     %.loc14_42.2: ref (i32, i32, i32) = array_index %v.var, %.loc14_42.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.call.loc14_35: init (i32, i32, i32) = call %F.ref.loc14_34() to %.loc14_42.3
// CHECK:STDOUT:   %F.ref.loc14_39: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc14_42.6: ref (i32, i32, i32) = splice_block %.loc14_42.5 {
// CHECK:STDOUT:     %.loc14_42.4: i32 = int_literal 1 [template = constants.%.10]
// CHECK:STDOUT:     %.loc14_42.5: ref (i32, i32, i32) = array_index %v.var, %.loc14_42.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.call.loc14_40: init (i32, i32, i32) = call %F.ref.loc14_39() to %.loc14_42.6
// CHECK:STDOUT:   %.loc14_42.7: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%F.call.loc14_35, %F.call.loc14_40)
// CHECK:STDOUT:   %.loc14_42.8: init [(i32, i32, i32); 2] = array_init (%F.call.loc14_35, %F.call.loc14_40) to %v.var
// CHECK:STDOUT:   %.loc14_43: init [(i32, i32, i32); 2] = converted %.loc14_42.7, %.loc14_42.8
// CHECK:STDOUT:   assign %v.var, %.loc14_43
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

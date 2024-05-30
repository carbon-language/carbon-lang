// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/expr_category/in_place_tuple_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/expr_category/in_place_tuple_init.carbon

fn F() -> (i32, i32);

fn G() -> (i32, i32) {
  var v: (i32, i32) = F();
  v = F();
  return F();
}

fn H() -> i32 {
  return G()[0];
}

// CHECK:STDOUT: --- in_place_tuple_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.3: G = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %H: type = fn_type @H [template]
// CHECK:STDOUT:   %struct.4: H = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc11_12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_20.1: (type, type) = tuple_literal (%int.make_type_32.loc11_12, %int.make_type_32.loc11_17)
// CHECK:STDOUT:     %.loc11_20.2: type = value_of_initializer %int.make_type_32.loc11_12 [template = i32]
// CHECK:STDOUT:     %.loc11_20.3: type = converted %int.make_type_32.loc11_12, %.loc11_20.2 [template = i32]
// CHECK:STDOUT:     %.loc11_20.4: type = value_of_initializer %int.make_type_32.loc11_17 [template = i32]
// CHECK:STDOUT:     %.loc11_20.5: type = converted %int.make_type_32.loc11_17, %.loc11_20.4 [template = i32]
// CHECK:STDOUT:     %.loc11_20.6: type = converted %.loc11_20.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref (i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc13_12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc13_17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc13_20.1: (type, type) = tuple_literal (%int.make_type_32.loc13_12, %int.make_type_32.loc13_17)
// CHECK:STDOUT:     %.loc13_20.2: type = value_of_initializer %int.make_type_32.loc13_12 [template = i32]
// CHECK:STDOUT:     %.loc13_20.3: type = converted %int.make_type_32.loc13_12, %.loc13_20.2 [template = i32]
// CHECK:STDOUT:     %.loc13_20.4: type = value_of_initializer %int.make_type_32.loc13_17 [template = i32]
// CHECK:STDOUT:     %.loc13_20.5: type = converted %int.make_type_32.loc13_17, %.loc13_20.4 [template = i32]
// CHECK:STDOUT:     %.loc13_20.6: type = converted %.loc13_20.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     @G.%return: ref (i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %H.decl: H = fn_decl @H [template = constants.%struct.4] {
// CHECK:STDOUT:     %int.make_type_32.loc19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc19_11.1: type = value_of_initializer %int.make_type_32.loc19 [template = i32]
// CHECK:STDOUT:     %.loc19_11.2: type = converted %int.make_type_32.loc19, %.loc19_11.1 [template = i32]
// CHECK:STDOUT:     @H.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc14_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_16: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc14_19.1: (type, type) = tuple_literal (%int.make_type_32.loc14_11, %int.make_type_32.loc14_16)
// CHECK:STDOUT:   %.loc14_19.2: type = value_of_initializer %int.make_type_32.loc14_11 [template = i32]
// CHECK:STDOUT:   %.loc14_19.3: type = converted %int.make_type_32.loc14_11, %.loc14_19.2 [template = i32]
// CHECK:STDOUT:   %.loc14_19.4: type = value_of_initializer %int.make_type_32.loc14_16 [template = i32]
// CHECK:STDOUT:   %.loc14_19.5: type = converted %int.make_type_32.loc14_16, %.loc14_19.4 [template = i32]
// CHECK:STDOUT:   %.loc14_19.6: type = converted %.loc14_19.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %v.var: ref (i32, i32) = var v
// CHECK:STDOUT:   %v: ref (i32, i32) = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc14: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc14_7: ref (i32, i32) = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc14: init (i32, i32) = call %F.ref.loc14() to %.loc14_7
// CHECK:STDOUT:   assign %v.var, %F.call.loc14
// CHECK:STDOUT:   %v.ref: ref (i32, i32) = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc15: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc15: ref (i32, i32) = splice_block %v.ref {}
// CHECK:STDOUT:   %F.call.loc15: init (i32, i32) = call %F.ref.loc15() to %.loc15
// CHECK:STDOUT:   assign %v.ref, %F.call.loc15
// CHECK:STDOUT:   %F.ref.loc16: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc13: ref (i32, i32) = splice_block %return {}
// CHECK:STDOUT:   %F.call.loc16: init (i32, i32) = call %F.ref.loc16() to %.loc13
// CHECK:STDOUT:   return %F.call.loc16 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc20_11.1: ref (i32, i32) = temporary_storage
// CHECK:STDOUT:   %G.call: init (i32, i32) = call %G.ref() to %.loc20_11.1
// CHECK:STDOUT:   %.loc20_14: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc20_11.2: ref (i32, i32) = temporary %.loc20_11.1, %G.call
// CHECK:STDOUT:   %.loc20_15.1: ref i32 = tuple_index %.loc20_11.2, %.loc20_14
// CHECK:STDOUT:   %.loc20_15.2: i32 = bind_value %.loc20_15.1
// CHECK:STDOUT:   return %.loc20_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

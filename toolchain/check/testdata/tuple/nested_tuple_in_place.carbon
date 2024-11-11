// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/nested_tuple_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/nested_tuple_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var v: ((i32, i32, i32), (i32, i32, i32)) = (F(), F());
}

fn H() {
  var v: (i32, (i32, i32, i32), i32) = (1, F(), 2);
}

// CHECK:STDOUT: --- nested_tuple_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%tuple.type.1, %tuple.type.1) [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (%tuple.type.2, %tuple.type.2) [template]
// CHECK:STDOUT:   %.1: type = ptr_type %tuple.type.2 [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (%.1, %.1) [template]
// CHECK:STDOUT:   %.2: type = ptr_type %tuple.type.5 [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.6: type = tuple_type (type, %tuple.type.1, type) [template]
// CHECK:STDOUT:   %tuple.type.7: type = tuple_type (i32, %tuple.type.2, i32) [template]
// CHECK:STDOUT:   %tuple.type.8: type = tuple_type (i32, %.1, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type %tuple.type.8 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc11_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_25.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc11_12, %int.make_type_32.loc11_17, %int.make_type_32.loc11_22)
// CHECK:STDOUT:     %.loc11_25.2: type = value_of_initializer %int.make_type_32.loc11_12 [template = i32]
// CHECK:STDOUT:     %.loc11_25.3: type = converted %int.make_type_32.loc11_12, %.loc11_25.2 [template = i32]
// CHECK:STDOUT:     %.loc11_25.4: type = value_of_initializer %int.make_type_32.loc11_17 [template = i32]
// CHECK:STDOUT:     %.loc11_25.5: type = converted %int.make_type_32.loc11_17, %.loc11_25.4 [template = i32]
// CHECK:STDOUT:     %.loc11_25.6: type = value_of_initializer %int.make_type_32.loc11_22 [template = i32]
// CHECK:STDOUT:     %.loc11_25.7: type = converted %int.make_type_32.loc11_22, %.loc11_25.6 [template = i32]
// CHECK:STDOUT:     %.loc11_25.8: type = converted %.loc11_25.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %tuple.type.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc14_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_25: %tuple.type.1 = tuple_literal (%int.make_type_32.loc14_12, %int.make_type_32.loc14_17, %int.make_type_32.loc14_22)
// CHECK:STDOUT:   %int.make_type_32.loc14_29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_39: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_42: %tuple.type.1 = tuple_literal (%int.make_type_32.loc14_29, %int.make_type_32.loc14_34, %int.make_type_32.loc14_39)
// CHECK:STDOUT:   %.loc14_43.1: %tuple.type.3 = tuple_literal (%.loc14_25, %.loc14_42)
// CHECK:STDOUT:   %.loc14_43.2: type = value_of_initializer %int.make_type_32.loc14_12 [template = i32]
// CHECK:STDOUT:   %.loc14_43.3: type = converted %int.make_type_32.loc14_12, %.loc14_43.2 [template = i32]
// CHECK:STDOUT:   %.loc14_43.4: type = value_of_initializer %int.make_type_32.loc14_17 [template = i32]
// CHECK:STDOUT:   %.loc14_43.5: type = converted %int.make_type_32.loc14_17, %.loc14_43.4 [template = i32]
// CHECK:STDOUT:   %.loc14_43.6: type = value_of_initializer %int.make_type_32.loc14_22 [template = i32]
// CHECK:STDOUT:   %.loc14_43.7: type = converted %int.make_type_32.loc14_22, %.loc14_43.6 [template = i32]
// CHECK:STDOUT:   %.loc14_43.8: type = converted %.loc14_25, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc14_43.9: type = value_of_initializer %int.make_type_32.loc14_29 [template = i32]
// CHECK:STDOUT:   %.loc14_43.10: type = converted %int.make_type_32.loc14_29, %.loc14_43.9 [template = i32]
// CHECK:STDOUT:   %.loc14_43.11: type = value_of_initializer %int.make_type_32.loc14_34 [template = i32]
// CHECK:STDOUT:   %.loc14_43.12: type = converted %int.make_type_32.loc14_34, %.loc14_43.11 [template = i32]
// CHECK:STDOUT:   %.loc14_43.13: type = value_of_initializer %int.make_type_32.loc14_39 [template = i32]
// CHECK:STDOUT:   %.loc14_43.14: type = converted %int.make_type_32.loc14_39, %.loc14_43.13 [template = i32]
// CHECK:STDOUT:   %.loc14_43.15: type = converted %.loc14_42, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc14_43.16: type = converted %.loc14_43.1, constants.%tuple.type.4 [template = constants.%tuple.type.4]
// CHECK:STDOUT:   %v.var: ref %tuple.type.4 = var v
// CHECK:STDOUT:   %v: ref %tuple.type.4 = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc14_48: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc14_56.1: ref %tuple.type.2 = tuple_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc14_49: init %tuple.type.2 = call %F.ref.loc14_48() to %.loc14_56.1
// CHECK:STDOUT:   %F.ref.loc14_53: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc14_56.2: ref %tuple.type.2 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc14_54: init %tuple.type.2 = call %F.ref.loc14_53() to %.loc14_56.2
// CHECK:STDOUT:   %.loc14_56.3: %tuple.type.4 = tuple_literal (%F.call.loc14_49, %F.call.loc14_54)
// CHECK:STDOUT:   %.loc14_56.4: init %tuple.type.4 = tuple_init (%F.call.loc14_49, %F.call.loc14_54) to %v.var
// CHECK:STDOUT:   %.loc14_57: init %tuple.type.4 = converted %.loc14_56.3, %.loc14_56.4
// CHECK:STDOUT:   assign %v.var, %.loc14_57
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc18_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc18_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc18_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc18_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_30: %tuple.type.1 = tuple_literal (%int.make_type_32.loc18_17, %int.make_type_32.loc18_22, %int.make_type_32.loc18_27)
// CHECK:STDOUT:   %int.make_type_32.loc18_33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_36.1: %tuple.type.6 = tuple_literal (%int.make_type_32.loc18_11, %.loc18_30, %int.make_type_32.loc18_33)
// CHECK:STDOUT:   %.loc18_36.2: type = value_of_initializer %int.make_type_32.loc18_11 [template = i32]
// CHECK:STDOUT:   %.loc18_36.3: type = converted %int.make_type_32.loc18_11, %.loc18_36.2 [template = i32]
// CHECK:STDOUT:   %.loc18_36.4: type = value_of_initializer %int.make_type_32.loc18_17 [template = i32]
// CHECK:STDOUT:   %.loc18_36.5: type = converted %int.make_type_32.loc18_17, %.loc18_36.4 [template = i32]
// CHECK:STDOUT:   %.loc18_36.6: type = value_of_initializer %int.make_type_32.loc18_22 [template = i32]
// CHECK:STDOUT:   %.loc18_36.7: type = converted %int.make_type_32.loc18_22, %.loc18_36.6 [template = i32]
// CHECK:STDOUT:   %.loc18_36.8: type = value_of_initializer %int.make_type_32.loc18_27 [template = i32]
// CHECK:STDOUT:   %.loc18_36.9: type = converted %int.make_type_32.loc18_27, %.loc18_36.8 [template = i32]
// CHECK:STDOUT:   %.loc18_36.10: type = converted %.loc18_30, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc18_36.11: type = value_of_initializer %int.make_type_32.loc18_33 [template = i32]
// CHECK:STDOUT:   %.loc18_36.12: type = converted %int.make_type_32.loc18_33, %.loc18_36.11 [template = i32]
// CHECK:STDOUT:   %.loc18_36.13: type = converted %.loc18_36.1, constants.%tuple.type.7 [template = constants.%tuple.type.7]
// CHECK:STDOUT:   %v.var: ref %tuple.type.7 = var v
// CHECK:STDOUT:   %v: ref %tuple.type.7 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc18_41: i32 = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc18_50.1: ref %tuple.type.2 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call: init %tuple.type.2 = call %F.ref() to %.loc18_50.1
// CHECK:STDOUT:   %.loc18_49: i32 = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_50.2: %tuple.type.7 = tuple_literal (%.loc18_41, %F.call, %.loc18_49)
// CHECK:STDOUT:   %.loc18_50.3: ref i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc18_50.4: init i32 = initialize_from %.loc18_41 to %.loc18_50.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_50.5: ref i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc18_50.6: init i32 = initialize_from %.loc18_49 to %.loc18_50.5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_50.7: init %tuple.type.7 = tuple_init (%.loc18_50.4, %F.call, %.loc18_50.6) to %v.var
// CHECK:STDOUT:   %.loc18_51: init %tuple.type.7 = converted %.loc18_50.2, %.loc18_50.7
// CHECK:STDOUT:   assign %v.var, %.loc18_51
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

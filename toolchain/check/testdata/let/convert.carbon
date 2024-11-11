// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/convert.carbon

fn F() -> i32 {
  var v: (i32, i32, i32) = (1, 2, 3);
  // Convert from object representation to value representation.
  let w: (i32, i32, i32) = v;
  return w.1;
}

// CHECK:STDOUT: --- convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.1: type = ptr_type %tuple.type.2 [template]
// CHECK:STDOUT:   %.2: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 3 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.2, %.3, %.4) [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32.loc11, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc12_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc12_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc12_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_24.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc12_11, %int.make_type_32.loc12_16, %int.make_type_32.loc12_21)
// CHECK:STDOUT:   %.loc12_24.2: type = value_of_initializer %int.make_type_32.loc12_11 [template = i32]
// CHECK:STDOUT:   %.loc12_24.3: type = converted %int.make_type_32.loc12_11, %.loc12_24.2 [template = i32]
// CHECK:STDOUT:   %.loc12_24.4: type = value_of_initializer %int.make_type_32.loc12_16 [template = i32]
// CHECK:STDOUT:   %.loc12_24.5: type = converted %int.make_type_32.loc12_16, %.loc12_24.4 [template = i32]
// CHECK:STDOUT:   %.loc12_24.6: type = value_of_initializer %int.make_type_32.loc12_21 [template = i32]
// CHECK:STDOUT:   %.loc12_24.7: type = converted %int.make_type_32.loc12_21, %.loc12_24.6 [template = i32]
// CHECK:STDOUT:   %.loc12_24.8: type = converted %.loc12_24.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %v.var: ref %tuple.type.2 = var v
// CHECK:STDOUT:   %v: ref %tuple.type.2 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc12_29: i32 = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_32: i32 = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_35: i32 = int_value 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_36.1: %tuple.type.2 = tuple_literal (%.loc12_29, %.loc12_32, %.loc12_35)
// CHECK:STDOUT:   %.loc12_36.2: ref i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc12_36.3: init i32 = initialize_from %.loc12_29 to %.loc12_36.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_36.4: ref i32 = tuple_access %v.var, element1
// CHECK:STDOUT:   %.loc12_36.5: init i32 = initialize_from %.loc12_32 to %.loc12_36.4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_36.6: ref i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc12_36.7: init i32 = initialize_from %.loc12_35 to %.loc12_36.6 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_36.8: init %tuple.type.2 = tuple_init (%.loc12_36.3, %.loc12_36.5, %.loc12_36.7) to %v.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc12_37: init %tuple.type.2 = converted %.loc12_36.1, %.loc12_36.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign %v.var, %.loc12_37
// CHECK:STDOUT:   %int.make_type_32.loc14_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_24.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc14_11, %int.make_type_32.loc14_16, %int.make_type_32.loc14_21)
// CHECK:STDOUT:   %.loc14_24.2: type = value_of_initializer %int.make_type_32.loc14_11 [template = i32]
// CHECK:STDOUT:   %.loc14_24.3: type = converted %int.make_type_32.loc14_11, %.loc14_24.2 [template = i32]
// CHECK:STDOUT:   %.loc14_24.4: type = value_of_initializer %int.make_type_32.loc14_16 [template = i32]
// CHECK:STDOUT:   %.loc14_24.5: type = converted %int.make_type_32.loc14_16, %.loc14_24.4 [template = i32]
// CHECK:STDOUT:   %.loc14_24.6: type = value_of_initializer %int.make_type_32.loc14_21 [template = i32]
// CHECK:STDOUT:   %.loc14_24.7: type = converted %int.make_type_32.loc14_21, %.loc14_24.6 [template = i32]
// CHECK:STDOUT:   %.loc14_24.8: type = converted %.loc14_24.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %v.ref: ref %tuple.type.2 = name_ref v, %v
// CHECK:STDOUT:   %.loc14_28.1: ref i32 = tuple_access %v.ref, element0
// CHECK:STDOUT:   %.loc14_28.2: i32 = bind_value %.loc14_28.1
// CHECK:STDOUT:   %.loc14_28.3: ref i32 = tuple_access %v.ref, element1
// CHECK:STDOUT:   %.loc14_28.4: i32 = bind_value %.loc14_28.3
// CHECK:STDOUT:   %.loc14_28.5: ref i32 = tuple_access %v.ref, element2
// CHECK:STDOUT:   %.loc14_28.6: i32 = bind_value %.loc14_28.5
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc14_28.2, %.loc14_28.4, %.loc14_28.6)
// CHECK:STDOUT:   %.loc14_29: %tuple.type.2 = converted %v.ref, %tuple
// CHECK:STDOUT:   %w: %tuple.type.2 = bind_name w, %.loc14_29
// CHECK:STDOUT:   %w.ref: %tuple.type.2 = name_ref w, %w
// CHECK:STDOUT:   %.loc15_12: i32 = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_11: i32 = tuple_access %w.ref, element1
// CHECK:STDOUT:   return %.loc15_11
// CHECK:STDOUT: }
// CHECK:STDOUT:

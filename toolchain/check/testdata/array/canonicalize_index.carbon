// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/canonicalize_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/canonicalize_index.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

var a: [i32; Add(1, 2)] = (1, 2, 3);
let b: [i32; 3]* = &a;

// CHECK:STDOUT: --- canonicalize_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [template]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.3: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.6: i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %.4 = tuple_value (%.1, %.2, %.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type = fn_decl @Add [template = constants.%Add] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc11_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32.loc11_11 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32.loc11_11, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_32.loc11_19 [template = i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_32.loc11_19, %.loc11_19.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_27.1: type = value_of_initializer %int.make_type_32.loc11_27 [template = i32]
// CHECK:STDOUT:     %.loc11_27.2: type = converted %int.make_type_32.loc11_27, %.loc11_27.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, %Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %.loc13_18: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_21: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %int.sadd: init i32 = call %Add.ref(%.loc13_18, %.loc13_21) [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_9.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_9.2: type = converted %int.make_type_32.loc13, %.loc13_9.1 [template = i32]
// CHECK:STDOUT:   %.loc13_23: type = array_type %int.sadd, i32 [template = constants.%.4]
// CHECK:STDOUT:   %a.var: ref %.4 = var a
// CHECK:STDOUT:   %a: ref %.4 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_14: i32 = int_value 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_9.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_9.2: type = converted %int.make_type_32.loc14, %.loc14_9.1 [template = i32]
// CHECK:STDOUT:   %.loc14_15: type = array_type %.loc14_14, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_16: type = ptr_type %.4 [template = constants.%.5]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_28: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_31: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_34: i32 = int_value 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_35.1: %tuple.type = tuple_literal (%.loc13_28, %.loc13_31, %.loc13_34)
// CHECK:STDOUT:   %.loc13_35.2: i32 = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_35.3: ref i32 = array_index file.%a.var, %.loc13_35.2
// CHECK:STDOUT:   %.loc13_35.4: init i32 = initialize_from %.loc13_28 to %.loc13_35.3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_35.5: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_35.6: ref i32 = array_index file.%a.var, %.loc13_35.5
// CHECK:STDOUT:   %.loc13_35.7: init i32 = initialize_from %.loc13_31 to %.loc13_35.6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_35.8: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_35.9: ref i32 = array_index file.%a.var, %.loc13_35.8
// CHECK:STDOUT:   %.loc13_35.10: init i32 = initialize_from %.loc13_34 to %.loc13_35.9 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_35.11: init %.4 = array_init (%.loc13_35.4, %.loc13_35.7, %.loc13_35.10) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_36: init %.4 = converted %.loc13_35.1, %.loc13_35.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc13_36
// CHECK:STDOUT:   %a.ref: ref %.4 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc14: %.5 = addr_of %a.ref
// CHECK:STDOUT:   %b: %.5 = bind_name b, %.loc14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

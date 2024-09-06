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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [template]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [template]
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
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .a = %a.loc13
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type = fn_decl @Add [template = constants.%Add] {
// CHECK:STDOUT:     %int.make_type_32.loc11_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32.loc11_11 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32.loc11_11, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc11_8.1: i32 = param a
// CHECK:STDOUT:     @Add.%a: i32 = bind_name a, %a.loc11_8.1
// CHECK:STDOUT:     %int.make_type_32.loc11_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_32.loc11_19 [template = i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_32.loc11_19, %.loc11_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc11_16.1: i32 = param b
// CHECK:STDOUT:     @Add.%b: i32 = bind_name b, %b.loc11_16.1
// CHECK:STDOUT:     %int.make_type_32.loc11_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_27.1: type = value_of_initializer %int.make_type_32.loc11_27 [template = i32]
// CHECK:STDOUT:     %.loc11_27.2: type = converted %int.make_type_32.loc11_27, %.loc11_27.1 [template = i32]
// CHECK:STDOUT:     @Add.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, %Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %.loc13_18: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_21: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %int.sadd: init i32 = call %Add.ref(%.loc13_18, %.loc13_21) [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_9.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_9.2: type = converted %int.make_type_32.loc13, %.loc13_9.1 [template = i32]
// CHECK:STDOUT:   %.loc13_23: type = array_type %int.sadd, i32 [template = constants.%.5]
// CHECK:STDOUT:   %a.var: ref %.5 = var a
// CHECK:STDOUT:   %a.loc13: ref %.5 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_14: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_9.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_9.2: type = converted %int.make_type_32.loc14, %.loc14_9.1 [template = i32]
// CHECK:STDOUT:   %.loc14_15: type = array_type %.loc14_14, i32 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_16: type = ptr_type %.5 [template = constants.%.6]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_28: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_31: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_34: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_35.1: %.7 = tuple_literal (%.loc13_28, %.loc13_31, %.loc13_34)
// CHECK:STDOUT:   %.loc13_35.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_35.3: ref i32 = array_index file.%a.var, %.loc13_35.2
// CHECK:STDOUT:   %.loc13_35.4: init i32 = initialize_from %.loc13_28 to %.loc13_35.3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_35.5: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_35.6: ref i32 = array_index file.%a.var, %.loc13_35.5
// CHECK:STDOUT:   %.loc13_35.7: init i32 = initialize_from %.loc13_31 to %.loc13_35.6 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_35.8: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_35.9: ref i32 = array_index file.%a.var, %.loc13_35.8
// CHECK:STDOUT:   %.loc13_35.10: init i32 = initialize_from %.loc13_34 to %.loc13_35.9 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_35.11: init %.5 = array_init (%.loc13_35.4, %.loc13_35.7, %.loc13_35.10) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_36: init %.5 = converted %.loc13_35.1, %.loc13_35.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc13_36
// CHECK:STDOUT:   %a.ref: ref %.5 = name_ref a, file.%a.loc13
// CHECK:STDOUT:   %.loc14: %.6 = addr_of %a.ref
// CHECK:STDOUT:   %b: %.6 = bind_name b, %.loc14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

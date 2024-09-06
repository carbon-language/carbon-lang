// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/base.carbon

var a: [i32; 1] = (1,);
var b: [f64; 2] = (11.1, 2.2,);
var c: [(); 5] = ((), (), (), (), (),);

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array.1: %.3 = tuple_value (%.2) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.9: type = array_type %.8, f64 [template]
// CHECK:STDOUT:   %.10: type = ptr_type %.9 [template]
// CHECK:STDOUT:   %.11: f64 = float_literal 11.100000000000001 [template]
// CHECK:STDOUT:   %.12: f64 = float_literal 2.2000000000000002 [template]
// CHECK:STDOUT:   %.13: type = tuple_type (f64, f64) [template]
// CHECK:STDOUT:   %array.2: %.9 = tuple_value (%.11, %.12) [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.15: type = array_type %.14, %.1 [template]
// CHECK:STDOUT:   %.16: type = ptr_type %.15 [template]
// CHECK:STDOUT:   %.17: type = tuple_type (%.1, %.1, %.1, %.1, %.1) [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %.18: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.19: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %array.3: %.15 = tuple_value (%tuple, %tuple, %tuple, %tuple, %tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Float = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Float.type = import_ref Core//prelude/types, inst+32, loaded [template = constants.%Float]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_9.1: i32 = int_literal 64 [template = constants.%.7]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc12_9.1) [template = f64]
// CHECK:STDOUT:   %.loc12_14: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc12_9.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc12_9.3: type = converted %float.make_type, %.loc12_9.2 [template = f64]
// CHECK:STDOUT:   %.loc12_15: type = array_type %.loc12_14, f64 [template = constants.%.9]
// CHECK:STDOUT:   %b.var: ref %.9 = var b
// CHECK:STDOUT:   %b: ref %.9 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc13_10.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc13_13: i32 = int_literal 5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %.loc13_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_14: type = array_type %.loc13_13, %.1 [template = constants.%.15]
// CHECK:STDOUT:   %c.var: ref %.15 = var c
// CHECK:STDOUT:   %c: ref %.15 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_22.1: %.5 = tuple_literal (%.loc11_20)
// CHECK:STDOUT:   %.loc11_22.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_22.3: ref i32 = array_index file.%a.var, %.loc11_22.2
// CHECK:STDOUT:   %.loc11_22.4: init i32 = initialize_from %.loc11_20 to %.loc11_22.3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_22.5: init %.3 = array_init (%.loc11_22.4) to file.%a.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc11_23: init %.3 = converted %.loc11_22.1, %.loc11_22.5 [template = constants.%array.1]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_23
// CHECK:STDOUT:   %.loc12_20: f64 = float_literal 11.100000000000001 [template = constants.%.11]
// CHECK:STDOUT:   %.loc12_26: f64 = float_literal 2.2000000000000002 [template = constants.%.12]
// CHECK:STDOUT:   %.loc12_30.1: %.13 = tuple_literal (%.loc12_20, %.loc12_26)
// CHECK:STDOUT:   %.loc12_30.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_30.3: ref f64 = array_index file.%b.var, %.loc12_30.2
// CHECK:STDOUT:   %.loc12_30.4: init f64 = initialize_from %.loc12_20 to %.loc12_30.3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc12_30.5: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_30.6: ref f64 = array_index file.%b.var, %.loc12_30.5
// CHECK:STDOUT:   %.loc12_30.7: init f64 = initialize_from %.loc12_26 to %.loc12_30.6 [template = constants.%.12]
// CHECK:STDOUT:   %.loc12_30.8: init %.9 = array_init (%.loc12_30.4, %.loc12_30.7) to file.%b.var [template = constants.%array.2]
// CHECK:STDOUT:   %.loc12_31: init %.9 = converted %.loc12_30.1, %.loc12_30.8 [template = constants.%array.2]
// CHECK:STDOUT:   assign file.%b.var, %.loc12_31
// CHECK:STDOUT:   %.loc13_20.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc13_24.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc13_28.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc13_32.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc13_36.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc13_38.1: %.17 = tuple_literal (%.loc13_20.1, %.loc13_24.1, %.loc13_28.1, %.loc13_32.1, %.loc13_36.1)
// CHECK:STDOUT:   %.loc13_38.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_38.3: ref %.1 = array_index file.%c.var, %.loc13_38.2
// CHECK:STDOUT:   %.loc13_20.2: init %.1 = tuple_init () to %.loc13_38.3 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_38.4: init %.1 = converted %.loc13_20.1, %.loc13_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_38.5: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_38.6: ref %.1 = array_index file.%c.var, %.loc13_38.5
// CHECK:STDOUT:   %.loc13_24.2: init %.1 = tuple_init () to %.loc13_38.6 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_38.7: init %.1 = converted %.loc13_24.1, %.loc13_24.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_38.8: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_38.9: ref %.1 = array_index file.%c.var, %.loc13_38.8
// CHECK:STDOUT:   %.loc13_28.2: init %.1 = tuple_init () to %.loc13_38.9 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_38.10: init %.1 = converted %.loc13_28.1, %.loc13_28.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_38.11: i32 = int_literal 3 [template = constants.%.18]
// CHECK:STDOUT:   %.loc13_38.12: ref %.1 = array_index file.%c.var, %.loc13_38.11
// CHECK:STDOUT:   %.loc13_32.2: init %.1 = tuple_init () to %.loc13_38.12 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_38.13: init %.1 = converted %.loc13_32.1, %.loc13_32.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_38.14: i32 = int_literal 4 [template = constants.%.19]
// CHECK:STDOUT:   %.loc13_38.15: ref %.1 = array_index file.%c.var, %.loc13_38.14
// CHECK:STDOUT:   %.loc13_36.2: init %.1 = tuple_init () to %.loc13_38.15 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_38.16: init %.1 = converted %.loc13_36.1, %.loc13_36.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_38.17: init %.15 = array_init (%.loc13_38.4, %.loc13_38.7, %.loc13_38.10, %.loc13_38.13, %.loc13_38.16) to file.%c.var [template = constants.%array.3]
// CHECK:STDOUT:   %.loc13_39: init %.15 = converted %.loc13_38.1, %.loc13_38.17 [template = constants.%array.3]
// CHECK:STDOUT:   assign file.%c.var, %.loc13_39
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

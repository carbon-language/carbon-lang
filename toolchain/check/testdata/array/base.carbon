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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.31: <specific function> = specific_function %.30, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.32: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %array.1: %.3 = tuple_value (%.32) [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.35: type = array_type %.34, f64 [template]
// CHECK:STDOUT:   %.37: f64 = float_literal 11.100000000000001 [template]
// CHECK:STDOUT:   %.38: f64 = float_literal 2.2000000000000002 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (f64, f64) [template]
// CHECK:STDOUT:   %array.2: %.35 = tuple_value (%.37, %.38) [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.40: type = array_type %.39, %empty_tuple.type [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %.42: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.43: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %array.3: %.40 = tuple_value (%empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Float = %import_ref.38
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
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc11_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc11_9.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_14: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_9.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.3: type = converted %int.make_type_signed, %.loc11_9.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_9.1: Core.IntLiteral = int_value 64 [template = constants.%.33]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc12_9.1) [template = f64]
// CHECK:STDOUT:   %.loc12_14: Core.IntLiteral = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_9.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc12_9.3: type = converted %float.make_type, %.loc12_9.2 [template = f64]
// CHECK:STDOUT:   %.loc12_15: type = array_type %.loc12_14, f64 [template = constants.%.35]
// CHECK:STDOUT:   %b.var: ref %.35 = var b
// CHECK:STDOUT:   %b: ref %.35 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc13_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_13: Core.IntLiteral = int_value 5 [template = constants.%.39]
// CHECK:STDOUT:   %.loc13_10.2: type = converted %.loc13_10.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc13_14: type = array_type %.loc13_13, %empty_tuple.type [template = constants.%.40]
// CHECK:STDOUT:   %c.var: ref %.40 = var c
// CHECK:STDOUT:   %c: ref %.40 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_22.1: %tuple.type.1 = tuple_literal (%.loc11_20)
// CHECK:STDOUT:   %.loc11_22.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_22.3: <bound method> = bound_method %.loc11_20, %.loc11_22.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_22.4: <specific function> = specific_function %.loc11_22.3, @Convert.2(constants.%.1) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc11_22.4(%.loc11_20) [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_22.5: init %i32 = converted %.loc11_20, %int.convert_checked [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_22.6: Core.IntLiteral = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_22.7: ref %i32 = array_index file.%a.var, %.loc11_22.6
// CHECK:STDOUT:   %.loc11_22.8: init %i32 = initialize_from %.loc11_22.5 to %.loc11_22.7 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_22.9: init %.3 = array_init (%.loc11_22.8) to file.%a.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc11_23: init %.3 = converted %.loc11_22.1, %.loc11_22.9 [template = constants.%array.1]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_23
// CHECK:STDOUT:   %.loc12_20: f64 = float_literal 11.100000000000001 [template = constants.%.37]
// CHECK:STDOUT:   %.loc12_26: f64 = float_literal 2.2000000000000002 [template = constants.%.38]
// CHECK:STDOUT:   %.loc12_30.1: %tuple.type.2 = tuple_literal (%.loc12_20, %.loc12_26)
// CHECK:STDOUT:   %.loc12_30.2: Core.IntLiteral = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_30.3: ref f64 = array_index file.%b.var, %.loc12_30.2
// CHECK:STDOUT:   %.loc12_30.4: init f64 = initialize_from %.loc12_20 to %.loc12_30.3 [template = constants.%.37]
// CHECK:STDOUT:   %.loc12_30.5: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_30.6: ref f64 = array_index file.%b.var, %.loc12_30.5
// CHECK:STDOUT:   %.loc12_30.7: init f64 = initialize_from %.loc12_26 to %.loc12_30.6 [template = constants.%.38]
// CHECK:STDOUT:   %.loc12_30.8: init %.35 = array_init (%.loc12_30.4, %.loc12_30.7) to file.%b.var [template = constants.%array.2]
// CHECK:STDOUT:   %.loc12_31: init %.35 = converted %.loc12_30.1, %.loc12_30.8 [template = constants.%array.2]
// CHECK:STDOUT:   assign file.%b.var, %.loc12_31
// CHECK:STDOUT:   %.loc13_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_24.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_28.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_36.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_38.1: %tuple.type.3 = tuple_literal (%.loc13_20.1, %.loc13_24.1, %.loc13_28.1, %.loc13_32.1, %.loc13_36.1)
// CHECK:STDOUT:   %.loc13_38.2: Core.IntLiteral = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_38.3: ref %empty_tuple.type = array_index file.%c.var, %.loc13_38.2
// CHECK:STDOUT:   %.loc13_20.2: init %empty_tuple.type = tuple_init () to %.loc13_38.3 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.4: init %empty_tuple.type = converted %.loc13_20.1, %.loc13_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.5: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_38.6: ref %empty_tuple.type = array_index file.%c.var, %.loc13_38.5
// CHECK:STDOUT:   %.loc13_24.2: init %empty_tuple.type = tuple_init () to %.loc13_38.6 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.7: init %empty_tuple.type = converted %.loc13_24.1, %.loc13_24.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.8: Core.IntLiteral = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_38.9: ref %empty_tuple.type = array_index file.%c.var, %.loc13_38.8
// CHECK:STDOUT:   %.loc13_28.2: init %empty_tuple.type = tuple_init () to %.loc13_38.9 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.10: init %empty_tuple.type = converted %.loc13_28.1, %.loc13_28.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.11: Core.IntLiteral = int_value 3 [template = constants.%.42]
// CHECK:STDOUT:   %.loc13_38.12: ref %empty_tuple.type = array_index file.%c.var, %.loc13_38.11
// CHECK:STDOUT:   %.loc13_32.2: init %empty_tuple.type = tuple_init () to %.loc13_38.12 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.13: init %empty_tuple.type = converted %.loc13_32.1, %.loc13_32.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.14: Core.IntLiteral = int_value 4 [template = constants.%.43]
// CHECK:STDOUT:   %.loc13_38.15: ref %empty_tuple.type = array_index file.%c.var, %.loc13_38.14
// CHECK:STDOUT:   %.loc13_36.2: init %empty_tuple.type = tuple_init () to %.loc13_38.15 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.16: init %empty_tuple.type = converted %.loc13_36.1, %.loc13_36.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.17: init %.40 = array_init (%.loc13_38.4, %.loc13_38.7, %.loc13_38.10, %.loc13_38.13, %.loc13_38.16) to file.%c.var [template = constants.%array.3]
// CHECK:STDOUT:   %.loc13_39: init %.40 = converted %.loc13_38.1, %.loc13_38.17 [template = constants.%array.3]
// CHECK:STDOUT:   assign file.%c.var, %.loc13_39
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_1.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array_type.b: type = array_type %int_1.5, %i32 [template]
// CHECK:STDOUT:   %tuple.type.9: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.cd: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.46: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.46 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5, %Convert.9 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %array.0: %array_type.b = tuple_value (%int_1.c) [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array_type.ce: type = array_type %int_2, f64 [template]
// CHECK:STDOUT:   %float.6: f64 = float_literal 11.100000000000001 [template]
// CHECK:STDOUT:   %float.9: f64 = float_literal 2.2000000000000002 [template]
// CHECK:STDOUT:   %tuple.type.b: type = tuple_type (f64, f64) [template]
// CHECK:STDOUT:   %array.6: %array_type.ce = tuple_value (%float.6, %float.9) [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %array_type.c1: type = array_type %int_5, %empty_tuple.type [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %array.1: %array_type.c1 = tuple_value (%empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a6
// CHECK:STDOUT:     .Float = %import_ref.20bd
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
// CHECK:STDOUT:   %a.var: ref %array_type.b = var a
// CHECK:STDOUT:   %a: ref %array_type.b = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %array_type.ce = var b
// CHECK:STDOUT:   %b: ref %array_type.ce = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref %array_type.c1 = var c
// CHECK:STDOUT:   %c: ref %array_type.c1 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %.loc11_22.1: %tuple.type.9 = tuple_literal (%int_1.loc11)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.cd = impl_witness_access constants.%impl_witness.5, element0 [template = constants.%Convert.9]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.loc11, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1.loc11) [template = constants.%int_1.c]
// CHECK:STDOUT:   %.loc11_22.2: init %i32 = converted %int_1.loc11, %int.convert_checked [template = constants.%int_1.c]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_22.3: ref %i32 = array_index file.%a.var, %int_0.loc11
// CHECK:STDOUT:   %.loc11_22.4: init %i32 = initialize_from %.loc11_22.2 to %.loc11_22.3 [template = constants.%int_1.c]
// CHECK:STDOUT:   %.loc11_22.5: init %array_type.b = array_init (%.loc11_22.4) to file.%a.var [template = constants.%array.0]
// CHECK:STDOUT:   %.loc11_23: init %array_type.b = converted %.loc11_22.1, %.loc11_22.5 [template = constants.%array.0]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_23
// CHECK:STDOUT:   %float.loc12_20: f64 = float_literal 11.100000000000001 [template = constants.%float.6]
// CHECK:STDOUT:   %float.loc12_26: f64 = float_literal 2.2000000000000002 [template = constants.%float.9]
// CHECK:STDOUT:   %.loc12_30.1: %tuple.type.b = tuple_literal (%float.loc12_20, %float.loc12_26)
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc12_30.2: ref f64 = array_index file.%b.var, %int_0.loc12
// CHECK:STDOUT:   %.loc12_30.3: init f64 = initialize_from %float.loc12_20 to %.loc12_30.2 [template = constants.%float.6]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %.loc12_30.4: ref f64 = array_index file.%b.var, %int_1.loc12
// CHECK:STDOUT:   %.loc12_30.5: init f64 = initialize_from %float.loc12_26 to %.loc12_30.4 [template = constants.%float.9]
// CHECK:STDOUT:   %.loc12_30.6: init %array_type.ce = array_init (%.loc12_30.3, %.loc12_30.5) to file.%b.var [template = constants.%array.6]
// CHECK:STDOUT:   %.loc12_31: init %array_type.ce = converted %.loc12_30.1, %.loc12_30.6 [template = constants.%array.6]
// CHECK:STDOUT:   assign file.%b.var, %.loc12_31
// CHECK:STDOUT:   %.loc13_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_24.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_28.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_36.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_38.1: %tuple.type.5 = tuple_literal (%.loc13_20.1, %.loc13_24.1, %.loc13_28.1, %.loc13_32.1, %.loc13_36.1)
// CHECK:STDOUT:   %int_0.loc13: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc13_38.2: ref %empty_tuple.type = array_index file.%c.var, %int_0.loc13
// CHECK:STDOUT:   %.loc13_20.2: init %empty_tuple.type = tuple_init () to %.loc13_38.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.3: init %empty_tuple.type = converted %.loc13_20.1, %.loc13_20.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %.loc13_38.4: ref %empty_tuple.type = array_index file.%c.var, %int_1.loc13
// CHECK:STDOUT:   %.loc13_24.2: init %empty_tuple.type = tuple_init () to %.loc13_38.4 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.5: init %empty_tuple.type = converted %.loc13_24.1, %.loc13_24.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc13_38.6: ref %empty_tuple.type = array_index file.%c.var, %int_2
// CHECK:STDOUT:   %.loc13_28.2: init %empty_tuple.type = tuple_init () to %.loc13_38.6 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.7: init %empty_tuple.type = converted %.loc13_28.1, %.loc13_28.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc13_38.8: ref %empty_tuple.type = array_index file.%c.var, %int_3
// CHECK:STDOUT:   %.loc13_32.2: init %empty_tuple.type = tuple_init () to %.loc13_38.8 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.9: init %empty_tuple.type = converted %.loc13_32.1, %.loc13_32.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc13_38.10: ref %empty_tuple.type = array_index file.%c.var, %int_4
// CHECK:STDOUT:   %.loc13_36.2: init %empty_tuple.type = tuple_init () to %.loc13_38.10 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.11: init %empty_tuple.type = converted %.loc13_36.1, %.loc13_36.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.12: init %array_type.c1 = array_init (%.loc13_38.3, %.loc13_38.5, %.loc13_38.7, %.loc13_38.9, %.loc13_38.11) to file.%c.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc13_39: init %array_type.c1 = converted %.loc13_38.1, %.loc13_38.12 [template = constants.%array.1]
// CHECK:STDOUT:   assign file.%c.var, %.loc13_39
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

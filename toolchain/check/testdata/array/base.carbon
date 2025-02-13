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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type.0cb: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %array.237: %array_type.0cb = tuple_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type.ce7: type = array_type %int_2, f64 [concrete]
// CHECK:STDOUT:   %float.6e4: f64 = float_literal 11.100000000000001 [concrete]
// CHECK:STDOUT:   %float.9f7: f64 = float_literal 2.2000000000000002 [concrete]
// CHECK:STDOUT:   %tuple.type.bdb: type = tuple_type (f64, f64) [concrete]
// CHECK:STDOUT:   %array.6a2: %array_type.ce7 = tuple_value (%float.6e4, %float.9f7) [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %array_type.c13: type = array_type %int_5, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type.5b2: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array.1cb: %array_type.c13 = tuple_value (%empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.0cb = binding_pattern a
// CHECK:STDOUT:     %.loc11_1: %array_type.0cb = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.0cb = var a
// CHECK:STDOUT:   %.loc11_15: type = splice_block %array_type.loc11 [concrete = constants.%array_type.0cb] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type.loc11: type = array_type %int_1, %i32 [concrete = constants.%array_type.0cb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.0cb = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %array_type.ce7 = binding_pattern b
// CHECK:STDOUT:     %.loc12_1: %array_type.ce7 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %array_type.ce7 = var b
// CHECK:STDOUT:   %.loc12_15: type = splice_block %array_type.loc12 [concrete = constants.%array_type.ce7] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %.loc12_9.1: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:     %.loc12_9.2: type = converted %float.make_type, %.loc12_9.1 [concrete = f64]
// CHECK:STDOUT:     %array_type.loc12: type = array_type %int_2, f64 [concrete = constants.%array_type.ce7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %array_type.ce7 = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %array_type.c13 = binding_pattern c
// CHECK:STDOUT:     %.loc13_1: %array_type.c13 = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %array_type.c13 = var c
// CHECK:STDOUT:   %.loc13_14: type = splice_block %array_type.loc13 [concrete = constants.%array_type.c13] {
// CHECK:STDOUT:     %.loc13_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:     %.loc13_10.2: type = converted %.loc13_10.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %array_type.loc13: type = array_type %int_5, %empty_tuple.type [concrete = constants.%array_type.c13]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %array_type.c13 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_22.1: %tuple.type.985 = tuple_literal (%int_1.loc11)
// CHECK:STDOUT:   %impl.elem0: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.loc11, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %bound_method, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %specific_fn(%int_1.loc11) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_22.2: init %i32 = converted %int_1.loc11, %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_22.3: ref %i32 = array_index file.%a.var, %int_0.loc11
// CHECK:STDOUT:   %.loc11_22.4: init %i32 = initialize_from %.loc11_22.2 to %.loc11_22.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_22.5: init %array_type.0cb = array_init (%.loc11_22.4) to file.%a.var [concrete = constants.%array.237]
// CHECK:STDOUT:   %.loc11_1: init %array_type.0cb = converted %.loc11_22.1, %.loc11_22.5 [concrete = constants.%array.237]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_1
// CHECK:STDOUT:   %float.loc12_20: f64 = float_literal 11.100000000000001 [concrete = constants.%float.6e4]
// CHECK:STDOUT:   %float.loc12_26: f64 = float_literal 2.2000000000000002 [concrete = constants.%float.9f7]
// CHECK:STDOUT:   %.loc12_30.1: %tuple.type.bdb = tuple_literal (%float.loc12_20, %float.loc12_26)
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc12_30.2: ref f64 = array_index file.%b.var, %int_0.loc12
// CHECK:STDOUT:   %.loc12_30.3: init f64 = initialize_from %float.loc12_20 to %.loc12_30.2 [concrete = constants.%float.6e4]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc12_30.4: ref f64 = array_index file.%b.var, %int_1.loc12
// CHECK:STDOUT:   %.loc12_30.5: init f64 = initialize_from %float.loc12_26 to %.loc12_30.4 [concrete = constants.%float.9f7]
// CHECK:STDOUT:   %.loc12_30.6: init %array_type.ce7 = array_init (%.loc12_30.3, %.loc12_30.5) to file.%b.var [concrete = constants.%array.6a2]
// CHECK:STDOUT:   %.loc12_1: init %array_type.ce7 = converted %.loc12_30.1, %.loc12_30.6 [concrete = constants.%array.6a2]
// CHECK:STDOUT:   assign file.%b.var, %.loc12_1
// CHECK:STDOUT:   %.loc13_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_24.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_28.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_36.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc13_38.1: %tuple.type.5b2 = tuple_literal (%.loc13_20.1, %.loc13_24.1, %.loc13_28.1, %.loc13_32.1, %.loc13_36.1)
// CHECK:STDOUT:   %int_0.loc13: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc13_38.2: ref %empty_tuple.type = array_index file.%c.var, %int_0.loc13
// CHECK:STDOUT:   %.loc13_20.2: init %empty_tuple.type = tuple_init () to %.loc13_38.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.3: init %empty_tuple.type = converted %.loc13_20.1, %.loc13_20.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc13_38.4: ref %empty_tuple.type = array_index file.%c.var, %int_1.loc13
// CHECK:STDOUT:   %.loc13_24.2: init %empty_tuple.type = tuple_init () to %.loc13_38.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.5: init %empty_tuple.type = converted %.loc13_24.1, %.loc13_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc13_38.6: ref %empty_tuple.type = array_index file.%c.var, %int_2
// CHECK:STDOUT:   %.loc13_28.2: init %empty_tuple.type = tuple_init () to %.loc13_38.6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.7: init %empty_tuple.type = converted %.loc13_28.1, %.loc13_28.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc13_38.8: ref %empty_tuple.type = array_index file.%c.var, %int_3
// CHECK:STDOUT:   %.loc13_32.2: init %empty_tuple.type = tuple_init () to %.loc13_38.8 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.9: init %empty_tuple.type = converted %.loc13_32.1, %.loc13_32.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc13_38.10: ref %empty_tuple.type = array_index file.%c.var, %int_4
// CHECK:STDOUT:   %.loc13_36.2: init %empty_tuple.type = tuple_init () to %.loc13_38.10 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.11: init %empty_tuple.type = converted %.loc13_36.1, %.loc13_36.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_38.12: init %array_type.c13 = array_init (%.loc13_38.3, %.loc13_38.5, %.loc13_38.7, %.loc13_38.9, %.loc13_38.11) to file.%c.var [concrete = constants.%array.1cb]
// CHECK:STDOUT:   %.loc13_1: init %array_type.c13 = converted %.loc13_38.1, %.loc13_38.12 [concrete = constants.%array.1cb]
// CHECK:STDOUT:   assign file.%c.var, %.loc13_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

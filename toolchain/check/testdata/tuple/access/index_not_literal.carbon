// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/index_not_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/index_not_literal.carbon

var a: (bool, i32) = (true, 34);
var b: i32 = a.({.index = 1}.index);
var c: bool = a.(0 as i32);
var d: i32 = a.({.index = 1 as i32}.index);

// CHECK:STDOUT: --- index_not_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.261: type = tuple_type (bool, %i32) [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_34.3f9: Core.IntLiteral = int_value 34 [concrete]
// CHECK:STDOUT:   %tuple.type.3c2: type = tuple_type (bool, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2b8: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.2b8 [concrete]
// CHECK:STDOUT:   %Convert.bound.82c: <bound method> = bound_method %int_34.3f9, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b6f: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.34d: <bound method> = bound_method %int_34.3f9, %Convert.specific_fn.b6f [concrete]
// CHECK:STDOUT:   %int_34.980: %i32 = int_value 34 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.261 = tuple_value (%true, %int_34.980) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.index.b1b: type = struct_type {.index: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.972: %struct_type.index.b1b = struct_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.4, @As(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.882: <witness> = impl_witness (imports.%Core.import_ref.78a), @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%impl_witness.882) [concrete]
// CHECK:STDOUT:   %.249: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.129: <bound method> = bound_method %int_0.5c6, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.aad: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.46b: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn.aad [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ffe: %ImplicitAs.type.2fd = facet_value %i32, (%impl_witness.023) [concrete]
// CHECK:STDOUT:   %.4d8: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.ffe [concrete]
// CHECK:STDOUT:   %Convert.bound.0fd: <bound method> = bound_method %int_0.6a9, %Convert.960 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.8a8: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.4b5: <bound method> = bound_method %int_0.6a9, %Convert.specific_fn.8a8 [concrete]
// CHECK:STDOUT:   %Convert.bound.c1b: <bound method> = bound_method %int_1.5b8, %Convert.197 [concrete]
// CHECK:STDOUT:   %bound_method.082: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn.aad [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.index.6ea: type = struct_type {.index: %i32} [concrete]
// CHECK:STDOUT:   %struct.63a: %struct_type.index.6ea = struct_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %Convert.bound.faf: <bound method> = bound_method %int_1.5d2, %Convert.960 [concrete]
// CHECK:STDOUT:   %bound_method.711: <bound method> = bound_method %int_1.5d2, %Convert.specific_fn.8a8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .As = %Core.As
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
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %tuple.type.261 = binding_pattern a
// CHECK:STDOUT:     %.loc11_1: %tuple.type.261 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %tuple.type.261 = var a
// CHECK:STDOUT:   %.loc11_18.1: type = splice_block %.loc11_18.5 [concrete = constants.%tuple.type.261] {
// CHECK:STDOUT:     %bool.make_type.loc11: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_18.2: %tuple.type.24b = tuple_literal (%bool.make_type.loc11, %i32.loc11)
// CHECK:STDOUT:     %.loc11_18.3: type = value_of_initializer %bool.make_type.loc11 [concrete = bool]
// CHECK:STDOUT:     %.loc11_18.4: type = converted %bool.make_type.loc11, %.loc11_18.3 [concrete = bool]
// CHECK:STDOUT:     %.loc11_18.5: type = converted %.loc11_18.2, constants.%tuple.type.261 [concrete = constants.%tuple.type.261]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %tuple.type.261 = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %.loc12_1: %i32 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %.loc12_8: type = splice_block %i32.loc12 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: bool = binding_pattern c
// CHECK:STDOUT:     %.loc13_1: bool = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref bool = var c
// CHECK:STDOUT:   %.loc13_8.1: type = splice_block %.loc13_8.3 [concrete = bool] {
// CHECK:STDOUT:     %bool.make_type.loc13: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc13_8.2: type = value_of_initializer %bool.make_type.loc13 [concrete = bool]
// CHECK:STDOUT:     %.loc13_8.3: type = converted %bool.make_type.loc13, %.loc13_8.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref bool = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %i32 = binding_pattern d
// CHECK:STDOUT:     %.loc14_1: %i32 = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %i32 = var d
// CHECK:STDOUT:   %.loc14_8: type = splice_block %i32.loc14 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete = constants.%int_34.3f9]
// CHECK:STDOUT:   %.loc11_31.1: %tuple.type.3c2 = tuple_literal (%true, %int_34)
// CHECK:STDOUT:   %tuple.elem0.loc11: ref bool = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_31.2: init bool = initialize_from %true to %tuple.elem0.loc11 [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc11: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_31.1: <bound method> = bound_method %int_34, %impl.elem0.loc11 [concrete = constants.%Convert.bound.82c]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:   %bound_method.loc11_31.2: <bound method> = bound_method %int_34, %specific_fn.loc11 [concrete = constants.%bound_method.34d]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %bound_method.loc11_31.2(%int_34) [concrete = constants.%int_34.980]
// CHECK:STDOUT:   %.loc11_31.3: init %i32 = converted %int_34, %int.convert_checked.loc11 [concrete = constants.%int_34.980]
// CHECK:STDOUT:   %tuple.elem1.loc11: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_31.4: init %i32 = initialize_from %.loc11_31.3 to %tuple.elem1.loc11 [concrete = constants.%int_34.980]
// CHECK:STDOUT:   %.loc11_31.5: init %tuple.type.261 = tuple_init (%.loc11_31.2, %.loc11_31.4) to file.%a.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc11_1: init %tuple.type.261 = converted %.loc11_31.1, %.loc11_31.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_1
// CHECK:STDOUT:   %a.ref.loc12: ref %tuple.type.261 = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc12_28.1: %struct_type.index.b1b = struct_literal (%int_1.loc12)
// CHECK:STDOUT:   %struct.loc12: %struct_type.index.b1b = struct_value (%int_1.loc12) [concrete = constants.%struct.972]
// CHECK:STDOUT:   %.loc12_28.2: %struct_type.index.b1b = converted %.loc12_28.1, %struct.loc12 [concrete = constants.%struct.972]
// CHECK:STDOUT:   %.loc12_29: Core.IntLiteral = struct_access %.loc12_28.2, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %a.ref.loc12, element1
// CHECK:STDOUT:   %.loc12_15: %i32 = bind_value %tuple.elem1.loc12
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15
// CHECK:STDOUT:   %a.ref.loc13: ref %tuple.type.261 = name_ref a, file.%a
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc13_20.1: %.249 = impl_witness_access constants.%impl_witness.882, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc13_20.1: <bound method> = bound_method %int_0, %impl.elem0.loc13_20.1 [concrete = constants.%Convert.bound.129]
// CHECK:STDOUT:   %specific_fn.loc13_20.1: <specific function> = specific_function %impl.elem0.loc13_20.1, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn.aad]
// CHECK:STDOUT:   %bound_method.loc13_20.2: <bound method> = bound_method %int_0, %specific_fn.loc13_20.1 [concrete = constants.%bound_method.46b]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.1: init %i32 = call %bound_method.loc13_20.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc13_20.1: %i32 = value_of_initializer %int.convert_checked.loc13_20.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc13_20.2: %i32 = converted %int_0, %.loc13_20.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc13_20.2: %.4d8 = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:   %bound_method.loc13_20.3: <bound method> = bound_method %.loc13_20.2, %impl.elem0.loc13_20.2 [concrete = constants.%Convert.bound.0fd]
// CHECK:STDOUT:   %specific_fn.loc13_20.2: <specific function> = specific_function %impl.elem0.loc13_20.2, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:   %bound_method.loc13_20.4: <bound method> = bound_method %.loc13_20.2, %specific_fn.loc13_20.2 [concrete = constants.%bound_method.4b5]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.2: init Core.IntLiteral = call %bound_method.loc13_20.4(%.loc13_20.2) [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc13_20.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc13_20.2 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc13_20.4: Core.IntLiteral = converted %.loc13_20.2, %.loc13_20.3 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %tuple.elem0.loc13: ref bool = tuple_access %a.ref.loc13, element0
// CHECK:STDOUT:   %.loc13_16: bool = bind_value %tuple.elem0.loc13
// CHECK:STDOUT:   assign file.%c.var, %.loc13_16
// CHECK:STDOUT:   %a.ref.loc14: ref %tuple.type.261 = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14_29: %.249 = impl_witness_access constants.%impl_witness.882, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc14_29.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_29 [concrete = constants.%Convert.bound.c1b]
// CHECK:STDOUT:   %specific_fn.loc14_29: <specific function> = specific_function %impl.elem0.loc14_29, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn.aad]
// CHECK:STDOUT:   %bound_method.loc14_29.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14_29 [concrete = constants.%bound_method.082]
// CHECK:STDOUT:   %int.convert_checked.loc14_29: init %i32 = call %bound_method.loc14_29.2(%int_1.loc14) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_29.1: %i32 = value_of_initializer %int.convert_checked.loc14_29 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_29.2: %i32 = converted %int_1.loc14, %.loc14_29.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.index.6ea = struct_literal (%.loc14_29.2)
// CHECK:STDOUT:   %struct.loc14: %struct_type.index.6ea = struct_value (%.loc14_29.2) [concrete = constants.%struct.63a]
// CHECK:STDOUT:   %.loc14_35.2: %struct_type.index.6ea = converted %.loc14_35.1, %struct.loc14 [concrete = constants.%struct.63a]
// CHECK:STDOUT:   %.loc14_36.1: %i32 = struct_access %.loc14_35.2, element0 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc14_36: %.4d8 = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:   %bound_method.loc14_36.1: <bound method> = bound_method %.loc14_36.1, %impl.elem0.loc14_36 [concrete = constants.%Convert.bound.faf]
// CHECK:STDOUT:   %specific_fn.loc14_36: <specific function> = specific_function %impl.elem0.loc14_36, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:   %bound_method.loc14_36.2: <bound method> = bound_method %.loc14_36.1, %specific_fn.loc14_36 [concrete = constants.%bound_method.711]
// CHECK:STDOUT:   %int.convert_checked.loc14_36: init Core.IntLiteral = call %bound_method.loc14_36.2(%.loc14_36.1) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc14_36.2: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_36 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc14_36.3: Core.IntLiteral = converted %.loc14_36.1, %.loc14_36.2 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access %a.ref.loc14, element1
// CHECK:STDOUT:   %.loc14_15: %i32 = bind_value %tuple.elem1.loc14
// CHECK:STDOUT:   assign file.%d.var, %.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

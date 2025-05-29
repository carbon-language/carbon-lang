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
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.261: type = tuple_type (bool, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.f98: type = pattern_type %tuple.type.261 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_34.3f9: Core.IntLiteral = int_value 34 [concrete]
// CHECK:STDOUT:   %tuple.type.3c2: type = tuple_type (bool, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access file.%a.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%a.var, element1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.708: type = fn_type @Convert.3, @impl.971(%From) [symbolic]
// CHECK:STDOUT:   %Convert.c68: %Convert.type.708 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.5, @impl.686(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound.82c: <bound method> = bound_method %int_34.3f9, %Convert.956 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
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
// CHECK:STDOUT:   %As.impl_witness.6b4: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%As.impl_witness.6b4) [concrete]
// CHECK:STDOUT:   %.982: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.129: <bound method> = bound_method %int_0.5c6, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.aad: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.46b: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn.aad [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a11: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1d9, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f49: %ImplicitAs.type.2fd = facet_value %i32, (%ImplicitAs.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.f49 [concrete]
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
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.85c: @impl.971.%Convert.type (%Convert.type.708) = import_ref Core//prelude/types/int, loc23_44, loaded [symbolic = @impl.971.%Convert (constants.%Convert.c68)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1d9 = impl_witness_table (%Core.import_ref.85c), @impl.971 [concrete]
// CHECK:STDOUT:   %Core.import_ref.78a: @impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude/types/int, loc28_39, loaded [symbolic = @impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @impl.686 [concrete]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.f98 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.f98 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %tuple.type.261 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc11_18.1: type = splice_block %.loc11_18.5 [concrete = constants.%tuple.type.261] {
// CHECK:STDOUT:     %bool.make_type.loc11: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_18.2: %tuple.type.24b = tuple_literal (%bool.make_type.loc11, %i32.loc11)
// CHECK:STDOUT:     %.loc11_18.3: type = value_of_initializer %bool.make_type.loc11 [concrete = bool]
// CHECK:STDOUT:     %.loc11_18.4: type = converted %bool.make_type.loc11, %.loc11_18.3 [concrete = bool]
// CHECK:STDOUT:     %.loc11_18.5: type = converted %.loc11_18.2, constants.%tuple.type.261 [concrete = constants.%tuple.type.261]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %tuple.type.261 = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc12: type = splice_block %i32.loc12 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.831 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref bool = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc13_8.1: type = splice_block %.loc13_8.3 [concrete = bool] {
// CHECK:STDOUT:     %bool.make_type.loc13: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc13_8.2: type = value_of_initializer %bool.make_type.loc13 [concrete = bool]
// CHECK:STDOUT:     %.loc13_8.3: type = converted %bool.make_type.loc13, %.loc13_8.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref bool = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.7ce = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %i32 = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc14: type = splice_block %i32.loc14 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete = constants.%int_34.3f9]
// CHECK:STDOUT:   %.loc11_31.1: %tuple.type.3c2 = tuple_literal (%true, %int_34)
// CHECK:STDOUT:   %tuple.elem0.loc11: ref bool = tuple_access file.%a.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc11_31.2: init bool = initialize_from %true to %tuple.elem0.loc11 [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc11: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_31.1: <bound method> = bound_method %int_34, %impl.elem0.loc11 [concrete = constants.%Convert.bound.82c]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:   %bound_method.loc11_31.2: <bound method> = bound_method %int_34, %specific_fn.loc11 [concrete = constants.%bound_method.34d]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %bound_method.loc11_31.2(%int_34) [concrete = constants.%int_34.980]
// CHECK:STDOUT:   %.loc11_31.3: init %i32 = converted %int_34, %int.convert_checked.loc11 [concrete = constants.%int_34.980]
// CHECK:STDOUT:   %tuple.elem1.loc11: ref %i32 = tuple_access file.%a.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc11_31.4: init %i32 = initialize_from %.loc11_31.3 to %tuple.elem1.loc11 [concrete = constants.%int_34.980]
// CHECK:STDOUT:   %.loc11_31.5: init %tuple.type.261 = tuple_init (%.loc11_31.2, %.loc11_31.4) to file.%a.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc11_1: init %tuple.type.261 = converted %.loc11_31.1, %.loc11_31.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_1
// CHECK:STDOUT:   %a.ref.loc12: ref %tuple.type.261 = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc12_28.1: %struct_type.index.b1b = struct_literal (%int_1.loc12)
// CHECK:STDOUT:   %struct.loc12: %struct_type.index.b1b = struct_value (%int_1.loc12) [concrete = constants.%struct.972]
// CHECK:STDOUT:   %.loc12_28.2: %struct_type.index.b1b = converted %.loc12_28.1, %struct.loc12 [concrete = constants.%struct.972]
// CHECK:STDOUT:   %.loc12_29: Core.IntLiteral = struct_access %.loc12_28.2, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %a.ref.loc12, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc12_15: %i32 = bind_value %tuple.elem1.loc12
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15
// CHECK:STDOUT:   %a.ref.loc13: ref %tuple.type.261 = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc13_20.1: %.982 = impl_witness_access constants.%As.impl_witness.6b4, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc13_20.1: <bound method> = bound_method %int_0, %impl.elem0.loc13_20.1 [concrete = constants.%Convert.bound.129]
// CHECK:STDOUT:   %specific_fn.loc13_20.1: <specific function> = specific_function %impl.elem0.loc13_20.1, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn.aad]
// CHECK:STDOUT:   %bound_method.loc13_20.2: <bound method> = bound_method %int_0, %specific_fn.loc13_20.1 [concrete = constants.%bound_method.46b]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.1: init %i32 = call %bound_method.loc13_20.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc13_20.1: %i32 = value_of_initializer %int.convert_checked.loc13_20.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc13_20.2: %i32 = converted %int_0, %.loc13_20.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc13_20.2: %.0ea = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:   %bound_method.loc13_20.3: <bound method> = bound_method %.loc13_20.2, %impl.elem0.loc13_20.2 [concrete = constants.%Convert.bound.0fd]
// CHECK:STDOUT:   %specific_fn.loc13_20.2: <specific function> = specific_function %impl.elem0.loc13_20.2, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:   %bound_method.loc13_20.4: <bound method> = bound_method %.loc13_20.2, %specific_fn.loc13_20.2 [concrete = constants.%bound_method.4b5]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.2: init Core.IntLiteral = call %bound_method.loc13_20.4(%.loc13_20.2) [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc13_20.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc13_20.2 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc13_20.4: Core.IntLiteral = converted %.loc13_20.2, %.loc13_20.3 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %tuple.elem0.loc13: ref bool = tuple_access %a.ref.loc13, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc13_16: bool = bind_value %tuple.elem0.loc13
// CHECK:STDOUT:   assign file.%c.var, %.loc13_16
// CHECK:STDOUT:   %a.ref.loc14: ref %tuple.type.261 = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14_29: %.982 = impl_witness_access constants.%As.impl_witness.6b4, element0 [concrete = constants.%Convert.197]
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
// CHECK:STDOUT:   %impl.elem0.loc14_36: %.0ea = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:   %bound_method.loc14_36.1: <bound method> = bound_method %.loc14_36.1, %impl.elem0.loc14_36 [concrete = constants.%Convert.bound.faf]
// CHECK:STDOUT:   %specific_fn.loc14_36: <specific function> = specific_function %impl.elem0.loc14_36, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:   %bound_method.loc14_36.2: <bound method> = bound_method %.loc14_36.1, %specific_fn.loc14_36 [concrete = constants.%bound_method.711]
// CHECK:STDOUT:   %int.convert_checked.loc14_36: init Core.IntLiteral = call %bound_method.loc14_36.2(%.loc14_36.1) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc14_36.2: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_36 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc14_36.3: Core.IntLiteral = converted %.loc14_36.1, %.loc14_36.2 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access %a.ref.loc14, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc14_15: %i32 = bind_value %tuple.elem1.loc14
// CHECK:STDOUT:   assign file.%d.var, %.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

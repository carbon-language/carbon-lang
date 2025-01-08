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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.c2e: type = tuple_type (bool, %i32) [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_34.3f9: Core.IntLiteral = int_value 34 [template]
// CHECK:STDOUT:   %tuple.type.3c2: type = tuple_type (bool, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.a9b: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.eae: <bound method> = bound_method %int_34.3f9, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.767: <specific function> = specific_function %Convert.bound.eae, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_34.bd3: %i32 = int_value 34 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.c2e = tuple_value (%true, %int_34.bd3) [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.index.b1b: type = struct_type {.index: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct.972: %struct_type.index.b1b = struct_value (%int_1.5b8) [template]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.1b9: type = fn_type @Convert.4, @As(%i32) [template]
// CHECK:STDOUT:   %impl_witness.c9a: <witness> = impl_witness (imports.%import_ref.6fa), @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.531: type = fn_type @Convert.5, @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.6c4: %Convert.type.531 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.47b: <bound method> = bound_method %int_0.5c6, %Convert.6c4 [template]
// CHECK:STDOUT:   %Convert.specific_fn.e13: <specific function> = specific_function %Convert.bound.47b, @Convert.5(%int_32) [template]
// CHECK:STDOUT:   %int_0.f61: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %impl_witness.dd3: <witness> = impl_witness (imports.%import_ref.a80), @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.d18: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.079: %Convert.type.d18 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.64e: <bound method> = bound_method %int_0.f61, %Convert.079 [template]
// CHECK:STDOUT:   %Convert.specific_fn.f03: <specific function> = specific_function %Convert.bound.64e, @Convert.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.e50: <bound method> = bound_method %int_1.5b8, %Convert.6c4 [template]
// CHECK:STDOUT:   %Convert.specific_fn.f7a: <specific function> = specific_function %Convert.bound.e50, @Convert.5(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.index.bb3: type = struct_type {.index: %i32} [template]
// CHECK:STDOUT:   %struct.a97: %struct_type.index.bb3 = struct_value (%int_1.c60) [template]
// CHECK:STDOUT:   %Convert.bound.7d6: <bound method> = bound_method %int_1.c60, %Convert.079 [template]
// CHECK:STDOUT:   %Convert.specific_fn.08c: <specific function> = specific_function %Convert.bound.7d6, @Convert.3(%int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.f7c
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     .As = %import_ref.166
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
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %a.var: ref %tuple.type.c2e = var a
// CHECK:STDOUT:   %a: ref %tuple.type.c2e = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref bool = var c
// CHECK:STDOUT:   %c: ref bool = bind_name c, %c.var
// CHECK:STDOUT:   %d.var: ref %i32 = var d
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [template = constants.%int_34.3f9]
// CHECK:STDOUT:   %.loc11_31.1: %tuple.type.3c2 = tuple_literal (%true, %int_34)
// CHECK:STDOUT:   %tuple.elem0.loc11: ref bool = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_31.2: init bool = initialize_from %true to %tuple.elem0.loc11 [template = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_34, %impl.elem0.loc11 [template = constants.%Convert.bound.eae]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.767]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_34) [template = constants.%int_34.bd3]
// CHECK:STDOUT:   %.loc11_31.3: init %i32 = converted %int_34, %int.convert_checked.loc11 [template = constants.%int_34.bd3]
// CHECK:STDOUT:   %tuple.elem1.loc11: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_31.4: init %i32 = initialize_from %.loc11_31.3 to %tuple.elem1.loc11 [template = constants.%int_34.bd3]
// CHECK:STDOUT:   %.loc11_31.5: init %tuple.type.c2e = tuple_init (%.loc11_31.2, %.loc11_31.4) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_32: init %tuple.type.c2e = converted %.loc11_31.1, %.loc11_31.5 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_32
// CHECK:STDOUT:   %a.ref.loc12: ref %tuple.type.c2e = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc12_28.1: %struct_type.index.b1b = struct_literal (%int_1.loc12)
// CHECK:STDOUT:   %struct.loc12: %struct_type.index.b1b = struct_value (%int_1.loc12) [template = constants.%struct.972]
// CHECK:STDOUT:   %.loc12_28.2: %struct_type.index.b1b = converted %.loc12_28.1, %struct.loc12 [template = constants.%struct.972]
// CHECK:STDOUT:   %.loc12_29: Core.IntLiteral = struct_access %.loc12_28.2, element0 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %a.ref.loc12, element1
// CHECK:STDOUT:   %.loc12_15: %i32 = bind_value %tuple.elem1.loc12
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15
// CHECK:STDOUT:   %a.ref.loc13: ref %tuple.type.c2e = name_ref a, file.%a
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc13_20.1: %Convert.type.1b9 = impl_witness_access constants.%impl_witness.c9a, element0 [template = constants.%Convert.6c4]
// CHECK:STDOUT:   %Convert.bound.loc13_20.1: <bound method> = bound_method %int_0, %impl.elem0.loc13_20.1 [template = constants.%Convert.bound.47b]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_20.1: <specific function> = specific_function %Convert.bound.loc13_20.1, @Convert.5(constants.%int_32) [template = constants.%Convert.specific_fn.e13]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.1: init %i32 = call %Convert.specific_fn.loc13_20.1(%int_0) [template = constants.%int_0.f61]
// CHECK:STDOUT:   %.loc13_20.1: %i32 = value_of_initializer %int.convert_checked.loc13_20.1 [template = constants.%int_0.f61]
// CHECK:STDOUT:   %.loc13_20.2: %i32 = converted %int_0, %.loc13_20.1 [template = constants.%int_0.f61]
// CHECK:STDOUT:   %impl.elem0.loc13_20.2: %Convert.type.a9b = impl_witness_access constants.%impl_witness.dd3, element0 [template = constants.%Convert.079]
// CHECK:STDOUT:   %Convert.bound.loc13_20.2: <bound method> = bound_method %.loc13_20.2, %impl.elem0.loc13_20.2 [template = constants.%Convert.bound.64e]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_20.2: <specific function> = specific_function %Convert.bound.loc13_20.2, @Convert.3(constants.%int_32) [template = constants.%Convert.specific_fn.f03]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.2: init Core.IntLiteral = call %Convert.specific_fn.loc13_20.2(%.loc13_20.2) [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc13_20.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc13_20.2 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc13_20.4: Core.IntLiteral = converted %.loc13_20.2, %.loc13_20.3 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %tuple.elem0.loc13: ref bool = tuple_access %a.ref.loc13, element0
// CHECK:STDOUT:   %.loc13_16: bool = bind_value %tuple.elem0.loc13
// CHECK:STDOUT:   assign file.%c.var, %.loc13_16
// CHECK:STDOUT:   %a.ref.loc14: ref %tuple.type.c2e = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14_29: %Convert.type.1b9 = impl_witness_access constants.%impl_witness.c9a, element0 [template = constants.%Convert.6c4]
// CHECK:STDOUT:   %Convert.bound.loc14_29: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_29 [template = constants.%Convert.bound.e50]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_29: <specific function> = specific_function %Convert.bound.loc14_29, @Convert.5(constants.%int_32) [template = constants.%Convert.specific_fn.f7a]
// CHECK:STDOUT:   %int.convert_checked.loc14_29: init %i32 = call %Convert.specific_fn.loc14_29(%int_1.loc14) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc14_29.1: %i32 = value_of_initializer %int.convert_checked.loc14_29 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc14_29.2: %i32 = converted %int_1.loc14, %.loc14_29.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.index.bb3 = struct_literal (%.loc14_29.2)
// CHECK:STDOUT:   %struct.loc14: %struct_type.index.bb3 = struct_value (%.loc14_29.2) [template = constants.%struct.a97]
// CHECK:STDOUT:   %.loc14_35.2: %struct_type.index.bb3 = converted %.loc14_35.1, %struct.loc14 [template = constants.%struct.a97]
// CHECK:STDOUT:   %.loc14_36.1: %i32 = struct_access %.loc14_35.2, element0 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %impl.elem0.loc14_36: %Convert.type.a9b = impl_witness_access constants.%impl_witness.dd3, element0 [template = constants.%Convert.079]
// CHECK:STDOUT:   %Convert.bound.loc14_36: <bound method> = bound_method %.loc14_36.1, %impl.elem0.loc14_36 [template = constants.%Convert.bound.7d6]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_36: <specific function> = specific_function %Convert.bound.loc14_36, @Convert.3(constants.%int_32) [template = constants.%Convert.specific_fn.08c]
// CHECK:STDOUT:   %int.convert_checked.loc14_36: init Core.IntLiteral = call %Convert.specific_fn.loc14_36(%.loc14_36.1) [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc14_36.2: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_36 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc14_36.3: Core.IntLiteral = converted %.loc14_36.1, %.loc14_36.2 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access %a.ref.loc14, element1
// CHECK:STDOUT:   %.loc14_15: %i32 = bind_value %tuple.elem1.loc14
// CHECK:STDOUT:   assign file.%d.var, %.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

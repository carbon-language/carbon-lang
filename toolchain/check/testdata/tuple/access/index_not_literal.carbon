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
// CHECK:STDOUT:   %tuple.type.c: type = tuple_type (bool, %i32) [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_34.3: Core.IntLiteral = int_value 34 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (bool, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.a: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.46: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.46 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.ea: <bound method> = bound_method %int_34.3, %Convert.9 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.ea, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_34.b: %i32 = int_value 34 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.c = tuple_value (%true, %int_34.b) [template]
// CHECK:STDOUT:   %int_1.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.index.b1: type = struct_type {.index: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct.9: %struct_type.index.b1 = struct_value (%int_1.5) [template]
// CHECK:STDOUT:   %int_0.5: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.4, @As(%i32) [template]
// CHECK:STDOUT:   %impl_witness.c9: <witness> = impl_witness (imports.%import_ref.6fa), @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.5, @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.6c4: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_0.5, %Convert.6c4 [template]
// CHECK:STDOUT:   %Convert.specific_fn.e: <specific function> = specific_function %Convert.bound.4, @Convert.5(%int_32) [template]
// CHECK:STDOUT:   %int_0.f: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %impl_witness.dd: <witness> = impl_witness (imports.%import_ref.a8), @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.d: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.07: %Convert.type.d = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_0.f, %Convert.07 [template]
// CHECK:STDOUT:   %Convert.specific_fn.f0: <specific function> = specific_function %Convert.bound.6, @Convert.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.e5: <bound method> = bound_method %int_1.5, %Convert.6c4 [template]
// CHECK:STDOUT:   %Convert.specific_fn.f7: <specific function> = specific_function %Convert.bound.e5, @Convert.5(%int_32) [template]
// CHECK:STDOUT:   %int_1.c: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.index.bb: type = struct_type {.index: %i32} [template]
// CHECK:STDOUT:   %struct.a: %struct_type.index.bb = struct_value (%int_1.c) [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_1.c, %Convert.07 [template]
// CHECK:STDOUT:   %Convert.specific_fn.0: <specific function> = specific_function %Convert.bound.7, @Convert.3(%int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.f7c
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a6
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
// CHECK:STDOUT:   %a.var: ref %tuple.type.c = var a
// CHECK:STDOUT:   %a: ref %tuple.type.c = bind_name a, %a.var
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
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [template = constants.%int_34.3]
// CHECK:STDOUT:   %.loc11_31.1: %tuple.type.3 = tuple_literal (%true, %int_34)
// CHECK:STDOUT:   %tuple.elem0.loc11: ref bool = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_31.2: init bool = initialize_from %true to %tuple.elem0.loc11 [template = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.cd = impl_witness_access constants.%impl_witness.5, element0 [template = constants.%Convert.9]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_34, %impl.elem0.loc11 [template = constants.%Convert.bound.ea]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_34) [template = constants.%int_34.b]
// CHECK:STDOUT:   %.loc11_31.3: init %i32 = converted %int_34, %int.convert_checked.loc11 [template = constants.%int_34.b]
// CHECK:STDOUT:   %tuple.elem1.loc11: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_31.4: init %i32 = initialize_from %.loc11_31.3 to %tuple.elem1.loc11 [template = constants.%int_34.b]
// CHECK:STDOUT:   %.loc11_31.5: init %tuple.type.c = tuple_init (%.loc11_31.2, %.loc11_31.4) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_32: init %tuple.type.c = converted %.loc11_31.1, %.loc11_31.5 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_32
// CHECK:STDOUT:   %a.ref.loc12: ref %tuple.type.c = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %.loc12_28.1: %struct_type.index.b1 = struct_literal (%int_1.loc12)
// CHECK:STDOUT:   %struct.loc12: %struct_type.index.b1 = struct_value (%int_1.loc12) [template = constants.%struct.9]
// CHECK:STDOUT:   %.loc12_28.2: %struct_type.index.b1 = converted %.loc12_28.1, %struct.loc12 [template = constants.%struct.9]
// CHECK:STDOUT:   %.loc12_29: Core.IntLiteral = struct_access %.loc12_28.2, element0 [template = constants.%int_1.5]
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %a.ref.loc12, element1
// CHECK:STDOUT:   %.loc12_15: %i32 = bind_value %tuple.elem1.loc12
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15
// CHECK:STDOUT:   %a.ref.loc13: ref %tuple.type.c = name_ref a, file.%a
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc13_20.1: %Convert.type.1 = impl_witness_access constants.%impl_witness.c9, element0 [template = constants.%Convert.6c4]
// CHECK:STDOUT:   %Convert.bound.loc13_20.1: <bound method> = bound_method %int_0, %impl.elem0.loc13_20.1 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_20.1: <specific function> = specific_function %Convert.bound.loc13_20.1, @Convert.5(constants.%int_32) [template = constants.%Convert.specific_fn.e]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.1: init %i32 = call %Convert.specific_fn.loc13_20.1(%int_0) [template = constants.%int_0.f]
// CHECK:STDOUT:   %.loc13_20.1: %i32 = value_of_initializer %int.convert_checked.loc13_20.1 [template = constants.%int_0.f]
// CHECK:STDOUT:   %.loc13_20.2: %i32 = converted %int_0, %.loc13_20.1 [template = constants.%int_0.f]
// CHECK:STDOUT:   %impl.elem0.loc13_20.2: %Convert.type.a = impl_witness_access constants.%impl_witness.dd, element0 [template = constants.%Convert.07]
// CHECK:STDOUT:   %Convert.bound.loc13_20.2: <bound method> = bound_method %.loc13_20.2, %impl.elem0.loc13_20.2 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_20.2: <specific function> = specific_function %Convert.bound.loc13_20.2, @Convert.3(constants.%int_32) [template = constants.%Convert.specific_fn.f0]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.2: init Core.IntLiteral = call %Convert.specific_fn.loc13_20.2(%.loc13_20.2) [template = constants.%int_0.5]
// CHECK:STDOUT:   %.loc13_20.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc13_20.2 [template = constants.%int_0.5]
// CHECK:STDOUT:   %.loc13_20.4: Core.IntLiteral = converted %.loc13_20.2, %.loc13_20.3 [template = constants.%int_0.5]
// CHECK:STDOUT:   %tuple.elem0.loc13: ref bool = tuple_access %a.ref.loc13, element0
// CHECK:STDOUT:   %.loc13_16: bool = bind_value %tuple.elem0.loc13
// CHECK:STDOUT:   assign file.%c.var, %.loc13_16
// CHECK:STDOUT:   %a.ref.loc14: ref %tuple.type.c = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14_29: %Convert.type.1 = impl_witness_access constants.%impl_witness.c9, element0 [template = constants.%Convert.6c4]
// CHECK:STDOUT:   %Convert.bound.loc14_29: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_29 [template = constants.%Convert.bound.e5]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_29: <specific function> = specific_function %Convert.bound.loc14_29, @Convert.5(constants.%int_32) [template = constants.%Convert.specific_fn.f7]
// CHECK:STDOUT:   %int.convert_checked.loc14_29: init %i32 = call %Convert.specific_fn.loc14_29(%int_1.loc14) [template = constants.%int_1.c]
// CHECK:STDOUT:   %.loc14_29.1: %i32 = value_of_initializer %int.convert_checked.loc14_29 [template = constants.%int_1.c]
// CHECK:STDOUT:   %.loc14_29.2: %i32 = converted %int_1.loc14, %.loc14_29.1 [template = constants.%int_1.c]
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.index.bb = struct_literal (%.loc14_29.2)
// CHECK:STDOUT:   %struct.loc14: %struct_type.index.bb = struct_value (%.loc14_29.2) [template = constants.%struct.a]
// CHECK:STDOUT:   %.loc14_35.2: %struct_type.index.bb = converted %.loc14_35.1, %struct.loc14 [template = constants.%struct.a]
// CHECK:STDOUT:   %.loc14_36.1: %i32 = struct_access %.loc14_35.2, element0 [template = constants.%int_1.c]
// CHECK:STDOUT:   %impl.elem0.loc14_36: %Convert.type.a = impl_witness_access constants.%impl_witness.dd, element0 [template = constants.%Convert.07]
// CHECK:STDOUT:   %Convert.bound.loc14_36: <bound method> = bound_method %.loc14_36.1, %impl.elem0.loc14_36 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_36: <specific function> = specific_function %Convert.bound.loc14_36, @Convert.3(constants.%int_32) [template = constants.%Convert.specific_fn.0]
// CHECK:STDOUT:   %int.convert_checked.loc14_36: init Core.IntLiteral = call %Convert.specific_fn.loc14_36(%.loc14_36.1) [template = constants.%int_1.5]
// CHECK:STDOUT:   %.loc14_36.2: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_36 [template = constants.%int_1.5]
// CHECK:STDOUT:   %.loc14_36.3: Core.IntLiteral = converted %.loc14_36.1, %.loc14_36.2 [template = constants.%int_1.5]
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access %a.ref.loc14, element1
// CHECK:STDOUT:   %.loc14_15: %i32 = bind_value %tuple.elem1.loc14
// CHECK:STDOUT:   assign file.%d.var, %.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

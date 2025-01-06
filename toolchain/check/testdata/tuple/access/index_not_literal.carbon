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
// CHECK:STDOUT:   %tuple.type.f: type = tuple_type (bool, %i32) [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_34.3: Core.IntLiteral = int_value 34 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (bool, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.c3: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.c8: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %impl_witness.bd: <witness> = impl_witness (imports.%import_ref.059), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.2c: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.54: %Convert.type.2c = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_34.3, %Convert.54 [template]
// CHECK:STDOUT:   %Convert.specific_fn.f: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_34.2: %i32 = int_value 34 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.f = tuple_value (%true, %int_34.2) [template]
// CHECK:STDOUT:   %int_1.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.index.b: type = struct_type {.index: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct.9: %struct_type.index.b = struct_value (%int_1.5) [template]
// CHECK:STDOUT:   %int_0.5: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.b: type = fn_type @Convert.4, @As(%i32) [template]
// CHECK:STDOUT:   %impl_witness.51: <witness> = impl_witness (imports.%import_ref.3e), @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.0: type = fn_type @Convert.5, @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.1: %Convert.type.0 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_0.5, %Convert.1 [template]
// CHECK:STDOUT:   %Convert.specific_fn.a: <specific function> = specific_function %Convert.bound.4, @Convert.5(%int_32) [template]
// CHECK:STDOUT:   %int_0.d: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %impl_witness.ea: <witness> = impl_witness (imports.%import_ref.102), @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.e: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.d, %Convert.e [template]
// CHECK:STDOUT:   %Convert.specific_fn.c9: <specific function> = specific_function %Convert.bound.1, @Convert.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.0: <bound method> = bound_method %int_1.5, %Convert.1 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.0, @Convert.5(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.index.7: type = struct_type {.index: %i32} [template]
// CHECK:STDOUT:   %struct.7: %struct_type.index.7 = struct_value (%int_1.2) [template]
// CHECK:STDOUT:   %Convert.bound.b: <bound method> = bound_method %int_1.2, %Convert.e [template]
// CHECK:STDOUT:   %Convert.specific_fn.c7: <specific function> = specific_function %Convert.bound.b, @Convert.3(%int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.057
// CHECK:STDOUT:     .Int = %import_ref.30a
// CHECK:STDOUT:     .ImplicitAs = %import_ref.41f
// CHECK:STDOUT:     .As = %import_ref.977
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
// CHECK:STDOUT:   %a.var: ref %tuple.type.f = var a
// CHECK:STDOUT:   %a: ref %tuple.type.f = bind_name a, %a.var
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
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.c3 = impl_witness_access constants.%impl_witness.bd, element0 [template = constants.%Convert.54]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_34, %impl.elem0.loc11 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.f]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_34) [template = constants.%int_34.2]
// CHECK:STDOUT:   %.loc11_31.3: init %i32 = converted %int_34, %int.convert_checked.loc11 [template = constants.%int_34.2]
// CHECK:STDOUT:   %tuple.elem1.loc11: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_31.4: init %i32 = initialize_from %.loc11_31.3 to %tuple.elem1.loc11 [template = constants.%int_34.2]
// CHECK:STDOUT:   %.loc11_31.5: init %tuple.type.f = tuple_init (%.loc11_31.2, %.loc11_31.4) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_32: init %tuple.type.f = converted %.loc11_31.1, %.loc11_31.5 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_32
// CHECK:STDOUT:   %a.ref.loc12: ref %tuple.type.f = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %.loc12_28.1: %struct_type.index.b = struct_literal (%int_1.loc12)
// CHECK:STDOUT:   %struct.loc12: %struct_type.index.b = struct_value (%int_1.loc12) [template = constants.%struct.9]
// CHECK:STDOUT:   %.loc12_28.2: %struct_type.index.b = converted %.loc12_28.1, %struct.loc12 [template = constants.%struct.9]
// CHECK:STDOUT:   %.loc12_29: Core.IntLiteral = struct_access %.loc12_28.2, element0 [template = constants.%int_1.5]
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %a.ref.loc12, element1
// CHECK:STDOUT:   %.loc12_15: %i32 = bind_value %tuple.elem1.loc12
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15
// CHECK:STDOUT:   %a.ref.loc13: ref %tuple.type.f = name_ref a, file.%a
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc13_20.1: %Convert.type.b = impl_witness_access constants.%impl_witness.51, element0 [template = constants.%Convert.1]
// CHECK:STDOUT:   %Convert.bound.loc13_20.1: <bound method> = bound_method %int_0, %impl.elem0.loc13_20.1 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_20.1: <specific function> = specific_function %Convert.bound.loc13_20.1, @Convert.5(constants.%int_32) [template = constants.%Convert.specific_fn.a]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.1: init %i32 = call %Convert.specific_fn.loc13_20.1(%int_0) [template = constants.%int_0.d]
// CHECK:STDOUT:   %.loc13_20.1: %i32 = value_of_initializer %int.convert_checked.loc13_20.1 [template = constants.%int_0.d]
// CHECK:STDOUT:   %.loc13_20.2: %i32 = converted %int_0, %.loc13_20.1 [template = constants.%int_0.d]
// CHECK:STDOUT:   %impl.elem0.loc13_20.2: %Convert.type.c8 = impl_witness_access constants.%impl_witness.ea, element0 [template = constants.%Convert.e]
// CHECK:STDOUT:   %Convert.bound.loc13_20.2: <bound method> = bound_method %.loc13_20.2, %impl.elem0.loc13_20.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_20.2: <specific function> = specific_function %Convert.bound.loc13_20.2, @Convert.3(constants.%int_32) [template = constants.%Convert.specific_fn.c9]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.2: init Core.IntLiteral = call %Convert.specific_fn.loc13_20.2(%.loc13_20.2) [template = constants.%int_0.5]
// CHECK:STDOUT:   %.loc13_20.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc13_20.2 [template = constants.%int_0.5]
// CHECK:STDOUT:   %.loc13_20.4: Core.IntLiteral = converted %.loc13_20.2, %.loc13_20.3 [template = constants.%int_0.5]
// CHECK:STDOUT:   %tuple.elem0.loc13: ref bool = tuple_access %a.ref.loc13, element0
// CHECK:STDOUT:   %.loc13_16: bool = bind_value %tuple.elem0.loc13
// CHECK:STDOUT:   assign file.%c.var, %.loc13_16
// CHECK:STDOUT:   %a.ref.loc14: ref %tuple.type.f = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14_29: %Convert.type.b = impl_witness_access constants.%impl_witness.51, element0 [template = constants.%Convert.1]
// CHECK:STDOUT:   %Convert.bound.loc14_29: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_29 [template = constants.%Convert.bound.0]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_29: <specific function> = specific_function %Convert.bound.loc14_29, @Convert.5(constants.%int_32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %int.convert_checked.loc14_29: init %i32 = call %Convert.specific_fn.loc14_29(%int_1.loc14) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_29.1: %i32 = value_of_initializer %int.convert_checked.loc14_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_29.2: %i32 = converted %int_1.loc14, %.loc14_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.index.7 = struct_literal (%.loc14_29.2)
// CHECK:STDOUT:   %struct.loc14: %struct_type.index.7 = struct_value (%.loc14_29.2) [template = constants.%struct.7]
// CHECK:STDOUT:   %.loc14_35.2: %struct_type.index.7 = converted %.loc14_35.1, %struct.loc14 [template = constants.%struct.7]
// CHECK:STDOUT:   %.loc14_36.1: %i32 = struct_access %.loc14_35.2, element0 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc14_36: %Convert.type.c8 = impl_witness_access constants.%impl_witness.ea, element0 [template = constants.%Convert.e]
// CHECK:STDOUT:   %Convert.bound.loc14_36: <bound method> = bound_method %.loc14_36.1, %impl.elem0.loc14_36 [template = constants.%Convert.bound.b]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_36: <specific function> = specific_function %Convert.bound.loc14_36, @Convert.3(constants.%int_32) [template = constants.%Convert.specific_fn.c7]
// CHECK:STDOUT:   %int.convert_checked.loc14_36: init Core.IntLiteral = call %Convert.specific_fn.loc14_36(%.loc14_36.1) [template = constants.%int_1.5]
// CHECK:STDOUT:   %.loc14_36.2: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_36 [template = constants.%int_1.5]
// CHECK:STDOUT:   %.loc14_36.3: Core.IntLiteral = converted %.loc14_36.1, %.loc14_36.2 [template = constants.%int_1.5]
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access %a.ref.loc14, element1
// CHECK:STDOUT:   %.loc14_15: %i32 = bind_value %tuple.elem1.loc14
// CHECK:STDOUT:   assign file.%d.var, %.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

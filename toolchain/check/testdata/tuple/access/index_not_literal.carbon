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
// CHECK:STDERR: index_not_literal.carbon:[[@LINE+4]]:15: error: tuple index must be a constant [TupleIndexNotConstant]
// CHECK:STDERR: var c: bool = a.(0 as i32);
// CHECK:STDERR:               ^~~~~~~~~~~~
// CHECK:STDERR:
var c: bool = a.(0 as i32);
// CHECK:STDERR: index_not_literal.carbon:[[@LINE+3]]:14: error: tuple index must be a constant [TupleIndexNotConstant]
// CHECK:STDERR: var d: i32 = a.({.index = 1 as i32}.index);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var d: i32 = a.({.index = 1 as i32}.index);

// CHECK:STDOUT: --- index_not_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (bool, %i32) [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (bool, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_34, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_34) [symbolic]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%true, %int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.index.1: type = struct_type {.index: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct.1: %struct_type.index.1 = struct_value (%int_1) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.4, @As(%i32) [template]
// CHECK:STDOUT:   %impl_witness.6: <witness> = impl_witness (%Convert.7), @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0, %Convert.7 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.5(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_0) [symbolic]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1, %Convert.7 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.5(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %Int = call %Convert.specific_fn.3(%int_1) [symbolic]
// CHECK:STDOUT:   %struct_type.index.2: type = struct_type {.index: %i32} [template]
// CHECK:STDOUT:   %struct.2: %struct_type.index.2 = struct_value (%int.convert_checked.3) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     .As = %import_ref.194
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
// CHECK:STDOUT:   %bool.make_type.loc11: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_18.1: %tuple.type.1 = tuple_literal (%bool.make_type.loc11, %i32.loc11)
// CHECK:STDOUT:   %.loc11_18.2: type = value_of_initializer %bool.make_type.loc11 [template = bool]
// CHECK:STDOUT:   %.loc11_18.3: type = converted %bool.make_type.loc11, %.loc11_18.2 [template = bool]
// CHECK:STDOUT:   %.loc11_18.4: type = converted %.loc11_18.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %bool.make_type.loc17: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc17_8.1: type = value_of_initializer %bool.make_type.loc17 [template = bool]
// CHECK:STDOUT:   %.loc17_8.2: type = converted %bool.make_type.loc17, %.loc17_8.1 [template = bool]
// CHECK:STDOUT:   %c.var: ref bool = var c
// CHECK:STDOUT:   %c: ref bool = bind_name c, %c.var
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %d.var: ref %i32 = var d
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [template = constants.%int_34]
// CHECK:STDOUT:   %.loc11_31.1: %tuple.type.3 = tuple_literal (%true, %int_34)
// CHECK:STDOUT:   %tuple.elem0: ref bool = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_31.2: init bool = initialize_from %true to %tuple.elem0 [template = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_34, %impl.elem0.loc11 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %Int = call %Convert.specific_fn.loc11(%int_34) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_31.3: init %i32 = converted %int_34, %int.convert_checked.loc11 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %tuple.elem1.loc11: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_31.4: init %i32 = initialize_from %.loc11_31.3 to %tuple.elem1.loc11 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_31.5: init %tuple.type.2 = tuple_init (%.loc11_31.2, %.loc11_31.4) to file.%a.var [symbolic = constants.%tuple]
// CHECK:STDOUT:   %.loc11_32: init %tuple.type.2 = converted %.loc11_31.1, %.loc11_31.5 [symbolic = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_32
// CHECK:STDOUT:   %a.ref.loc12: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc12_28.1: %struct_type.index.1 = struct_literal (%int_1.loc12)
// CHECK:STDOUT:   %struct.loc12: %struct_type.index.1 = struct_value (%int_1.loc12) [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_28.2: %struct_type.index.1 = converted %.loc12_28.1, %struct.loc12 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_29: Core.IntLiteral = struct_access %.loc12_28.2, element0 [template = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %a.ref.loc12, element1
// CHECK:STDOUT:   %.loc12_15: %i32 = bind_value %tuple.elem1.loc12
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15
// CHECK:STDOUT:   %a.ref.loc17: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc17: %Convert.type.11 = impl_witness_access constants.%impl_witness.6, element0 [symbolic = constants.%Convert.7]
// CHECK:STDOUT:   %Convert.bound.loc17: <bound method> = bound_method %int_0, %impl.elem0.loc17 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc17: <specific function> = specific_function %Convert.bound.loc17, @Convert.5(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %Int = call %Convert.specific_fn.loc17(%int_0) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc17_20.1: %i32 = value_of_initializer %int.convert_checked.loc17 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc17_20.2: %i32 = converted %int_0, %.loc17_20.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %a.ref.loc21: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc21: %Convert.type.11 = impl_witness_access constants.%impl_witness.6, element0 [symbolic = constants.%Convert.7]
// CHECK:STDOUT:   %Convert.bound.loc21: <bound method> = bound_method %int_1.loc21, %impl.elem0.loc21 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc21: <specific function> = specific_function %Convert.bound.loc21, @Convert.5(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %Int = call %Convert.specific_fn.loc21(%int_1.loc21) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc21_29.1: %i32 = value_of_initializer %int.convert_checked.loc21 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc21_29.2: %i32 = converted %int_1.loc21, %.loc21_29.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc21_35.1: %struct_type.index.2 = struct_literal (%.loc21_29.2)
// CHECK:STDOUT:   %struct.loc21: %struct_type.index.2 = struct_value (%.loc21_29.2) [symbolic = constants.%struct.2]
// CHECK:STDOUT:   %.loc21_35.2: %struct_type.index.2 = converted %.loc21_35.1, %struct.loc21 [symbolic = constants.%struct.2]
// CHECK:STDOUT:   %.loc21_36: %i32 = struct_access %.loc21_35.2, element0 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

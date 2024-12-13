// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/index_not_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/index_not_literal.carbon

var a: [i32; 3] = (1, 2, 3);
var b: i32 = a[{.index = 2}.index];

// CHECK:STDOUT: --- index_not_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %i32 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_2) [symbolic]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %iN = call %Convert.specific_fn.3(%int_3) [symbolic]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int.convert_checked.1, %int.convert_checked.2, %int.convert_checked.3) [symbolic]
// CHECK:STDOUT:   %struct_type.index: type = struct_type {.index: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int_2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int_32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_signed.loc11, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_signed.loc12, %.loc12_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc11_23: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type = tuple_literal (%int_1.loc11_20, %int_2.loc11_23, %int_3)
// CHECK:STDOUT:   %impl.elem0.loc11_27.1: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_27.1: <bound method> = bound_method %int_1.loc11_20, %impl.elem0.loc11_27.1 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.1: <specific function> = specific_function %Convert.bound.loc11_27.1, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.1: init %iN = call %Convert.specific_fn.loc11_27.1(%int_1.loc11_20) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_27.2: init %i32 = converted %int_1.loc11_20, %int.convert_checked.loc11_27.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_27.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc11_27.4: init %i32 = initialize_from %.loc11_27.2 to %.loc11_27.3 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc11_27.2: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_27.2: <bound method> = bound_method %int_2.loc11_23, %impl.elem0.loc11_27.2 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.2: <specific function> = specific_function %Convert.bound.loc11_27.2, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.2: init %iN = call %Convert.specific_fn.loc11_27.2(%int_2.loc11_23) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc11_27.5: init %i32 = converted %int_2.loc11_23, %int.convert_checked.loc11_27.2 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int_1.loc11_27: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc11_27.6: ref %i32 = array_index file.%a.var, %int_1.loc11_27
// CHECK:STDOUT:   %.loc11_27.7: init %i32 = initialize_from %.loc11_27.5 to %.loc11_27.6 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %impl.elem0.loc11_27.3: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_27.3: <bound method> = bound_method %int_3, %impl.elem0.loc11_27.3 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.3: <specific function> = specific_function %Convert.bound.loc11_27.3, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.3: init %iN = call %Convert.specific_fn.loc11_27.3(%int_3) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc11_27.8: init %i32 = converted %int_3, %int.convert_checked.loc11_27.3 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int_2.loc11_27: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc11_27.9: ref %i32 = array_index file.%a.var, %int_2.loc11_27
// CHECK:STDOUT:   %.loc11_27.10: init %i32 = initialize_from %.loc11_27.8 to %.loc11_27.9 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc11_27.11: init %array_type = array_init (%.loc11_27.4, %.loc11_27.7, %.loc11_27.10) to file.%a.var [symbolic = constants.%array]
// CHECK:STDOUT:   %.loc11_28: init %array_type = converted %.loc11_27.1, %.loc11_27.11 [symbolic = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %int_2.loc12: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc12_27.1: %struct_type.index = struct_literal (%int_2.loc12)
// CHECK:STDOUT:   %struct: %struct_type.index = struct_value (%int_2.loc12) [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_27.2: %struct_type.index = converted %.loc12_27.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_28.1: Core.IntLiteral = struct_access %.loc12_27.2, element0 [template = constants.%int_2]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_34.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_34.2: type = converted %int.make_type_signed, %.loc12_34.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc12: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc12: <bound method> = bound_method %.loc12_28.1, %impl.elem0.loc12 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12: <specific function> = specific_function %Convert.bound.loc12, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %iN = call %Convert.specific_fn.loc12(%.loc12_28.1) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12_28.2: %i32 = value_of_initializer %int.convert_checked.loc12 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12_28.3: %i32 = converted %.loc12_28.1, %.loc12_28.2 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12_34.3: ref %i32 = array_index %a.ref, %.loc12_28.3
// CHECK:STDOUT:   %.loc12_34.4: %i32 = bind_value %.loc12_34.3
// CHECK:STDOUT:   assign file.%b.var, %.loc12_34.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/two_elements.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/two_elements.carbon

let v: (i32, i32) = (4, 102);
let w: (i32, i32) = v;

var x: (i32, i32) = (4, 102);
var y: (i32, i32) = x;

// CHECK:STDOUT: --- two_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %int_102: Core.IntLiteral = int_value 102 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3), @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_4, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_4) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_102, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_102) [symbolic]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int.convert_checked.1, %int.convert_checked.2) [symbolic]
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
// CHECK:STDOUT:     .v = @__global_init.%v
// CHECK:STDOUT:     .w = @__global_init.%w
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32.loc11_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_9: init type = call constants.%Int(%int_32.loc11_9) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc11_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_14: init type = call constants.%Int(%int_32.loc11_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_9, %int.make_type_signed.loc11_14)
// CHECK:STDOUT:   %.loc11_17.2: type = value_of_initializer %int.make_type_signed.loc11_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.3: type = converted %int.make_type_signed.loc11_9, %.loc11_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.4: type = value_of_initializer %int.make_type_signed.loc11_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.5: type = converted %int.make_type_signed.loc11_14, %.loc11_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.6: type = converted %.loc11_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %int_32.loc12_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12_9: init type = call constants.%Int(%int_32.loc12_9) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc12_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12_14: init type = call constants.%Int(%int_32.loc12_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc12_9, %int.make_type_signed.loc12_14)
// CHECK:STDOUT:   %.loc12_17.2: type = value_of_initializer %int.make_type_signed.loc12_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_17.3: type = converted %int.make_type_signed.loc12_9, %.loc12_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_17.4: type = value_of_initializer %int.make_type_signed.loc12_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_17.5: type = converted %int.make_type_signed.loc12_14, %.loc12_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_17.6: type = converted %.loc12_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %int_32.loc14_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_9: init type = call constants.%Int(%int_32.loc14_9) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_14: init type = call constants.%Int(%int_32.loc14_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc14_9, %int.make_type_signed.loc14_14)
// CHECK:STDOUT:   %.loc14_17.2: type = value_of_initializer %int.make_type_signed.loc14_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.3: type = converted %int.make_type_signed.loc14_9, %.loc14_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.4: type = value_of_initializer %int.make_type_signed.loc14_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.5: type = converted %int.make_type_signed.loc14_14, %.loc14_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.6: type = converted %.loc14_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %x.var: ref %tuple.type.2 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.2 = bind_name x, %x.var
// CHECK:STDOUT:   %int_32.loc15_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15_9: init type = call constants.%Int(%int_32.loc15_9) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc15_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15_14: init type = call constants.%Int(%int_32.loc15_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc15_9, %int.make_type_signed.loc15_14)
// CHECK:STDOUT:   %.loc15_17.2: type = value_of_initializer %int.make_type_signed.loc15_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_17.3: type = converted %int.make_type_signed.loc15_9, %.loc15_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_17.4: type = value_of_initializer %int.make_type_signed.loc15_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_17.5: type = converted %int.make_type_signed.loc15_14, %.loc15_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_17.6: type = converted %.loc15_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %y.var: ref %tuple.type.2 = var y
// CHECK:STDOUT:   %y: ref %tuple.type.2 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_4.loc11: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %int_102.loc11: Core.IntLiteral = int_value 102 [template = constants.%int_102]
// CHECK:STDOUT:   %.loc11_28.1: %tuple.type.3 = tuple_literal (%int_4.loc11, %int_102.loc11)
// CHECK:STDOUT:   %impl.elem0.loc11_28.1: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_28.1: <bound method> = bound_method %int_4.loc11, %impl.elem0.loc11_28.1 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_28.1: <specific function> = specific_function %Convert.bound.loc11_28.1, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_28.1: init %iN = call %Convert.specific_fn.loc11_28.1(%int_4.loc11) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_28.2: %i32 = value_of_initializer %int.convert_checked.loc11_28.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_28.3: %i32 = converted %int_4.loc11, %.loc11_28.2 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc11_28.2: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11_28.2: <bound method> = bound_method %int_102.loc11, %impl.elem0.loc11_28.2 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_28.2: <specific function> = specific_function %Convert.bound.loc11_28.2, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_28.2: init %iN = call %Convert.specific_fn.loc11_28.2(%int_102.loc11) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc11_28.4: %i32 = value_of_initializer %int.convert_checked.loc11_28.2 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc11_28.5: %i32 = converted %int_102.loc11, %.loc11_28.4 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc11_28.3, %.loc11_28.5) [symbolic = constants.%tuple]
// CHECK:STDOUT:   %.loc11_29: %tuple.type.2 = converted %.loc11_28.1, %tuple [symbolic = constants.%tuple]
// CHECK:STDOUT:   %v: %tuple.type.2 = bind_name v, %.loc11_29
// CHECK:STDOUT:   %v.ref: %tuple.type.2 = name_ref v, %v
// CHECK:STDOUT:   %w: %tuple.type.2 = bind_name w, %v.ref
// CHECK:STDOUT:   %int_4.loc14: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %int_102.loc14: Core.IntLiteral = int_value 102 [template = constants.%int_102]
// CHECK:STDOUT:   %.loc14_28.1: %tuple.type.3 = tuple_literal (%int_4.loc14, %int_102.loc14)
// CHECK:STDOUT:   %impl.elem0.loc14_28.1: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc14_28.1: <bound method> = bound_method %int_4.loc14, %impl.elem0.loc14_28.1 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_28.1: <specific function> = specific_function %Convert.bound.loc14_28.1, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_28.1: init %iN = call %Convert.specific_fn.loc14_28.1(%int_4.loc14) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc14_28.2: init %i32 = converted %int_4.loc14, %int.convert_checked.loc14_28.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %tuple.elem0.loc14: ref %i32 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_28.3: init %i32 = initialize_from %.loc14_28.2 to %tuple.elem0.loc14 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc14_28.2: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc14_28.2: <bound method> = bound_method %int_102.loc14, %impl.elem0.loc14_28.2 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_28.2: <specific function> = specific_function %Convert.bound.loc14_28.2, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_28.2: init %iN = call %Convert.specific_fn.loc14_28.2(%int_102.loc14) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc14_28.4: init %i32 = converted %int_102.loc14, %int.convert_checked.loc14_28.2 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_28.5: init %i32 = initialize_from %.loc14_28.4 to %tuple.elem1.loc14 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc14_28.6: init %tuple.type.2 = tuple_init (%.loc14_28.3, %.loc14_28.5) to file.%x.var [symbolic = constants.%tuple]
// CHECK:STDOUT:   %.loc14_29: init %tuple.type.2 = converted %.loc14_28.1, %.loc14_28.6 [symbolic = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_29
// CHECK:STDOUT:   %x.ref: ref %tuple.type.2 = name_ref x, file.%x
// CHECK:STDOUT:   %tuple.elem0.loc15_21.1: ref %i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %.loc15_21.1: %i32 = bind_value %tuple.elem0.loc15_21.1
// CHECK:STDOUT:   %tuple.elem0.loc15_21.2: ref %i32 = tuple_access file.%y.var, element0
// CHECK:STDOUT:   %.loc15_21.2: init %i32 = initialize_from %.loc15_21.1 to %tuple.elem0.loc15_21.2
// CHECK:STDOUT:   %tuple.elem1.loc15_21.1: ref %i32 = tuple_access %x.ref, element1
// CHECK:STDOUT:   %.loc15_21.3: %i32 = bind_value %tuple.elem1.loc15_21.1
// CHECK:STDOUT:   %tuple.elem1.loc15_21.2: ref %i32 = tuple_access file.%y.var, element1
// CHECK:STDOUT:   %.loc15_21.4: init %i32 = initialize_from %.loc15_21.3 to %tuple.elem1.loc15_21.2
// CHECK:STDOUT:   %.loc15_21.5: init %tuple.type.2 = tuple_init (%.loc15_21.2, %.loc15_21.4) to file.%y.var
// CHECK:STDOUT:   %.loc15_22: init %tuple.type.2 = converted %x.ref, %.loc15_21.5
// CHECK:STDOUT:   assign file.%y.var, %.loc15_22
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

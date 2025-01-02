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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int_4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %int_102.1: Core.IntLiteral = int_value 102 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.19: <witness> = impl_witness (imports.%import_ref.14), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_4.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_102.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_102.2: %i32 = int_value 102 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int_4.2, %int_102.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
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
// CHECK:STDOUT:   %x.var: ref %tuple.type.2 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.2 = bind_name x, %x.var
// CHECK:STDOUT:   %y.var: ref %tuple.type.2 = var y
// CHECK:STDOUT:   %y: ref %tuple.type.2 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_4.loc11: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %int_102.loc11: Core.IntLiteral = int_value 102 [template = constants.%int_102.1]
// CHECK:STDOUT:   %.loc11_28.1: %tuple.type.3 = tuple_literal (%int_4.loc11, %int_102.loc11)
// CHECK:STDOUT:   %impl.elem0.loc11_28.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_28.1: <bound method> = bound_method %int_4.loc11, %impl.elem0.loc11_28.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_28.1: <specific function> = specific_function %Convert.bound.loc11_28.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_28.1: init %i32 = call %Convert.specific_fn.loc11_28.1(%int_4.loc11) [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc11_28.2: %i32 = value_of_initializer %int.convert_checked.loc11_28.1 [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc11_28.3: %i32 = converted %int_4.loc11, %.loc11_28.2 [template = constants.%int_4.2]
// CHECK:STDOUT:   %impl.elem0.loc11_28.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_28.2: <bound method> = bound_method %int_102.loc11, %impl.elem0.loc11_28.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_28.2: <specific function> = specific_function %Convert.bound.loc11_28.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_28.2: init %i32 = call %Convert.specific_fn.loc11_28.2(%int_102.loc11) [template = constants.%int_102.2]
// CHECK:STDOUT:   %.loc11_28.4: %i32 = value_of_initializer %int.convert_checked.loc11_28.2 [template = constants.%int_102.2]
// CHECK:STDOUT:   %.loc11_28.5: %i32 = converted %int_102.loc11, %.loc11_28.4 [template = constants.%int_102.2]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc11_28.3, %.loc11_28.5) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_29: %tuple.type.2 = converted %.loc11_28.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %v: %tuple.type.2 = bind_name v, %.loc11_29
// CHECK:STDOUT:   %v.ref: %tuple.type.2 = name_ref v, %v
// CHECK:STDOUT:   %w: %tuple.type.2 = bind_name w, %v.ref
// CHECK:STDOUT:   %int_4.loc14: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %int_102.loc14: Core.IntLiteral = int_value 102 [template = constants.%int_102.1]
// CHECK:STDOUT:   %.loc14_28.1: %tuple.type.3 = tuple_literal (%int_4.loc14, %int_102.loc14)
// CHECK:STDOUT:   %impl.elem0.loc14_28.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_28.1: <bound method> = bound_method %int_4.loc14, %impl.elem0.loc14_28.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_28.1: <specific function> = specific_function %Convert.bound.loc14_28.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_28.1: init %i32 = call %Convert.specific_fn.loc14_28.1(%int_4.loc14) [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc14_28.2: init %i32 = converted %int_4.loc14, %int.convert_checked.loc14_28.1 [template = constants.%int_4.2]
// CHECK:STDOUT:   %tuple.elem0.loc14: ref %i32 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_28.3: init %i32 = initialize_from %.loc14_28.2 to %tuple.elem0.loc14 [template = constants.%int_4.2]
// CHECK:STDOUT:   %impl.elem0.loc14_28.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_28.2: <bound method> = bound_method %int_102.loc14, %impl.elem0.loc14_28.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_28.2: <specific function> = specific_function %Convert.bound.loc14_28.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_28.2: init %i32 = call %Convert.specific_fn.loc14_28.2(%int_102.loc14) [template = constants.%int_102.2]
// CHECK:STDOUT:   %.loc14_28.4: init %i32 = converted %int_102.loc14, %int.convert_checked.loc14_28.2 [template = constants.%int_102.2]
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_28.5: init %i32 = initialize_from %.loc14_28.4 to %tuple.elem1.loc14 [template = constants.%int_102.2]
// CHECK:STDOUT:   %.loc14_28.6: init %tuple.type.2 = tuple_init (%.loc14_28.3, %.loc14_28.5) to file.%x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_29: init %tuple.type.2 = converted %.loc14_28.1, %.loc14_28.6 [template = constants.%tuple]
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

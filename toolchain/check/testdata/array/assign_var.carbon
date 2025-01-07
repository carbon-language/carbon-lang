// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/assign_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/assign_var.carbon

var a: (i32, i32, i32) = (1, 2, 3);
var b: [i32; 3] = a;

// CHECK:STDOUT: --- assign_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.dc60: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %int_1.5b81: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc8: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1baa: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.37fc: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1a: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd1: <bound method> = bound_method %int_1.5b81, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b730: <specific function> = specific_function %Convert.bound.afd1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c605: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.f0b0: <bound method> = bound_method %int_2.ecc8, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e1: <specific function> = specific_function %Convert.bound.f0b0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.1667: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.0db0: <bound method> = bound_method %int_3.1baa, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4567: <specific function> = specific_function %Convert.bound.0db0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.25ba: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.dc60 = tuple_value (%int_1.c605, %int_2.1667, %int_3.25ba) [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1baa, %i32 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
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
// CHECK:STDOUT:   %a.var: ref %tuple.type.dc60 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.dc60 = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %array_type = var b
// CHECK:STDOUT:   %b: ref %array_type = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %int_2.loc11: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1baa]
// CHECK:STDOUT:   %.loc11_34.1: %tuple.type.37fc = tuple_literal (%int_1.loc11, %int_2.loc11, %int_3)
// CHECK:STDOUT:   %impl.elem0.loc11_34.1: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_34.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_34.1 [template = constants.%Convert.bound.afd1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_34.1: <specific function> = specific_function %Convert.bound.loc11_34.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b730]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.1: init %i32 = call %Convert.specific_fn.loc11_34.1(%int_1.loc11) [template = constants.%int_1.c605]
// CHECK:STDOUT:   %.loc11_34.2: init %i32 = converted %int_1.loc11, %int.convert_checked.loc11_34.1 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %tuple.elem0.loc11: ref %i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_34.3: init %i32 = initialize_from %.loc11_34.2 to %tuple.elem0.loc11 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %impl.elem0.loc11_34.2: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_34.2: <bound method> = bound_method %int_2.loc11, %impl.elem0.loc11_34.2 [template = constants.%Convert.bound.f0b0]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_34.2: <specific function> = specific_function %Convert.bound.loc11_34.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e1]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.2: init %i32 = call %Convert.specific_fn.loc11_34.2(%int_2.loc11) [template = constants.%int_2.1667]
// CHECK:STDOUT:   %.loc11_34.4: init %i32 = converted %int_2.loc11, %int.convert_checked.loc11_34.2 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %tuple.elem1.loc11: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_34.5: init %i32 = initialize_from %.loc11_34.4 to %tuple.elem1.loc11 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %impl.elem0.loc11_34.3: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_34.3: <bound method> = bound_method %int_3, %impl.elem0.loc11_34.3 [template = constants.%Convert.bound.0db0]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_34.3: <specific function> = specific_function %Convert.bound.loc11_34.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4567]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.3: init %i32 = call %Convert.specific_fn.loc11_34.3(%int_3) [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %.loc11_34.6: init %i32 = converted %int_3, %int.convert_checked.loc11_34.3 [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %tuple.elem2.loc11: ref %i32 = tuple_access file.%a.var, element2
// CHECK:STDOUT:   %.loc11_34.7: init %i32 = initialize_from %.loc11_34.6 to %tuple.elem2.loc11 [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %.loc11_34.8: init %tuple.type.dc60 = tuple_init (%.loc11_34.3, %.loc11_34.5, %.loc11_34.7) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_35: init %tuple.type.dc60 = converted %.loc11_34.1, %.loc11_34.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_35
// CHECK:STDOUT:   %a.ref: ref %tuple.type.dc60 = name_ref a, file.%a
// CHECK:STDOUT:   %tuple.elem0.loc12: ref %i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc12_19.1: %i32 = bind_value %tuple.elem0.loc12
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc12_19.2: ref %i32 = array_index file.%b.var, %int_0
// CHECK:STDOUT:   %.loc12_19.3: init %i32 = initialize_from %.loc12_19.1 to %.loc12_19.2
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %a.ref, element1
// CHECK:STDOUT:   %.loc12_19.4: %i32 = bind_value %tuple.elem1.loc12
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %.loc12_19.5: ref %i32 = array_index file.%b.var, %int_1.loc12
// CHECK:STDOUT:   %.loc12_19.6: init %i32 = initialize_from %.loc12_19.4 to %.loc12_19.5
// CHECK:STDOUT:   %tuple.elem2.loc12: ref %i32 = tuple_access %a.ref, element2
// CHECK:STDOUT:   %.loc12_19.7: %i32 = bind_value %tuple.elem2.loc12
// CHECK:STDOUT:   %int_2.loc12: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %.loc12_19.8: ref %i32 = array_index file.%b.var, %int_2.loc12
// CHECK:STDOUT:   %.loc12_19.9: init %i32 = initialize_from %.loc12_19.7 to %.loc12_19.8
// CHECK:STDOUT:   %.loc12_19.10: init %array_type = array_init (%.loc12_19.3, %.loc12_19.6, %.loc12_19.9) to file.%b.var
// CHECK:STDOUT:   %.loc12_20: init %array_type = converted %a.ref, %.loc12_19.10
// CHECK:STDOUT:   assign file.%b.var, %.loc12_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

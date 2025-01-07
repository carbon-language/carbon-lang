// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/tuple_as_element.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/tuple_as_element.carbon

var x: {.a: i32, .b: (i32,)} = {.a = 1, .b = (2,)};
var y: {.a: i32, .b: (i32,)} = x;

// CHECK:STDOUT: --- tuple_as_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.0e4: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %struct_type.a.b.d03: type = struct_type {.a: %i32, .b: %tuple.type.0e4} [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %struct_type.a.b.057: type = struct_type {.a: Core.IntLiteral, .b: %tuple.type.985} [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.f0b: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e: <specific function> = specific_function %Convert.bound.f0b, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.166: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.0e4 = tuple_value (%int_2.166) [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b.d03 = struct_value (%int_1.c60, %tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.d03 = var x
// CHECK:STDOUT:   %x: ref %struct_type.a.b.d03 = bind_name x, %x.var
// CHECK:STDOUT:   %y.var: ref %struct_type.a.b.d03 = var y
// CHECK:STDOUT:   %y: ref %struct_type.a.b.d03 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_49.1: %tuple.type.985 = tuple_literal (%int_2)
// CHECK:STDOUT:   %.loc11_50.1: %struct_type.a.b.057 = struct_literal (%int_1, %.loc11_49.1)
// CHECK:STDOUT:   %impl.elem0.loc11_50: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_50: <bound method> = bound_method %int_1, %impl.elem0.loc11_50 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_50: <specific function> = specific_function %Convert.bound.loc11_50, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc11_50: init %i32 = call %Convert.specific_fn.loc11_50(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc11_50.2: init %i32 = converted %int_1, %int.convert_checked.loc11_50 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc11_50.3: ref %i32 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc11_50.4: init %i32 = initialize_from %.loc11_50.2 to %.loc11_50.3 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %impl.elem0.loc11_49: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_49: <bound method> = bound_method %int_2, %impl.elem0.loc11_49 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_49: <specific function> = specific_function %Convert.bound.loc11_49, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc11_49: init %i32 = call %Convert.specific_fn.loc11_49(%int_2) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc11_49.2: init %i32 = converted %int_2, %int.convert_checked.loc11_49 [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc11_50.5: ref %tuple.type.0e4 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_49.3: init %tuple.type.0e4 = tuple_init (%.loc11_49.2) to %.loc11_50.5 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_50.6: init %tuple.type.0e4 = converted %.loc11_49.1, %.loc11_49.3 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_50.7: init %tuple.type.0e4 = initialize_from %.loc11_50.6 to %.loc11_50.5 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_50.8: init %struct_type.a.b.d03 = struct_init (%.loc11_50.4, %.loc11_50.7) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_51: init %struct_type.a.b.d03 = converted %.loc11_50.1, %.loc11_50.8 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_51
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.d03 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_32.1: ref %i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc12_32.2: %i32 = bind_value %.loc12_32.1
// CHECK:STDOUT:   %.loc12_32.3: ref %i32 = struct_access file.%y.var, element0
// CHECK:STDOUT:   %.loc12_32.4: init %i32 = initialize_from %.loc12_32.2 to %.loc12_32.3
// CHECK:STDOUT:   %.loc12_32.5: ref %tuple.type.0e4 = struct_access %x.ref, element1
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc12_32.5, element0
// CHECK:STDOUT:   %.loc12_32.6: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   %.loc12_32.7: ref %tuple.type.0e4 = struct_access file.%y.var, element1
// CHECK:STDOUT:   %.loc12_32.8: init %tuple.type.0e4 = tuple_init (%.loc12_32.6) to %.loc12_32.7
// CHECK:STDOUT:   %.loc12_32.9: init %tuple.type.0e4 = converted %.loc12_32.5, %.loc12_32.8
// CHECK:STDOUT:   %.loc12_32.10: init %tuple.type.0e4 = initialize_from %.loc12_32.9 to %.loc12_32.7
// CHECK:STDOUT:   %.loc12_32.11: init %struct_type.a.b.d03 = struct_init (%.loc12_32.4, %.loc12_32.10) to file.%y.var
// CHECK:STDOUT:   %.loc12_33: init %struct_type.a.b.d03 = converted %x.ref, %.loc12_32.11
// CHECK:STDOUT:   assign file.%y.var, %.loc12_33
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

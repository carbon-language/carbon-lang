// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/two_entries.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/two_entries.carbon

let v: {.a: i32, .b: i32} = {.a = 1, .b = 2};
let w: {.a: i32, .b: i32} = v;

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
var y: {.a: i32, .b: i32} = x;

// CHECK:STDOUT: --- two_entries.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.a.b.3de: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
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
// CHECK:STDOUT:   %struct: %struct_type.a.b.3de = struct_value (%int_1.c60, %int_2.166) [template]
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
// CHECK:STDOUT:     .v = @__global_init.%v
// CHECK:STDOUT:     .w = @__global_init.%w
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.3de = var x
// CHECK:STDOUT:   %x: ref %struct_type.a.b.3de = bind_name x, %x.var
// CHECK:STDOUT:   %y.var: ref %struct_type.a.b.3de = var y
// CHECK:STDOUT:   %y: ref %struct_type.a.b.3de = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc11: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_44.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc11, %int_2.loc11)
// CHECK:STDOUT:   %impl.elem0.loc11_44.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_44.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_44.1 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_44.1: <specific function> = specific_function %Convert.bound.loc11_44.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc11_44.1: init %i32 = call %Convert.specific_fn.loc11_44.1(%int_1.loc11) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc11_44.2: %i32 = value_of_initializer %int.convert_checked.loc11_44.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc11_44.3: %i32 = converted %int_1.loc11, %.loc11_44.2 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %impl.elem0.loc11_44.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_44.2: <bound method> = bound_method %int_2.loc11, %impl.elem0.loc11_44.2 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_44.2: <specific function> = specific_function %Convert.bound.loc11_44.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc11_44.2: init %i32 = call %Convert.specific_fn.loc11_44.2(%int_2.loc11) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc11_44.4: %i32 = value_of_initializer %int.convert_checked.loc11_44.2 [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc11_44.5: %i32 = converted %int_2.loc11, %.loc11_44.4 [template = constants.%int_2.166]
// CHECK:STDOUT:   %struct: %struct_type.a.b.3de = struct_value (%.loc11_44.3, %.loc11_44.5) [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_45: %struct_type.a.b.3de = converted %.loc11_44.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %v: %struct_type.a.b.3de = bind_name v, %.loc11_45
// CHECK:STDOUT:   %v.ref: %struct_type.a.b.3de = name_ref v, %v
// CHECK:STDOUT:   %w: %struct_type.a.b.3de = bind_name w, %v.ref
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc14: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc14_44.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc14, %int_2.loc14)
// CHECK:STDOUT:   %impl.elem0.loc14_44.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc14_44.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_44.1 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_44.1: <specific function> = specific_function %Convert.bound.loc14_44.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc14_44.1: init %i32 = call %Convert.specific_fn.loc14_44.1(%int_1.loc14) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc14_44.2: init %i32 = converted %int_1.loc14, %int.convert_checked.loc14_44.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc14_44.3: ref %i32 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_44.4: init %i32 = initialize_from %.loc14_44.2 to %.loc14_44.3 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %impl.elem0.loc14_44.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc14_44.2: <bound method> = bound_method %int_2.loc14, %impl.elem0.loc14_44.2 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_44.2: <specific function> = specific_function %Convert.bound.loc14_44.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc14_44.2: init %i32 = call %Convert.specific_fn.loc14_44.2(%int_2.loc14) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc14_44.5: init %i32 = converted %int_2.loc14, %int.convert_checked.loc14_44.2 [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc14_44.6: ref %i32 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_44.7: init %i32 = initialize_from %.loc14_44.5 to %.loc14_44.6 [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc14_44.8: init %struct_type.a.b.3de = struct_init (%.loc14_44.4, %.loc14_44.7) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc14_45: init %struct_type.a.b.3de = converted %.loc14_44.1, %.loc14_44.8 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_45
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.3de = name_ref x, file.%x
// CHECK:STDOUT:   %.loc15_29.1: ref %i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc15_29.2: %i32 = bind_value %.loc15_29.1
// CHECK:STDOUT:   %.loc15_29.3: ref %i32 = struct_access file.%y.var, element0
// CHECK:STDOUT:   %.loc15_29.4: init %i32 = initialize_from %.loc15_29.2 to %.loc15_29.3
// CHECK:STDOUT:   %.loc15_29.5: ref %i32 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc15_29.6: %i32 = bind_value %.loc15_29.5
// CHECK:STDOUT:   %.loc15_29.7: ref %i32 = struct_access file.%y.var, element1
// CHECK:STDOUT:   %.loc15_29.8: init %i32 = initialize_from %.loc15_29.6 to %.loc15_29.7
// CHECK:STDOUT:   %.loc15_29.9: init %struct_type.a.b.3de = struct_init (%.loc15_29.4, %.loc15_29.8) to file.%y.var
// CHECK:STDOUT:   %.loc15_30: init %struct_type.a.b.3de = converted %x.ref, %.loc15_29.9
// CHECK:STDOUT:   assign file.%y.var, %.loc15_30
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

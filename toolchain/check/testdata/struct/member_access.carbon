// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/member_access.carbon

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
var y: i32 = x.b;
var z: i32 = y;

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.a.b.95f: type = struct_type {.a: f64, .b: %i32} [template]
// CHECK:STDOUT:   %float: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.a.b.fbb: type = struct_type {.a: f64, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b.95f = struct_value (%float, %int_1.c60) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Float = %import_ref.20bd
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
// CHECK:STDOUT:     .z = %z
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.95f = var x
// CHECK:STDOUT:   %x: ref %struct_type.a.b.95f = bind_name x, %x.var
// CHECK:STDOUT:   %y.var: ref %i32 = var y
// CHECK:STDOUT:   %y: ref %i32 = bind_name y, %y.var
// CHECK:STDOUT:   %z.var: ref %i32 = var z
// CHECK:STDOUT:   %z: ref %i32 = bind_name z, %z.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %float: f64 = float_literal 0 [template = constants.%float]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_46.1: %struct_type.a.b.fbb = struct_literal (%float, %int_1)
// CHECK:STDOUT:   %.loc11_46.2: ref f64 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc11_46.3: init f64 = initialize_from %float to %.loc11_46.2 [template = constants.%float]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc11_46.4: init %i32 = converted %int_1, %int.convert_checked [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc11_46.5: ref %i32 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_46.6: init %i32 = initialize_from %.loc11_46.4 to %.loc11_46.5 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc11_46.7: init %struct_type.a.b.95f = struct_init (%.loc11_46.3, %.loc11_46.6) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_47: init %struct_type.a.b.95f = converted %.loc11_46.1, %.loc11_46.7 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_47
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.95f = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_15.1: ref %i32 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc12_15.2: %i32 = bind_value %.loc12_15.1
// CHECK:STDOUT:   assign file.%y.var, %.loc12_15.2
// CHECK:STDOUT:   %y.ref: ref %i32 = name_ref y, file.%y
// CHECK:STDOUT:   %.loc13: %i32 = bind_value %y.ref
// CHECK:STDOUT:   assign file.%z.var, %.loc13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

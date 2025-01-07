// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/element_access.carbon

var a: (i32,) = (12,);
var b: (i32,) = a;
var c: i32 = b.0;

// CHECK:STDOUT: --- element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.0e4: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_12.6a3, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.43d: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.0e4 = tuple_value (%int_12.43d) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %a.var: ref %tuple.type.0e4 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.0e4 = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %tuple.type.0e4 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.0e4 = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref %i32 = var c
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.6a3]
// CHECK:STDOUT:   %.loc11_21.1: %tuple.type.985 = tuple_literal (%int_12)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_12, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_12) [template = constants.%int_12.43d]
// CHECK:STDOUT:   %.loc11_21.2: init %i32 = converted %int_12, %int.convert_checked [template = constants.%int_12.43d]
// CHECK:STDOUT:   %.loc11_21.3: init %tuple.type.0e4 = tuple_init (%.loc11_21.2) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_22: init %tuple.type.0e4 = converted %.loc11_21.1, %.loc11_21.3 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_22
// CHECK:STDOUT:   %a.ref: ref %tuple.type.0e4 = name_ref a, file.%a
// CHECK:STDOUT:   %tuple.elem0.loc12: ref %i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc12_17.1: %i32 = bind_value %tuple.elem0.loc12
// CHECK:STDOUT:   %.loc12_17.2: init %tuple.type.0e4 = tuple_init (%.loc12_17.1) to file.%b.var
// CHECK:STDOUT:   %.loc12_18: init %tuple.type.0e4 = converted %a.ref, %.loc12_17.2
// CHECK:STDOUT:   assign file.%b.var, %.loc12_18
// CHECK:STDOUT:   %b.ref: ref %tuple.type.0e4 = name_ref b, file.%b
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc13: ref %i32 = tuple_access %b.ref, element0
// CHECK:STDOUT:   %.loc13: %i32 = bind_value %tuple.elem0.loc13
// CHECK:STDOUT:   assign file.%c.var, %.loc13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

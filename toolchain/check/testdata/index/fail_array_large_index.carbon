// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_array_large_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_array_large_index.carbon

var a: [i32; 1] = (12,);

// CHECK:STDERR: fail_array_large_index.carbon:[[@LINE+4]]:16: error: array index `1` is past the end of type `[i32; 1]` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a[1];
// CHECK:STDERR:                ^
// CHECK:STDERR:
var b: i32 = a[1];

// CHECK:STDERR: fail_array_large_index.carbon:[[@LINE+3]]:16: error: array index `2147483647` is past the end of type `[i32; 1]` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var c: i32 = a[0x7FFF_FFFF];
// CHECK:STDERR:                ^~~~~~~~~~~
var c: i32 = a[0x7FFF_FFFF];

// CHECK:STDOUT: --- fail_array_large_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %i32 [template]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.8aa: <bound method> = bound_method %int_12.6a3, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.545: <specific function> = specific_function %Convert.bound.8aa, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.43d: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_12.43d) [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2147483647.d89: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2147483647.d89, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.a96: <specific function> = specific_function %Convert.bound.ef9, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.2ba: %i32 = int_value 2147483647 [template]
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
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref %i32 = var c
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.6a3]
// CHECK:STDOUT:   %.loc11_23.1: %tuple.type = tuple_literal (%int_12)
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_12, %impl.elem0.loc11 [template = constants.%Convert.bound.8aa]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.545]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_12) [template = constants.%int_12.43d]
// CHECK:STDOUT:   %.loc11_23.2: init %i32 = converted %int_12, %int.convert_checked.loc11 [template = constants.%int_12.43d]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_23.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc11_23.4: init %i32 = initialize_from %.loc11_23.2 to %.loc11_23.3 [template = constants.%int_12.43d]
// CHECK:STDOUT:   %.loc11_23.5: init %array_type = array_init (%.loc11_23.4) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_24: init %array_type = converted %.loc11_23.1, %.loc11_23.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_24
// CHECK:STDOUT:   %a.ref.loc17: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc17: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc17: <bound method> = bound_method %int_1, %impl.elem0.loc17 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc17: <specific function> = specific_function %Convert.bound.loc17, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %Convert.specific_fn.loc17(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc17_16.1: %i32 = value_of_initializer %int.convert_checked.loc17 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc17_16.2: %i32 = converted %int_1, %.loc17_16.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc17_17.1: ref %i32 = array_index %a.ref.loc17, %.loc17_16.2 [template = <error>]
// CHECK:STDOUT:   %.loc17_17.2: %i32 = bind_value %.loc17_17.1
// CHECK:STDOUT:   assign file.%b.var, %.loc17_17.2
// CHECK:STDOUT:   %a.ref.loc22: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.d89]
// CHECK:STDOUT:   %int_32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc22: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc22: <bound method> = bound_method %int_2147483647, %impl.elem0.loc22 [template = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %Convert.specific_fn.loc22: <specific function> = specific_function %Convert.bound.loc22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.a96]
// CHECK:STDOUT:   %int.convert_checked.loc22: init %i32 = call %Convert.specific_fn.loc22(%int_2147483647) [template = constants.%int_2147483647.2ba]
// CHECK:STDOUT:   %.loc22_16.1: %i32 = value_of_initializer %int.convert_checked.loc22 [template = constants.%int_2147483647.2ba]
// CHECK:STDOUT:   %.loc22_16.2: %i32 = converted %int_2147483647, %.loc22_16.1 [template = constants.%int_2147483647.2ba]
// CHECK:STDOUT:   %.loc22_27.1: ref %i32 = array_index %a.ref.loc22, %.loc22_16.2 [template = <error>]
// CHECK:STDOUT:   %.loc22_27.2: %i32 = bind_value %.loc22_27.1
// CHECK:STDOUT:   assign file.%c.var, %.loc22_27.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

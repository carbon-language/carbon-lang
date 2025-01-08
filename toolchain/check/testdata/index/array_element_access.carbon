// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/array_element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/array_element_access.carbon

var a: [i32; 2] = (12, 24);
var b: i32 = 1;
var c: i32 = a[0];
var d: i32 = a[b];

// CHECK:STDOUT: --- array_element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [template]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_24.e3c: Core.IntLiteral = int_value 24 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.8aa: <bound method> = bound_method %int_12.6a3, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.545: <specific function> = specific_function %Convert.bound.8aa, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.43d: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.c54: <bound method> = bound_method %int_24.e3c, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4df: <specific function> = specific_function %Convert.bound.c54, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_24.6ea: %i32 = int_value 24 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_12.43d, %int_24.6ea) [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.b41: <bound method> = bound_method %int_0.5c6, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c35: <specific function> = specific_function %Convert.bound.b41, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.f61: %i32 = int_value 0 [template]
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
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref %i32 = var c
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var
// CHECK:STDOUT:   %d.var: ref %i32 = var d
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.6a3]
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [template = constants.%int_24.e3c]
// CHECK:STDOUT:   %.loc11_26.1: %tuple.type = tuple_literal (%int_12, %int_24)
// CHECK:STDOUT:   %impl.elem0.loc11_26.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_26.1: <bound method> = bound_method %int_12, %impl.elem0.loc11_26.1 [template = constants.%Convert.bound.8aa]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_26.1: <specific function> = specific_function %Convert.bound.loc11_26.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.545]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.1: init %i32 = call %Convert.specific_fn.loc11_26.1(%int_12) [template = constants.%int_12.43d]
// CHECK:STDOUT:   %.loc11_26.2: init %i32 = converted %int_12, %int.convert_checked.loc11_26.1 [template = constants.%int_12.43d]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc11_26.3: ref %i32 = array_index file.%a.var, %int_0.loc11
// CHECK:STDOUT:   %.loc11_26.4: init %i32 = initialize_from %.loc11_26.2 to %.loc11_26.3 [template = constants.%int_12.43d]
// CHECK:STDOUT:   %impl.elem0.loc11_26.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_26.2: <bound method> = bound_method %int_24, %impl.elem0.loc11_26.2 [template = constants.%Convert.bound.c54]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_26.2: <specific function> = specific_function %Convert.bound.loc11_26.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4df]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.2: init %i32 = call %Convert.specific_fn.loc11_26.2(%int_24) [template = constants.%int_24.6ea]
// CHECK:STDOUT:   %.loc11_26.5: init %i32 = converted %int_24, %int.convert_checked.loc11_26.2 [template = constants.%int_24.6ea]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_26.6: ref %i32 = array_index file.%a.var, %int_1.loc11
// CHECK:STDOUT:   %.loc11_26.7: init %i32 = initialize_from %.loc11_26.5 to %.loc11_26.6 [template = constants.%int_24.6ea]
// CHECK:STDOUT:   %.loc11_26.8: init %array_type = array_init (%.loc11_26.4, %.loc11_26.7) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_27: init %array_type = converted %.loc11_26.1, %.loc11_26.8 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_27
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc12: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc12: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc12: <specific function> = specific_function %Convert.bound.loc12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %Convert.specific_fn.loc12(%int_1.loc12) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc12: init %i32 = converted %int_1.loc12, %int.convert_checked.loc12 [template = constants.%int_1.c60]
// CHECK:STDOUT:   assign file.%b.var, %.loc12
// CHECK:STDOUT:   %a.ref.loc13: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %int_0.loc13: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc13: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc13: <bound method> = bound_method %int_0.loc13, %impl.elem0.loc13 [template = constants.%Convert.bound.b41]
// CHECK:STDOUT:   %Convert.specific_fn.loc13: <specific function> = specific_function %Convert.bound.loc13, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c35]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %Convert.specific_fn.loc13(%int_0.loc13) [template = constants.%int_0.f61]
// CHECK:STDOUT:   %.loc13_16.1: %i32 = value_of_initializer %int.convert_checked.loc13 [template = constants.%int_0.f61]
// CHECK:STDOUT:   %.loc13_16.2: %i32 = converted %int_0.loc13, %.loc13_16.1 [template = constants.%int_0.f61]
// CHECK:STDOUT:   %.loc13_17.1: ref %i32 = array_index %a.ref.loc13, %.loc13_16.2
// CHECK:STDOUT:   %.loc13_17.2: %i32 = bind_value %.loc13_17.1
// CHECK:STDOUT:   assign file.%c.var, %.loc13_17.2
// CHECK:STDOUT:   %a.ref.loc14: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %b.ref: ref %i32 = name_ref b, file.%b
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_16: %i32 = bind_value %b.ref
// CHECK:STDOUT:   %.loc14_17.1: ref %i32 = array_index %a.ref.loc14, %.loc14_16
// CHECK:STDOUT:   %.loc14_17.2: %i32 = bind_value %.loc14_17.1
// CHECK:STDOUT:   assign file.%d.var, %.loc14_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

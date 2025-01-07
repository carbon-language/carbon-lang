// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/nested_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/nested_tuple.carbon

var x: ((i32, i32), i32) = ((12, 76), 6);

// CHECK:STDOUT: --- nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.471: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.5a7: type = tuple_type (%tuple.type.471, %i32) [template]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_76.872: Core.IntLiteral = int_value 76 [template]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.acf: type = tuple_type (%tuple.type.f94, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.8aa: <bound method> = bound_method %int_12.6a3, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.545: <specific function> = specific_function %Convert.bound.8aa, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.43d: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %Convert.bound.169: <bound method> = bound_method %int_76.872, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5a1: <specific function> = specific_function %Convert.bound.169, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_76.fed: %i32 = int_value 76 [template]
// CHECK:STDOUT:   %tuple.664: %tuple.type.471 = tuple_value (%int_12.43d, %int_76.fed) [template]
// CHECK:STDOUT:   %Convert.bound.b1c: <bound method> = bound_method %int_6.462, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b26: <specific function> = specific_function %Convert.bound.b1c, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.b03: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %tuple.1cb: %tuple.type.5a7 = tuple_value (%tuple.664, %int_6.b03) [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %x.var: ref %tuple.type.5a7 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.5a7 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.6a3]
// CHECK:STDOUT:   %int_76: Core.IntLiteral = int_value 76 [template = constants.%int_76.872]
// CHECK:STDOUT:   %.loc11_36.1: %tuple.type.f94 = tuple_literal (%int_12, %int_76)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.462]
// CHECK:STDOUT:   %.loc11_40.1: %tuple.type.acf = tuple_literal (%.loc11_36.1, %int_6)
// CHECK:STDOUT:   %impl.elem0.loc11_36.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_36.1: <bound method> = bound_method %int_12, %impl.elem0.loc11_36.1 [template = constants.%Convert.bound.8aa]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_36.1: <specific function> = specific_function %Convert.bound.loc11_36.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.545]
// CHECK:STDOUT:   %int.convert_checked.loc11_36.1: init %i32 = call %Convert.specific_fn.loc11_36.1(%int_12) [template = constants.%int_12.43d]
// CHECK:STDOUT:   %.loc11_36.2: init %i32 = converted %int_12, %int.convert_checked.loc11_36.1 [template = constants.%int_12.43d]
// CHECK:STDOUT:   %tuple.elem0.loc11_40: ref %tuple.type.471 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc11_36: ref %i32 = tuple_access %tuple.elem0.loc11_40, element0
// CHECK:STDOUT:   %.loc11_36.3: init %i32 = initialize_from %.loc11_36.2 to %tuple.elem0.loc11_36 [template = constants.%int_12.43d]
// CHECK:STDOUT:   %impl.elem0.loc11_36.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_36.2: <bound method> = bound_method %int_76, %impl.elem0.loc11_36.2 [template = constants.%Convert.bound.169]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_36.2: <specific function> = specific_function %Convert.bound.loc11_36.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5a1]
// CHECK:STDOUT:   %int.convert_checked.loc11_36.2: init %i32 = call %Convert.specific_fn.loc11_36.2(%int_76) [template = constants.%int_76.fed]
// CHECK:STDOUT:   %.loc11_36.4: init %i32 = converted %int_76, %int.convert_checked.loc11_36.2 [template = constants.%int_76.fed]
// CHECK:STDOUT:   %tuple.elem1.loc11_36: ref %i32 = tuple_access %tuple.elem0.loc11_40, element1
// CHECK:STDOUT:   %.loc11_36.5: init %i32 = initialize_from %.loc11_36.4 to %tuple.elem1.loc11_36 [template = constants.%int_76.fed]
// CHECK:STDOUT:   %.loc11_36.6: init %tuple.type.471 = tuple_init (%.loc11_36.3, %.loc11_36.5) to %tuple.elem0.loc11_40 [template = constants.%tuple.664]
// CHECK:STDOUT:   %.loc11_40.2: init %tuple.type.471 = converted %.loc11_36.1, %.loc11_36.6 [template = constants.%tuple.664]
// CHECK:STDOUT:   %impl.elem0.loc11_40: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_40: <bound method> = bound_method %int_6, %impl.elem0.loc11_40 [template = constants.%Convert.bound.b1c]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_40: <specific function> = specific_function %Convert.bound.loc11_40, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b26]
// CHECK:STDOUT:   %int.convert_checked.loc11_40: init %i32 = call %Convert.specific_fn.loc11_40(%int_6) [template = constants.%int_6.b03]
// CHECK:STDOUT:   %.loc11_40.3: init %i32 = converted %int_6, %int.convert_checked.loc11_40 [template = constants.%int_6.b03]
// CHECK:STDOUT:   %tuple.elem1.loc11_40: ref %i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_40.4: init %i32 = initialize_from %.loc11_40.3 to %tuple.elem1.loc11_40 [template = constants.%int_6.b03]
// CHECK:STDOUT:   %.loc11_40.5: init %tuple.type.5a7 = tuple_init (%.loc11_40.2, %.loc11_40.4) to file.%x.var [template = constants.%tuple.1cb]
// CHECK:STDOUT:   %.loc11_41: init %tuple.type.5a7 = converted %.loc11_40.1, %.loc11_40.5 [template = constants.%tuple.1cb]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_41
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

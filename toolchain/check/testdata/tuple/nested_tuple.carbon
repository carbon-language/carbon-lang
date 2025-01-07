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
// CHECK:STDOUT:   %tuple.type.4719: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.5a75: type = tuple_type (%tuple.type.4719, %i32) [template]
// CHECK:STDOUT:   %int_12.6a39: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_76.872e: Core.IntLiteral = int_value 76 [template]
// CHECK:STDOUT:   %tuple.type.f949: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_6.4624: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.acff: type = tuple_type (%tuple.type.f949, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1a: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.8aa1: <bound method> = bound_method %int_12.6a39, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5452: <specific function> = specific_function %Convert.bound.8aa1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.43d2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %Convert.bound.1693: <bound method> = bound_method %int_76.872e, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5a19: <specific function> = specific_function %Convert.bound.1693, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_76.fed8: %i32 = int_value 76 [template]
// CHECK:STDOUT:   %tuple.6646: %tuple.type.4719 = tuple_value (%int_12.43d2, %int_76.fed8) [template]
// CHECK:STDOUT:   %Convert.bound.b1cb: <bound method> = bound_method %int_6.4624, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b269: <specific function> = specific_function %Convert.bound.b1cb, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.b036: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %tuple.1cb1: %tuple.type.5a75 = tuple_value (%tuple.6646, %int_6.b036) [template]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %x.var: ref %tuple.type.5a75 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.5a75 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.6a39]
// CHECK:STDOUT:   %int_76: Core.IntLiteral = int_value 76 [template = constants.%int_76.872e]
// CHECK:STDOUT:   %.loc11_36.1: %tuple.type.f949 = tuple_literal (%int_12, %int_76)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.4624]
// CHECK:STDOUT:   %.loc11_40.1: %tuple.type.acff = tuple_literal (%.loc11_36.1, %int_6)
// CHECK:STDOUT:   %impl.elem0.loc11_36.1: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_36.1: <bound method> = bound_method %int_12, %impl.elem0.loc11_36.1 [template = constants.%Convert.bound.8aa1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_36.1: <specific function> = specific_function %Convert.bound.loc11_36.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5452]
// CHECK:STDOUT:   %int.convert_checked.loc11_36.1: init %i32 = call %Convert.specific_fn.loc11_36.1(%int_12) [template = constants.%int_12.43d2]
// CHECK:STDOUT:   %.loc11_36.2: init %i32 = converted %int_12, %int.convert_checked.loc11_36.1 [template = constants.%int_12.43d2]
// CHECK:STDOUT:   %tuple.elem0.loc11_40: ref %tuple.type.4719 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc11_36: ref %i32 = tuple_access %tuple.elem0.loc11_40, element0
// CHECK:STDOUT:   %.loc11_36.3: init %i32 = initialize_from %.loc11_36.2 to %tuple.elem0.loc11_36 [template = constants.%int_12.43d2]
// CHECK:STDOUT:   %impl.elem0.loc11_36.2: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_36.2: <bound method> = bound_method %int_76, %impl.elem0.loc11_36.2 [template = constants.%Convert.bound.1693]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_36.2: <specific function> = specific_function %Convert.bound.loc11_36.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5a19]
// CHECK:STDOUT:   %int.convert_checked.loc11_36.2: init %i32 = call %Convert.specific_fn.loc11_36.2(%int_76) [template = constants.%int_76.fed8]
// CHECK:STDOUT:   %.loc11_36.4: init %i32 = converted %int_76, %int.convert_checked.loc11_36.2 [template = constants.%int_76.fed8]
// CHECK:STDOUT:   %tuple.elem1.loc11_36: ref %i32 = tuple_access %tuple.elem0.loc11_40, element1
// CHECK:STDOUT:   %.loc11_36.5: init %i32 = initialize_from %.loc11_36.4 to %tuple.elem1.loc11_36 [template = constants.%int_76.fed8]
// CHECK:STDOUT:   %.loc11_36.6: init %tuple.type.4719 = tuple_init (%.loc11_36.3, %.loc11_36.5) to %tuple.elem0.loc11_40 [template = constants.%tuple.6646]
// CHECK:STDOUT:   %.loc11_40.2: init %tuple.type.4719 = converted %.loc11_36.1, %.loc11_36.6 [template = constants.%tuple.6646]
// CHECK:STDOUT:   %impl.elem0.loc11_40: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc11_40: <bound method> = bound_method %int_6, %impl.elem0.loc11_40 [template = constants.%Convert.bound.b1cb]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_40: <specific function> = specific_function %Convert.bound.loc11_40, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b269]
// CHECK:STDOUT:   %int.convert_checked.loc11_40: init %i32 = call %Convert.specific_fn.loc11_40(%int_6) [template = constants.%int_6.b036]
// CHECK:STDOUT:   %.loc11_40.3: init %i32 = converted %int_6, %int.convert_checked.loc11_40 [template = constants.%int_6.b036]
// CHECK:STDOUT:   %tuple.elem1.loc11_40: ref %i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_40.4: init %i32 = initialize_from %.loc11_40.3 to %tuple.elem1.loc11_40 [template = constants.%int_6.b036]
// CHECK:STDOUT:   %.loc11_40.5: init %tuple.type.5a75 = tuple_init (%.loc11_40.2, %.loc11_40.4) to file.%x.var [template = constants.%tuple.1cb1]
// CHECK:STDOUT:   %.loc11_41: init %tuple.type.5a75 = converted %.loc11_40.1, %.loc11_40.5 [template = constants.%tuple.1cb1]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_41
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

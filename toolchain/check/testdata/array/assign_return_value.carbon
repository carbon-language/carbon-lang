// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/assign_return_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/assign_return_value.carbon

fn F() -> (i32,) { return (0,); }

fn Run() {
  var t: [i32; 1] = F();
}

// CHECK:STDOUT: --- assign_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.0e4: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.f61: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.0e4 = tuple_value (%int_0.f61) [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %i32 [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %tuple.type.0e4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.0e4 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_16.1: %tuple.type.85c = tuple_literal (%i32)
// CHECK:STDOUT:     %.loc11_16.2: type = converted %.loc11_16.1, constants.%tuple.type.0e4 [template = constants.%tuple.type.0e4]
// CHECK:STDOUT:     %return.param: ref %tuple.type.0e4 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.0e4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %tuple.type.0e4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc11_30.1: %tuple.type.985 = tuple_literal (%int_0)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.f61]
// CHECK:STDOUT:   %.loc11_30.2: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.f61]
// CHECK:STDOUT:   %.loc11_30.3: %i32 = converted %int_0, %.loc11_30.2 [template = constants.%int_0.f61]
// CHECK:STDOUT:   %tuple: %tuple.type.0e4 = tuple_value (%.loc11_30.3) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_31: %tuple.type.0e4 = converted %.loc11_30.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc11_31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %t.var: ref %array_type = var t
// CHECK:STDOUT:   %t: ref %array_type = bind_name t, %t.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %tuple.type.0e4 = call %F.ref()
// CHECK:STDOUT:   %.loc14_23.1: ref %tuple.type.0e4 = temporary_storage
// CHECK:STDOUT:   %.loc14_23.2: ref %tuple.type.0e4 = temporary %.loc14_23.1, %F.call
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc14_23.2, element0
// CHECK:STDOUT:   %.loc14_23.3: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc14_23.4: ref %i32 = array_index %t.var, %int_0
// CHECK:STDOUT:   %.loc14_23.5: init %i32 = initialize_from %.loc14_23.3 to %.loc14_23.4
// CHECK:STDOUT:   %.loc14_23.6: init %array_type = array_init (%.loc14_23.5) to %t.var
// CHECK:STDOUT:   %.loc14_24: init %array_type = converted %F.call, %.loc14_23.6
// CHECK:STDOUT:   assign %t.var, %.loc14_24
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/partially_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/partially_const.carbon

fn Make(n: i32) -> {.a: i32, .b: i32, .c: i32} {
  return {.a = 0, .b = n, .c = 0};
}

// CHECK:STDOUT: --- partially_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.0b6: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.1ee: type = struct_type {.a: Core.IntLiteral, .b: %i32, .c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: %struct_type.a.b.c.0b6 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %struct_type.a.b.c.0b6 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_43: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_43: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete = constants.%struct_type.a.b.c.0b6]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %i32.loc11_12 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %struct_type.a.b.c.0b6 = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.a.b.c.0b6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make(%n.param_patt: %i32) -> %return.param_patt: %struct_type.a.b.c.0b6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc12_16: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %int_0.loc12_32: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc12_33.1: %struct_type.a.b.c.1ee = struct_literal (%int_0.loc12_16, %n.ref, %int_0.loc12_32)
// CHECK:STDOUT:   %impl.elem0.loc12_33.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_33.1: <bound method> = bound_method %int_0.loc12_16, %impl.elem0.loc12_33.1 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc12_33.1: <specific function> = specific_function %impl.elem0.loc12_33.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_33.2: <bound method> = bound_method %int_0.loc12_16, %specific_fn.loc12_33.1 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc12_33.1: init %i32 = call %bound_method.loc12_33.2(%int_0.loc12_16) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc12_33.2: init %i32 = converted %int_0.loc12_16, %int.convert_checked.loc12_33.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc12_33.3: ref %i32 = struct_access %return, element0
// CHECK:STDOUT:   %.loc12_33.4: init %i32 = initialize_from %.loc12_33.2 to %.loc12_33.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc12_33.5: ref %i32 = struct_access %return, element1
// CHECK:STDOUT:   %.loc12_33.6: init %i32 = initialize_from %n.ref to %.loc12_33.5
// CHECK:STDOUT:   %impl.elem0.loc12_33.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_33.3: <bound method> = bound_method %int_0.loc12_32, %impl.elem0.loc12_33.2 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc12_33.2: <specific function> = specific_function %impl.elem0.loc12_33.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_33.4: <bound method> = bound_method %int_0.loc12_32, %specific_fn.loc12_33.2 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc12_33.2: init %i32 = call %bound_method.loc12_33.4(%int_0.loc12_32) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc12_33.7: init %i32 = converted %int_0.loc12_32, %int.convert_checked.loc12_33.2 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc12_33.8: ref %i32 = struct_access %return, element2
// CHECK:STDOUT:   %.loc12_33.9: init %i32 = initialize_from %.loc12_33.7 to %.loc12_33.8 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc12_33.10: init %struct_type.a.b.c.0b6 = struct_init (%.loc12_33.4, %.loc12_33.6, %.loc12_33.9) to %return
// CHECK:STDOUT:   %.loc12_34: init %struct_type.a.b.c.0b6 = converted %.loc12_33.1, %.loc12_33.10
// CHECK:STDOUT:   return %.loc12_34 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/nested.carbon

fn F(a: bool, b: bool, c: bool) -> i32 {
  return if a then if b then 1 else 2 else if c then 3 else 4;
}

// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.f0b: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e: <specific function> = specific_function %Convert.bound.f0b, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.166: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.0db: <bound method> = bound_method %int_3.1ba, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.456: <specific function> = specific_function %Convert.bound.0db, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.25b: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.fe5: <bound method> = bound_method %int_4.0c1, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c6f: <specific function> = specific_function %Convert.bound.fe5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2ec: %i32 = int_value 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.f7c
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: bool = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: bool = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %c.patt: bool = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: bool = value_param_pattern %c.patt, runtime_param2
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %.loc11_9.1: type = splice_block %.loc11_9.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type.loc11_9: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc11_9.2: type = value_of_initializer %bool.make_type.loc11_9 [template = bool]
// CHECK:STDOUT:       %.loc11_9.3: type = converted %bool.make_type.loc11_9, %.loc11_9.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: bool = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %.loc11_18.1: type = splice_block %.loc11_18.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type.loc11_18: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc11_18.2: type = value_of_initializer %bool.make_type.loc11_18 [template = bool]
// CHECK:STDOUT:       %.loc11_18.3: type = converted %bool.make_type.loc11_18, %.loc11_18.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: bool = value_param runtime_param2
// CHECK:STDOUT:     %.loc11_27.1: type = splice_block %.loc11_27.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type.loc11_27: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc11_27.2: type = value_of_initializer %bool.make_type.loc11_27 [template = bool]
// CHECK:STDOUT:       %.loc11_27.3: type = converted %bool.make_type.loc11_27, %.loc11_27.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: bool = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param3
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: bool, %b.param_patt: bool, %c.param_patt: bool) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %a
// CHECK:STDOUT:   if %a.ref br !if.expr.then.loc12_10 else br !if.expr.else.loc12_10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12_10:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then.loc12_20 else br !if.expr.else.loc12_20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12_20:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc12_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc12_25: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc12_25: <bound method> = bound_method %int_1, %impl.elem0.loc12_25 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_25: <specific function> = specific_function %Convert.bound.loc12_25, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc12_25: init %i32 = call %Convert.specific_fn.loc12_25(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc12_25.1: %i32 = value_of_initializer %int.convert_checked.loc12_25 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc12_25.2: %i32 = converted %int_1, %.loc12_25.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   br !if.expr.result.loc12_20(%.loc12_25.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12_20:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc12_32: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc12_32: <bound method> = bound_method %int_2, %impl.elem0.loc12_32 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_32: <specific function> = specific_function %Convert.bound.loc12_32, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc12_32: init %i32 = call %Convert.specific_fn.loc12_32(%int_2) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc12_32.1: %i32 = value_of_initializer %int.convert_checked.loc12_32 [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc12_32.2: %i32 = converted %int_2, %.loc12_32.1 [template = constants.%int_2.166]
// CHECK:STDOUT:   br !if.expr.result.loc12_20(%.loc12_32.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12_20:
// CHECK:STDOUT:   %.loc12_20: %i32 = block_arg !if.expr.result.loc12_20
// CHECK:STDOUT:   br !if.expr.result.loc12_10(%.loc12_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12_10:
// CHECK:STDOUT:   %c.ref: bool = name_ref c, %c
// CHECK:STDOUT:   if %c.ref br !if.expr.then.loc12_44 else br !if.expr.else.loc12_44
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12_44:
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_32.loc12_49: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12_49: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc12_49: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc12_49: <bound method> = bound_method %int_3, %impl.elem0.loc12_49 [template = constants.%Convert.bound.0db]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_49: <specific function> = specific_function %Convert.bound.loc12_49, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.456]
// CHECK:STDOUT:   %int.convert_checked.loc12_49: init %i32 = call %Convert.specific_fn.loc12_49(%int_3) [template = constants.%int_3.25b]
// CHECK:STDOUT:   %.loc12_49.1: %i32 = value_of_initializer %int.convert_checked.loc12_49 [template = constants.%int_3.25b]
// CHECK:STDOUT:   %.loc12_49.2: %i32 = converted %int_3, %.loc12_49.1 [template = constants.%int_3.25b]
// CHECK:STDOUT:   br !if.expr.result.loc12_44(%.loc12_49.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12_44:
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc12_56: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc12_56: <bound method> = bound_method %int_4, %impl.elem0.loc12_56 [template = constants.%Convert.bound.fe5]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_56: <specific function> = specific_function %Convert.bound.loc12_56, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c6f]
// CHECK:STDOUT:   %int.convert_checked.loc12_56: init %i32 = call %Convert.specific_fn.loc12_56(%int_4) [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %.loc12_56.1: %i32 = value_of_initializer %int.convert_checked.loc12_56 [template = constants.%int_4.2ec]
// CHECK:STDOUT:   %.loc12_56.2: %i32 = converted %int_4, %.loc12_56.1 [template = constants.%int_4.2ec]
// CHECK:STDOUT:   br !if.expr.result.loc12_44(%.loc12_56.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12_44:
// CHECK:STDOUT:   %.loc12_44: %i32 = block_arg !if.expr.result.loc12_44
// CHECK:STDOUT:   br !if.expr.result.loc12_10(%.loc12_44)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12_10:
// CHECK:STDOUT:   %.loc12_10: %i32 = block_arg !if.expr.result.loc12_10
// CHECK:STDOUT:   return %.loc12_10
// CHECK:STDOUT: }
// CHECK:STDOUT:

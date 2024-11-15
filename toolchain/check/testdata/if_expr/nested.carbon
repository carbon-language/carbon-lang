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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.28: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.28, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.31, %Convert.15 [template]
// CHECK:STDOUT:   %.33: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.34, %Convert.15 [template]
// CHECK:STDOUT:   %.36: i32 = int_value 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
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
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc11_9: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc11_9.1: type = value_of_initializer %bool.make_type.loc11_9 [template = bool]
// CHECK:STDOUT:     %.loc11_9.2: type = converted %bool.make_type.loc11_9, %.loc11_9.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc11_18: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc11_18.1: type = value_of_initializer %bool.make_type.loc11_18 [template = bool]
// CHECK:STDOUT:     %.loc11_18.2: type = converted %bool.make_type.loc11_18, %.loc11_18.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc11_27: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc11_27.1: type = value_of_initializer %bool.make_type.loc11_27 [template = bool]
// CHECK:STDOUT:     %.loc11_27.2: type = converted %bool.make_type.loc11_27, %.loc11_27.1 [template = bool]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_36.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_36.2: type = converted %int.make_type_32, %.loc11_36.1 [template = i32]
// CHECK:STDOUT:     %a.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %a: bool = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: bool = value_param runtime_param2
// CHECK:STDOUT:     %c: bool = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param3
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: bool, %b.param_patt: bool, %c.param_patt: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %a
// CHECK:STDOUT:   if %a.ref br !if.expr.then.loc12_10 else br !if.expr.else.loc12_10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12_10:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then.loc12_20 else br !if.expr.else.loc12_20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12_20:
// CHECK:STDOUT:   %.loc12_30: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12_25.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_25.2: <bound method> = bound_method %.loc12_30, %.loc12_25.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc12_25: init i32 = call %.loc12_25.2(%.loc12_30) [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_25.3: i32 = value_of_initializer %int.convert_checked.loc12_25 [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_25.4: i32 = converted %.loc12_30, %.loc12_25.3 [template = constants.%.27]
// CHECK:STDOUT:   br !if.expr.result.loc12_20(%.loc12_25.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12_20:
// CHECK:STDOUT:   %.loc12_37: Core.IntLiteral = int_value 2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_32.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_32.2: <bound method> = bound_method %.loc12_37, %.loc12_32.1 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc12_32: init i32 = call %.loc12_32.2(%.loc12_37) [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_32.3: i32 = value_of_initializer %int.convert_checked.loc12_32 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_32.4: i32 = converted %.loc12_37, %.loc12_32.3 [template = constants.%.30]
// CHECK:STDOUT:   br !if.expr.result.loc12_20(%.loc12_32.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12_20:
// CHECK:STDOUT:   %.loc12_20: i32 = block_arg !if.expr.result.loc12_20
// CHECK:STDOUT:   br !if.expr.result.loc12_10(%.loc12_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12_10:
// CHECK:STDOUT:   %c.ref: bool = name_ref c, %c
// CHECK:STDOUT:   if %c.ref br !if.expr.then.loc12_44 else br !if.expr.else.loc12_44
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12_44:
// CHECK:STDOUT:   %.loc12_54: Core.IntLiteral = int_value 3 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_49.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_49.2: <bound method> = bound_method %.loc12_54, %.loc12_49.1 [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc12_49: init i32 = call %.loc12_49.2(%.loc12_54) [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_49.3: i32 = value_of_initializer %int.convert_checked.loc12_49 [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_49.4: i32 = converted %.loc12_54, %.loc12_49.3 [template = constants.%.33]
// CHECK:STDOUT:   br !if.expr.result.loc12_44(%.loc12_49.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12_44:
// CHECK:STDOUT:   %.loc12_61: Core.IntLiteral = int_value 4 [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_56.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_56.2: <bound method> = bound_method %.loc12_61, %.loc12_56.1 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc12_56: init i32 = call %.loc12_56.2(%.loc12_61) [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_56.3: i32 = value_of_initializer %int.convert_checked.loc12_56 [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_56.4: i32 = converted %.loc12_61, %.loc12_56.3 [template = constants.%.36]
// CHECK:STDOUT:   br !if.expr.result.loc12_44(%.loc12_56.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12_44:
// CHECK:STDOUT:   %.loc12_44: i32 = block_arg !if.expr.result.loc12_44
// CHECK:STDOUT:   br !if.expr.result.loc12_10(%.loc12_44)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12_10:
// CHECK:STDOUT:   %.loc12_10: i32 = block_arg !if.expr.result.loc12_10
// CHECK:STDOUT:   return %.loc12_10
// CHECK:STDOUT: }
// CHECK:STDOUT:

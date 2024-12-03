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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.16: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.17: <specific function> = specific_function %.16, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.18: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %.19: <specific function> = specific_function %.18, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int.3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.20: <bound method> = bound_method %int.3.1, %Convert.14 [template]
// CHECK:STDOUT:   %.21: <specific function> = specific_function %.20, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %int.4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.22: <bound method> = bound_method %int.4.1, %Convert.14 [template]
// CHECK:STDOUT:   %.23: <specific function> = specific_function %.22, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
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
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param3
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
// CHECK:STDOUT:     %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_36.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_36.2: type = converted %int.make_type_signed, %.loc11_36.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %a: bool = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: bool = value_param runtime_param2
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
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc12_25.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_25.2: <bound method> = bound_method %int.1, %.loc12_25.1 [template = constants.%.16]
// CHECK:STDOUT:   %.loc12_25.3: <specific function> = specific_function %.loc12_25.2, @Convert.2(constants.%int.32) [template = constants.%.17]
// CHECK:STDOUT:   %int.convert_checked.loc12_25: init %i32 = call %.loc12_25.3(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc12_25.4: %i32 = value_of_initializer %int.convert_checked.loc12_25 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc12_25.5: %i32 = converted %int.1, %.loc12_25.4 [template = constants.%int.1.2]
// CHECK:STDOUT:   br !if.expr.result.loc12_20(%.loc12_25.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12_20:
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc12_32.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_32.2: <bound method> = bound_method %int.2, %.loc12_32.1 [template = constants.%.18]
// CHECK:STDOUT:   %.loc12_32.3: <specific function> = specific_function %.loc12_32.2, @Convert.2(constants.%int.32) [template = constants.%.19]
// CHECK:STDOUT:   %int.convert_checked.loc12_32: init %i32 = call %.loc12_32.3(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc12_32.4: %i32 = value_of_initializer %int.convert_checked.loc12_32 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc12_32.5: %i32 = converted %int.2, %.loc12_32.4 [template = constants.%int.2.2]
// CHECK:STDOUT:   br !if.expr.result.loc12_20(%.loc12_32.5)
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
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc12_49.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_49.2: <bound method> = bound_method %int.3, %.loc12_49.1 [template = constants.%.20]
// CHECK:STDOUT:   %.loc12_49.3: <specific function> = specific_function %.loc12_49.2, @Convert.2(constants.%int.32) [template = constants.%.21]
// CHECK:STDOUT:   %int.convert_checked.loc12_49: init %i32 = call %.loc12_49.3(%int.3) [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc12_49.4: %i32 = value_of_initializer %int.convert_checked.loc12_49 [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc12_49.5: %i32 = converted %int.3, %.loc12_49.4 [template = constants.%int.3.2]
// CHECK:STDOUT:   br !if.expr.result.loc12_44(%.loc12_49.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12_44:
// CHECK:STDOUT:   %int.4: Core.IntLiteral = int_value 4 [template = constants.%int.4.1]
// CHECK:STDOUT:   %.loc12_56.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_56.2: <bound method> = bound_method %int.4, %.loc12_56.1 [template = constants.%.22]
// CHECK:STDOUT:   %.loc12_56.3: <specific function> = specific_function %.loc12_56.2, @Convert.2(constants.%int.32) [template = constants.%.23]
// CHECK:STDOUT:   %int.convert_checked.loc12_56: init %i32 = call %.loc12_56.3(%int.4) [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc12_56.4: %i32 = value_of_initializer %int.convert_checked.loc12_56 [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc12_56.5: %i32 = converted %int.4, %.loc12_56.4 [template = constants.%int.4.2]
// CHECK:STDOUT:   br !if.expr.result.loc12_44(%.loc12_56.5)
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

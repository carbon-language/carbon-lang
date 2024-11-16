// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/struct.carbon

fn Main() -> {.a: i32} {
  return {.a = 3};
}

// CHECK:STDOUT: --- struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %.2} [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %.2 = int_value 3 [template]
// CHECK:STDOUT:   %struct: %.3 = struct_value (%.30) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_19.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_19.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_19.3: type = converted %int.make_type_signed, %.loc11_19.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_22: type = struct_type {.a: %.2} [template = constants.%.3]
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_16: Core.IntLiteral = int_value 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_17.1: %.5 = struct_literal (%.loc12_16)
// CHECK:STDOUT:   %.loc12_17.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_17.3: <bound method> = bound_method %.loc12_16, %.loc12_17.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_17.4: <specific function> = specific_function %.loc12_17.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %.loc12_17.4(%.loc12_16) [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_17.5: %.2 = value_of_initializer %int.convert_checked [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_17.6: %.2 = converted %.loc12_16, %.loc12_17.5 [template = constants.%.30]
// CHECK:STDOUT:   %struct: %.3 = struct_value (%.loc12_17.6) [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_18: %.3 = converted %.loc12_17.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   return %.loc12_18
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %.1: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int.3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %int.3.1, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value (%int.3.2) [template]
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
// CHECK:STDOUT:     %return.patt: %.1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.1 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_signed, %.loc11_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_22: type = struct_type {.a: %i32} [template = constants.%.1]
// CHECK:STDOUT:     %return.param: ref %.1 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc12_17.1: %.2 = struct_literal (%int.3)
// CHECK:STDOUT:   %.loc12_17.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_17.3: <bound method> = bound_method %int.3, %.loc12_17.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_17.4: <specific function> = specific_function %.loc12_17.3, @Convert.2(constants.%int.32) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc12_17.4(%int.3) [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc12_17.5: %i32 = value_of_initializer %int.convert_checked [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc12_17.6: %i32 = converted %int.3, %.loc12_17.5 [template = constants.%int.3.2]
// CHECK:STDOUT:   %struct: %.1 = struct_value (%.loc12_17.6) [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_18: %.1 = converted %.loc12_17.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   return %.loc12_18
// CHECK:STDOUT: }
// CHECK:STDOUT:

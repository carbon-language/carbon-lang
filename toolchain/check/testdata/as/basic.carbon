// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/basic.carbon

fn Main() -> i32 {
  return 1 as i32;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @As(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.7, @impl.5(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.7(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
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
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11: init type = call constants.%Int(%.loc11_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_14.2: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_14.3: type = converted %int.make_type_signed.loc11, %.loc11_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_10: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_15.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.3: type = converted %int.make_type_signed.loc12, %.loc12_15.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_12.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_12.2: <bound method> = bound_method %.loc12_10, %.loc12_12.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_12.3: <specific function> = specific_function %.loc12_12.2, @Convert.7(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc12_12.3(%.loc12_10) [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_12.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_12.5: %i32 = converted %.loc12_10, %.loc12_12.4 [template = constants.%.29]
// CHECK:STDOUT:   return %.loc12_12.5
// CHECK:STDOUT: }
// CHECK:STDOUT:

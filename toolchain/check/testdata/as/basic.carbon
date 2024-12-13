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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @As(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.7, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.9) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %Convert.9 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.7(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %iN = call %Convert.specific_fn(%int_1) [symbolic]
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
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11: init type = call constants.%Int(%int_32.loc11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_14.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_14.2: type = converted %int.make_type_signed.loc11, %.loc11_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.2: type = converted %int.make_type_signed.loc12, %.loc12_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.9]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.7(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %iN = call %Convert.specific_fn(%int_1) [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc12_12.1: %i32 = value_of_initializer %int.convert_checked [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc12_12.2: %i32 = converted %int_1, %.loc12_12.1 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   return %.loc12_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

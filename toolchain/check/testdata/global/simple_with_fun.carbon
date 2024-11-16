// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/global/simple_with_fun.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/global/simple_with_fun.carbon

fn test_a() -> i32 {
  return 0;
}

var a: i32 = test_a();

// CHECK:STDOUT: --- simple_with_fun.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %test_a.type: type = fn_type @test_a [template]
// CHECK:STDOUT:   %test_a: %test_a.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.28: %.2 = int_value 0 [template]
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
// CHECK:STDOUT:     .test_a = %test_a.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %test_a.decl: %test_a.type = fn_decl @test_a [template = constants.%test_a] {
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_16.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_16.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_16.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_16.3: type = converted %int.make_type_signed, %.loc11_16.2 [template = constants.%.2]
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc15_8.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_8.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_8.3: type = converted %int.make_type_signed, %.loc15_8.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref %.2 = var a
// CHECK:STDOUT:   %a: ref %.2 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @test_a() -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_10: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_11.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_11.2: <bound method> = bound_method %.loc12_10, %.loc12_11.1 [template = constants.%.26]
// CHECK:STDOUT:   %.loc12_11.3: <specific function> = specific_function %.loc12_11.2, @Convert.2(constants.%.1) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %.loc12_11.3(%.loc12_10) [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_11.4: %.2 = value_of_initializer %int.convert_checked [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_11.5: %.2 = converted %.loc12_10, %.loc12_11.4 [template = constants.%.28]
// CHECK:STDOUT:   return %.loc12_11.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %test_a.ref: %test_a.type = name_ref test_a, file.%test_a.decl [template = constants.%test_a]
// CHECK:STDOUT:   %test_a.call: init %.2 = call %test_a.ref()
// CHECK:STDOUT:   assign file.%a.var, %test_a.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/i32.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/i32.carbon

fn Echo(a: i32) -> i32 {
  return a;
}

fn Main() {
  var b: i32 = Echo(1);
}

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Echo.type: type = fn_type @Echo [template]
// CHECK:STDOUT:   %Echo: %Echo.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.28: %.2 = int_value 1 [template]
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
// CHECK:STDOUT:     .Echo = %Echo.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Echo.decl: %Echo.type = fn_decl @Echo [template = constants.%Echo] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_12: init type = call constants.%Int(%.loc11_12.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_12.2: type = value_of_initializer %int.make_type_signed.loc11_12 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_12.3: type = converted %int.make_type_signed.loc11_12, %.loc11_12.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_20.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_20: init type = call constants.%Int(%.loc11_20.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_20.2: type = value_of_initializer %int.make_type_signed.loc11_20 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_20.3: type = converted %int.make_type_signed.loc11_20, %.loc11_20.2 [template = constants.%.2]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a.param_patt: %.2) -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %.2 = name_ref a, %a
// CHECK:STDOUT:   return %a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc16_10.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_10.3: type = converted %int.make_type_signed, %.loc16_10.2 [template = constants.%.2]
// CHECK:STDOUT:   %b.var: ref %.2 = var b
// CHECK:STDOUT:   %b: ref %.2 = bind_name b, %b.var
// CHECK:STDOUT:   %Echo.ref: %Echo.type = name_ref Echo, file.%Echo.decl [template = constants.%Echo]
// CHECK:STDOUT:   %.loc16_21.1: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc16_21.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_21.3: <bound method> = bound_method %.loc16_21.1, %.loc16_21.2 [template = constants.%.26]
// CHECK:STDOUT:   %.loc16_21.4: <specific function> = specific_function %.loc16_21.3, @Convert.2(constants.%.1) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %.loc16_21.4(%.loc16_21.1) [template = constants.%.28]
// CHECK:STDOUT:   %.loc16_21.5: %.2 = value_of_initializer %int.convert_checked [template = constants.%.28]
// CHECK:STDOUT:   %.loc16_21.6: %.2 = converted %.loc16_21.1, %.loc16_21.5 [template = constants.%.28]
// CHECK:STDOUT:   %Echo.call: init %.2 = call %Echo.ref(%.loc16_21.6)
// CHECK:STDOUT:   assign %b.var, %Echo.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

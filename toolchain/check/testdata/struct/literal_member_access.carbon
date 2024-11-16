// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/literal_member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/literal_member_access.carbon

fn G() -> {.x: i32, .y: i32, .z: i32};

fn F() -> i32 {
  return {.a = 1, .b = G(), .c = 3}.b.y;
}

// CHECK:STDOUT: --- literal_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: %.2, .y: %.2, .z: %.2} [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: Core.IntLiteral, .b: %.3, .c: Core.IntLiteral} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_16.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_16: init type = call constants.%Int(%.loc11_16.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_16.2: type = value_of_initializer %int.make_type_signed.loc11_16 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_16.3: type = converted %int.make_type_signed.loc11_16, %.loc11_16.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_25.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_25: init type = call constants.%Int(%.loc11_25.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_25.2: type = value_of_initializer %int.make_type_signed.loc11_25 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_25.3: type = converted %int.make_type_signed.loc11_25, %.loc11_25.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_34.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_34: init type = call constants.%Int(%.loc11_34.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_34.2: type = value_of_initializer %int.make_type_signed.loc11_34 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_34.3: type = converted %int.make_type_signed.loc11_34, %.loc11_34.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_37: type = struct_type {.x: %.2, .y: %.2, .z: %.2} [template = constants.%.3]
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc13_11.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_11.3: type = converted %int.make_type_signed, %.loc13_11.2 [template = constants.%.2]
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_16: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc14_25.1: ref %.3 = temporary_storage
// CHECK:STDOUT:   %G.call: init %.3 = call %G.ref() to %.loc14_25.1
// CHECK:STDOUT:   %.loc14_34: Core.IntLiteral = int_value 3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_35.1: %.7 = struct_literal (%.loc14_16, %G.call, %.loc14_34)
// CHECK:STDOUT:   %.loc14_25.2: ref %.3 = temporary %.loc14_25.1, %G.call
// CHECK:STDOUT:   %.loc14_25.3: ref %.2 = struct_access %.loc14_25.2, element0
// CHECK:STDOUT:   %.loc14_25.4: %.2 = bind_value %.loc14_25.3
// CHECK:STDOUT:   %.loc14_25.5: ref %.2 = struct_access %.loc14_25.2, element1
// CHECK:STDOUT:   %.loc14_25.6: %.2 = bind_value %.loc14_25.5
// CHECK:STDOUT:   %.loc14_25.7: ref %.2 = struct_access %.loc14_25.2, element2
// CHECK:STDOUT:   %.loc14_25.8: %.2 = bind_value %.loc14_25.7
// CHECK:STDOUT:   %struct.loc14_25: %.3 = struct_value (%.loc14_25.4, %.loc14_25.6, %.loc14_25.8)
// CHECK:STDOUT:   %.loc14_35.2: %.3 = converted %G.call, %struct.loc14_25
// CHECK:STDOUT:   %struct.loc14_35: %.7 = struct_value (%.loc14_16, %.loc14_35.2, %.loc14_34)
// CHECK:STDOUT:   %.loc14_35.3: %.7 = converted %.loc14_35.1, %struct.loc14_35
// CHECK:STDOUT:   %.loc14_36: %.3 = struct_access %.loc14_35.3, element1
// CHECK:STDOUT:   %.loc14_38: %.2 = struct_access %.loc14_36, element1
// CHECK:STDOUT:   return %.loc14_38
// CHECK:STDOUT: }
// CHECK:STDOUT:

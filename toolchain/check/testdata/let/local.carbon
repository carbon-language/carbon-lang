// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/local.carbon

fn F(a: i32) -> i32 {
  let b: i32 = a;
  return b;
}

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_9: init type = call constants.%Int(%.loc11_9.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_9.2: type = value_of_initializer %int.make_type_signed.loc11_9 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_9.3: type = converted %int.make_type_signed.loc11_9, %.loc11_9.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_17.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_17: init type = call constants.%Int(%.loc11_17.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_17.2: type = value_of_initializer %int.make_type_signed.loc11_17 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_17.3: type = converted %int.make_type_signed.loc11_17, %.loc11_17.2 [template = constants.%.2]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %.2) -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_10.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed.loc12, %.loc12_10.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.ref: %.2 = name_ref a, %a
// CHECK:STDOUT:   %b: %.2 = bind_name b, %a.ref
// CHECK:STDOUT:   %b.ref: %.2 = name_ref b, %b
// CHECK:STDOUT:   return %b.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:

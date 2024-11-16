// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/nested_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/nested_const.carbon

// TODO: The `const` in the return type should not be necessary.
fn F(p: const (const (const i32*)*)) -> const i32 {
  return **p;
}

// CHECK:STDOUT: --- nested_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: type = const_type %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = const_type %.4 [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.7: type = const_type %.6 [template]
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
// CHECK:STDOUT:     %p.patt: %.7 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.7 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_29: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc12_29: init type = call constants.%Int(%.loc12_29) [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_23.1: type = value_of_initializer %int.make_type_signed.loc12_29 [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_23.2: type = converted %int.make_type_signed.loc12_29, %.loc12_23.1 [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_23.3: type = const_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc12_32: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %.loc12_16: type = const_type %.4 [template = constants.%.5]
// CHECK:STDOUT:     %.loc12_34: type = ptr_type %.5 [template = constants.%.6]
// CHECK:STDOUT:     %.loc12_9: type = const_type %.6 [template = constants.%.7]
// CHECK:STDOUT:     %.loc12_47: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc12_47: init type = call constants.%Int(%.loc12_47) [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_41.1: type = value_of_initializer %int.make_type_signed.loc12_47 [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_41.2: type = converted %int.make_type_signed.loc12_47, %.loc12_41.1 [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_41.3: type = const_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %p.param: %.7 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.7 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param_patt: %.7) -> %.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.7 = name_ref p, %p
// CHECK:STDOUT:   %.loc13_11.1: ref %.5 = deref %p.ref
// CHECK:STDOUT:   %.loc13_11.2: %.5 = bind_value %.loc13_11.1
// CHECK:STDOUT:   %.loc13_10.1: ref %.3 = deref %.loc13_11.2
// CHECK:STDOUT:   %.loc13_10.2: %.3 = bind_value %.loc13_10.1
// CHECK:STDOUT:   return %.loc13_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

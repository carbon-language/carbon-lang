// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/collapse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/collapse.carbon

// OK, `const (const i32)` is the same type as `const i32`.
// CHECK:STDERR: collapse.carbon:[[@LINE+3]]:25: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: fn F(p: const i32**) -> const (const i32)** {
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~
fn F(p: const i32**) -> const (const i32)** {
  return p;
}

// CHECK:STDOUT: --- collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: type = const_type %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [template]
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
// CHECK:STDOUT:     %p.patt: %.5 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.5 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.5 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_15: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc15_15: init type = call constants.%Int(%.loc15_15) [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_9.1: type = value_of_initializer %int.make_type_signed.loc15_15 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_9.2: type = converted %int.make_type_signed.loc15_15, %.loc15_9.1 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_9.3: type = const_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc15_18: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %.loc15_19: type = ptr_type %.4 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_38: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc15_38: init type = call constants.%Int(%.loc15_38) [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_32.1: type = value_of_initializer %int.make_type_signed.loc15_38 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_32.2: type = converted %int.make_type_signed.loc15_38, %.loc15_32.1 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_32.3: type = const_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc15_25: type = const_type %.3 [template = constants.%.3]
// CHECK:STDOUT:     %.loc15_42: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %.loc15_43: type = ptr_type %.4 [template = constants.%.5]
// CHECK:STDOUT:     %p.param: %.5 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.5 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.5 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param_patt: %.5) -> %.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.5 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:

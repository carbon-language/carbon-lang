// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/fail_collapse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/fail_collapse.carbon

// CHECK:STDERR: fail_collapse.carbon:[[@LINE+4]]:9: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: fn G(p: const (const i32)**) -> i32** {
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(p: const (const i32)**) -> i32** {
  // CHECK:STDERR: fail_collapse.carbon:[[@LINE+6]]:3: error: cannot implicitly convert from `const i32**` to `i32**` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_collapse.carbon:[[@LINE+3]]:3: note: type `const i32**` does not implement interface `ImplicitAs(i32**)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  return p;
}

// CHECK:STDOUT: --- fail_collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = const_type %i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %.4 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.4 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.6 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.6 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_22: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc15_22: init type = call constants.%Int(%.loc15_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_16.1: type = value_of_initializer %int.make_type_signed.loc15_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_16.2: type = converted %int.make_type_signed.loc15_22, %.loc15_16.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_16.3: type = const_type %i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_9: type = const_type %.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_26: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc15_27: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %.loc15_33: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc15_33: init type = call constants.%Int(%.loc15_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_36.1: type = value_of_initializer %int.make_type_signed.loc15_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_36.2: type = converted %int.make_type_signed.loc15_33, %.loc15_36.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_36.3: type = ptr_type %i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_37: type = ptr_type %.5 [template = constants.%.6]
// CHECK:STDOUT:     %p.param: %.4 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.4 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.6 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %.4) -> %.6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc22: %.6 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

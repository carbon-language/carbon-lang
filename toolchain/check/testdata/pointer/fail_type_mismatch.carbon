// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_type_mismatch.carbon

fn ConstMismatch(p: const {}*) -> const ({}*) {
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+6]]:3: error: cannot implicitly convert from `const {}*` to `const ({}*)` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:3: note: type `const {}*` does not implement interface `ImplicitAs(const ({}*))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  return p;
}

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = const_type %.1 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.5: type = const_type %.4 [template]
// CHECK:STDOUT:   %ConstMismatch.type: type = fn_type @ConstMismatch [template]
// CHECK:STDOUT:   %ConstMismatch: %ConstMismatch.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ConstMismatch = %ConstMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ConstMismatch.decl: %ConstMismatch.type = fn_decl @ConstMismatch [template = constants.%ConstMismatch] {
// CHECK:STDOUT:     %p.patt: %.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.5 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_28: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc11_21.1: type = converted %.loc11_28, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_21.2: type = const_type %.1 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_29: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc11_43: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc11_44.1: type = converted %.loc11_43, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_44.2: type = ptr_type %.1 [template = constants.%.4]
// CHECK:STDOUT:     %.loc11_35: type = const_type %.4 [template = constants.%.5]
// CHECK:STDOUT:     %p.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.3 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.5 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstMismatch(%p.param_patt: %.3) -> %.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc18: %.5 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

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
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:3: note: type `const {}*` does not implement interface `Core.ImplicitAs(const ({}*))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  return p;
}

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %const.c: type = const_type %empty_struct_type [template]
// CHECK:STDOUT:   %ptr.b: type = ptr_type %const.c [template]
// CHECK:STDOUT:   %ptr.c: type = ptr_type %empty_struct_type [template]
// CHECK:STDOUT:   %const.9: type = const_type %ptr.c [template]
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
// CHECK:STDOUT:     %p.patt: %ptr.b = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.b = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %const.9 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %const.9 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_43: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc11_44: type = converted %.loc11_43, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc11_44: type = ptr_type %empty_struct_type [template = constants.%ptr.c]
// CHECK:STDOUT:     %const.loc11_35: type = const_type %ptr.c [template = constants.%const.9]
// CHECK:STDOUT:     %p.param: %ptr.b = value_param runtime_param0
// CHECK:STDOUT:     %.loc11_29: type = splice_block %ptr.loc11_29 [template = constants.%ptr.b] {
// CHECK:STDOUT:       %.loc11_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc11_21: type = converted %.loc11_28, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:       %const.loc11_21: type = const_type %empty_struct_type [template = constants.%const.c]
// CHECK:STDOUT:       %ptr.loc11_29: type = ptr_type %const.c [template = constants.%ptr.b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.b = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.9 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %const.9 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstMismatch(%p.param_patt: %ptr.b) -> %const.9 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.b = name_ref p, %p
// CHECK:STDOUT:   %.loc18: %const.9 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

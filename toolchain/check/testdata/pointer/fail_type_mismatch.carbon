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
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `const {}*` to `const ({}*)` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:3: note: type `const {}*` does not implement interface `Core.ImplicitAs(const ({}*))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %const.c48: type = const_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.bf9: type = ptr_type %const.c48 [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %const.987: type = const_type %ptr.c28 [concrete]
// CHECK:STDOUT:   %ConstMismatch.type: type = fn_type @ConstMismatch [concrete]
// CHECK:STDOUT:   %ConstMismatch: %ConstMismatch.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ConstMismatch = %ConstMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ConstMismatch.decl: %ConstMismatch.type = fn_decl @ConstMismatch [concrete = constants.%ConstMismatch] {
// CHECK:STDOUT:     %p.patt: %ptr.bf9 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.bf9 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %const.987 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %const.987 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_43: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc11_44: type = converted %.loc11_43, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %ptr.loc11_44: type = ptr_type %.loc11_44 [concrete = constants.%ptr.c28]
// CHECK:STDOUT:     %const.loc11_35: type = const_type %ptr.loc11_44 [concrete = constants.%const.987]
// CHECK:STDOUT:     %p.param: %ptr.bf9 = value_param call_param0
// CHECK:STDOUT:     %.loc11_29: type = splice_block %ptr.loc11_29 [concrete = constants.%ptr.bf9] {
// CHECK:STDOUT:       %.loc11_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc11_21: type = converted %.loc11_28, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %const.loc11_21: type = const_type %.loc11_21 [concrete = constants.%const.c48]
// CHECK:STDOUT:       %ptr.loc11_29: type = ptr_type %const.loc11_21 [concrete = constants.%ptr.bf9]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.bf9 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.987 = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.987 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstMismatch(%p.param_patt: %ptr.bf9) -> %const.987 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.bf9 = name_ref p, %p
// CHECK:STDOUT:   %.loc19: %const.987 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

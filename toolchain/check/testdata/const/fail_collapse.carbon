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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %const: type = const_type %i32 [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %const [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %ptr.1 [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %ptr.4: type = ptr_type %ptr.3 [template]
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
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.4 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15_22: init type = call constants.%Int(%int_32.loc15_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_16.1: type = value_of_initializer %int.make_type_signed.loc15_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_16.2: type = converted %int.make_type_signed.loc15_22, %.loc15_16.1 [template = constants.%i32]
// CHECK:STDOUT:     %const.loc15_16: type = const_type %i32 [template = constants.%const]
// CHECK:STDOUT:     %const.loc15_9: type = const_type %const [template = constants.%const]
// CHECK:STDOUT:     %ptr.loc15_26: type = ptr_type %const [template = constants.%ptr.1]
// CHECK:STDOUT:     %ptr.loc15_27: type = ptr_type %ptr.1 [template = constants.%ptr.2]
// CHECK:STDOUT:     %int_32.loc15_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15_33: init type = call constants.%Int(%int_32.loc15_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_36.1: type = value_of_initializer %int.make_type_signed.loc15_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_36.2: type = converted %int.make_type_signed.loc15_33, %.loc15_36.1 [template = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15_36: type = ptr_type %i32 [template = constants.%ptr.3]
// CHECK:STDOUT:     %ptr.loc15_37: type = ptr_type %ptr.3 [template = constants.%ptr.4]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.4 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %ptr.2) -> %ptr.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc22: %ptr.4 = converted %p.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

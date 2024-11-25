// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/fail_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/fail_generic.carbon

// TODO: Should this be valid?
fn F(a: i32) -> i32 {
  let T:! type = i32;
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+7]]:3: error: cannot implicitly convert from `Core.IntLiteral` to `T` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   let x: T = 5;
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+4]]:3: note: type `Core.IntLiteral` does not implement interface `ImplicitAs(T)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let x: T = 5;
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let x: T = 5;
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+6]]:3: error: cannot implicitly convert from `T` to `i32` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return x;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+3]]:3: note: type `T` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return x;
  // CHECK:STDERR:   ^~~~~~~~~
  return x;
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 5 [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc12_9: init type = call constants.%Int(%.loc12_9.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_9.2: type = value_of_initializer %int.make_type_signed.loc12_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_9.3: type = converted %int.make_type_signed.loc12_9, %.loc12_9.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_17.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc12_17: init type = call constants.%Int(%.loc12_17.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_17.2: type = value_of_initializer %int.make_type_signed.loc12_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_17.3: type = converted %int.make_type_signed.loc12_17, %.loc12_17.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_18: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_18) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_21.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_21.2: type = converted %int.make_type_signed.loc13, %.loc13_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, %.loc13_21.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc21_14: Core.IntLiteral = int_value 5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc21_15: %T = converted %.loc21_14, <error> [template = <error>]
// CHECK:STDOUT:   %x: %T = bind_name x, <error>
// CHECK:STDOUT:   %x.ref: %T = name_ref x, %x
// CHECK:STDOUT:   %.loc28: %i32 = converted %x.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

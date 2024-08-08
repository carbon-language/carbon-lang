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
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+8]]:3: ERROR: Name `ImplicitAs` not found.
  // CHECK:STDERR:   let x: T = 5;
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `missing or invalid operator interface, also avoid duplicate diagnostic if prelude is unavailable`.
  // CHECK:STDERR:   let x: T = 5;
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let x: T = 5;
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+7]]:3: ERROR: Name `ImplicitAs` not found.
  // CHECK:STDERR:   return x;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `missing or invalid operator interface, also avoid duplicate diagnostic if prelude is unavailable`.
  // CHECK:STDERR:   return x;
  // CHECK:STDERR:   ^~~~~~~~~
  return x;
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.2: i32 = int_literal 5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc12_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_9.1: type = value_of_initializer %int.make_type_32.loc12_9 [template = i32]
// CHECK:STDOUT:     %.loc12_9.2: type = converted %int.make_type_32.loc12_9, %.loc12_9.1 [template = i32]
// CHECK:STDOUT:     %a.loc12_6.1: i32 = param a
// CHECK:STDOUT:     @F.%a: i32 = bind_name a, %a.loc12_6.1
// CHECK:STDOUT:     %int.make_type_32.loc12_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_17.1: type = value_of_initializer %int.make_type_32.loc12_17 [template = i32]
// CHECK:STDOUT:     %.loc12_17.2: type = converted %int.make_type_32.loc12_17, %.loc12_17.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc13_21.2: type = converted %int.make_type_32, %.loc13_21.1 [template = i32]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0, %.loc13_21.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc22_14: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc22_15: %T = converted %.loc22_14, <error> [template = <error>]
// CHECK:STDOUT:   %x: %T = bind_name x, <error>
// CHECK:STDOUT:   %x.ref: %T = name_ref x, %x
// CHECK:STDOUT:   %.loc30: i32 = converted %x.ref, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_deref_not_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_deref_not_pointer.carbon

fn Deref(n: i32) {
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:3: ERROR: Cannot dereference operand of non-pointer type `i32`.
  // CHECK:STDERR:   *n;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *n;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:4: ERROR: Cannot apply `->` operator to non-pointer type `i32`.
  // CHECK:STDERR:   n->foo;
  // CHECK:STDERR:    ^~
  // CHECK:STDERR:
  n->foo;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:3: ERROR: Cannot dereference operand of non-pointer type `()`.
  // CHECK:STDERR:   *();
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *();
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:5: ERROR: Cannot apply `->` operator to non-pointer type `()`.
  // CHECK:STDERR:   ()->foo;
  // CHECK:STDERR:     ^~
  // CHECK:STDERR:
  ()->foo;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:3: ERROR: Cannot dereference operand of non-pointer type `{}`.
  // CHECK:STDERR:   *{};
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *{};
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:5: ERROR: Cannot apply `->` operator to non-pointer type `{}`.
  // CHECK:STDERR:   {}->foo;
  // CHECK:STDERR:     ^~
  {}->foo;
}

// CHECK:STDOUT: --- fail_deref_not_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Deref.type: type = fn_type @Deref [template]
// CHECK:STDOUT:   %Deref: %Deref.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %struct: %.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
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
// CHECK:STDOUT:     .Deref = %Deref.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Deref.decl: %Deref.type = fn_decl @Deref [template = constants.%Deref] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_13.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_13.2: type = converted %int.make_type_32, %.loc11_13.1 [template = i32]
// CHECK:STDOUT:     %n.loc11_10.1: i32 = param n, runtime_param0
// CHECK:STDOUT:     @Deref.%n: i32 = bind_name n, %n.loc11_10.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Deref(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref.loc16: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc16: ref <error> = deref %n.ref.loc16
// CHECK:STDOUT:   %n.ref.loc21: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc21: ref <error> = deref %n.ref.loc21
// CHECK:STDOUT:   %.loc26_5.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple.loc26: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc26_5.2: %.1 = converted %.loc26_5.1, %tuple.loc26 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc26_3: ref <error> = deref %.loc26_5.2
// CHECK:STDOUT:   %.loc31_4.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple.loc31: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc31_4.2: %.1 = converted %.loc31_4.1, %tuple.loc31 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc31_5: ref <error> = deref %.loc31_4.2
// CHECK:STDOUT:   %.loc36_5.1: %.2 = struct_literal ()
// CHECK:STDOUT:   %struct.loc36: %.2 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc36_5.2: %.2 = converted %.loc36_5.1, %struct.loc36 [template = constants.%struct]
// CHECK:STDOUT:   %.loc36_3: ref <error> = deref %.loc36_5.2
// CHECK:STDOUT:   %.loc40_4.1: %.2 = struct_literal ()
// CHECK:STDOUT:   %struct.loc40: %.2 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc40_4.2: %.2 = converted %.loc40_4.1, %struct.loc40 [template = constants.%struct]
// CHECK:STDOUT:   %.loc40_5: ref <error> = deref %.loc40_4.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

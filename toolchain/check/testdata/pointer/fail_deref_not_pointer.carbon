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
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:3: error: cannot dereference operand of non-pointer type `i32` [DerefOfNonPointer]
  // CHECK:STDERR:   *n;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *n;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:4: error: cannot apply `->` operator to non-pointer type `i32` [ArrowOperatorOfNonPointer]
  // CHECK:STDERR:   n->foo;
  // CHECK:STDERR:    ^~
  // CHECK:STDERR:
  n->foo;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:3: error: cannot dereference operand of non-pointer type `()` [DerefOfNonPointer]
  // CHECK:STDERR:   *();
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *();
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:5: error: cannot apply `->` operator to non-pointer type `()` [ArrowOperatorOfNonPointer]
  // CHECK:STDERR:   ()->foo;
  // CHECK:STDERR:     ^~
  // CHECK:STDERR:
  ()->foo;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:3: error: cannot dereference operand of non-pointer type `{}` [DerefOfNonPointer]
  // CHECK:STDERR:   *{};
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *{};
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:5: error: cannot apply `->` operator to non-pointer type `{}` [ArrowOperatorOfNonPointer]
  // CHECK:STDERR:   {}->foo;
  // CHECK:STDERR:     ^~
  {}->foo;
}

// CHECK:STDOUT: --- fail_deref_not_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Deref.type: type = fn_type @Deref [template]
// CHECK:STDOUT:   %Deref: %Deref.type = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %struct: %.2 = struct_value () [template]
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
// CHECK:STDOUT:     .Deref = %Deref.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Deref.decl: %Deref.type = fn_decl @Deref [template = constants.%Deref] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_13.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_13.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_13.3: type = converted %int.make_type_signed, %.loc11_13.2 [template = constants.%i32]
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Deref(%n.param_patt: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref.loc16: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc16: ref <error> = deref %n.ref.loc16
// CHECK:STDOUT:   %n.ref.loc21: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc21: ref <error> = deref %n.ref.loc21
// CHECK:STDOUT:   %foo.ref.loc21: <error> = name_ref foo, <error> [template = <error>]
// CHECK:STDOUT:   %.loc26_5.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple.loc26: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc26_5.2: %empty_tuple.type = converted %.loc26_5.1, %empty_tuple.loc26 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc26_3: ref <error> = deref %.loc26_5.2
// CHECK:STDOUT:   %.loc31_4.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple.loc31: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31_4.2: %empty_tuple.type = converted %.loc31_4.1, %empty_tuple.loc31 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31_5: ref <error> = deref %.loc31_4.2
// CHECK:STDOUT:   %foo.ref.loc31: <error> = name_ref foo, <error> [template = <error>]
// CHECK:STDOUT:   %.loc36_5.1: %.2 = struct_literal ()
// CHECK:STDOUT:   %struct.loc36: %.2 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc36_5.2: %.2 = converted %.loc36_5.1, %struct.loc36 [template = constants.%struct]
// CHECK:STDOUT:   %.loc36_3: ref <error> = deref %.loc36_5.2
// CHECK:STDOUT:   %.loc40_4.1: %.2 = struct_literal ()
// CHECK:STDOUT:   %struct.loc40: %.2 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc40_4.2: %.2 = converted %.loc40_4.1, %struct.loc40 [template = constants.%struct]
// CHECK:STDOUT:   %.loc40_5: ref <error> = deref %.loc40_4.2
// CHECK:STDOUT:   %foo.ref.loc40: <error> = name_ref foo, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

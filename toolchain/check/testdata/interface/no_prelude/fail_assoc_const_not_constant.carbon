// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_not_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_not_constant.carbon

interface I {
  // CHECK:STDERR: fail_assoc_const_not_constant.carbon:[[@LINE+7]]:7: error: pattern in associated constant declaration must be a single `:!` binding
  // CHECK:STDERR:   let a: i32;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_assoc_const_not_constant.carbon:[[@LINE+3]]:10: error: package `Core` implicitly referenced here, but not found
  // CHECK:STDERR:   let a: i32;
  // CHECK:STDERR:          ^~~
  let a: i32;
}

// We shouldn't issue further errors on uses of the invalid name.
alias UseA = I.a;

// Ideally we would still diagnose this, but it's OK that we don't.
alias UseOther = I.other;

// CHECK:STDOUT: --- fail_assoc_const_not_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .UseA = %UseA
// CHECK:STDOUT:     .UseOther = %UseOther
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %I.ref.loc23: type = name_ref I, %I.decl [template = constants.%I.type]
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, <error> [template = <error>]
// CHECK:STDOUT:   %UseA: <error> = bind_alias UseA, <error> [template = <error>]
// CHECK:STDOUT:   %I.ref.loc26: type = name_ref I, %I.decl [template = constants.%I.type]
// CHECK:STDOUT:   %other.ref: <error> = name_ref other, <error> [template = <error>]
// CHECK:STDOUT:   %UseOther: <error> = bind_alias UseOther, <error> [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:

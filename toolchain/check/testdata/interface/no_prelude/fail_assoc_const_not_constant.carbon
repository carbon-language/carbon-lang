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
  // CHECK:STDERR: fail_assoc_const_not_constant.carbon:[[@LINE+4]]:7: error: pattern in associated constant declaration must be a single `:!` binding [ExpectedSymbolicBindingInAssociatedConstant]
  // CHECK:STDERR:   let a: {.b: ()};
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let a: {.b: ()};
}

// We shouldn't issue further errors on uses of the invalid name.
alias UseA = I.a;

// Ideally we would still diagnose this, but it's OK that we don't.
alias UseOther = I.other;

// CHECK:STDOUT: --- fail_assoc_const_not_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .UseA = %UseA
// CHECK:STDOUT:     .UseOther = %UseOther
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %I.ref.loc20: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %a.ref: %struct_type.b = name_ref a, <unexpected>.inst22.loc16_7
// CHECK:STDOUT:   %UseA: %struct_type.b = bind_alias UseA, <unexpected>.inst22.loc16_7
// CHECK:STDOUT:   %I.ref.loc23: type = name_ref I, %I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   %other.ref: <error> = name_ref other, <error> [concrete = <error>]
// CHECK:STDOUT:   %UseOther: <error> = bind_alias UseOther, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .a = <unexpected>.inst22.loc16_7
// CHECK:STDOUT:   .other = <poisoned>
// CHECK:STDOUT:   has_error
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:

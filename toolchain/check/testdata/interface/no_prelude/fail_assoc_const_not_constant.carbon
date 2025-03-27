// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
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
// CHECK:STDERR: fail_assoc_const_not_constant.carbon:[[@LINE+7]]:14: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: alias UseA = I.a;
// CHECK:STDERR:              ^~~
// CHECK:STDERR: fail_assoc_const_not_constant.carbon:[[@LINE-12]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
alias UseA = I.a;

// Ideally we would still diagnose this, but it's OK that we don't.
// CHECK:STDERR: fail_assoc_const_not_constant.carbon:[[@LINE+7]]:18: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: alias UseOther = I.other;
// CHECK:STDERR:                  ^~~~~~~
// CHECK:STDERR: fail_assoc_const_not_constant.carbon:[[@LINE-22]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
alias UseOther = I.other;

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_not_binding.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_not_binding.carbon

interface I {
  // CHECK:STDERR: fail_assoc_const_not_binding.carbon:[[@LINE+3]]:3: error: semantics TODO: `tuple pattern in let/var`
  // CHECK:STDERR:   let (T:! type, U:! type);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  let (T:! type, U:! type);
}

// CHECK:STDOUT: --- fail_assoc_const_not_binding.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = <unexpected>.inst+3
// CHECK:STDOUT:   witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:

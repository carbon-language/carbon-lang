// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_address_of_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_address_of_error.carbon

fn Test() {
  // CHECK:STDERR: fail_address_of_error.carbon:[[@LINE+4]]:4: error: name `undeclared` not found [NameNotFound]
  // CHECK:STDERR:   &undeclared;
  // CHECK:STDERR:    ^~~~~~~~~~
  // CHECK:STDERR:
  &undeclared;
  // CHECK:STDERR: fail_address_of_error.carbon:[[@LINE+8]]:6: error: name `undeclared` not found [NameNotFound]
  // CHECK:STDERR:   &(&undeclared);
  // CHECK:STDERR:      ^~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_address_of_error.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &(&undeclared);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &(&undeclared);
}

// CHECK:STDOUT: --- fail_address_of_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .undeclared = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %undeclared.ref.loc16: <error> = name_ref undeclared, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc16: <error> = addr_of %undeclared.ref.loc16 [concrete = <error>]
// CHECK:STDOUT:   %undeclared.ref.loc25: <error> = name_ref undeclared, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc25_5: <error> = addr_of %undeclared.ref.loc25 [concrete = <error>]
// CHECK:STDOUT:   %addr.loc25_3: <error> = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

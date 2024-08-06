// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_redeclare_member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_redeclare_member.carbon

interface Interface {
  fn F();
  // CHECK:STDERR: fail_redeclare_member.carbon:[[@LINE+6]]:3: ERROR: Duplicate name being declared in the same scope.
  // CHECK:STDERR:   fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_redeclare_member.carbon:[[@LINE-4]]:3: Name is previously declared here.
  // CHECK:STDERR:   fn F();
  // CHECK:STDERR:   ^~~~~~~
  fn F();
}

// CHECK:STDOUT: --- fail_redeclare_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Interface [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Interface[%Self] [symbolic]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %F.type [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @Interface.%F.decl [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1, @Interface[%Self] [symbolic]
// CHECK:STDOUT:   %.5: %.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface[%Self.1: %.1] {
// CHECK:STDOUT:   %Self.2: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Interface[%Self.2] [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Interface.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.1, @Interface.%F.type (%F.type) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @Interface.%.1 (%.3) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:   %.type: type = fn_type @.1, @Interface[%Self.2] [symbolic = %.type (constants.%.type)]
// CHECK:STDOUT:   %.3: @Interface.%.type (%.type) = struct_value () [symbolic = %.3 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.decl: @Interface.%F.type (%F.type) = fn_decl @F [symbolic = %F (constants.%F)] {}
// CHECK:STDOUT:   %.loc12: @Interface.%.1 (%.3) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:   %.decl: @Interface.%.type (%.type) = fn_decl @.1 [symbolic = %.3 (constants.%.5)] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.1
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Interface.%Self.1: %.1) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(@Interface.%Self.1: %.1) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface[constants.%Self] {
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface[@Interface.%Self.2] {
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:

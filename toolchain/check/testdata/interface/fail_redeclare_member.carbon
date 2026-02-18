// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_redeclare_member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_redeclare_member.carbon

interface Interface {
  fn F();
  // CHECK:STDERR: fail_redeclare_member.carbon:[[@LINE+7]]:3: error: redeclaration of `fn F` is redundant [RedeclRedundant]
  // CHECK:STDERR:   fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_redeclare_member.carbon:[[@LINE-4]]:3: note: previously declared here [RedeclPrevDecl]
  // CHECK:STDERR:   fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  fn F();
}

// CHECK:STDOUT: --- fail_redeclare_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self: %Interface.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F.type: type = fn_type @Interface.WithSelf.F, @Interface(%Self) [symbolic]
// CHECK:STDOUT:   %Interface.WithSelf.F: %Interface.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface [concrete]
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, @Interface.%Interface.WithSelf.F.decl.loc16 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self.loc15: %Interface.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @Interface [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Interface.WithSelf.F.decl.loc16: @Interface.%Interface.WithSelf.F.type (%Interface.WithSelf.F.type) = fn_decl @Interface.WithSelf.F [symbolic = @Interface.%Interface.WithSelf.F (constants.%Interface.WithSelf.F)] {} {}
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.WithSelf.F.decl.loc16 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Interface.WithSelf.F.decl.loc24: @Interface.%Interface.WithSelf.F.type (%Interface.WithSelf.F.type) = fn_decl @Interface.WithSelf.F [symbolic = @Interface.%Interface.WithSelf.F (constants.%Interface.WithSelf.F)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc15
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%Interface.WithSelf.F.decl.loc16)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.WithSelf.F(@Interface.%Self.loc15: %Interface.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:

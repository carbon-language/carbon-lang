// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/fail_adapt_bad_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/fail_adapt_bad_type.carbon

// --- fail_incomplete_type.carbon

library "[[@TEST_NAME]]";

class Incomplete;

class AdaptIncomplete {
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE+7]]:3: error: adapted type `Incomplete` is an incomplete type [IncompleteTypeInAdaptDecl]
  // CHECK:STDERR:   adapt Incomplete;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE-6]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  adapt Incomplete;
}

// CHECK:STDOUT: --- fail_incomplete_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [concrete]
// CHECK:STDOUT:   %AdaptIncomplete: type = class_type @AdaptIncomplete [concrete]
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
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .AdaptIncomplete = %AdaptIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [concrete = constants.%Incomplete] {} {}
// CHECK:STDOUT:   %AdaptIncomplete.decl: type = class_decl @AdaptIncomplete [concrete = constants.%AdaptIncomplete] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [concrete = constants.%Incomplete]
// CHECK:STDOUT:   adapt_decl <error> [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptIncomplete
// CHECK:STDOUT:   .Incomplete = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:

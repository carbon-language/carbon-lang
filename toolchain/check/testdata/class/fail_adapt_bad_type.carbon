// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_incomplete_type.carbon

library "incomplete_type" api;

class Incomplete;

class AdaptIncomplete {
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE+6]]:3: ERROR: Adapted type `Incomplete` is an incomplete type.
  // CHECK:STDERR:   adapt Incomplete;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE-6]]:1: Class was forward declared here.
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  adapt Incomplete;
}

// CHECK:STDOUT: --- fail_incomplete_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %AdaptIncomplete: type = class_type @AdaptIncomplete [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .AdaptIncomplete = %AdaptIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {}
// CHECK:STDOUT:   %AdaptIncomplete.decl: type = class_decl @AdaptIncomplete [template = constants.%AdaptIncomplete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptIncomplete {
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, file.%Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:   adapt_decl <error>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptIncomplete
// CHECK:STDOUT: }
// CHECK:STDOUT:

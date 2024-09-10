// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_import_incomplete_return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_import_incomplete_return.carbon

// --- fail_incomplete_return.carbon

library "[[@TEST_NAME]]";

class C;
class D;

fn ReturnCUnused() -> C;
fn ReturnCUsed() -> C;
fn ReturnDUnused() -> D;
fn ReturnDUsed() -> D;

fn Call() {
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE+10]]:3: ERROR: Function returns incomplete type `C`.
  // CHECK:STDERR:   ReturnCUsed();
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE-12]]:1: Class was forward declared here.
  // CHECK:STDERR: class C;
  // CHECK:STDERR: ^~~~~~~~
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE-11]]:18: Return type declared here.
  // CHECK:STDERR: fn ReturnCUsed() -> C;
  // CHECK:STDERR:                  ^~~~
  // CHECK:STDERR:
  ReturnCUsed();
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE+10]]:3: ERROR: Function returns incomplete type `D`.
  // CHECK:STDERR:   ReturnDUsed();
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE-22]]:1: Class was forward declared here.
  // CHECK:STDERR: class D;
  // CHECK:STDERR: ^~~~~~~~
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE-20]]:18: Return type declared here.
  // CHECK:STDERR: fn ReturnDUsed() -> D;
  // CHECK:STDERR:                  ^~~~
  // CHECK:STDERR:
  ReturnDUsed();
}

class D {}

// --- fail_use_imported.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_use_imported.carbon:[[@LINE+3]]:1: ERROR: Imported API 'Main//api' not found.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
import library "api";

fn CallFAndGIncomplete() {
  ReturnCUnused();
  ReturnCUsed();
  ReturnDUnused();
  ReturnDUsed();
}

// CHECK:STDOUT: --- fail_incomplete_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %ReturnCUnused.type: type = fn_type @ReturnCUnused [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %ReturnCUnused: %ReturnCUnused.type = struct_value () [template]
// CHECK:STDOUT:   %ReturnCUsed.type: type = fn_type @ReturnCUsed [template]
// CHECK:STDOUT:   %ReturnCUsed: %ReturnCUsed.type = struct_value () [template]
// CHECK:STDOUT:   %ReturnDUnused.type: type = fn_type @ReturnDUnused [template]
// CHECK:STDOUT:   %ReturnDUnused: %ReturnDUnused.type = struct_value () [template]
// CHECK:STDOUT:   %ReturnDUsed.type: type = fn_type @ReturnDUsed [template]
// CHECK:STDOUT:   %ReturnDUsed: %ReturnDUsed.type = struct_value () [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl.loc5
// CHECK:STDOUT:     .ReturnCUnused = %ReturnCUnused.decl
// CHECK:STDOUT:     .ReturnCUsed = %ReturnCUsed.decl
// CHECK:STDOUT:     .ReturnDUnused = %ReturnDUnused.decl
// CHECK:STDOUT:     .ReturnDUsed = %ReturnDUsed.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl.loc5: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %ReturnCUnused.decl: %ReturnCUnused.type = fn_decl @ReturnCUnused [template = constants.%ReturnCUnused] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @ReturnCUnused.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnCUsed.decl: %ReturnCUsed.type = fn_decl @ReturnCUsed [template = constants.%ReturnCUsed] {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @ReturnCUsed.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnDUnused.decl: %ReturnDUnused.type = fn_decl @ReturnDUnused [template = constants.%ReturnDUnused] {
// CHECK:STDOUT:     %D.ref.loc9: type = name_ref D, %D.decl.loc5 [template = constants.%D]
// CHECK:STDOUT:     @ReturnDUnused.%return: ref %D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnDUsed.decl: %ReturnDUsed.type = fn_decl @ReturnDUsed [template = constants.%ReturnDUsed] {
// CHECK:STDOUT:     %D.ref.loc10: type = name_ref D, %D.decl.loc5 [template = constants.%D]
// CHECK:STDOUT:     @ReturnDUsed.%return: ref %D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {}
// CHECK:STDOUT:   %D.decl.loc37: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnCUnused() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnCUsed() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnDUnused() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnDUsed() -> %D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnCUsed.ref: %ReturnCUsed.type = name_ref ReturnCUsed, file.%ReturnCUsed.decl [template = constants.%ReturnCUsed]
// CHECK:STDOUT:   %ReturnCUsed.call: init <error> = call %ReturnCUsed.ref()
// CHECK:STDOUT:   %ReturnDUsed.ref: %ReturnDUsed.type = name_ref ReturnDUsed, file.%ReturnDUsed.decl [template = constants.%ReturnDUsed]
// CHECK:STDOUT:   %ReturnDUsed.call: init <error> = call %ReturnDUsed.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_imported.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CallFAndGIncomplete.type: type = fn_type @CallFAndGIncomplete [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %CallFAndGIncomplete: %CallFAndGIncomplete.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .CallFAndGIncomplete = %CallFAndGIncomplete.decl
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %CallFAndGIncomplete.decl: %CallFAndGIncomplete.type = fn_decl @CallFAndGIncomplete [template = constants.%CallFAndGIncomplete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFAndGIncomplete() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnCUnused.ref: <error> = name_ref ReturnCUnused, <error> [template = <error>]
// CHECK:STDOUT:   %ReturnCUsed.ref: <error> = name_ref ReturnCUsed, <error> [template = <error>]
// CHECK:STDOUT:   %ReturnDUnused.ref: <error> = name_ref ReturnDUnused, <error> [template = <error>]
// CHECK:STDOUT:   %ReturnDUsed.ref: <error> = name_ref ReturnDUsed, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

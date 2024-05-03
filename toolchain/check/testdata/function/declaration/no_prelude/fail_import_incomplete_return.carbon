// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_incomplete_return.carbon

library "api" api;

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

library "use_imported" api;

import library "api";

fn CallFAndGIncomplete() {
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE+15]]:3: ERROR: Function returns incomplete type `C`.
  // CHECK:STDERR:   ReturnCUnused();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE-6]]:1: In import.
  // CHECK:STDERR: import library "api";
  // CHECK:STDERR: ^~~~~~
  // CHECK:STDERR: fail_incomplete_return.carbon:4:1: Class was forward declared here.
  // CHECK:STDERR: class C;
  // CHECK:STDERR: ^~~~~~~~
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE-12]]:1: In import.
  // CHECK:STDERR: import library "api";
  // CHECK:STDERR: ^~~~~~
  // CHECK:STDERR: fail_incomplete_return.carbon:7:20: Return type declared here.
  // CHECK:STDERR: fn ReturnCUnused() -> C;
  // CHECK:STDERR:                    ^~~~
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
// CHECK:STDOUT:   %ReturnCUnused: type = fn_type @ReturnCUnused [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: ReturnCUnused = struct_value () [template]
// CHECK:STDOUT:   %ReturnCUsed: type = fn_type @ReturnCUsed [template]
// CHECK:STDOUT:   %struct.2: ReturnCUsed = struct_value () [template]
// CHECK:STDOUT:   %ReturnDUnused: type = fn_type @ReturnDUnused [template]
// CHECK:STDOUT:   %struct.3: ReturnDUnused = struct_value () [template]
// CHECK:STDOUT:   %ReturnDUsed: type = fn_type @ReturnDUsed [template]
// CHECK:STDOUT:   %struct.4: ReturnDUsed = struct_value () [template]
// CHECK:STDOUT:   %Call: type = fn_type @Call [template]
// CHECK:STDOUT:   %struct.5: Call = struct_value () [template]
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
// CHECK:STDOUT:   %ReturnCUnused.decl: ReturnCUnused = fn_decl @ReturnCUnused [template = constants.%struct.1] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @ReturnCUnused.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnCUsed.decl: ReturnCUsed = fn_decl @ReturnCUsed [template = constants.%struct.2] {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @ReturnCUsed.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnDUnused.decl: ReturnDUnused = fn_decl @ReturnDUnused [template = constants.%struct.3] {
// CHECK:STDOUT:     %D.ref.loc9: type = name_ref D, %D.decl.loc5 [template = constants.%D]
// CHECK:STDOUT:     @ReturnDUnused.%return: ref D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnDUsed.decl: ReturnDUsed = fn_decl @ReturnDUsed [template = constants.%struct.4] {
// CHECK:STDOUT:     %D.ref.loc10: type = name_ref D, %D.decl.loc5 [template = constants.%D]
// CHECK:STDOUT:     @ReturnDUsed.%return: ref D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: Call = fn_decl @Call [template = constants.%struct.5] {}
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
// CHECK:STDOUT: fn @ReturnCUnused() -> C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnCUsed() -> C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnDUnused() -> D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnDUsed() -> D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnCUsed.ref: ReturnCUsed = name_ref ReturnCUsed, file.%ReturnCUsed.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %ReturnCUsed.call: init <error> = call %ReturnCUsed.ref()
// CHECK:STDOUT:   %ReturnDUsed.ref: ReturnDUsed = name_ref ReturnDUsed, file.%ReturnDUsed.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %ReturnDUsed.call: init <error> = call %ReturnDUsed.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_imported.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CallFAndGIncomplete: type = fn_type @CallFAndGIncomplete [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: CallFAndGIncomplete = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %ReturnCUnused: type = fn_type @ReturnCUnused [template]
// CHECK:STDOUT:   %struct.2: ReturnCUnused = struct_value () [template]
// CHECK:STDOUT:   %ReturnCUsed: type = fn_type @ReturnCUsed [template]
// CHECK:STDOUT:   %struct.3: ReturnCUsed = struct_value () [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %ReturnDUnused: type = fn_type @ReturnDUnused [template]
// CHECK:STDOUT:   %struct.4: ReturnDUnused = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %ReturnDUsed: type = fn_type @ReturnDUsed [template]
// CHECK:STDOUT:   %struct.5: ReturnDUsed = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.1
// CHECK:STDOUT:     .D = %import_ref.2
// CHECK:STDOUT:     .ReturnCUnused = %import_ref.3
// CHECK:STDOUT:     .ReturnCUsed = %import_ref.4
// CHECK:STDOUT:     .ReturnDUnused = %import_ref.5
// CHECK:STDOUT:     .ReturnDUsed = %import_ref.6
// CHECK:STDOUT:     .Call = %import_ref.7
// CHECK:STDOUT:     .CallFAndGIncomplete = %CallFAndGIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: ReturnCUnused = import_ref ir1, inst+7, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.4: ReturnCUsed = import_ref ir1, inst+13, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.5: ReturnDUnused = import_ref ir1, inst+18, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   %import_ref.6: ReturnDUsed = import_ref ir1, inst+23, loaded [template = constants.%struct.5]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+26, unloaded
// CHECK:STDOUT:   %CallFAndGIncomplete.decl: CallFAndGIncomplete = fn_decl @CallFAndGIncomplete [template = constants.%struct.1] {}
// CHECK:STDOUT:   %import_ref.8 = import_ref ir1, inst+4, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFAndGIncomplete() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnCUnused.ref: ReturnCUnused = name_ref ReturnCUnused, file.%import_ref.3 [template = constants.%struct.2]
// CHECK:STDOUT:   %ReturnCUnused.call: init <error> = call %ReturnCUnused.ref()
// CHECK:STDOUT:   %ReturnCUsed.ref: ReturnCUsed = name_ref ReturnCUsed, file.%import_ref.4 [template = constants.%struct.3]
// CHECK:STDOUT:   %ReturnCUsed.call: init <error> = call %ReturnCUsed.ref()
// CHECK:STDOUT:   %ReturnDUnused.ref: ReturnDUnused = name_ref ReturnDUnused, file.%import_ref.5 [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc24_16.1: ref D = temporary_storage
// CHECK:STDOUT:   %ReturnDUnused.call: init D = call %ReturnDUnused.ref() to %.loc24_16.1
// CHECK:STDOUT:   %.loc24_16.2: ref D = temporary %.loc24_16.1, %ReturnDUnused.call
// CHECK:STDOUT:   %ReturnDUsed.ref: ReturnDUsed = name_ref ReturnDUsed, file.%import_ref.6 [template = constants.%struct.5]
// CHECK:STDOUT:   %ReturnDUsed.call: init <error> = call %ReturnDUsed.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnCUnused() -> C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnCUsed() -> C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnDUnused() -> D;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnDUsed() -> D;
// CHECK:STDOUT:

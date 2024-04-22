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
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl.loc5
// CHECK:STDOUT:     .ReturnCUnused = %ReturnCUnused
// CHECK:STDOUT:     .ReturnCUsed = %ReturnCUsed
// CHECK:STDOUT:     .ReturnDUnused = %ReturnDUnused
// CHECK:STDOUT:     .ReturnDUsed = %ReturnDUsed
// CHECK:STDOUT:     .Call = %Call
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl.loc5: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %ReturnCUnused: <function> = fn_decl @ReturnCUnused [template] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @ReturnCUnused.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnCUsed: <function> = fn_decl @ReturnCUsed [template] {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @ReturnCUsed.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnDUnused: <function> = fn_decl @ReturnDUnused [template] {
// CHECK:STDOUT:     %D.ref.loc9: type = name_ref D, %D.decl.loc5 [template = constants.%D]
// CHECK:STDOUT:     @ReturnDUnused.%return: ref D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnDUsed: <function> = fn_decl @ReturnDUsed [template] {
// CHECK:STDOUT:     %D.ref.loc10: type = name_ref D, %D.decl.loc5 [template = constants.%D]
// CHECK:STDOUT:     @ReturnDUsed.%return: ref D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call: <function> = fn_decl @Call [template] {}
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
// CHECK:STDOUT:   %ReturnCUsed.ref: <function> = name_ref ReturnCUsed, file.%ReturnCUsed [template = file.%ReturnCUsed]
// CHECK:STDOUT:   %ReturnCUsed.call: init <error> = call %ReturnCUsed.ref()
// CHECK:STDOUT:   %ReturnDUsed.ref: <function> = name_ref ReturnDUsed, file.%ReturnDUsed [template = file.%ReturnDUsed]
// CHECK:STDOUT:   %ReturnDUsed.call: init <error> = call %ReturnDUsed.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_imported.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
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
// CHECK:STDOUT:     .CallFAndGIncomplete = %CallFAndGIncomplete
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir2, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir2, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir2, inst+7, loc_14 [template = imports.%ReturnCUnused]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir2, inst+10, loc_18 [template = imports.%ReturnCUsed]
// CHECK:STDOUT:   %import_ref.5: <function> = import_ref ir2, inst+13, loc_22 [template = imports.%ReturnDUnused]
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir2, inst+16, loc_26 [template = imports.%ReturnDUsed]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir2, inst+17, unloaded
// CHECK:STDOUT:   %CallFAndGIncomplete: <function> = fn_decl @CallFAndGIncomplete [template] {}
// CHECK:STDOUT:   %import_ref.8 = import_ref ir2, inst+4, unloaded
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %ReturnCUnused.ref: <function> = name_ref ReturnCUnused, file.%import_ref.3 [template = imports.%ReturnCUnused]
// CHECK:STDOUT:   %ReturnCUnused.call: init <error> = call %ReturnCUnused.ref()
// CHECK:STDOUT:   %ReturnCUsed.ref: <function> = name_ref ReturnCUsed, file.%import_ref.4 [template = imports.%ReturnCUsed]
// CHECK:STDOUT:   %ReturnCUsed.call: init <error> = call %ReturnCUsed.ref()
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %ReturnDUnused.ref: <function> = name_ref ReturnDUnused, file.%import_ref.5 [template = imports.%ReturnDUnused]
// CHECK:STDOUT:   %.loc24_16.1: ref D = temporary_storage
// CHECK:STDOUT:   %ReturnDUnused.call: init D = call %ReturnDUnused.ref() to %.loc24_16.1
// CHECK:STDOUT:   %.loc24_16.2: ref D = temporary %.loc24_16.1, %ReturnDUnused.call
// CHECK:STDOUT:   %ReturnDUsed.ref: <function> = name_ref ReturnDUsed, file.%import_ref.6 [template = imports.%ReturnDUsed]
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

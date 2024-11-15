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
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE+10]]:3: error: function returns incomplete type `C` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   ReturnCUsed();
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE-12]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR: ^~~~~~~~
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE-11]]:18: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: fn ReturnCUsed() -> C;
  // CHECK:STDERR:                  ^~~~
  // CHECK:STDERR:
  ReturnCUsed();
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE+10]]:3: error: function returns incomplete type `D` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   ReturnDUsed();
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE-22]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class D;
  // CHECK:STDERR: ^~~~~~~~
  // CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE-20]]:18: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: fn ReturnDUsed() -> D;
  // CHECK:STDERR:                  ^~~~
  // CHECK:STDERR:
  ReturnDUsed();
}

class D {}

// --- fail_use_imported.carbon

library "[[@TEST_NAME]]";

import library "incomplete_return";

fn CallFAndGIncomplete() {
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE+12]]:3: error: function returns incomplete type `C` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   ReturnCUnused();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE-6]]:1: in import [InImport]
  // CHECK:STDERR: fail_incomplete_return.carbon:4:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR: ^~~~~~~~
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE-10]]:1: in import [InImport]
  // CHECK:STDERR: fail_incomplete_return.carbon:7:20: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: fn ReturnCUnused() -> C;
  // CHECK:STDERR:                    ^~~~
  // CHECK:STDERR:
  ReturnCUnused();
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE+11]]:3: error: function returns incomplete type `C` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   ReturnCUsed();
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: fail_incomplete_return.carbon:4:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR: ^~~~~~~~
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE-23]]:1: in import [InImport]
  // CHECK:STDERR: fail_incomplete_return.carbon:8:18: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: fn ReturnCUsed() -> C;
  // CHECK:STDERR:                  ^~~~
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
// CHECK:STDOUT:   %ReturnCUnused: %ReturnCUnused.type = struct_value () [template]
// CHECK:STDOUT:   %ReturnCUsed.type: type = fn_type @ReturnCUsed [template]
// CHECK:STDOUT:   %ReturnCUsed: %ReturnCUsed.type = struct_value () [template]
// CHECK:STDOUT:   %ReturnDUnused.type: type = fn_type @ReturnDUnused [template]
// CHECK:STDOUT:   %ReturnDUnused: %ReturnDUnused.type = struct_value () [template]
// CHECK:STDOUT:   %ReturnDUsed.type: type = fn_type @ReturnDUsed [template]
// CHECK:STDOUT:   %ReturnDUsed: %ReturnDUsed.type = struct_value () [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
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
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl.loc5: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %ReturnCUnused.decl: %ReturnCUnused.type = fn_decl @ReturnCUnused [template = constants.%ReturnCUnused] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnCUsed.decl: %ReturnCUsed.type = fn_decl @ReturnCUsed [template = constants.%ReturnCUsed] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnDUnused.decl: %ReturnDUnused.type = fn_decl @ReturnDUnused [template = constants.%ReturnDUnused] {
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl.loc5 [template = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnDUsed.decl: %ReturnDUsed.type = fn_decl @ReturnDUsed [template = constants.%ReturnDUsed] {
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl.loc5 [template = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {} {}
// CHECK:STDOUT:   %D.decl.loc37: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc37: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
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
// CHECK:STDOUT:   %CallFAndGIncomplete: %CallFAndGIncomplete.type = struct_value () [template]
// CHECK:STDOUT:   %ReturnCUnused.type: type = fn_type @ReturnCUnused [template]
// CHECK:STDOUT:   %ReturnCUnused: %ReturnCUnused.type = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %ReturnCUsed.type: type = fn_type @ReturnCUsed [template]
// CHECK:STDOUT:   %ReturnCUsed: %ReturnCUsed.type = struct_value () [template]
// CHECK:STDOUT:   %ReturnDUnused.type: type = fn_type @ReturnDUnused [template]
// CHECK:STDOUT:   %ReturnDUnused: %ReturnDUnused.type = struct_value () [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %ReturnDUsed.type: type = fn_type @ReturnDUsed [template]
// CHECK:STDOUT:   %ReturnDUsed: %ReturnDUsed.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//incomplete_return, inst+1, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//incomplete_return, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: %ReturnCUnused.type = import_ref Main//incomplete_return, inst+10, loaded [template = constants.%ReturnCUnused]
// CHECK:STDOUT:   %import_ref.4: %ReturnCUsed.type = import_ref Main//incomplete_return, inst+19, loaded [template = constants.%ReturnCUsed]
// CHECK:STDOUT:   %import_ref.5: %ReturnDUnused.type = import_ref Main//incomplete_return, inst+27, loaded [template = constants.%ReturnDUnused]
// CHECK:STDOUT:   %import_ref.6: %ReturnDUsed.type = import_ref Main//incomplete_return, inst+35, loaded [template = constants.%ReturnDUsed]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//incomplete_return, inst+38, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//incomplete_return, inst+4, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:     .D = imports.%import_ref.2
// CHECK:STDOUT:     .ReturnCUnused = imports.%import_ref.3
// CHECK:STDOUT:     .ReturnCUsed = imports.%import_ref.4
// CHECK:STDOUT:     .ReturnDUnused = imports.%import_ref.5
// CHECK:STDOUT:     .ReturnDUsed = imports.%import_ref.6
// CHECK:STDOUT:     .Call = imports.%import_ref.7
// CHECK:STDOUT:     .CallFAndGIncomplete = %CallFAndGIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %CallFAndGIncomplete.decl: %CallFAndGIncomplete.type = fn_decl @CallFAndGIncomplete [template = constants.%CallFAndGIncomplete] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFAndGIncomplete() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnCUnused.ref: %ReturnCUnused.type = name_ref ReturnCUnused, imports.%import_ref.3 [template = constants.%ReturnCUnused]
// CHECK:STDOUT:   %ReturnCUnused.call: init <error> = call %ReturnCUnused.ref()
// CHECK:STDOUT:   %ReturnCUsed.ref: %ReturnCUsed.type = name_ref ReturnCUsed, imports.%import_ref.4 [template = constants.%ReturnCUsed]
// CHECK:STDOUT:   %ReturnCUsed.call: init <error> = call %ReturnCUsed.ref()
// CHECK:STDOUT:   %ReturnDUnused.ref: %ReturnDUnused.type = name_ref ReturnDUnused, imports.%import_ref.5 [template = constants.%ReturnDUnused]
// CHECK:STDOUT:   %.loc32_16.1: ref %D = temporary_storage
// CHECK:STDOUT:   %ReturnDUnused.call: init %D = call %ReturnDUnused.ref() to %.loc32_16.1
// CHECK:STDOUT:   %.loc32_16.2: ref %D = temporary %.loc32_16.1, %ReturnDUnused.call
// CHECK:STDOUT:   %ReturnDUsed.ref: %ReturnDUsed.type = name_ref ReturnDUsed, imports.%import_ref.6 [template = constants.%ReturnDUsed]
// CHECK:STDOUT:   %.loc33_14.1: ref %D = temporary_storage
// CHECK:STDOUT:   %ReturnDUsed.call: init %D = call %ReturnDUsed.ref() to %.loc33_14.1
// CHECK:STDOUT:   %.loc33_14.2: ref %D = temporary %.loc33_14.1, %ReturnDUsed.call
// CHECK:STDOUT:   return
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

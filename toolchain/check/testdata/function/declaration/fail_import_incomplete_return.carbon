// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/fail_import_incomplete_return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/fail_import_incomplete_return.carbon

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
  // CHECK:STDERR:   ^~~~~~~~~~~~~
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
  // CHECK:STDERR:   ^~~~~~~~~~~~~
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
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
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
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE+12]]:3: error: function returns incomplete type `C` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   ReturnCUsed();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: fail_incomplete_return.carbon:4:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR: ^~~~~~~~
  // CHECK:STDERR: fail_use_imported.carbon:[[@LINE-23]]:1: in import [InImport]
  // CHECK:STDERR: fail_incomplete_return.carbon:8:18: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: fn ReturnCUsed() -> C;
  // CHECK:STDERR:                  ^~~~
  // CHECK:STDERR:
  ReturnCUsed();
  ReturnDUnused();
  ReturnDUsed();
}

// CHECK:STDOUT: --- fail_incomplete_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ReturnCUnused.type: type = fn_type @ReturnCUnused [concrete]
// CHECK:STDOUT:   %ReturnCUnused: %ReturnCUnused.type = struct_value () [concrete]
// CHECK:STDOUT:   %ReturnCUsed.type: type = fn_type @ReturnCUsed [concrete]
// CHECK:STDOUT:   %ReturnCUsed: %ReturnCUsed.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %ReturnDUnused.type: type = fn_type @ReturnDUnused [concrete]
// CHECK:STDOUT:   %ReturnDUnused: %ReturnDUnused.type = struct_value () [concrete]
// CHECK:STDOUT:   %ReturnDUsed.type: type = fn_type @ReturnDUsed [concrete]
// CHECK:STDOUT:   %ReturnDUsed: %ReturnDUsed.type = struct_value () [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @D.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.a94: type = pattern_type %ptr.19c [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl.loc5
// CHECK:STDOUT:     .ReturnCUnused = %ReturnCUnused.decl
// CHECK:STDOUT:     .ReturnCUsed = %ReturnCUsed.decl
// CHECK:STDOUT:     .ReturnDUnused = %ReturnDUnused.decl
// CHECK:STDOUT:     .ReturnDUsed = %ReturnDUsed.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl.loc5: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %ReturnCUnused.decl: %ReturnCUnused.type = fn_decl @ReturnCUnused [concrete = constants.%ReturnCUnused] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnCUsed.decl: %ReturnCUsed.type = fn_decl @ReturnCUsed [concrete = constants.%ReturnCUsed] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnDUnused.decl: %ReturnDUnused.type = fn_decl @ReturnDUnused [concrete = constants.%ReturnDUnused] {
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl.loc5 [concrete = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnDUsed.decl: %ReturnDUsed.type = fn_decl @ReturnDUsed [concrete = constants.%ReturnDUsed] {
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl.loc5 [concrete = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT:   %D.decl.loc37: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Destroy.impl: constants.%D as constants.%Destroy.type {
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.decl: %D.as.Destroy.impl.Op.type = fn_decl @D.as.Destroy.impl.Op [concrete = constants.%D.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a94 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a94 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc37: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.19c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %self: %ptr.19c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @D.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:   %ReturnCUsed.ref: %ReturnCUsed.type = name_ref ReturnCUsed, file.%ReturnCUsed.decl [concrete = constants.%ReturnCUsed]
// CHECK:STDOUT:   %ReturnCUsed.call: init <error> = call %ReturnCUsed.ref()
// CHECK:STDOUT:   %ReturnDUsed.ref: %ReturnDUsed.type = name_ref ReturnDUsed, file.%ReturnDUsed.decl [concrete = constants.%ReturnDUsed]
// CHECK:STDOUT:   %ReturnDUsed.call: init <error> = call %ReturnDUsed.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Destroy.impl.Op(%self.param: %ptr.19c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use_imported.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CallFAndGIncomplete.type: type = fn_type @CallFAndGIncomplete [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %CallFAndGIncomplete: %CallFAndGIncomplete.type = struct_value () [concrete]
// CHECK:STDOUT:   %ReturnCUnused.type: type = fn_type @ReturnCUnused [concrete]
// CHECK:STDOUT:   %ReturnCUnused: %ReturnCUnused.type = struct_value () [concrete]
// CHECK:STDOUT:   %ReturnCUsed.type: type = fn_type @ReturnCUsed [concrete]
// CHECK:STDOUT:   %ReturnCUsed: %ReturnCUsed.type = struct_value () [concrete]
// CHECK:STDOUT:   %ReturnDUnused.type: type = fn_type @ReturnDUnused [concrete]
// CHECK:STDOUT:   %ReturnDUnused: %ReturnDUnused.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.b3dde2.1: type = class_type @D.1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ReturnDUsed.type: type = fn_type @ReturnDUsed [concrete]
// CHECK:STDOUT:   %ReturnDUsed: %ReturnDUsed.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.b6f: <witness> = impl_witness imports.%Destroy.impl_witness_table.cd7 [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D.b3dde2.1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C = import_ref Main//incomplete_return, C, unloaded
// CHECK:STDOUT:   %Main.D = import_ref Main//incomplete_return, D, unloaded
// CHECK:STDOUT:   %Main.ReturnCUnused: %ReturnCUnused.type = import_ref Main//incomplete_return, ReturnCUnused, loaded [concrete = constants.%ReturnCUnused]
// CHECK:STDOUT:   %Main.ReturnCUsed: %ReturnCUsed.type = import_ref Main//incomplete_return, ReturnCUsed, loaded [concrete = constants.%ReturnCUsed]
// CHECK:STDOUT:   %Main.ReturnDUnused: %ReturnDUnused.type = import_ref Main//incomplete_return, ReturnDUnused, loaded [concrete = constants.%ReturnDUnused]
// CHECK:STDOUT:   %Main.ReturnDUsed: %ReturnDUsed.type = import_ref Main//incomplete_return, ReturnDUsed, loaded [concrete = constants.%ReturnDUsed]
// CHECK:STDOUT:   %Main.Call = import_ref Main//incomplete_return, Call, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//incomplete_return, loc37_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.cabe25.1 = import_ref Main//incomplete_return, inst21 [no loc], unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.7ba: <witness> = import_ref Main//incomplete_return, loc37_9, loaded [concrete = constants.%Destroy.impl_witness.b6f]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//incomplete_return, loc37_10, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.cabe25.2 = import_ref Main//incomplete_return, inst21 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.4aa: type = import_ref Main//incomplete_return, inst21 [no loc], loaded [concrete = constants.%D.b3dde2.1]
// CHECK:STDOUT:   %Main.import_ref.cb9: type = import_ref Main//incomplete_return, inst68 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.283: %D.as.Destroy.impl.Op.type = import_ref Main//incomplete_return, loc37_9, loaded [concrete = constants.%D.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.cd7 = impl_witness_table (%Main.import_ref.283), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .ReturnCUnused = imports.%Main.ReturnCUnused
// CHECK:STDOUT:     .ReturnCUsed = imports.%Main.ReturnCUsed
// CHECK:STDOUT:     .ReturnDUnused = imports.%Main.ReturnDUnused
// CHECK:STDOUT:     .ReturnDUsed = imports.%Main.ReturnDUsed
// CHECK:STDOUT:     .Call = imports.%Main.Call
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .CallFAndGIncomplete = %CallFAndGIncomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %CallFAndGIncomplete.decl: %CallFAndGIncomplete.type = fn_decl @CallFAndGIncomplete [concrete = constants.%CallFAndGIncomplete] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Destroy.impl: imports.%Main.import_ref.4aa as imports.%Main.import_ref.cb9 [from "fail_incomplete_return.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.7ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "fail_incomplete_return.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: class @D.1 [from "fail_incomplete_return.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.cabe25.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D.2 [from "fail_incomplete_return.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.cabe25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFAndGIncomplete() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnCUnused.ref: %ReturnCUnused.type = name_ref ReturnCUnused, imports.%Main.ReturnCUnused [concrete = constants.%ReturnCUnused]
// CHECK:STDOUT:   %ReturnCUnused.call: init <error> = call %ReturnCUnused.ref()
// CHECK:STDOUT:   %ReturnCUsed.ref: %ReturnCUsed.type = name_ref ReturnCUsed, imports.%Main.ReturnCUsed [concrete = constants.%ReturnCUsed]
// CHECK:STDOUT:   %ReturnCUsed.call: init <error> = call %ReturnCUsed.ref()
// CHECK:STDOUT:   %ReturnDUnused.ref: %ReturnDUnused.type = name_ref ReturnDUnused, imports.%Main.ReturnDUnused [concrete = constants.%ReturnDUnused]
// CHECK:STDOUT:   %.loc33_17.1: ref %D.b3dde2.1 = temporary_storage
// CHECK:STDOUT:   %ReturnDUnused.call: init %D.b3dde2.1 = call %ReturnDUnused.ref() to %.loc33_17.1
// CHECK:STDOUT:   %.loc33_17.2: ref %D.b3dde2.1 = temporary %.loc33_17.1, %ReturnDUnused.call
// CHECK:STDOUT:   %ReturnDUsed.ref: %ReturnDUsed.type = name_ref ReturnDUsed, imports.%Main.ReturnDUsed [concrete = constants.%ReturnDUsed]
// CHECK:STDOUT:   %.loc34_15.1: ref %D.b3dde2.1 = temporary_storage
// CHECK:STDOUT:   %ReturnDUsed.call: init %D.b3dde2.1 = call %ReturnDUsed.ref() to %.loc34_15.1
// CHECK:STDOUT:   %.loc34_15.2: ref %D.b3dde2.1 = temporary %.loc34_15.1, %ReturnDUsed.call
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.bound.loc34: <bound method> = bound_method %.loc34_15.1, constants.%D.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc34: %ptr.19c = addr_of %.loc34_15.1
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.call.loc34: init %empty_tuple.type = call %D.as.Destroy.impl.Op.bound.loc34(%addr.loc34)
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.bound.loc33: <bound method> = bound_method %.loc33_17.1, constants.%D.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc33: %ptr.19c = addr_of %.loc33_17.1
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.call.loc33: init %empty_tuple.type = call %D.as.Destroy.impl.Op.bound.loc33(%addr.loc33)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnCUnused [from "fail_incomplete_return.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnCUsed [from "fail_incomplete_return.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnDUnused [from "fail_incomplete_return.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnDUsed [from "fail_incomplete_return.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Destroy.impl.Op = "no_op" [from "fail_incomplete_return.carbon"];
// CHECK:STDOUT:

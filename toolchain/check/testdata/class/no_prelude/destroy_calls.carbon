// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/destroy_calls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/destroy_calls.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class NoAddr {
  fn Make() -> NoAddr;
  fn destroy[self: Self]();
}

class ExplicitReturn {
  fn Make() -> ExplicitReturn;
  fn destroy[self: Self]() -> ();
}

class WithAddr {
  fn Make() -> WithAddr;
  fn destroy[addr self: Self*]();
}

// --- implicit_return.carbon

library "[[@TEST_NAME]]";
import library "types";

fn F() {
  var no_addr: NoAddr;
  var explicit_return: ExplicitReturn;
  var with_addr: WithAddr;
}

// --- nested_scope.carbon

library "[[@TEST_NAME]]";
import library "types";

fn F() {
  var no_addr: NoAddr;
  var explicit_return: ExplicitReturn;
  var with_addr: WithAddr;
  if (true) {
    // TODO: Destroy in nested scope.
    var in_scope: NoAddr;
  }
}

// --- temp.carbon

library "[[@TEST_NAME]]";
import library "types";

fn F() {
  // TODO: The scoping of these destroy calls is incorrect. Maybe we need to
  // establish statement scopes?
  NoAddr.Make();
  ExplicitReturn.Make();
  WithAddr.Make();
}

// --- fail_recovery.carbon

library "[[@TEST_NAME]]";

class NoSelf {
  // CHECK:STDERR: fail_recovery.carbon:[[@LINE+4]]:3: error: missing implicit `self` parameter [DestroyFunctionMissingSelf]
  // CHECK:STDERR:   fn destroy();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  fn destroy();
}

class Args {
  // CHECK:STDERR: fail_recovery.carbon:[[@LINE+4]]:26: error: unexpected parameter [DestroyFunctionNonEmptyExplicitParams]
  // CHECK:STDERR:   fn destroy[self: Self](x: ());
  // CHECK:STDERR:                          ^~~~~
  // CHECK:STDERR:
  fn destroy[self: Self](x: ());
}

fn F() {
  var a: NoSelf;
  var b: Args;
}

// --- fail_not_breaking_generics.carbon
// CHECK:STDERR: fail_not_breaking_generics.carbon: error: value of type `<dependent type>` is not callable [CallToNonCallable]
// CHECK:STDERR:

class C(template T:! type) {}

fn F(template T:! type) {
  var v: C(T);
}

fn G() { F({}); }

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %Make.type.bc9: type = fn_type @Make.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make.1be: %Make.type.bc9 = struct_value () [concrete]
// CHECK:STDOUT:   %destroy.type.bc5: type = fn_type @destroy.1 [concrete]
// CHECK:STDOUT:   %destroy.60f: %destroy.type.bc5 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %Make.type.378: type = fn_type @Make.2 [concrete]
// CHECK:STDOUT:   %Make.960: %Make.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %destroy.type.dfa: type = fn_type @destroy.2 [concrete]
// CHECK:STDOUT:   %destroy.539: %destroy.type.dfa = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %Make.type.e14: type = fn_type @Make.3 [concrete]
// CHECK:STDOUT:   %Make.b0a: %Make.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %destroy.type.02f: type = fn_type @destroy.3 [concrete]
// CHECK:STDOUT:   %destroy.8d0: %destroy.type.02f = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NoAddr = %NoAddr.decl
// CHECK:STDOUT:     .ExplicitReturn = %ExplicitReturn.decl
// CHECK:STDOUT:     .WithAddr = %WithAddr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NoAddr.decl: type = class_decl @NoAddr [concrete = constants.%NoAddr] {} {}
// CHECK:STDOUT:   %ExplicitReturn.decl: type = class_decl @ExplicitReturn [concrete = constants.%ExplicitReturn] {} {}
// CHECK:STDOUT:   %WithAddr.decl: type = class_decl @WithAddr [concrete = constants.%WithAddr] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoAddr {
// CHECK:STDOUT:   %Make.decl: %Make.type.bc9 = fn_decl @Make.1 [concrete = constants.%Make.1be] {
// CHECK:STDOUT:     %return.patt: %NoAddr = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %NoAddr = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %NoAddr.ref: type = name_ref NoAddr, file.%NoAddr.decl [concrete = constants.%NoAddr]
// CHECK:STDOUT:     %return.param: ref %NoAddr = out_param call_param0
// CHECK:STDOUT:     %return: ref %NoAddr = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %destroy.decl: %destroy.type.bc5 = fn_decl @destroy.1 [concrete = constants.%destroy.60f] {
// CHECK:STDOUT:     %self.patt: %NoAddr = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %NoAddr = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NoAddr = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:     %self: %NoAddr = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoAddr
// CHECK:STDOUT:   .NoAddr = <poisoned>
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .destroy = %destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ExplicitReturn {
// CHECK:STDOUT:   %Make.decl: %Make.type.378 = fn_decl @Make.2 [concrete = constants.%Make.960] {
// CHECK:STDOUT:     %return.patt: %ExplicitReturn = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ExplicitReturn = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ExplicitReturn.ref: type = name_ref ExplicitReturn, file.%ExplicitReturn.decl [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:     %return.param: ref %ExplicitReturn = out_param call_param0
// CHECK:STDOUT:     %return: ref %ExplicitReturn = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %destroy.decl: %destroy.type.dfa = fn_decl @destroy.2 [concrete = constants.%destroy.539] {
// CHECK:STDOUT:     %self.patt: %ExplicitReturn = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ExplicitReturn = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_32.2: type = converted %.loc11_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param: %ExplicitReturn = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:     %self: %ExplicitReturn = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ExplicitReturn
// CHECK:STDOUT:   .ExplicitReturn = <poisoned>
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .destroy = %destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WithAddr {
// CHECK:STDOUT:   %Make.decl: %Make.type.e14 = fn_decl @Make.3 [concrete = constants.%Make.b0a] {
// CHECK:STDOUT:     %return.patt: %WithAddr = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %WithAddr = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %WithAddr.ref: type = name_ref WithAddr, file.%WithAddr.decl [concrete = constants.%WithAddr]
// CHECK:STDOUT:     %return.param: ref %WithAddr = out_param call_param0
// CHECK:STDOUT:     %return: ref %WithAddr = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %destroy.decl: %destroy.type.02f = fn_decl @destroy.3 [concrete = constants.%destroy.8d0] {
// CHECK:STDOUT:     %self.patt: %ptr = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc16_14: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc16_29: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%WithAddr
// CHECK:STDOUT:   .WithAddr = <poisoned>
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .destroy = %destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.1() -> %NoAddr;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.1[%self.param_patt: %NoAddr]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %ExplicitReturn;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.2[%self.param_patt: %ExplicitReturn]() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.3() -> %WithAddr;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.3[addr %self.param_patt: %ptr]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %destroy.type.bc5: type = fn_type @destroy.1 [concrete]
// CHECK:STDOUT:   %destroy.60f: %destroy.type.bc5 = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %destroy.type.dfa: type = fn_type @destroy.2 [concrete]
// CHECK:STDOUT:   %destroy.539: %destroy.type.dfa = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %destroy.type.02f: type = fn_type @destroy.3 [concrete]
// CHECK:STDOUT:   %destroy.8d0: %destroy.type.02f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NoAddr: type = import_ref Main//types, NoAddr, loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.ExplicitReturn: type = import_ref Main//types, ExplicitReturn, loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.WithAddr: type = import_ref Main//types, WithAddr, loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5b = import_ref Main//types, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.7fe: %destroy.type.bc5 = import_ref Main//types, loc6_27, loaded [concrete = constants.%destroy.60f]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst38 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b14 = import_ref Main//types, loc10_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.83a: %destroy.type.dfa = import_ref Main//types, loc11_33, loaded [concrete = constants.%destroy.539]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst63 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//types, loc15_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.54e: %destroy.type.02f = import_ref Main//types, loc16_33, loaded [concrete = constants.%destroy.8d0]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NoAddr = imports.%Main.NoAddr
// CHECK:STDOUT:     .ExplicitReturn = imports.%Main.ExplicitReturn
// CHECK:STDOUT:     .WithAddr = imports.%Main.WithAddr
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoAddr [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.f42
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.b5b
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.7fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ExplicitReturn [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ee7
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.b14
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.83a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WithAddr [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.95d
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.1cb
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.54e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %no_addr.patt: %NoAddr = binding_pattern no_addr
// CHECK:STDOUT:     %.loc6_3.1: %NoAddr = var_pattern %no_addr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %no_addr.var: ref %NoAddr = var no_addr
// CHECK:STDOUT:   %destroy.ref.1: %destroy.type.bc5 = name_ref destroy, imports.%Main.import_ref.7fe [concrete = constants.%destroy.60f]
// CHECK:STDOUT:   %destroy.bound.1: <bound method> = bound_method %no_addr.var, %destroy.ref.1
// CHECK:STDOUT:   %NoAddr.ref: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %no_addr: ref %NoAddr = bind_name no_addr, %no_addr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %explicit_return.patt: %ExplicitReturn = binding_pattern explicit_return
// CHECK:STDOUT:     %.loc7_3.1: %ExplicitReturn = var_pattern %explicit_return.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %explicit_return.var: ref %ExplicitReturn = var explicit_return
// CHECK:STDOUT:   %destroy.ref.2: %destroy.type.dfa = name_ref destroy, imports.%Main.import_ref.83a [concrete = constants.%destroy.539]
// CHECK:STDOUT:   %destroy.bound.2: <bound method> = bound_method %explicit_return.var, %destroy.ref.2
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %explicit_return: ref %ExplicitReturn = bind_name explicit_return, %explicit_return.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %with_addr.patt: %WithAddr = binding_pattern with_addr
// CHECK:STDOUT:     %.loc8: %WithAddr = var_pattern %with_addr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %with_addr.var: ref %WithAddr = var with_addr
// CHECK:STDOUT:   %destroy.ref.3: %destroy.type.02f = name_ref destroy, imports.%Main.import_ref.54e [concrete = constants.%destroy.8d0]
// CHECK:STDOUT:   %destroy.bound.3: <bound method> = bound_method %with_addr.var, %destroy.ref.3
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %with_addr: ref %WithAddr = bind_name with_addr, %with_addr.var
// CHECK:STDOUT:   %addr: %ptr.b4e = addr_of %with_addr.var
// CHECK:STDOUT:   %destroy.call.1: init %empty_tuple.type = call %destroy.bound.3(%addr)
// CHECK:STDOUT:   %.loc7_3.2: %ExplicitReturn = bind_value %explicit_return.var
// CHECK:STDOUT:   %destroy.call.2: init %empty_tuple.type = call %destroy.bound.2(%.loc7_3.2)
// CHECK:STDOUT:   %.loc6_3.2: %NoAddr = bind_value %no_addr.var
// CHECK:STDOUT:   %destroy.call.3: init %empty_tuple.type = call %destroy.bound.1(%.loc6_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.1[%self.param_patt: %NoAddr]() [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.2[%self.param_patt: %ExplicitReturn]() -> %empty_tuple.type [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.3[addr <unexpected>.inst83: %ptr.b4e]() [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %destroy.type.bc5: type = fn_type @destroy.1 [concrete]
// CHECK:STDOUT:   %destroy.60f: %destroy.type.bc5 = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %destroy.type.dfa: type = fn_type @destroy.2 [concrete]
// CHECK:STDOUT:   %destroy.539: %destroy.type.dfa = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %destroy.type.02f: type = fn_type @destroy.3 [concrete]
// CHECK:STDOUT:   %destroy.8d0: %destroy.type.02f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NoAddr: type = import_ref Main//types, NoAddr, loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.ExplicitReturn: type = import_ref Main//types, ExplicitReturn, loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.WithAddr: type = import_ref Main//types, WithAddr, loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5b = import_ref Main//types, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.7fe: %destroy.type.bc5 = import_ref Main//types, loc6_27, loaded [concrete = constants.%destroy.60f]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst38 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b14 = import_ref Main//types, loc10_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.83a: %destroy.type.dfa = import_ref Main//types, loc11_33, loaded [concrete = constants.%destroy.539]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst63 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//types, loc15_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.54e: %destroy.type.02f = import_ref Main//types, loc16_33, loaded [concrete = constants.%destroy.8d0]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NoAddr = imports.%Main.NoAddr
// CHECK:STDOUT:     .ExplicitReturn = imports.%Main.ExplicitReturn
// CHECK:STDOUT:     .WithAddr = imports.%Main.WithAddr
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoAddr [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.f42
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.b5b
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.7fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ExplicitReturn [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ee7
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.b14
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.83a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WithAddr [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.95d
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.1cb
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.54e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %no_addr.patt: %NoAddr = binding_pattern no_addr
// CHECK:STDOUT:     %.loc6_3.1: %NoAddr = var_pattern %no_addr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %no_addr.var: ref %NoAddr = var no_addr
// CHECK:STDOUT:   %destroy.ref.1: %destroy.type.bc5 = name_ref destroy, imports.%Main.import_ref.7fe [concrete = constants.%destroy.60f]
// CHECK:STDOUT:   %destroy.bound.1: <bound method> = bound_method %no_addr.var, %destroy.ref.1
// CHECK:STDOUT:   %NoAddr.ref.loc6: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %no_addr: ref %NoAddr = bind_name no_addr, %no_addr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %explicit_return.patt: %ExplicitReturn = binding_pattern explicit_return
// CHECK:STDOUT:     %.loc7_3.1: %ExplicitReturn = var_pattern %explicit_return.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %explicit_return.var: ref %ExplicitReturn = var explicit_return
// CHECK:STDOUT:   %destroy.ref.2: %destroy.type.dfa = name_ref destroy, imports.%Main.import_ref.83a [concrete = constants.%destroy.539]
// CHECK:STDOUT:   %destroy.bound.2: <bound method> = bound_method %explicit_return.var, %destroy.ref.2
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %explicit_return: ref %ExplicitReturn = bind_name explicit_return, %explicit_return.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %with_addr.patt: %WithAddr = binding_pattern with_addr
// CHECK:STDOUT:     %.loc8: %WithAddr = var_pattern %with_addr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %with_addr.var: ref %WithAddr = var with_addr
// CHECK:STDOUT:   %destroy.ref.3: %destroy.type.02f = name_ref destroy, imports.%Main.import_ref.54e [concrete = constants.%destroy.8d0]
// CHECK:STDOUT:   %destroy.bound.3: <bound method> = bound_method %with_addr.var, %destroy.ref.3
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %with_addr: ref %WithAddr = bind_name with_addr, %with_addr.var
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %in_scope.patt: %NoAddr = binding_pattern in_scope
// CHECK:STDOUT:     %.loc11: %NoAddr = var_pattern %in_scope.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %in_scope.var: ref %NoAddr = var in_scope
// CHECK:STDOUT:   %destroy.ref.4: %destroy.type.bc5 = name_ref destroy, imports.%Main.import_ref.7fe [concrete = constants.%destroy.60f]
// CHECK:STDOUT:   %destroy.bound.4: <bound method> = bound_method %in_scope.var, %destroy.ref.4
// CHECK:STDOUT:   %NoAddr.ref.loc11: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %in_scope: ref %NoAddr = bind_name in_scope, %in_scope.var
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %addr: %ptr.b4e = addr_of %with_addr.var
// CHECK:STDOUT:   %destroy.call.1: init %empty_tuple.type = call %destroy.bound.3(%addr)
// CHECK:STDOUT:   %.loc7_3.2: %ExplicitReturn = bind_value %explicit_return.var
// CHECK:STDOUT:   %destroy.call.2: init %empty_tuple.type = call %destroy.bound.2(%.loc7_3.2)
// CHECK:STDOUT:   %.loc6_3.2: %NoAddr = bind_value %no_addr.var
// CHECK:STDOUT:   %destroy.call.3: init %empty_tuple.type = call %destroy.bound.1(%.loc6_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.1[%self.param_patt: %NoAddr]() [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.2[%self.param_patt: %ExplicitReturn]() -> %empty_tuple.type [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.3[addr <unexpected>.inst83: %ptr.b4e]() [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- temp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Make.type.bc9: type = fn_type @Make.1 [concrete]
// CHECK:STDOUT:   %Make.1be: %Make.type.bc9 = struct_value () [concrete]
// CHECK:STDOUT:   %destroy.type.bc5: type = fn_type @destroy.1 [concrete]
// CHECK:STDOUT:   %destroy.60f: %destroy.type.bc5 = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %Make.type.378: type = fn_type @Make.2 [concrete]
// CHECK:STDOUT:   %Make.960: %Make.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %destroy.type.dfa: type = fn_type @destroy.2 [concrete]
// CHECK:STDOUT:   %destroy.539: %destroy.type.dfa = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %Make.type.e14: type = fn_type @Make.3 [concrete]
// CHECK:STDOUT:   %Make.b0a: %Make.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %destroy.type.02f: type = fn_type @destroy.3 [concrete]
// CHECK:STDOUT:   %destroy.8d0: %destroy.type.02f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NoAddr: type = import_ref Main//types, NoAddr, loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.ExplicitReturn: type = import_ref Main//types, ExplicitReturn, loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.WithAddr: type = import_ref Main//types, WithAddr, loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.784: %Make.type.bc9 = import_ref Main//types, loc5_22, loaded [concrete = constants.%Make.1be]
// CHECK:STDOUT:   %Main.import_ref.7fe: %destroy.type.bc5 = import_ref Main//types, loc6_27, loaded [concrete = constants.%destroy.60f]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst38 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8e0: %Make.type.378 = import_ref Main//types, loc10_30, loaded [concrete = constants.%Make.960]
// CHECK:STDOUT:   %Main.import_ref.83a: %destroy.type.dfa = import_ref Main//types, loc11_33, loaded [concrete = constants.%destroy.539]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst63 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.974: %Make.type.e14 = import_ref Main//types, loc15_24, loaded [concrete = constants.%Make.b0a]
// CHECK:STDOUT:   %Main.import_ref.54e: %destroy.type.02f = import_ref Main//types, loc16_33, loaded [concrete = constants.%destroy.8d0]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NoAddr = imports.%Main.NoAddr
// CHECK:STDOUT:     .ExplicitReturn = imports.%Main.ExplicitReturn
// CHECK:STDOUT:     .WithAddr = imports.%Main.WithAddr
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoAddr [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.f42
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.784
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.7fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ExplicitReturn [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ee7
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.8e0
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.83a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WithAddr [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.95d
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.974
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.54e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NoAddr.ref: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Make.ref.loc8: %Make.type.bc9 = name_ref Make, imports.%Main.import_ref.784 [concrete = constants.%Make.1be]
// CHECK:STDOUT:   %.loc8_15.1: ref %NoAddr = temporary_storage
// CHECK:STDOUT:   %destroy.ref.1: %destroy.type.bc5 = name_ref destroy, imports.%Main.import_ref.7fe [concrete = constants.%destroy.60f]
// CHECK:STDOUT:   %destroy.bound.1: <bound method> = bound_method %.loc8_15.1, %destroy.ref.1
// CHECK:STDOUT:   %Make.call.loc8: init %NoAddr = call %Make.ref.loc8() to %.loc8_15.1
// CHECK:STDOUT:   %.loc8_15.2: ref %NoAddr = temporary %.loc8_15.1, %Make.call.loc8
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Make.ref.loc9: %Make.type.378 = name_ref Make, imports.%Main.import_ref.8e0 [concrete = constants.%Make.960]
// CHECK:STDOUT:   %.loc9_23.1: ref %ExplicitReturn = temporary_storage
// CHECK:STDOUT:   %destroy.ref.2: %destroy.type.dfa = name_ref destroy, imports.%Main.import_ref.83a [concrete = constants.%destroy.539]
// CHECK:STDOUT:   %destroy.bound.2: <bound method> = bound_method %.loc9_23.1, %destroy.ref.2
// CHECK:STDOUT:   %Make.call.loc9: init %ExplicitReturn = call %Make.ref.loc9() to %.loc9_23.1
// CHECK:STDOUT:   %.loc9_23.2: ref %ExplicitReturn = temporary %.loc9_23.1, %Make.call.loc9
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Make.ref.loc10: %Make.type.e14 = name_ref Make, imports.%Main.import_ref.974 [concrete = constants.%Make.b0a]
// CHECK:STDOUT:   %.loc10_17.1: ref %WithAddr = temporary_storage
// CHECK:STDOUT:   %destroy.ref.3: %destroy.type.02f = name_ref destroy, imports.%Main.import_ref.54e [concrete = constants.%destroy.8d0]
// CHECK:STDOUT:   %destroy.bound.3: <bound method> = bound_method %.loc10_17.1, %destroy.ref.3
// CHECK:STDOUT:   %Make.call.loc10: init %WithAddr = call %Make.ref.loc10() to %.loc10_17.1
// CHECK:STDOUT:   %.loc10_17.2: ref %WithAddr = temporary %.loc10_17.1, %Make.call.loc10
// CHECK:STDOUT:   %addr: %ptr.b4e = addr_of %.loc10_17.1
// CHECK:STDOUT:   %destroy.call.1: init %empty_tuple.type = call %destroy.bound.3(%addr)
// CHECK:STDOUT:   %.loc9_23.3: %ExplicitReturn = bind_value %.loc9_23.1
// CHECK:STDOUT:   %destroy.call.2: init %empty_tuple.type = call %destroy.bound.2(%.loc9_23.3)
// CHECK:STDOUT:   %.loc8_15.3: %NoAddr = bind_value %.loc8_15.1
// CHECK:STDOUT:   %destroy.call.3: init %empty_tuple.type = call %destroy.bound.1(%.loc8_15.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.1() -> %NoAddr [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.1[%self.param_patt: %NoAddr]() [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %ExplicitReturn [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.2[%self.param_patt: %ExplicitReturn]() -> %empty_tuple.type [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.3() -> %WithAddr [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.3[addr <unexpected>.inst93: %ptr.b4e]() [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_recovery.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoSelf: type = class_type @NoSelf [concrete]
// CHECK:STDOUT:   %destroy.type.d69: type = fn_type @destroy.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %destroy.ab7: %destroy.type.d69 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Args: type = class_type @Args [concrete]
// CHECK:STDOUT:   %destroy.type.ba0: type = fn_type @destroy.2 [concrete]
// CHECK:STDOUT:   %destroy.723: %destroy.type.ba0 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NoSelf = %NoSelf.decl
// CHECK:STDOUT:     .Args = %Args.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NoSelf.decl: type = class_decl @NoSelf [concrete = constants.%NoSelf] {} {}
// CHECK:STDOUT:   %Args.decl: type = class_decl @Args [concrete = constants.%Args] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoSelf {
// CHECK:STDOUT:   %destroy.decl: %destroy.type.d69 = fn_decl @destroy.1 [concrete = constants.%destroy.ab7] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoSelf
// CHECK:STDOUT:   .destroy = %destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Args {
// CHECK:STDOUT:   %destroy.decl: %destroy.type.ba0 = fn_decl @destroy.2 [concrete = constants.%destroy.723] {
// CHECK:STDOUT:     %self.patt: %Args = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Args = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Args = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Args [concrete = constants.%Args]
// CHECK:STDOUT:     %self: %Args = bind_name self, %self.param
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param1
// CHECK:STDOUT:     %.loc17_30.1: type = splice_block %.loc17_30.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc17_30.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc17_30.3: type = converted %.loc17_30.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Args
// CHECK:STDOUT:   .destroy = %destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.2[%self.param_patt: %Args]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %NoSelf = binding_pattern a
// CHECK:STDOUT:     %.loc21: %NoSelf = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %NoSelf = var a
// CHECK:STDOUT:   %destroy.ref.1: %destroy.type.d69 = name_ref destroy, @NoSelf.%destroy.decl [concrete = constants.%destroy.ab7]
// CHECK:STDOUT:   %NoSelf.ref: type = name_ref NoSelf, file.%NoSelf.decl [concrete = constants.%NoSelf]
// CHECK:STDOUT:   %a: ref %NoSelf = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %Args = binding_pattern b
// CHECK:STDOUT:     %.loc22_3.1: %Args = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Args = var b
// CHECK:STDOUT:   %destroy.ref.2: %destroy.type.ba0 = name_ref destroy, @Args.%destroy.decl [concrete = constants.%destroy.723]
// CHECK:STDOUT:   %destroy.bound: <bound method> = bound_method %b.var, %destroy.ref.2
// CHECK:STDOUT:   %Args.ref: type = name_ref Args, file.%Args.decl [concrete = constants.%Args]
// CHECK:STDOUT:   %b: ref %Args = bind_name b, %b.var
// CHECK:STDOUT:   %.loc22_3.2: %Args = bind_value %b.var
// CHECK:STDOUT:   %destroy.call.1: init %empty_tuple.type = call %destroy.bound(%.loc22_3.2)
// CHECK:STDOUT:   %destroy.call.2: init %empty_tuple.type = call %destroy.ref.1()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_breaking_generics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.f2e [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %inst.as_compatible: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.6f0: ref %C.7a7 = as_compatible @F.%v.var
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.432: <error> = splice_block <error> [concrete = <error>] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc4_18.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_18.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_18.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_15.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_18.1: type) {
// CHECK:STDOUT:   %T.loc4_18.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_18.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_18.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_18.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_15.1: type) {
// CHECK:STDOUT:   %T.loc6_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_15.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_13.2: type = class_type @C, @C(%T.loc6_15.2) [template = %C.loc7_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%C.loc7_13.2 (%C.f2e) [template = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %.3: <instruction> = refine_type_action %v.var, @F.%C.loc7_13.2 (%C.f2e) [template]
// CHECK:STDOUT:   %.4: <instruction> = access_optional_member_action %.1, destroy [template]
// CHECK:STDOUT:   %.5: type = type_of_inst %.4 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc6_15.1: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%C.loc7_13.2 (%C.f2e) = binding_pattern v
// CHECK:STDOUT:       %.loc7_3: @F.%C.loc7_13.2 (%C.f2e) = var_pattern %v.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%C.loc7_13.2 (%C.f2e) = var v
// CHECK:STDOUT:     %.1: @F.%C.loc7_13.2 (%C.f2e) = splice_inst %.3
// CHECK:STDOUT:     %.2: @F.%.5 (@F.%.5) = splice_inst %.4
// CHECK:STDOUT:     %.loc7_13: type = splice_block %C.loc7_13.1 [template = %C.loc7_13.2 (constants.%C.f2e)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_15.1 [template = %T.loc6_15.2 (constants.%T)]
// CHECK:STDOUT:       %C.loc7_13.1: type = class_type @C, @C(constants.%T) [template = %C.loc7_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%C.loc7_13.2 (%C.f2e) = bind_name v, %v.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc10_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_14: type = converted %.loc10_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%empty_struct_type) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_15.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@F.%T.loc6_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc6_15.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc6_15.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_13.2 => constants.%C.7a7
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %.3 => constants.%inst.as_compatible
// CHECK:STDOUT:   %.4 => constants.%inst.splice_block
// CHECK:STDOUT:   %.5 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc4_18.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc4_18.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

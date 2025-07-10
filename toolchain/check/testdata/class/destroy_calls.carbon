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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/destroy_calls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/destroy_calls.carbon

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

// --- not_breaking_generics.carbon
library "[[@TEST_NAME]]";

class C(template T:! type) {}

fn F(template T:! type) {
  var v: C(T);
}

fn G() { F({}); }

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %NoAddr [concrete]
// CHECK:STDOUT:   %Make.type.bc9: type = fn_type @Make.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make.1be: %Make.type.bc9 = struct_value () [concrete]
// CHECK:STDOUT:   %destroy.type.bc5: type = fn_type @destroy.1 [concrete]
// CHECK:STDOUT:   %destroy.60f: %destroy.type.bc5 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %Make.type.378: type = fn_type @Make.2 [concrete]
// CHECK:STDOUT:   %Make.960: %Make.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %destroy.type.dfa: type = fn_type @destroy.2 [concrete]
// CHECK:STDOUT:   %destroy.539: %destroy.type.dfa = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %WithAddr [concrete]
// CHECK:STDOUT:   %Make.type.e14: type = fn_type @Make.3 [concrete]
// CHECK:STDOUT:   %Make.b0a: %Make.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.4a0: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %destroy.type.02f: type = fn_type @destroy.3 [concrete]
// CHECK:STDOUT:   %destroy.8d0: %destroy.type.02f = struct_value () [concrete]
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
// CHECK:STDOUT:     .NoAddr = %NoAddr.decl
// CHECK:STDOUT:     .ExplicitReturn = %ExplicitReturn.decl
// CHECK:STDOUT:     .WithAddr = %WithAddr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NoAddr.decl: type = class_decl @NoAddr [concrete = constants.%NoAddr] {} {}
// CHECK:STDOUT:   %ExplicitReturn.decl: type = class_decl @ExplicitReturn [concrete = constants.%ExplicitReturn] {} {}
// CHECK:STDOUT:   %WithAddr.decl: type = class_decl @WithAddr [concrete = constants.%WithAddr] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoAddr {
// CHECK:STDOUT:   %Make.decl: %Make.type.bc9 = fn_decl @Make.1 [concrete = constants.%Make.1be] {
// CHECK:STDOUT:     %return.patt: %pattern_type.88f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.88f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %NoAddr.ref: type = name_ref NoAddr, file.%NoAddr.decl [concrete = constants.%NoAddr]
// CHECK:STDOUT:     %return.param: ref %NoAddr = out_param call_param0
// CHECK:STDOUT:     %return: ref %NoAddr = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %destroy.decl: %destroy.type.bc5 = fn_decl @destroy.1 [concrete = constants.%destroy.60f] {
// CHECK:STDOUT:     %self.patt: %pattern_type.88f = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.88f = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NoAddr = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:     %self: %NoAddr = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT:     %return.patt: %pattern_type.611 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.611 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ExplicitReturn.ref: type = name_ref ExplicitReturn, file.%ExplicitReturn.decl [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:     %return.param: ref %ExplicitReturn = out_param call_param0
// CHECK:STDOUT:     %return: ref %ExplicitReturn = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %destroy.decl: %destroy.type.dfa = fn_decl @destroy.2 [concrete = constants.%destroy.539] {
// CHECK:STDOUT:     %self.patt: %pattern_type.611 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.611 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_32.2: type = converted %.loc11_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param: %ExplicitReturn = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:     %self: %ExplicitReturn = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT:     %return.patt: %pattern_type.f93 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f93 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %WithAddr.ref: type = name_ref WithAddr, file.%WithAddr.decl [concrete = constants.%WithAddr]
// CHECK:STDOUT:     %return.param: ref %WithAddr = out_param call_param0
// CHECK:STDOUT:     %return: ref %WithAddr = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %destroy.decl: %destroy.type.02f = fn_decl @destroy.3 [concrete = constants.%destroy.8d0] {
// CHECK:STDOUT:     %self.patt: %pattern_type.4a0 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4a0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16_14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc16_29: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT: fn @destroy.1(%self.param: %NoAddr);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2() -> %ExplicitReturn;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.2(%self.param: %ExplicitReturn) -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.3() -> %WithAddr;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.3(%self.param: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %NoAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %WithAddr [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.8f4: type = fn_type @Op.2, @impl(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.bec: %Op.type.8f4 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %Op.specific_fn.b42: <specific function> = specific_function %Op.bec, @Op.2(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.type.641: type = fn_type @Op.2, @impl(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.8e6: %Op.type.641 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b0a: type = ptr_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %Op.specific_fn.523: <specific function> = specific_function %Op.8e6, @Op.2(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.type.73f: type = fn_type @Op.2, @impl(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.d78: %Op.type.73f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4b8: type = ptr_type %NoAddr [concrete]
// CHECK:STDOUT:   %Op.specific_fn.8fe: <specific function> = specific_function %Op.d78, @Op.2(%NoAddr) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NoAddr: type = import_ref Main//types, NoAddr, loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.ExplicitReturn: type = import_ref Main//types, ExplicitReturn, loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.WithAddr: type = import_ref Main//types, WithAddr, loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5b = import_ref Main//types, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.1ed = import_ref Main//types, loc6_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst42 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b14 = import_ref Main//types, loc10_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.f79 = import_ref Main//types, loc11_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst70 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//types, loc15_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.675 = import_ref Main//types, loc16_33, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NoAddr = imports.%Main.NoAddr
// CHECK:STDOUT:     .ExplicitReturn = imports.%Main.ExplicitReturn
// CHECK:STDOUT:     .WithAddr = imports.%Main.WithAddr
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
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
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.1ed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ExplicitReturn [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ee7
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.b14
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WithAddr [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.95d
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.1cb
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.675
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %no_addr.patt: %pattern_type.88f = binding_pattern no_addr [concrete]
// CHECK:STDOUT:     %no_addr.var_patt: %pattern_type.88f = var_pattern %no_addr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %no_addr.var: ref %NoAddr = var %no_addr.var_patt
// CHECK:STDOUT:   %NoAddr.ref: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %no_addr: ref %NoAddr = bind_name no_addr, %no_addr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %explicit_return.patt: %pattern_type.611 = binding_pattern explicit_return [concrete]
// CHECK:STDOUT:     %explicit_return.var_patt: %pattern_type.611 = var_pattern %explicit_return.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %explicit_return.var: ref %ExplicitReturn = var %explicit_return.var_patt
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %explicit_return: ref %ExplicitReturn = bind_name explicit_return, %explicit_return.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %with_addr.patt: %pattern_type.f93 = binding_pattern with_addr [concrete]
// CHECK:STDOUT:     %with_addr.var_patt: %pattern_type.f93 = var_pattern %with_addr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %with_addr.var: ref %WithAddr = var %with_addr.var_patt
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %with_addr: ref %WithAddr = bind_name with_addr, %with_addr.var
// CHECK:STDOUT:   %Op.bound.loc8: <bound method> = bound_method %with_addr.var, constants.%Op.bec
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.bec, @Op.2(constants.%WithAddr) [concrete = constants.%Op.specific_fn.b42]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %with_addr.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc8: %ptr.b4e = addr_of %with_addr.var
// CHECK:STDOUT:   %no_op.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
// CHECK:STDOUT:   %Op.bound.loc7: <bound method> = bound_method %explicit_return.var, constants.%Op.8e6
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.8e6, @Op.2(constants.%ExplicitReturn) [concrete = constants.%Op.specific_fn.523]
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %explicit_return.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc7: %ptr.b0a = addr_of %explicit_return.var
// CHECK:STDOUT:   %no_op.loc7: init %empty_tuple.type = call %bound_method.loc7(%addr.loc7)
// CHECK:STDOUT:   %Op.bound.loc6: <bound method> = bound_method %no_addr.var, constants.%Op.d78
// CHECK:STDOUT:   %Op.specific_fn.3: <specific function> = specific_function constants.%Op.d78, @Op.2(constants.%NoAddr) [concrete = constants.%Op.specific_fn.8fe]
// CHECK:STDOUT:   %bound_method.loc6: <bound method> = bound_method %no_addr.var, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc6: %ptr.4b8 = addr_of %no_addr.var
// CHECK:STDOUT:   %no_op.loc6: init %empty_tuple.type = call %bound_method.loc6(%addr.loc6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %NoAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %WithAddr [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.73f: type = fn_type @Op.2, @impl(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.d78: %Op.type.73f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4b8: type = ptr_type %NoAddr [concrete]
// CHECK:STDOUT:   %Op.specific_fn.8fe: <specific function> = specific_function %Op.d78, @Op.2(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.type.8f4: type = fn_type @Op.2, @impl(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.bec: %Op.type.8f4 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %Op.specific_fn.b42: <specific function> = specific_function %Op.bec, @Op.2(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.type.641: type = fn_type @Op.2, @impl(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.8e6: %Op.type.641 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b0a: type = ptr_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %Op.specific_fn.523: <specific function> = specific_function %Op.8e6, @Op.2(%ExplicitReturn) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NoAddr: type = import_ref Main//types, NoAddr, loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.ExplicitReturn: type = import_ref Main//types, ExplicitReturn, loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.WithAddr: type = import_ref Main//types, WithAddr, loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5b = import_ref Main//types, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.1ed = import_ref Main//types, loc6_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst42 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b14 = import_ref Main//types, loc10_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.f79 = import_ref Main//types, loc11_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst70 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//types, loc15_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.675 = import_ref Main//types, loc16_33, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NoAddr = imports.%Main.NoAddr
// CHECK:STDOUT:     .ExplicitReturn = imports.%Main.ExplicitReturn
// CHECK:STDOUT:     .WithAddr = imports.%Main.WithAddr
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
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
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.1ed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ExplicitReturn [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ee7
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.b14
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WithAddr [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.95d
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.1cb
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.675
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %no_addr.patt: %pattern_type.88f = binding_pattern no_addr [concrete]
// CHECK:STDOUT:     %no_addr.var_patt: %pattern_type.88f = var_pattern %no_addr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %no_addr.var: ref %NoAddr = var %no_addr.var_patt
// CHECK:STDOUT:   %NoAddr.ref.loc6: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %no_addr: ref %NoAddr = bind_name no_addr, %no_addr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %explicit_return.patt: %pattern_type.611 = binding_pattern explicit_return [concrete]
// CHECK:STDOUT:     %explicit_return.var_patt: %pattern_type.611 = var_pattern %explicit_return.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %explicit_return.var: ref %ExplicitReturn = var %explicit_return.var_patt
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %explicit_return: ref %ExplicitReturn = bind_name explicit_return, %explicit_return.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %with_addr.patt: %pattern_type.f93 = binding_pattern with_addr [concrete]
// CHECK:STDOUT:     %with_addr.var_patt: %pattern_type.f93 = var_pattern %with_addr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %with_addr.var: ref %WithAddr = var %with_addr.var_patt
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %with_addr: ref %WithAddr = bind_name with_addr, %with_addr.var
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %in_scope.patt: %pattern_type.88f = binding_pattern in_scope [concrete]
// CHECK:STDOUT:     %in_scope.var_patt: %pattern_type.88f = var_pattern %in_scope.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %in_scope.var: ref %NoAddr = var %in_scope.var_patt
// CHECK:STDOUT:   %NoAddr.ref.loc10: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %in_scope: ref %NoAddr = bind_name in_scope, %in_scope.var
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %Op.bound.loc10: <bound method> = bound_method %in_scope.var, constants.%Op.d78
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.d78, @Op.2(constants.%NoAddr) [concrete = constants.%Op.specific_fn.8fe]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %in_scope.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10: %ptr.4b8 = addr_of %in_scope.var
// CHECK:STDOUT:   %no_op.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10)
// CHECK:STDOUT:   %Op.bound.loc8: <bound method> = bound_method %with_addr.var, constants.%Op.bec
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.bec, @Op.2(constants.%WithAddr) [concrete = constants.%Op.specific_fn.b42]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %with_addr.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8: %ptr.b4e = addr_of %with_addr.var
// CHECK:STDOUT:   %no_op.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
// CHECK:STDOUT:   %Op.bound.loc7: <bound method> = bound_method %explicit_return.var, constants.%Op.8e6
// CHECK:STDOUT:   %Op.specific_fn.3: <specific function> = specific_function constants.%Op.8e6, @Op.2(constants.%ExplicitReturn) [concrete = constants.%Op.specific_fn.523]
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %explicit_return.var, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc7: %ptr.b0a = addr_of %explicit_return.var
// CHECK:STDOUT:   %no_op.loc7: init %empty_tuple.type = call %bound_method.loc7(%addr.loc7)
// CHECK:STDOUT:   %Op.bound.loc6: <bound method> = bound_method %no_addr.var, constants.%Op.d78
// CHECK:STDOUT:   %Op.specific_fn.4: <specific function> = specific_function constants.%Op.d78, @Op.2(constants.%NoAddr) [concrete = constants.%Op.specific_fn.8fe]
// CHECK:STDOUT:   %bound_method.loc6: <bound method> = bound_method %no_addr.var, %Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc6: %ptr.4b8 = addr_of %no_addr.var
// CHECK:STDOUT:   %no_op.loc6: init %empty_tuple.type = call %bound_method.loc6(%addr.loc6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- temp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Make.type.bc9: type = fn_type @Make.1 [concrete]
// CHECK:STDOUT:   %Make.1be: %Make.type.bc9 = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %Make.type.378: type = fn_type @Make.2 [concrete]
// CHECK:STDOUT:   %Make.960: %Make.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %Make.type.e14: type = fn_type @Make.3 [concrete]
// CHECK:STDOUT:   %Make.b0a: %Make.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.8f4: type = fn_type @Op.2, @impl(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.bec: %Op.type.8f4 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %Op.specific_fn.b42: <specific function> = specific_function %Op.bec, @Op.2(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.type.641: type = fn_type @Op.2, @impl(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.8e6: %Op.type.641 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b0a: type = ptr_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %Op.specific_fn.523: <specific function> = specific_function %Op.8e6, @Op.2(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.type.73f: type = fn_type @Op.2, @impl(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.d78: %Op.type.73f = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4b8: type = ptr_type %NoAddr [concrete]
// CHECK:STDOUT:   %Op.specific_fn.8fe: <specific function> = specific_function %Op.d78, @Op.2(%NoAddr) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NoAddr: type = import_ref Main//types, NoAddr, loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.ExplicitReturn: type = import_ref Main//types, ExplicitReturn, loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.WithAddr: type = import_ref Main//types, WithAddr, loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.784: %Make.type.bc9 = import_ref Main//types, loc5_22, loaded [concrete = constants.%Make.1be]
// CHECK:STDOUT:   %Main.import_ref.1ed = import_ref Main//types, loc6_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst42 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8e0: %Make.type.378 = import_ref Main//types, loc10_30, loaded [concrete = constants.%Make.960]
// CHECK:STDOUT:   %Main.import_ref.f79 = import_ref Main//types, loc11_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst70 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.974: %Make.type.e14 = import_ref Main//types, loc15_24, loaded [concrete = constants.%Make.b0a]
// CHECK:STDOUT:   %Main.import_ref.675 = import_ref Main//types, loc16_33, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NoAddr = imports.%Main.NoAddr
// CHECK:STDOUT:     .ExplicitReturn = imports.%Main.ExplicitReturn
// CHECK:STDOUT:     .WithAddr = imports.%Main.WithAddr
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
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
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.1ed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ExplicitReturn [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.ee7
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.8e0
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.f79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WithAddr [from "types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.95d
// CHECK:STDOUT:   .Make = imports.%Main.import_ref.974
// CHECK:STDOUT:   .destroy = imports.%Main.import_ref.675
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NoAddr.ref: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Make.ref.loc8: %Make.type.bc9 = name_ref Make, imports.%Main.import_ref.784 [concrete = constants.%Make.1be]
// CHECK:STDOUT:   %.loc8_15.1: ref %NoAddr = temporary_storage
// CHECK:STDOUT:   %Make.call.loc8: init %NoAddr = call %Make.ref.loc8() to %.loc8_15.1
// CHECK:STDOUT:   %.loc8_15.2: ref %NoAddr = temporary %.loc8_15.1, %Make.call.loc8
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Make.ref.loc9: %Make.type.378 = name_ref Make, imports.%Main.import_ref.8e0 [concrete = constants.%Make.960]
// CHECK:STDOUT:   %.loc9_23.1: ref %ExplicitReturn = temporary_storage
// CHECK:STDOUT:   %Make.call.loc9: init %ExplicitReturn = call %Make.ref.loc9() to %.loc9_23.1
// CHECK:STDOUT:   %.loc9_23.2: ref %ExplicitReturn = temporary %.loc9_23.1, %Make.call.loc9
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Make.ref.loc10: %Make.type.e14 = name_ref Make, imports.%Main.import_ref.974 [concrete = constants.%Make.b0a]
// CHECK:STDOUT:   %.loc10_17.1: ref %WithAddr = temporary_storage
// CHECK:STDOUT:   %Make.call.loc10: init %WithAddr = call %Make.ref.loc10() to %.loc10_17.1
// CHECK:STDOUT:   %.loc10_17.2: ref %WithAddr = temporary %.loc10_17.1, %Make.call.loc10
// CHECK:STDOUT:   %Op.bound.loc10: <bound method> = bound_method %.loc10_17.1, constants.%Op.bec
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.bec, @Op.2(constants.%WithAddr) [concrete = constants.%Op.specific_fn.b42]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_17.1, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10: %ptr.b4e = addr_of %.loc10_17.1
// CHECK:STDOUT:   %no_op.loc10: init %empty_tuple.type = call %bound_method.loc10(%addr.loc10)
// CHECK:STDOUT:   %Op.bound.loc9: <bound method> = bound_method %.loc9_23.1, constants.%Op.8e6
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.8e6, @Op.2(constants.%ExplicitReturn) [concrete = constants.%Op.specific_fn.523]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_23.1, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc9: %ptr.b0a = addr_of %.loc9_23.1
// CHECK:STDOUT:   %no_op.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9)
// CHECK:STDOUT:   %Op.bound.loc8: <bound method> = bound_method %.loc8_15.1, constants.%Op.d78
// CHECK:STDOUT:   %Op.specific_fn.3: <specific function> = specific_function constants.%Op.d78, @Op.2(constants.%NoAddr) [concrete = constants.%Op.specific_fn.8fe]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_15.1, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8: %ptr.4b8 = addr_of %.loc8_15.1
// CHECK:STDOUT:   %no_op.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.1 [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2 [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.3 [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_recovery.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoSelf: type = class_type @NoSelf [concrete]
// CHECK:STDOUT:   %destroy.type.d69: type = fn_type @destroy.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %destroy.ab7: %destroy.type.d69 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Args: type = class_type @Args [concrete]
// CHECK:STDOUT:   %pattern_type.a81: type = pattern_type %Args [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %destroy.type.ba0: type = fn_type @destroy.2 [concrete]
// CHECK:STDOUT:   %destroy.723: %destroy.type.ba0 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9f4: type = pattern_type %NoSelf [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.59d: type = fn_type @Op.2, @impl(%Args) [concrete]
// CHECK:STDOUT:   %Op.a71: %Op.type.59d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.7b2: type = ptr_type %Args [concrete]
// CHECK:STDOUT:   %Op.specific_fn.242: <specific function> = specific_function %Op.a71, @Op.2(%Args) [concrete]
// CHECK:STDOUT:   %Op.type.f20: type = fn_type @Op.2, @impl(%NoSelf) [concrete]
// CHECK:STDOUT:   %Op.2e0: %Op.type.f20 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5ab: type = ptr_type %NoSelf [concrete]
// CHECK:STDOUT:   %Op.specific_fn.a58: <specific function> = specific_function %Op.2e0, @Op.2(%NoSelf) [concrete]
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
// CHECK:STDOUT:     .NoSelf = %NoSelf.decl
// CHECK:STDOUT:     .Args = %Args.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NoSelf.decl: type = class_decl @NoSelf [concrete = constants.%NoSelf] {} {}
// CHECK:STDOUT:   %Args.decl: type = class_decl @Args [concrete = constants.%Args] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoSelf {
// CHECK:STDOUT:   %destroy.decl: %destroy.type.d69 = fn_decl @destroy.1 [concrete = constants.%destroy.ab7] {} {}
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoSelf
// CHECK:STDOUT:   .destroy = %destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Args {
// CHECK:STDOUT:   %destroy.decl: %destroy.type.ba0 = fn_decl @destroy.2 [concrete = constants.%destroy.723] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a81 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a81 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern %x.patt, call_param1 [concrete]
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Args
// CHECK:STDOUT:   .destroy = %destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @destroy.2(%self.param: %Args, %x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9f4 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.9f4 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %NoSelf = var %a.var_patt
// CHECK:STDOUT:   %NoSelf.ref: type = name_ref NoSelf, file.%NoSelf.decl [concrete = constants.%NoSelf]
// CHECK:STDOUT:   %a: ref %NoSelf = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.a81 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.a81 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Args = var %b.var_patt
// CHECK:STDOUT:   %Args.ref: type = name_ref Args, file.%Args.decl [concrete = constants.%Args]
// CHECK:STDOUT:   %b: ref %Args = bind_name b, %b.var
// CHECK:STDOUT:   %Op.bound.loc22: <bound method> = bound_method %b.var, constants.%Op.a71
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.a71, @Op.2(constants.%Args) [concrete = constants.%Op.specific_fn.242]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %b.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc22: %ptr.7b2 = addr_of %b.var
// CHECK:STDOUT:   %no_op.loc22: init %empty_tuple.type = call %bound_method.loc22(%addr.loc22)
// CHECK:STDOUT:   %Op.bound.loc21: <bound method> = bound_method %a.var, constants.%Op.2e0
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.2e0, @Op.2(constants.%NoSelf) [concrete = constants.%Op.specific_fn.a58]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %a.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc21: %ptr.5ab = addr_of %a.var
// CHECK:STDOUT:   %no_op.loc21: init %empty_tuple.type = call %bound_method.loc21(%addr.loc21)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- not_breaking_generics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3d5d.1: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T.8b3d5d.1) [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.389: <witness> = require_complete_type %C.f2e [template]
// CHECK:STDOUT:   %pattern_type.e5e: type = pattern_type %C.f2e [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T.8b3d5d.2: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T.8b3d5d.2) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.7d2: type = ptr_type %C.f2e [template]
// CHECK:STDOUT:   %require_complete.448: <witness> = require_complete_type %ptr.7d2 [template]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.f2e, @Destroy [template]
// CHECK:STDOUT:   %Destroy.facet.d66: %Destroy.type = facet_value %C.f2e, (%Destroy.lookup_impl_witness) [template]
// CHECK:STDOUT:   %.eb5: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.d66 [template]
// CHECK:STDOUT:   %impl.elem0: %.eb5 = impl_witness_access %Destroy.lookup_impl_witness, element0 [template]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Destroy.facet.d66) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.99a: type = pattern_type %C.7a7 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.a3f: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl(%C.7a7) [concrete]
// CHECK:STDOUT:   %Op.type.add: type = fn_type @Op.2, @impl(%C.7a7) [concrete]
// CHECK:STDOUT:   %Op.b1c: %Op.type.add = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.308: type = ptr_type %C.7a7 [concrete]
// CHECK:STDOUT:   %complete_type.903: <witness> = complete_type_witness %ptr.308 [concrete]
// CHECK:STDOUT:   %Destroy.facet.f20: %Destroy.type = facet_value %C.7a7, (%Destroy.impl_witness.a3f) [concrete]
// CHECK:STDOUT:   %.2d6: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.f20 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.b1c, @Op.2(%C.7a7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_18.1: type = bind_symbolic_name T, 0, template [template = %T.loc3_18.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc5_15.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc3_18.1: type) {
// CHECK:STDOUT:   %T.loc3_18.2: type = bind_symbolic_name T, 0, template [template = %T.loc3_18.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_15.1: type) {
// CHECK:STDOUT:   %T.loc5_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc5_15.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc6_13.2: type = class_type @C, @C(%T.loc5_15.2) [template = %C.loc6_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %require_complete.loc6_13: <witness> = require_complete_type %C.loc6_13.2 [template = %require_complete.loc6_13 (constants.%require_complete.389)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc6_13.2 [template = %pattern_type (constants.%pattern_type.e5e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.loc6_13.2, @Destroy [template = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.loc6_13.2, (%Destroy.lookup_impl_witness) [template = %Destroy.facet (constants.%Destroy.facet.d66)]
// CHECK:STDOUT:   %.loc6_3.2: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [template = %.loc6_3.2 (constants.%.eb5)]
// CHECK:STDOUT:   %impl.elem0.loc6_3.2: @F.%.loc6_3.2 (%.eb5) = impl_witness_access %Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc6_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_3.2: <specific function> = specific_impl_function %impl.elem0.loc6_3.2, @Op.1(%Destroy.facet) [template = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C.loc6_13.2 [template = %ptr (constants.%ptr.7d2)]
// CHECK:STDOUT:   %require_complete.loc6_3: <witness> = require_complete_type %ptr [template = %require_complete.loc6_3 (constants.%require_complete.448)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.e5e) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.e5e) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%C.loc6_13.2 (%C.f2e) = var %v.var_patt
// CHECK:STDOUT:     %.loc6_13: type = splice_block %C.loc6_13.1 [template = %C.loc6_13.2 (constants.%C.f2e)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc5_15.1 [template = %T.loc5_15.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:       %C.loc6_13.1: type = class_type @C, @C(constants.%T.8b3d5d.1) [template = %C.loc6_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%C.loc6_13.2 (%C.f2e) = bind_name v, %v.var
// CHECK:STDOUT:     %impl.elem0.loc6_3.1: @F.%.loc6_3.2 (%.eb5) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc6_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc6_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_3.1: <specific function> = specific_impl_function %impl.elem0.loc6_3.1, @Op.1(constants.%Destroy.facet.d66) [template = %specific_impl_fn.loc6_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc6_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc6_3.1
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.7d2) = addr_of %v.var
// CHECK:STDOUT:     %.loc6_3.1: init %empty_tuple.type = call %bound_method.loc6_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc9_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_14: type = converted %.loc9_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%empty_struct_type) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.8b3d5d.1) {
// CHECK:STDOUT:   %T.loc3_18.2 => constants.%T.8b3d5d.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3d5d.1) {
// CHECK:STDOUT:   %T.loc5_15.2 => constants.%T.8b3d5d.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc5_15.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc6_13.2 => constants.%C.7a7
// CHECK:STDOUT:   %require_complete.loc6_13 => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.99a
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.a3f
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.f20
// CHECK:STDOUT:   %.loc6_3.2 => constants.%.2d6
// CHECK:STDOUT:   %impl.elem0.loc6_3.2 => constants.%Op.b1c
// CHECK:STDOUT:   %specific_impl_fn.loc6_3.2 => constants.%Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.308
// CHECK:STDOUT:   %require_complete.loc6_3 => constants.%complete_type.903
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc3_18.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:

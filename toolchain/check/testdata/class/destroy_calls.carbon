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
// CHECK:STDOUT:   %NoAddr.Make.type: type = fn_type @NoAddr.Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NoAddr.Make: %NoAddr.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr.destroy.type: type = fn_type @NoAddr.destroy [concrete]
// CHECK:STDOUT:   %NoAddr.destroy: %NoAddr.destroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.5ce: <witness> = impl_witness @NoAddr.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.4b8: type = ptr_type %NoAddr [concrete]
// CHECK:STDOUT:   %pattern_type.ba5: type = pattern_type %ptr.4b8 [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.type: type = fn_type @NoAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op: %NoAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %ExplicitReturn.Make.type: type = fn_type @ExplicitReturn.Make [concrete]
// CHECK:STDOUT:   %ExplicitReturn.Make: %ExplicitReturn.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %ExplicitReturn.destroy.type: type = fn_type @ExplicitReturn.destroy [concrete]
// CHECK:STDOUT:   %ExplicitReturn.destroy: %ExplicitReturn.destroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.6ac: <witness> = impl_witness @ExplicitReturn.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.b0a: type = ptr_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.e7a: type = pattern_type %ptr.b0a [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.type: type = fn_type @ExplicitReturn.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op: %ExplicitReturn.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %WithAddr [concrete]
// CHECK:STDOUT:   %WithAddr.Make.type: type = fn_type @WithAddr.Make [concrete]
// CHECK:STDOUT:   %WithAddr.Make: %WithAddr.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.4a0: type = pattern_type %ptr.b4e [concrete]
// CHECK:STDOUT:   %WithAddr.destroy.type: type = fn_type @WithAddr.destroy [concrete]
// CHECK:STDOUT:   %WithAddr.destroy: %WithAddr.destroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.934: <witness> = impl_witness @WithAddr.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.type: type = fn_type @WithAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op: %WithAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT: impl @NoAddr.as.Destroy.impl: constants.%NoAddr as constants.%Destroy.type {
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.decl: %NoAddr.as.Destroy.impl.Op.type = fn_decl @NoAddr.as.Destroy.impl.Op [concrete = constants.%NoAddr.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.ba5 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ba5 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.4b8 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:     %self: %ptr.4b8 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %NoAddr.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @NoAddr.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ExplicitReturn.as.Destroy.impl: constants.%ExplicitReturn as constants.%Destroy.type {
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.decl: %ExplicitReturn.as.Destroy.impl.Op.type = fn_decl @ExplicitReturn.as.Destroy.impl.Op [concrete = constants.%ExplicitReturn.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.e7a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e7a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.b0a = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:     %self: %ptr.b0a = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %ExplicitReturn.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @ExplicitReturn.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @WithAddr.as.Destroy.impl: constants.%WithAddr as constants.%Destroy.type {
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.decl: %WithAddr.as.Destroy.impl.Op.type = fn_decl @WithAddr.as.Destroy.impl.Op [concrete = constants.%WithAddr.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.4a0 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4a0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.b4e = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:     %self: %ptr.b4e = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %WithAddr.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @WithAddr.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoAddr {
// CHECK:STDOUT:   %NoAddr.Make.decl: %NoAddr.Make.type = fn_decl @NoAddr.Make [concrete = constants.%NoAddr.Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.88f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.88f = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %NoAddr.ref: type = name_ref NoAddr, file.%NoAddr.decl [concrete = constants.%NoAddr]
// CHECK:STDOUT:     %return.param: ref %NoAddr = out_param call_param0
// CHECK:STDOUT:     %return: ref %NoAddr = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NoAddr.destroy.decl: %NoAddr.destroy.type = fn_decl @NoAddr.destroy [concrete = constants.%NoAddr.destroy] {
// CHECK:STDOUT:     %self.patt: %pattern_type.88f = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.88f = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NoAddr = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:     %self: %NoAddr = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @NoAddr.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@NoAddr.as.Destroy.impl.%NoAddr.as.Destroy.impl.Op.decl), @NoAddr.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.5ce]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoAddr
// CHECK:STDOUT:   .NoAddr = <poisoned>
// CHECK:STDOUT:   .Make = %NoAddr.Make.decl
// CHECK:STDOUT:   .destroy = %NoAddr.destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ExplicitReturn {
// CHECK:STDOUT:   %ExplicitReturn.Make.decl: %ExplicitReturn.Make.type = fn_decl @ExplicitReturn.Make [concrete = constants.%ExplicitReturn.Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.611 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.611 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ExplicitReturn.ref: type = name_ref ExplicitReturn, file.%ExplicitReturn.decl [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:     %return.param: ref %ExplicitReturn = out_param call_param0
// CHECK:STDOUT:     %return: ref %ExplicitReturn = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ExplicitReturn.destroy.decl: %ExplicitReturn.destroy.type = fn_decl @ExplicitReturn.destroy [concrete = constants.%ExplicitReturn.destroy] {
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
// CHECK:STDOUT:   impl_decl @ExplicitReturn.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@ExplicitReturn.as.Destroy.impl.%ExplicitReturn.as.Destroy.impl.Op.decl), @ExplicitReturn.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.6ac]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ExplicitReturn
// CHECK:STDOUT:   .ExplicitReturn = <poisoned>
// CHECK:STDOUT:   .Make = %ExplicitReturn.Make.decl
// CHECK:STDOUT:   .destroy = %ExplicitReturn.destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WithAddr {
// CHECK:STDOUT:   %WithAddr.Make.decl: %WithAddr.Make.type = fn_decl @WithAddr.Make [concrete = constants.%WithAddr.Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.f93 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f93 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %WithAddr.ref: type = name_ref WithAddr, file.%WithAddr.decl [concrete = constants.%WithAddr]
// CHECK:STDOUT:     %return.param: ref %WithAddr = out_param call_param0
// CHECK:STDOUT:     %return: ref %WithAddr = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WithAddr.destroy.decl: %WithAddr.destroy.type = fn_decl @WithAddr.destroy [concrete = constants.%WithAddr.destroy] {
// CHECK:STDOUT:     %self.patt: %pattern_type.4a0 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4a0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16_14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.b4e = value_param call_param0
// CHECK:STDOUT:     %.loc16_29: type = splice_block %ptr [concrete = constants.%ptr.b4e] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.b4e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.b4e = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @WithAddr.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@WithAddr.as.Destroy.impl.%WithAddr.as.Destroy.impl.Op.decl), @WithAddr.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.934]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%WithAddr
// CHECK:STDOUT:   .WithAddr = <poisoned>
// CHECK:STDOUT:   .Make = %WithAddr.Make.decl
// CHECK:STDOUT:   .destroy = %WithAddr.destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoAddr.Make() -> %NoAddr;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoAddr.destroy(%self.param: %NoAddr);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoAddr.as.Destroy.impl.Op(%self.param: %ptr.4b8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitReturn.Make() -> %ExplicitReturn;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitReturn.destroy(%self.param: %ExplicitReturn) -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitReturn.as.Destroy.impl.Op(%self.param: %ptr.b0a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithAddr.Make() -> %WithAddr;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithAddr.destroy(%self.param: %ptr.b4e);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithAddr.as.Destroy.impl.Op(%self.param: %ptr.b4e) = "no_op";
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
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %NoAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %WithAddr [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.91f: <witness> = impl_witness imports.%Destroy.impl_witness_table.4f4 [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.type: type = fn_type @WithAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op: %WithAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.f3a: <witness> = impl_witness imports.%Destroy.impl_witness_table.6ea [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.type: type = fn_type @ExplicitReturn.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op: %ExplicitReturn.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b0a: type = ptr_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.17d: <witness> = impl_witness imports.%Destroy.impl_witness_table.b7c [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.type: type = fn_type @NoAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op: %NoAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4b8: type = ptr_type %NoAddr [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5b = import_ref Main//types, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.1ed = import_ref Main//types, loc6_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst80 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b14 = import_ref Main//types, loc10_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.f79 = import_ref Main//types, loc11_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst123 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//types, loc15_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.675 = import_ref Main//types, loc16_33, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.15b: <witness> = import_ref Main//types, loc4_14, loaded [concrete = constants.%Destroy.impl_witness.17d]
// CHECK:STDOUT:   %Main.import_ref.09f: type = import_ref Main//types, inst19 [no loc], loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.import_ref.cb9298.1: type = import_ref Main//types, inst40 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.022: <witness> = import_ref Main//types, loc9_22, loaded [concrete = constants.%Destroy.impl_witness.f3a]
// CHECK:STDOUT:   %Main.import_ref.2e8: type = import_ref Main//types, inst80 [no loc], loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.import_ref.cb9298.2: type = import_ref Main//types, inst40 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.d7d: <witness> = import_ref Main//types, loc14_16, loaded [concrete = constants.%Destroy.impl_witness.91f]
// CHECK:STDOUT:   %Main.import_ref.0a2: type = import_ref Main//types, inst123 [no loc], loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Main.import_ref.cb9298.3: type = import_ref Main//types, inst40 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.344: %WithAddr.as.Destroy.impl.Op.type = import_ref Main//types, loc14_16, loaded [concrete = constants.%WithAddr.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.4f4 = impl_witness_table (%Main.import_ref.344), @WithAddr.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.d37: %ExplicitReturn.as.Destroy.impl.Op.type = import_ref Main//types, loc9_22, loaded [concrete = constants.%ExplicitReturn.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.6ea = impl_witness_table (%Main.import_ref.d37), @ExplicitReturn.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.b27: %NoAddr.as.Destroy.impl.Op.type = import_ref Main//types, loc4_14, loaded [concrete = constants.%NoAddr.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.b7c = impl_witness_table (%Main.import_ref.b27), @NoAddr.as.Destroy.impl [concrete]
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
// CHECK:STDOUT: impl @NoAddr.as.Destroy.impl: imports.%Main.import_ref.09f as imports.%Main.import_ref.cb9298.1 [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.15b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ExplicitReturn.as.Destroy.impl: imports.%Main.import_ref.2e8 as imports.%Main.import_ref.cb9298.2 [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.022
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @WithAddr.as.Destroy.impl: imports.%Main.import_ref.0a2 as imports.%Main.import_ref.cb9298.3 [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.d7d
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
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.bound: <bound method> = bound_method %with_addr.var, constants.%WithAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc8: %ptr.b4e = addr_of %with_addr.var
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %WithAddr.as.Destroy.impl.Op.bound(%addr.loc8)
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.bound: <bound method> = bound_method %explicit_return.var, constants.%ExplicitReturn.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc7: %ptr.b0a = addr_of %explicit_return.var
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.call: init %empty_tuple.type = call %ExplicitReturn.as.Destroy.impl.Op.bound(%addr.loc7)
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.bound: <bound method> = bound_method %no_addr.var, constants.%NoAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc6: %ptr.4b8 = addr_of %no_addr.var
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %NoAddr.as.Destroy.impl.Op.bound(%addr.loc6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithAddr.as.Destroy.impl.Op = "no_op" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitReturn.as.Destroy.impl.Op = "no_op" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoAddr.as.Destroy.impl.Op = "no_op" [from "types.carbon"];
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
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %NoAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %WithAddr [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.17d: <witness> = impl_witness imports.%Destroy.impl_witness_table.b7c [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.type: type = fn_type @NoAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op: %NoAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4b8: type = ptr_type %NoAddr [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.91f: <witness> = impl_witness imports.%Destroy.impl_witness_table.4f4 [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.type: type = fn_type @WithAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op: %WithAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.f3a: <witness> = impl_witness imports.%Destroy.impl_witness_table.6ea [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.type: type = fn_type @ExplicitReturn.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op: %ExplicitReturn.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b0a: type = ptr_type %ExplicitReturn [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5b = import_ref Main//types, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.1ed = import_ref Main//types, loc6_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst80 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b14 = import_ref Main//types, loc10_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.f79 = import_ref Main//types, loc11_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst123 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//types, loc15_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.675 = import_ref Main//types, loc16_33, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.15b: <witness> = import_ref Main//types, loc4_14, loaded [concrete = constants.%Destroy.impl_witness.17d]
// CHECK:STDOUT:   %Main.import_ref.09f: type = import_ref Main//types, inst19 [no loc], loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.import_ref.cb9298.1: type = import_ref Main//types, inst40 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.022: <witness> = import_ref Main//types, loc9_22, loaded [concrete = constants.%Destroy.impl_witness.f3a]
// CHECK:STDOUT:   %Main.import_ref.2e8: type = import_ref Main//types, inst80 [no loc], loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.import_ref.cb9298.2: type = import_ref Main//types, inst40 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.d7d: <witness> = import_ref Main//types, loc14_16, loaded [concrete = constants.%Destroy.impl_witness.91f]
// CHECK:STDOUT:   %Main.import_ref.0a2: type = import_ref Main//types, inst123 [no loc], loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Main.import_ref.cb9298.3: type = import_ref Main//types, inst40 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.b27: %NoAddr.as.Destroy.impl.Op.type = import_ref Main//types, loc4_14, loaded [concrete = constants.%NoAddr.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.b7c = impl_witness_table (%Main.import_ref.b27), @NoAddr.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.344: %WithAddr.as.Destroy.impl.Op.type = import_ref Main//types, loc14_16, loaded [concrete = constants.%WithAddr.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.4f4 = impl_witness_table (%Main.import_ref.344), @WithAddr.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.d37: %ExplicitReturn.as.Destroy.impl.Op.type = import_ref Main//types, loc9_22, loaded [concrete = constants.%ExplicitReturn.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.6ea = impl_witness_table (%Main.import_ref.d37), @ExplicitReturn.as.Destroy.impl [concrete]
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
// CHECK:STDOUT: impl @NoAddr.as.Destroy.impl: imports.%Main.import_ref.09f as imports.%Main.import_ref.cb9298.1 [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.15b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ExplicitReturn.as.Destroy.impl: imports.%Main.import_ref.2e8 as imports.%Main.import_ref.cb9298.2 [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.022
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @WithAddr.as.Destroy.impl: imports.%Main.import_ref.0a2 as imports.%Main.import_ref.cb9298.3 [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.d7d
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
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %in_scope.var, constants.%NoAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc10: %ptr.4b8 = addr_of %in_scope.var
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %NoAddr.as.Destroy.impl.Op.bound.loc10(%addr.loc10)
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.bound: <bound method> = bound_method %with_addr.var, constants.%WithAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc8: %ptr.b4e = addr_of %with_addr.var
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %WithAddr.as.Destroy.impl.Op.bound(%addr.loc8)
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.bound: <bound method> = bound_method %explicit_return.var, constants.%ExplicitReturn.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc7: %ptr.b0a = addr_of %explicit_return.var
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.call: init %empty_tuple.type = call %ExplicitReturn.as.Destroy.impl.Op.bound(%addr.loc7)
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.bound.loc6: <bound method> = bound_method %no_addr.var, constants.%NoAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc6: %ptr.4b8 = addr_of %no_addr.var
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.call.loc6: init %empty_tuple.type = call %NoAddr.as.Destroy.impl.Op.bound.loc6(%addr.loc6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoAddr.as.Destroy.impl.Op = "no_op" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithAddr.as.Destroy.impl.Op = "no_op" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitReturn.as.Destroy.impl.Op = "no_op" [from "types.carbon"];
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
// CHECK:STDOUT:   %NoAddr.Make.type: type = fn_type @NoAddr.Make [concrete]
// CHECK:STDOUT:   %NoAddr.Make: %NoAddr.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %ExplicitReturn.Make.type: type = fn_type @ExplicitReturn.Make [concrete]
// CHECK:STDOUT:   %ExplicitReturn.Make: %ExplicitReturn.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %WithAddr.Make.type: type = fn_type @WithAddr.Make [concrete]
// CHECK:STDOUT:   %WithAddr.Make: %WithAddr.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.91f: <witness> = impl_witness imports.%Destroy.impl_witness_table.4f4 [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.type: type = fn_type @WithAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op: %WithAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.f3a: <witness> = impl_witness imports.%Destroy.impl_witness_table.6ea [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.type: type = fn_type @ExplicitReturn.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op: %ExplicitReturn.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b0a: type = ptr_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.17d: <witness> = impl_witness imports.%Destroy.impl_witness_table.b7c [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.type: type = fn_type @NoAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op: %NoAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4b8: type = ptr_type %NoAddr [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst19 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.784: %NoAddr.Make.type = import_ref Main//types, loc5_22, loaded [concrete = constants.%NoAddr.Make]
// CHECK:STDOUT:   %Main.import_ref.1ed = import_ref Main//types, loc6_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst80 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8e0: %ExplicitReturn.Make.type = import_ref Main//types, loc10_30, loaded [concrete = constants.%ExplicitReturn.Make]
// CHECK:STDOUT:   %Main.import_ref.f79 = import_ref Main//types, loc11_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst123 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.974: %WithAddr.Make.type = import_ref Main//types, loc15_24, loaded [concrete = constants.%WithAddr.Make]
// CHECK:STDOUT:   %Main.import_ref.675 = import_ref Main//types, loc16_33, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.15b: <witness> = import_ref Main//types, loc4_14, loaded [concrete = constants.%Destroy.impl_witness.17d]
// CHECK:STDOUT:   %Main.import_ref.09f: type = import_ref Main//types, inst19 [no loc], loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.import_ref.cb9298.1: type = import_ref Main//types, inst40 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.022: <witness> = import_ref Main//types, loc9_22, loaded [concrete = constants.%Destroy.impl_witness.f3a]
// CHECK:STDOUT:   %Main.import_ref.2e8: type = import_ref Main//types, inst80 [no loc], loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.import_ref.cb9298.2: type = import_ref Main//types, inst40 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.d7d: <witness> = import_ref Main//types, loc14_16, loaded [concrete = constants.%Destroy.impl_witness.91f]
// CHECK:STDOUT:   %Main.import_ref.0a2: type = import_ref Main//types, inst123 [no loc], loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Main.import_ref.cb9298.3: type = import_ref Main//types, inst40 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Main.import_ref.344: %WithAddr.as.Destroy.impl.Op.type = import_ref Main//types, loc14_16, loaded [concrete = constants.%WithAddr.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.4f4 = impl_witness_table (%Main.import_ref.344), @WithAddr.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.d37: %ExplicitReturn.as.Destroy.impl.Op.type = import_ref Main//types, loc9_22, loaded [concrete = constants.%ExplicitReturn.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.6ea = impl_witness_table (%Main.import_ref.d37), @ExplicitReturn.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.b27: %NoAddr.as.Destroy.impl.Op.type = import_ref Main//types, loc4_14, loaded [concrete = constants.%NoAddr.as.Destroy.impl.Op]
// CHECK:STDOUT:   %Destroy.impl_witness_table.b7c = impl_witness_table (%Main.import_ref.b27), @NoAddr.as.Destroy.impl [concrete]
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
// CHECK:STDOUT: impl @NoAddr.as.Destroy.impl: imports.%Main.import_ref.09f as imports.%Main.import_ref.cb9298.1 [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.15b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ExplicitReturn.as.Destroy.impl: imports.%Main.import_ref.2e8 as imports.%Main.import_ref.cb9298.2 [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.022
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @WithAddr.as.Destroy.impl: imports.%Main.import_ref.0a2 as imports.%Main.import_ref.cb9298.3 [from "types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.d7d
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
// CHECK:STDOUT:   %Make.ref.loc8: %NoAddr.Make.type = name_ref Make, imports.%Main.import_ref.784 [concrete = constants.%NoAddr.Make]
// CHECK:STDOUT:   %.loc8_15.1: ref %NoAddr = temporary_storage
// CHECK:STDOUT:   %NoAddr.Make.call: init %NoAddr = call %Make.ref.loc8() to %.loc8_15.1
// CHECK:STDOUT:   %.loc8_15.2: ref %NoAddr = temporary %.loc8_15.1, %NoAddr.Make.call
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Make.ref.loc9: %ExplicitReturn.Make.type = name_ref Make, imports.%Main.import_ref.8e0 [concrete = constants.%ExplicitReturn.Make]
// CHECK:STDOUT:   %.loc9_23.1: ref %ExplicitReturn = temporary_storage
// CHECK:STDOUT:   %ExplicitReturn.Make.call: init %ExplicitReturn = call %Make.ref.loc9() to %.loc9_23.1
// CHECK:STDOUT:   %.loc9_23.2: ref %ExplicitReturn = temporary %.loc9_23.1, %ExplicitReturn.Make.call
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Make.ref.loc10: %WithAddr.Make.type = name_ref Make, imports.%Main.import_ref.974 [concrete = constants.%WithAddr.Make]
// CHECK:STDOUT:   %.loc10_17.1: ref %WithAddr = temporary_storage
// CHECK:STDOUT:   %WithAddr.Make.call: init %WithAddr = call %Make.ref.loc10() to %.loc10_17.1
// CHECK:STDOUT:   %.loc10_17.2: ref %WithAddr = temporary %.loc10_17.1, %WithAddr.Make.call
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_17.1, constants.%WithAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc10: %ptr.b4e = addr_of %.loc10_17.1
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %WithAddr.as.Destroy.impl.Op.bound(%addr.loc10)
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc9_23.1, constants.%ExplicitReturn.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc9: %ptr.b0a = addr_of %.loc9_23.1
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.call: init %empty_tuple.type = call %ExplicitReturn.as.Destroy.impl.Op.bound(%addr.loc9)
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_15.1, constants.%NoAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc8: %ptr.4b8 = addr_of %.loc8_15.1
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %NoAddr.as.Destroy.impl.Op.bound(%addr.loc8)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoAddr.Make [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitReturn.Make [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithAddr.Make [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WithAddr.as.Destroy.impl.Op = "no_op" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ExplicitReturn.as.Destroy.impl.Op = "no_op" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoAddr.as.Destroy.impl.Op = "no_op" [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_recovery.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NoSelf: type = class_type @NoSelf [concrete]
// CHECK:STDOUT:   %NoSelf.destroy.type: type = fn_type @NoSelf.destroy [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NoSelf.destroy: %NoSelf.destroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.d11: <witness> = impl_witness @NoSelf.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.5ab: type = ptr_type %NoSelf [concrete]
// CHECK:STDOUT:   %pattern_type.a98: type = pattern_type %ptr.5ab [concrete]
// CHECK:STDOUT:   %NoSelf.as.Destroy.impl.Op.type: type = fn_type @NoSelf.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %NoSelf.as.Destroy.impl.Op: %NoSelf.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Args: type = class_type @Args [concrete]
// CHECK:STDOUT:   %pattern_type.a81: type = pattern_type %Args [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Args.destroy.type: type = fn_type @Args.destroy [concrete]
// CHECK:STDOUT:   %Args.destroy: %Args.destroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.318: <witness> = impl_witness @Args.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.7b2: type = ptr_type %Args [concrete]
// CHECK:STDOUT:   %pattern_type.8cf: type = pattern_type %ptr.7b2 [concrete]
// CHECK:STDOUT:   %Args.as.Destroy.impl.Op.type: type = fn_type @Args.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Args.as.Destroy.impl.Op: %Args.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9f4: type = pattern_type %NoSelf [concrete]
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
// CHECK:STDOUT: impl @NoSelf.as.Destroy.impl: constants.%NoSelf as constants.%Destroy.type {
// CHECK:STDOUT:   %NoSelf.as.Destroy.impl.Op.decl: %NoSelf.as.Destroy.impl.Op.type = fn_decl @NoSelf.as.Destroy.impl.Op [concrete = constants.%NoSelf.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a98 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a98 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.5ab = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoSelf [concrete = constants.%NoSelf]
// CHECK:STDOUT:     %self: %ptr.5ab = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %NoSelf.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @NoSelf.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Args.as.Destroy.impl: constants.%Args as constants.%Destroy.type {
// CHECK:STDOUT:   %Args.as.Destroy.impl.Op.decl: %Args.as.Destroy.impl.Op.type = fn_decl @Args.as.Destroy.impl.Op [concrete = constants.%Args.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.8cf = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8cf = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc12: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.7b2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Args [concrete = constants.%Args]
// CHECK:STDOUT:     %self: %ptr.7b2 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Args.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Args.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoSelf {
// CHECK:STDOUT:   %NoSelf.destroy.decl: %NoSelf.destroy.type = fn_decl @NoSelf.destroy [concrete = constants.%NoSelf.destroy] {} {}
// CHECK:STDOUT:   impl_decl @NoSelf.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@NoSelf.as.Destroy.impl.%NoSelf.as.Destroy.impl.Op.decl), @NoSelf.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.d11]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoSelf
// CHECK:STDOUT:   .destroy = %NoSelf.destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Args {
// CHECK:STDOUT:   %Args.destroy.decl: %Args.destroy.type = fn_decl @Args.destroy [concrete = constants.%Args.destroy] {
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
// CHECK:STDOUT:   impl_decl @Args.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Args.as.Destroy.impl.%Args.as.Destroy.impl.Op.decl), @Args.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.318]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Args
// CHECK:STDOUT:   .destroy = %Args.destroy.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoSelf.destroy();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoSelf.as.Destroy.impl.Op(%self.param: %ptr.5ab) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Args.destroy(%self.param: %Args, %x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Args.as.Destroy.impl.Op(%self.param: %ptr.7b2) = "no_op";
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
// CHECK:STDOUT:   %Args.as.Destroy.impl.Op.bound: <bound method> = bound_method %b.var, constants.%Args.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc22: %ptr.7b2 = addr_of %b.var
// CHECK:STDOUT:   %Args.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Args.as.Destroy.impl.Op.bound(%addr.loc22)
// CHECK:STDOUT:   %NoSelf.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%NoSelf.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc21: %ptr.5ab = addr_of %a.var
// CHECK:STDOUT:   %NoSelf.as.Destroy.impl.Op.call: init %empty_tuple.type = call %NoSelf.as.Destroy.impl.Op.bound(%addr.loc21)
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.a08: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T.8b3d5d.1) [template]
// CHECK:STDOUT:   %ptr.7d2: type = ptr_type %C.f2e [template]
// CHECK:STDOUT:   %pattern_type.1d2: type = pattern_type %ptr.7d2 [template]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.7a1: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T.8b3d5d.1) [template]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.d65: %C.as.Destroy.impl.Op.type.7a1 = struct_value () [template]
// CHECK:STDOUT:   %Destroy.facet.d7f: %Destroy.type = facet_value %C.f2e, (%Destroy.impl_witness.a08) [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.389: <witness> = require_complete_type %C.f2e [template]
// CHECK:STDOUT:   %pattern_type.e5e: type = pattern_type %C.f2e [template]
// CHECK:STDOUT:   %.a2f: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.d7f [template]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.67e: <specific function> = specific_function %C.as.Destroy.impl.Op.d65, @C.as.Destroy.impl.Op(%T.8b3d5d.1) [template]
// CHECK:STDOUT:   %require_complete.448: <witness> = require_complete_type %ptr.7d2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.99a: type = pattern_type %C.7a7 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.b34: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Destroy.facet.5e7: %Destroy.type = facet_value %C.7a7, (%Destroy.impl_witness.b34) [concrete]
// CHECK:STDOUT:   %.ec8: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.5e7 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.f57: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.a9a: %C.as.Destroy.impl.Op.type.f57 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.308: type = ptr_type %C.7a7 [concrete]
// CHECK:STDOUT:   %pattern_type.558: type = pattern_type %ptr.308 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.d47: <specific function> = specific_function %C.as.Destroy.impl.Op.a9a, @C.as.Destroy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type.903: <witness> = complete_type_witness %ptr.308 [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_18.2: type = bind_symbolic_name T, 0, template [template = %T.loc3_18.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc5_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc5_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.%T.loc3_18.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template = %T (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T) [template = %Destroy.impl_witness (constants.%Destroy.impl_witness.a08)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T) [template = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type.7a1)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type.7a1) = struct_value () [template = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.d65)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%C.f2e as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type.7a1) = fn_decl @C.as.Destroy.impl.Op [template = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.d65)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.1d2) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.1d2) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc3_28.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2) = value_param call_param0
// CHECK:STDOUT:       %.loc3_28.2: type = splice_block %Self.ref [template = %C (constants.%C.f2e)] {
// CHECK:STDOUT:         %.loc3_28.3: type = specific_constant constants.%C.f2e, @C(constants.%T.8b3d5d.1) [template = %C (constants.%C.f2e)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc3_28.3 [template = %C (constants.%C.f2e)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc3_18.2: type) {
// CHECK:STDOUT:   %T.loc3_18.1: type = bind_symbolic_name T, 0, template [template = %T.loc3_18.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%T.8b3d5d.1) [template = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.a08)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.%T.loc3_18.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template = %T (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [template = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [template = %ptr (constants.%ptr.7d2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [template = %pattern_type (constants.%pattern_type.1d2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.7d2)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_15.2: type) {
// CHECK:STDOUT:   %T.loc5_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc5_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc6_13.2: type = class_type @C, @C(%T.loc5_15.1) [template = %C.loc6_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %require_complete.loc6_13: <witness> = require_complete_type %C.loc6_13.2 [template = %require_complete.loc6_13 (constants.%require_complete.389)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc6_13.2 [template = %pattern_type (constants.%pattern_type.e5e)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T.loc5_15.1) [template = %Destroy.impl_witness (constants.%Destroy.impl_witness.a08)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.loc6_13.2, (%Destroy.impl_witness) [template = %Destroy.facet (constants.%Destroy.facet.d7f)]
// CHECK:STDOUT:   %.loc6_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [template = %.loc6_3 (constants.%.a2f)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T.loc5_15.1) [template = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type.7a1)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @F.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type.7a1) = struct_value () [template = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.d65)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %C.as.Destroy.impl.Op, @C.as.Destroy.impl.Op(%T.loc5_15.1) [template = %C.as.Destroy.impl.Op.specific_fn (constants.%C.as.Destroy.impl.Op.specific_fn.67e)]
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
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc5_15.2 [template = %T.loc5_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:       %C.loc6_13.1: type = class_type @C, @C(constants.%T.8b3d5d.1) [template = %C.loc6_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%C.loc6_13.2 (%C.f2e) = bind_name v, %v.var
// CHECK:STDOUT:     %impl.elem0: @F.%.loc6_3 (%.a2f) = impl_witness_access constants.%Destroy.impl_witness.a08, element0 [template = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.d65)]
// CHECK:STDOUT:     %bound_method.loc6_3.1: <bound method> = bound_method %v.var, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Destroy.impl.Op(constants.%T.8b3d5d.1) [template = %C.as.Destroy.impl.Op.specific_fn (constants.%C.as.Destroy.impl.Op.specific_fn.67e)]
// CHECK:STDOUT:     %bound_method.loc6_3.2: <bound method> = bound_method %v.var, %specific_fn
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.7d2) = addr_of %v.var
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc6_3.2(%addr)
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
// CHECK:STDOUT:   %T.loc3_18.1 => constants.%T.8b3d5d.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%T.8b3d5d.1) {
// CHECK:STDOUT:   %T => constants.%T.8b3d5d.1
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.a08
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.7a1
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.d65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%T.8b3d5d.1) {
// CHECK:STDOUT:   %T => constants.%T.8b3d5d.1
// CHECK:STDOUT:   %C => constants.%C.f2e
// CHECK:STDOUT:   %ptr => constants.%ptr.7d2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3d5d.1) {
// CHECK:STDOUT:   %T.loc5_15.1 => constants.%T.8b3d5d.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc5_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc6_13.2 => constants.%C.7a7
// CHECK:STDOUT:   %require_complete.loc6_13 => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.99a
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.b34
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.5e7
// CHECK:STDOUT:   %.loc6_3 => constants.%.ec8
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.f57
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.a9a
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn => constants.%C.as.Destroy.impl.Op.specific_fn.d47
// CHECK:STDOUT:   %ptr => constants.%ptr.308
// CHECK:STDOUT:   %require_complete.loc6_3 => constants.%complete_type.903
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc3_18.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.b34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %C => constants.%C.7a7
// CHECK:STDOUT:   %ptr => constants.%ptr.308
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.558
// CHECK:STDOUT: }
// CHECK:STDOUT:

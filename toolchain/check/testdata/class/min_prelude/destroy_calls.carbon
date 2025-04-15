// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/destroy_calls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/destroy_calls.carbon

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

// --- fail_not_breaking_generics.carbon

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
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.725 [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.706: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.aaa: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.type.73f: type = fn_type @Op.2, @impl(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.d78: %Op.type.73f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.2b6: %Destroy.type = facet_value %NoAddr, (%Destroy.impl_witness.aaa) [concrete]
// CHECK:STDOUT:   %.7a4: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.2b6 [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.40a: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.type.641: type = fn_type @Op.2, @impl(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.8e6: %Op.type.641 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.c42: %Destroy.type = facet_value %ExplicitReturn, (%Destroy.impl_witness.40a) [concrete]
// CHECK:STDOUT:   %.b47: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.c42 [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.824: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.type.8f4: type = fn_type @Op.2, @impl(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.bec: %Op.type.8f4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.6cc: %Destroy.type = facet_value %WithAddr, (%Destroy.impl_witness.824) [concrete]
// CHECK:STDOUT:   %.372: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.6cc [concrete]
// CHECK:STDOUT:   %Op.specific_fn.b42: <specific function> = specific_function %Op.bec, @Op.2(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.specific_fn.523: <specific function> = specific_function %Op.8e6, @Op.2(%ExplicitReturn) [concrete]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5b = import_ref Main//types, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.1ed = import_ref Main//types, loc6_27, unloaded
// CHECK:STDOUT:   %Core.import_ref.40f = import_ref Core//prelude, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.f99: %Destroy.assoc_type = import_ref Core//prelude, loc9_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.150: %Destroy.type = import_ref Core//prelude, inst17 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Core.import_ref.4b4: <witness> = import_ref Core//prelude, loc11_37, loaded [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc11_24, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.56e: type = import_ref Core//prelude, loc11_29, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst41 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b14 = import_ref Main//types, loc10_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.f79 = import_ref Main//types, loc11_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst67 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//types, loc15_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.675 = import_ref Main//types, loc16_33, unloaded
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
// CHECK:STDOUT: interface @Destroy [from "include_files/destroy.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.40f
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.f99
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness constants.%Destroy.impl_witness_table, @impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T) [symbolic = %Op.type (constants.%Op.type.bc9)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.bc9) = struct_value () [symbolic = %Op (constants.%Op.46f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.56e {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.4b4
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %no_addr.patt: %NoAddr = binding_pattern no_addr
// CHECK:STDOUT:     %.loc6_3.1: %NoAddr = var_pattern %no_addr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %no_addr.var: ref %NoAddr = var no_addr
// CHECK:STDOUT:   %impl.elem0.loc6: %.7a4 = impl_witness_access constants.%Destroy.impl_witness.aaa, element0 [concrete = constants.%Op.d78]
// CHECK:STDOUT:   %bound_method.loc6: <bound method> = bound_method %no_addr.var, %impl.elem0.loc6
// CHECK:STDOUT:   %NoAddr.ref: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %no_addr: ref %NoAddr = bind_name no_addr, %no_addr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %explicit_return.patt: %ExplicitReturn = binding_pattern explicit_return
// CHECK:STDOUT:     %.loc7_3.1: %ExplicitReturn = var_pattern %explicit_return.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %explicit_return.var: ref %ExplicitReturn = var explicit_return
// CHECK:STDOUT:   %impl.elem0.loc7: %.b47 = impl_witness_access constants.%Destroy.impl_witness.40a, element0 [concrete = constants.%Op.8e6]
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %explicit_return.var, %impl.elem0.loc7
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %explicit_return: ref %ExplicitReturn = bind_name explicit_return, %explicit_return.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %with_addr.patt: %WithAddr = binding_pattern with_addr
// CHECK:STDOUT:     %.loc8_3.1: %WithAddr = var_pattern %with_addr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %with_addr.var: ref %WithAddr = var with_addr
// CHECK:STDOUT:   %impl.elem0.loc8: %.372 = impl_witness_access constants.%Destroy.impl_witness.824, element0 [concrete = constants.%Op.bec]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %with_addr.var, %impl.elem0.loc8
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %with_addr: ref %WithAddr = bind_name with_addr, %with_addr.var
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Op.2(constants.%WithAddr) [concrete = constants.%Op.specific_fn.b42]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %with_addr.var, %specific_fn.loc8
// CHECK:STDOUT:   %.loc8_3.2: %WithAddr = bind_value %with_addr.var
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc8_3.2)
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Op.2(constants.%ExplicitReturn) [concrete = constants.%Op.specific_fn.523]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %explicit_return.var, %specific_fn.loc7
// CHECK:STDOUT:   %.loc7_3.2: %ExplicitReturn = bind_value %explicit_return.var
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc7_3.2)
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Op.2(constants.%NoAddr) [concrete = constants.%Op.specific_fn.8fe]
// CHECK:STDOUT:   %bound_method.3: <bound method> = bound_method %no_addr.var, %specific_fn.loc6
// CHECK:STDOUT:   %.loc6_3.2: %NoAddr = bind_value %no_addr.var
// CHECK:STDOUT:   %no_op.3: init %empty_tuple.type = call %bound_method.3(%.loc6_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.150: %Destroy.type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.706
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%NoAddr) {
// CHECK:STDOUT:   %T => constants.%NoAddr
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.aaa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.73f
// CHECK:STDOUT:   %Op => constants.%Op.d78
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%ExplicitReturn) {
// CHECK:STDOUT:   %T => constants.%ExplicitReturn
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.40a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.641
// CHECK:STDOUT:   %Op => constants.%Op.8e6
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%WithAddr) {
// CHECK:STDOUT:   %T => constants.%WithAddr
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.824
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.8f4
// CHECK:STDOUT:   %Op => constants.%Op.bec
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%WithAddr) {
// CHECK:STDOUT:   %T => constants.%WithAddr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%ExplicitReturn) {
// CHECK:STDOUT:   %T => constants.%ExplicitReturn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%NoAddr) {
// CHECK:STDOUT:   %T => constants.%NoAddr
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.725 [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.706: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.aaa: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.type.73f: type = fn_type @Op.2, @impl(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.d78: %Op.type.73f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.2b6: %Destroy.type = facet_value %NoAddr, (%Destroy.impl_witness.aaa) [concrete]
// CHECK:STDOUT:   %.7a4: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.2b6 [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.40a: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.type.641: type = fn_type @Op.2, @impl(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.8e6: %Op.type.641 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.c42: %Destroy.type = facet_value %ExplicitReturn, (%Destroy.impl_witness.40a) [concrete]
// CHECK:STDOUT:   %.b47: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.c42 [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.824: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.type.8f4: type = fn_type @Op.2, @impl(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.bec: %Op.type.8f4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.6cc: %Destroy.type = facet_value %WithAddr, (%Destroy.impl_witness.824) [concrete]
// CHECK:STDOUT:   %.372: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.6cc [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Op.specific_fn.8fe: <specific function> = specific_function %Op.d78, @Op.2(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.specific_fn.b42: <specific function> = specific_function %Op.bec, @Op.2(%WithAddr) [concrete]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5b = import_ref Main//types, loc5_22, unloaded
// CHECK:STDOUT:   %Main.import_ref.1ed = import_ref Main//types, loc6_27, unloaded
// CHECK:STDOUT:   %Core.import_ref.40f = import_ref Core//prelude, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.f99: %Destroy.assoc_type = import_ref Core//prelude, loc9_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.150: %Destroy.type = import_ref Core//prelude, inst17 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Core.import_ref.4b4: <witness> = import_ref Core//prelude, loc11_37, loaded [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc11_24, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.56e: type = import_ref Core//prelude, loc11_29, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst41 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b14 = import_ref Main//types, loc10_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.f79 = import_ref Main//types, loc11_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst67 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1cb = import_ref Main//types, loc15_24, unloaded
// CHECK:STDOUT:   %Main.import_ref.675 = import_ref Main//types, loc16_33, unloaded
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
// CHECK:STDOUT: interface @Destroy [from "include_files/destroy.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.40f
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.f99
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness constants.%Destroy.impl_witness_table, @impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T) [symbolic = %Op.type (constants.%Op.type.bc9)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.bc9) = struct_value () [symbolic = %Op (constants.%Op.46f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.56e {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.4b4
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %no_addr.patt: %NoAddr = binding_pattern no_addr
// CHECK:STDOUT:     %.loc6_3.1: %NoAddr = var_pattern %no_addr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %no_addr.var: ref %NoAddr = var no_addr
// CHECK:STDOUT:   %impl.elem0.loc6: %.7a4 = impl_witness_access constants.%Destroy.impl_witness.aaa, element0 [concrete = constants.%Op.d78]
// CHECK:STDOUT:   %bound_method.loc6: <bound method> = bound_method %no_addr.var, %impl.elem0.loc6
// CHECK:STDOUT:   %NoAddr.ref.loc6: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %no_addr: ref %NoAddr = bind_name no_addr, %no_addr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %explicit_return.patt: %ExplicitReturn = binding_pattern explicit_return
// CHECK:STDOUT:     %.loc7_3.1: %ExplicitReturn = var_pattern %explicit_return.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %explicit_return.var: ref %ExplicitReturn = var explicit_return
// CHECK:STDOUT:   %impl.elem0.loc7: %.b47 = impl_witness_access constants.%Destroy.impl_witness.40a, element0 [concrete = constants.%Op.8e6]
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %explicit_return.var, %impl.elem0.loc7
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %explicit_return: ref %ExplicitReturn = bind_name explicit_return, %explicit_return.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %with_addr.patt: %WithAddr = binding_pattern with_addr
// CHECK:STDOUT:     %.loc8_3.1: %WithAddr = var_pattern %with_addr.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %with_addr.var: ref %WithAddr = var with_addr
// CHECK:STDOUT:   %impl.elem0.loc8: %.372 = impl_witness_access constants.%Destroy.impl_witness.824, element0 [concrete = constants.%Op.bec]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %with_addr.var, %impl.elem0.loc8
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %with_addr: ref %WithAddr = bind_name with_addr, %with_addr.var
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %in_scope.patt: %NoAddr = binding_pattern in_scope
// CHECK:STDOUT:     %.loc10_5.1: %NoAddr = var_pattern %in_scope.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %in_scope.var: ref %NoAddr = var in_scope
// CHECK:STDOUT:   %impl.elem0.loc10: %.7a4 = impl_witness_access constants.%Destroy.impl_witness.aaa, element0 [concrete = constants.%Op.d78]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %in_scope.var, %impl.elem0.loc10
// CHECK:STDOUT:   %NoAddr.ref.loc10: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %in_scope: ref %NoAddr = bind_name in_scope, %in_scope.var
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Op.2(constants.%NoAddr) [concrete = constants.%Op.specific_fn.8fe]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %in_scope.var, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_5.2: %NoAddr = bind_value %in_scope.var
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc10_5.2)
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Op.2(constants.%WithAddr) [concrete = constants.%Op.specific_fn.b42]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %with_addr.var, %specific_fn.loc8
// CHECK:STDOUT:   %.loc8_3.2: %WithAddr = bind_value %with_addr.var
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc8_3.2)
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Op.2(constants.%ExplicitReturn) [concrete = constants.%Op.specific_fn.523]
// CHECK:STDOUT:   %bound_method.3: <bound method> = bound_method %explicit_return.var, %specific_fn.loc7
// CHECK:STDOUT:   %.loc7_3.2: %ExplicitReturn = bind_value %explicit_return.var
// CHECK:STDOUT:   %no_op.3: init %empty_tuple.type = call %bound_method.3(%.loc7_3.2)
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Op.2(constants.%NoAddr) [concrete = constants.%Op.specific_fn.8fe]
// CHECK:STDOUT:   %bound_method.4: <bound method> = bound_method %no_addr.var, %specific_fn.loc6
// CHECK:STDOUT:   %.loc6_3.2: %NoAddr = bind_value %no_addr.var
// CHECK:STDOUT:   %no_op.4: init %empty_tuple.type = call %bound_method.4(%.loc6_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.150: %Destroy.type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.706
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%NoAddr) {
// CHECK:STDOUT:   %T => constants.%NoAddr
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.aaa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.73f
// CHECK:STDOUT:   %Op => constants.%Op.d78
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%ExplicitReturn) {
// CHECK:STDOUT:   %T => constants.%ExplicitReturn
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.40a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.641
// CHECK:STDOUT:   %Op => constants.%Op.8e6
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%WithAddr) {
// CHECK:STDOUT:   %T => constants.%WithAddr
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.824
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.8f4
// CHECK:STDOUT:   %Op => constants.%Op.bec
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%NoAddr) {
// CHECK:STDOUT:   %T => constants.%NoAddr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%WithAddr) {
// CHECK:STDOUT:   %T => constants.%WithAddr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%ExplicitReturn) {
// CHECK:STDOUT:   %T => constants.%ExplicitReturn
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Make.type.bc9: type = fn_type @Make.1 [concrete]
// CHECK:STDOUT:   %Make.1be: %Make.type.bc9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.725 [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.706: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.aaa: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.type.73f: type = fn_type @Op.2, @impl(%NoAddr) [concrete]
// CHECK:STDOUT:   %Op.d78: %Op.type.73f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.2b6: %Destroy.type = facet_value %NoAddr, (%Destroy.impl_witness.aaa) [concrete]
// CHECK:STDOUT:   %.7a4: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.2b6 [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %Make.type.378: type = fn_type @Make.2 [concrete]
// CHECK:STDOUT:   %Make.960: %Make.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.40a: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.type.641: type = fn_type @Op.2, @impl(%ExplicitReturn) [concrete]
// CHECK:STDOUT:   %Op.8e6: %Op.type.641 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.c42: %Destroy.type = facet_value %ExplicitReturn, (%Destroy.impl_witness.40a) [concrete]
// CHECK:STDOUT:   %.b47: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.c42 [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %Make.type.e14: type = fn_type @Make.3 [concrete]
// CHECK:STDOUT:   %Make.b0a: %Make.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.824: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.type.8f4: type = fn_type @Op.2, @impl(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.bec: %Op.type.8f4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.6cc: %Destroy.type = facet_value %WithAddr, (%Destroy.impl_witness.824) [concrete]
// CHECK:STDOUT:   %.372: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.6cc [concrete]
// CHECK:STDOUT:   %Op.specific_fn.b42: <specific function> = specific_function %Op.bec, @Op.2(%WithAddr) [concrete]
// CHECK:STDOUT:   %Op.specific_fn.523: <specific function> = specific_function %Op.8e6, @Op.2(%ExplicitReturn) [concrete]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//types, loc7_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.f42 = import_ref Main//types, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.784: %Make.type.bc9 = import_ref Main//types, loc5_22, loaded [concrete = constants.%Make.1be]
// CHECK:STDOUT:   %Main.import_ref.1ed = import_ref Main//types, loc6_27, unloaded
// CHECK:STDOUT:   %Core.import_ref.40f = import_ref Core//prelude, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.f99: %Destroy.assoc_type = import_ref Core//prelude, loc9_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.150: %Destroy.type = import_ref Core//prelude, inst17 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Core.import_ref.4b4: <witness> = import_ref Core//prelude, loc11_37, loaded [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc11_24, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.56e: type = import_ref Core//prelude, loc11_29, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//types, loc12_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.ee7 = import_ref Main//types, inst41 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8e0: %Make.type.378 = import_ref Main//types, loc10_30, loaded [concrete = constants.%Make.960]
// CHECK:STDOUT:   %Main.import_ref.f79 = import_ref Main//types, loc11_33, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//types, loc17_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.95d = import_ref Main//types, inst67 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.974: %Make.type.e14 = import_ref Main//types, loc15_24, loaded [concrete = constants.%Make.b0a]
// CHECK:STDOUT:   %Main.import_ref.675 = import_ref Main//types, loc16_33, unloaded
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
// CHECK:STDOUT: interface @Destroy [from "include_files/destroy.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.40f
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.f99
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness constants.%Destroy.impl_witness_table, @impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T) [symbolic = %Op.type (constants.%Op.type.bc9)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.bc9) = struct_value () [symbolic = %Op (constants.%Op.46f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.56e {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.4b4
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.7a4 = impl_witness_access constants.%Destroy.impl_witness.aaa, element0 [concrete = constants.%Op.d78]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_15.1, %impl.elem0.loc8
// CHECK:STDOUT:   %Make.call.loc8: init %NoAddr = call %Make.ref.loc8() to %.loc8_15.1
// CHECK:STDOUT:   %.loc8_15.2: ref %NoAddr = temporary %.loc8_15.1, %Make.call.loc8
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Make.ref.loc9: %Make.type.378 = name_ref Make, imports.%Main.import_ref.8e0 [concrete = constants.%Make.960]
// CHECK:STDOUT:   %.loc9_23.1: ref %ExplicitReturn = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc9: %.b47 = impl_witness_access constants.%Destroy.impl_witness.40a, element0 [concrete = constants.%Op.8e6]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %.loc9_23.1, %impl.elem0.loc9
// CHECK:STDOUT:   %Make.call.loc9: init %ExplicitReturn = call %Make.ref.loc9() to %.loc9_23.1
// CHECK:STDOUT:   %.loc9_23.2: ref %ExplicitReturn = temporary %.loc9_23.1, %Make.call.loc9
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Make.ref.loc10: %Make.type.e14 = name_ref Make, imports.%Main.import_ref.974 [concrete = constants.%Make.b0a]
// CHECK:STDOUT:   %.loc10_17.1: ref %WithAddr = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc10: %.372 = impl_witness_access constants.%Destroy.impl_witness.824, element0 [concrete = constants.%Op.bec]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_17.1, %impl.elem0.loc10
// CHECK:STDOUT:   %Make.call.loc10: init %WithAddr = call %Make.ref.loc10() to %.loc10_17.1
// CHECK:STDOUT:   %.loc10_17.2: ref %WithAddr = temporary %.loc10_17.1, %Make.call.loc10
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Op.2(constants.%WithAddr) [concrete = constants.%Op.specific_fn.b42]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.loc10_17.1, %specific_fn.loc10
// CHECK:STDOUT:   %.loc10_17.3: %WithAddr = bind_value %.loc10_17.1
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc10_17.3)
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9, @Op.2(constants.%ExplicitReturn) [concrete = constants.%Op.specific_fn.523]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.loc9_23.1, %specific_fn.loc9
// CHECK:STDOUT:   %.loc9_23.3: %ExplicitReturn = bind_value %.loc9_23.1
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc9_23.3)
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Op.2(constants.%NoAddr) [concrete = constants.%Op.specific_fn.8fe]
// CHECK:STDOUT:   %bound_method.3: <bound method> = bound_method %.loc8_15.1, %specific_fn.loc8
// CHECK:STDOUT:   %.loc8_15.3: %NoAddr = bind_value %.loc8_15.1
// CHECK:STDOUT:   %no_op.3: init %empty_tuple.type = call %bound_method.3(%.loc8_15.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.1 [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.150: %Destroy.type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.2 [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make.3 [from "types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.706
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%NoAddr) {
// CHECK:STDOUT:   %T => constants.%NoAddr
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.aaa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.73f
// CHECK:STDOUT:   %Op => constants.%Op.d78
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%ExplicitReturn) {
// CHECK:STDOUT:   %T => constants.%ExplicitReturn
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.40a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.641
// CHECK:STDOUT:   %Op => constants.%Op.8e6
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%WithAddr) {
// CHECK:STDOUT:   %T => constants.%WithAddr
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.824
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.8f4
// CHECK:STDOUT:   %Op => constants.%Op.bec
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%WithAddr) {
// CHECK:STDOUT:   %T => constants.%WithAddr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%ExplicitReturn) {
// CHECK:STDOUT:   %T => constants.%ExplicitReturn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%NoAddr) {
// CHECK:STDOUT:   %T => constants.%NoAddr
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.725 [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.706: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.da8: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%NoSelf) [concrete]
// CHECK:STDOUT:   %Op.type.f20: type = fn_type @Op.2, @impl(%NoSelf) [concrete]
// CHECK:STDOUT:   %Op.2e0: %Op.type.f20 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.f87: %Destroy.type = facet_value %NoSelf, (%Destroy.impl_witness.da8) [concrete]
// CHECK:STDOUT:   %.960: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.f87 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ba4: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%Args) [concrete]
// CHECK:STDOUT:   %Op.type.59d: type = fn_type @Op.2, @impl(%Args) [concrete]
// CHECK:STDOUT:   %Op.a71: %Op.type.59d = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.7fd: %Destroy.type = facet_value %Args, (%Destroy.impl_witness.ba4) [concrete]
// CHECK:STDOUT:   %.dcb: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.7fd [concrete]
// CHECK:STDOUT:   %Op.specific_fn.242: <specific function> = specific_function %Op.a71, @Op.2(%Args) [concrete]
// CHECK:STDOUT:   %Op.specific_fn.a58: <specific function> = specific_function %Op.2e0, @Op.2(%NoSelf) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.40f = import_ref Core//prelude, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.f99: %Destroy.assoc_type = import_ref Core//prelude, loc9_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.150: %Destroy.type = import_ref Core//prelude, inst17 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Core.import_ref.4b4: <witness> = import_ref Core//prelude, loc11_37, loaded [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc11_24, loaded [symbolic = @impl.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.56e: type = import_ref Core//prelude, loc11_29, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T)]
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
// CHECK:STDOUT: interface @Destroy [from "include_files/destroy.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.40f
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.f99
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness constants.%Destroy.impl_witness_table, @impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T) [symbolic = %Op.type (constants.%Op.type.bc9)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.bc9) = struct_value () [symbolic = %Op (constants.%Op.46f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.56e {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.4b4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoSelf {
// CHECK:STDOUT:   %destroy.decl: %destroy.type.d69 = fn_decl @destroy.1 [concrete = constants.%destroy.ab7] {} {}
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT: fn @destroy.2(%self.param: %Args, %x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %NoSelf = binding_pattern a
// CHECK:STDOUT:     %.loc21_3.1: %NoSelf = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %NoSelf = var a
// CHECK:STDOUT:   %impl.elem0.loc21: %.960 = impl_witness_access constants.%Destroy.impl_witness.da8, element0 [concrete = constants.%Op.2e0]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %a.var, %impl.elem0.loc21
// CHECK:STDOUT:   %NoSelf.ref: type = name_ref NoSelf, file.%NoSelf.decl [concrete = constants.%NoSelf]
// CHECK:STDOUT:   %a: ref %NoSelf = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %Args = binding_pattern b
// CHECK:STDOUT:     %.loc22_3.1: %Args = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %Args = var b
// CHECK:STDOUT:   %impl.elem0.loc22: %.dcb = impl_witness_access constants.%Destroy.impl_witness.ba4, element0 [concrete = constants.%Op.a71]
// CHECK:STDOUT:   %bound_method.loc22: <bound method> = bound_method %b.var, %impl.elem0.loc22
// CHECK:STDOUT:   %Args.ref: type = name_ref Args, file.%Args.decl [concrete = constants.%Args]
// CHECK:STDOUT:   %b: ref %Args = bind_name b, %b.var
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Op.2(constants.%Args) [concrete = constants.%Op.specific_fn.242]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %b.var, %specific_fn.loc22
// CHECK:STDOUT:   %.loc22_3.2: %Args = bind_value %b.var
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc22_3.2)
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Op.2(constants.%NoSelf) [concrete = constants.%Op.specific_fn.a58]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %a.var, %specific_fn.loc21
// CHECK:STDOUT:   %.loc21_3.2: %NoSelf = bind_value %a.var
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc21_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.150: %Destroy.type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.706
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%NoSelf) {
// CHECK:STDOUT:   %T => constants.%NoSelf
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.da8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.f20
// CHECK:STDOUT:   %Op => constants.%Op.2e0
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%Args) {
// CHECK:STDOUT:   %T => constants.%Args
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.ba4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.59d
// CHECK:STDOUT:   %Op => constants.%Op.a71
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%Args) {
// CHECK:STDOUT:   %T => constants.%Args
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%NoSelf) {
// CHECK:STDOUT:   %T => constants.%NoSelf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_breaking_generics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3d5d.1: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %T.patt.e017e3.1: type = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T.8b3d5d.1) [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.389: <witness> = require_complete_type %C.f2e [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.725 [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %T.8b3d5d.2: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e017e3.2: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.706: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%T.8b3d5d.2) [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T.8b3d5d.2) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3d5d.2 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.6a7: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%C.f2e) [template]
// CHECK:STDOUT:   %Op.type.4a9: type = fn_type @Op.2, @impl(%C.f2e) [template]
// CHECK:STDOUT:   %Op.c24: %Op.type.4a9 = struct_value () [template]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.f2e, @Destroy [template]
// CHECK:STDOUT:   %Destroy.facet.d66: %Destroy.type = facet_value %C.f2e, (%Destroy.lookup_impl_witness) [template]
// CHECK:STDOUT:   %.eb5: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.d66 [template]
// CHECK:STDOUT:   %impl.elem0: %.eb5 = impl_witness_access %Destroy.lookup_impl_witness, element0 [template]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Destroy.facet.d66) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.a3f: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%C.7a7) [concrete]
// CHECK:STDOUT:   %Op.type.add: type = fn_type @Op.2, @impl(%C.7a7) [concrete]
// CHECK:STDOUT:   %Op.b1c: %Op.type.add = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.f20: %Destroy.type = facet_value %C.7a7, (%Destroy.impl_witness.a3f) [concrete]
// CHECK:STDOUT:   %.2d6: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.f20 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.b1c, @Op.2(%C.7a7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.40f = import_ref Core//prelude, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.f99: %Destroy.assoc_type = import_ref Core//prelude, loc9_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.Op = import_ref Core//prelude, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.150: %Destroy.type = import_ref Core//prelude, inst17 [no loc], loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Core.import_ref.4b4: <witness> = import_ref Core//prelude, loc11_37, loaded [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T.8b3d5d.2)]
// CHECK:STDOUT:   %Core.import_ref.583: type = import_ref Core//prelude, loc11_24, loaded [symbolic = @impl.%T (constants.%T.8b3d5d.2)]
// CHECK:STDOUT:   %Core.import_ref.56e: type = import_ref Core//prelude, loc11_29, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//prelude, loc11_14, loaded [symbolic = @impl.%T (constants.%T.8b3d5d.2)]
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
// CHECK:STDOUT:     %T.patt.loc2_18.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc2_18.2 (constants.%T.patt.e017e3.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc2_18.1: type = bind_symbolic_name T, 0, template [template = %T.loc2_18.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt.e017e3.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Destroy [from "include_files/destroy.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Core.import_ref.40f
// CHECK:STDOUT:   .Op = imports.%Core.import_ref.f99
// CHECK:STDOUT:   witness = (imports.%Core.Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(imports.%Core.import_ref.5ab3ec.1: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3d5d.2)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e017e3.2)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness constants.%Destroy.impl_witness_table, @impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.706)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T) [symbolic = %Op.type (constants.%Op.type.bc9)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.bc9) = struct_value () [symbolic = %Op (constants.%Op.46f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%Core.import_ref.583 as imports.%Core.import_ref.56e {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%Core.import_ref.4b4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc2_18.1: type) {
// CHECK:STDOUT:   %T.loc2_18.2: type = bind_symbolic_name T, 0, template [template = %T.loc2_18.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   %T.patt.loc2_18.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc2_18.2 (constants.%T.patt.e017e3.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt.e017e3.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc5_13.2: type = class_type @C, @C(%T.loc4_15.2) [template = %C.loc5_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc5_13.2 [template = %require_complete (constants.%require_complete.389)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.loc5_13.2, @Destroy [template = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.loc5_13.2, (%Destroy.lookup_impl_witness) [template = %Destroy.facet (constants.%Destroy.facet.d66)]
// CHECK:STDOUT:   %.loc5_3.3: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [template = %.loc5_3.3 (constants.%.eb5)]
// CHECK:STDOUT:   %impl.elem0.loc5_3.2: @F.%.loc5_3.3 (%.eb5) = impl_witness_access %Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc5_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc5_3.2: <specific function> = specific_impl_function %impl.elem0.loc5_3.2, @Op.1(%Destroy.facet) [template = %specific_impl_fn.loc5_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%C.loc5_13.2 (%C.f2e) = binding_pattern v
// CHECK:STDOUT:       %.loc5_3.1: @F.%C.loc5_13.2 (%C.f2e) = var_pattern %v.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%C.loc5_13.2 (%C.f2e) = var v
// CHECK:STDOUT:     %impl.elem0.loc5_3.1: @F.%.loc5_3.3 (%.eb5) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc5_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc5: <bound method> = bound_method %v.var, %impl.elem0.loc5_3.1
// CHECK:STDOUT:     %.loc5_13: type = splice_block %C.loc5_13.1 [template = %C.loc5_13.2 (constants.%C.f2e)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:       %C.loc5_13.1: type = class_type @C, @C(constants.%T.8b3d5d.1) [template = %C.loc5_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%C.loc5_13.2 (%C.f2e) = bind_name v, %v.var
// CHECK:STDOUT:     %specific_impl_fn.loc5_3.1: <specific function> = specific_impl_function %impl.elem0.loc5_3.1, @Op.1(constants.%Destroy.facet.d66) [template = %specific_impl_fn.loc5_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.1: <bound method> = bound_method %v.var, %specific_impl_fn.loc5_3.1
// CHECK:STDOUT:     %.loc5_3.2: @F.%C.loc5_13.2 (%C.f2e) = bind_value %v.var
// CHECK:STDOUT:     %.1: init %empty_tuple.type = call %bound_method.1(%.loc5_3.2)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(imports.%Core.import_ref.150: %Destroy.type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(imports.%Core.import_ref.5ab3ec.2: type) [from "include_files/destroy.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3d5d.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc8_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_14: type = converted %.loc8_13, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%empty_struct_type) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.8b3d5d.1) {
// CHECK:STDOUT:   %T.loc2_18.2 => constants.%T.8b3d5d.1
// CHECK:STDOUT:   %T.patt.loc2_18.2 => constants.%T.patt.e017e3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3d5d.1) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T.8b3d5d.1
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt.e017e3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T.8b3d5d.2) {
// CHECK:STDOUT:   %T => constants.%T.8b3d5d.2
// CHECK:STDOUT:   %T.patt => constants.%T.patt.e017e3.2
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.706
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T.8b3d5d.2) {
// CHECK:STDOUT:   %T => constants.%T.8b3d5d.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%C.f2e) {
// CHECK:STDOUT:   %T => constants.%C.f2e
// CHECK:STDOUT:   %T.patt => constants.%T.patt.e017e3.2
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.6a7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.4a9
// CHECK:STDOUT:   %Op => constants.%Op.c24
// CHECK:STDOUT:   %require_complete => constants.%require_complete.389
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Destroy.facet.d66) {
// CHECK:STDOUT:   %Self => constants.%Destroy.facet.d66
// CHECK:STDOUT:   %Self.as_type => constants.%C.f2e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@F.%T.loc4_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(@F.%Destroy.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt.e017e3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc5_13.2 => constants.%C.7a7
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.a3f
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.f20
// CHECK:STDOUT:   %.loc5_3.3 => constants.%.2d6
// CHECK:STDOUT:   %impl.elem0.loc5_3.2 => constants.%Op.b1c
// CHECK:STDOUT:   %specific_impl_fn.loc5_3.2 => constants.%Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc2_18.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc2_18.2 => constants.%T.patt.e017e3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%C.7a7) {
// CHECK:STDOUT:   %T => constants.%C.7a7
// CHECK:STDOUT:   %T.patt => constants.%T.patt.e017e3.2
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.a3f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type => constants.%Op.type.add
// CHECK:STDOUT:   %Op => constants.%Op.b1c
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Destroy.facet.f20) {
// CHECK:STDOUT:   %Self => constants.%Destroy.facet.f20
// CHECK:STDOUT:   %Self.as_type => constants.%C.7a7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%C.7a7) {
// CHECK:STDOUT:   %T => constants.%C.7a7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %Op.type.439: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.417: %Op.type.439 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, @Destroy.%Op.decl [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness file.%Destroy.impl_witness_table, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.type.b22: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.d6d: %Op.type.b22 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %T, (%Destroy.impl_witness) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Destroy = %Destroy.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.decl: type = interface_decl @Destroy [concrete = constants.%Destroy.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_14.1 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:     %Destroy.ref: type = name_ref Destroy, file.%Destroy.decl [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@impl.%Op.decl), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @impl(constants.%T) [symbolic = @impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Destroy {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op.decl: %Op.type.439 = fn_decl @Op.1 [concrete = constants.%Op.417] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_15.1: type = splice_block %.loc9_15.2 [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %Destroy.type = name_ref Self, @Destroy.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc9_15.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc9_15.2: type = converted %Self.ref, %Self.as_type.loc9_15.2 [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, %Op.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T.loc11_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness file.%Destroy.impl_witness_table, @impl(%T.loc11_14.2) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Op.type: type = fn_type @Op.2, @impl(%T.loc11_14.2) [symbolic = %Op.type (constants.%Op.type.b22)]
// CHECK:STDOUT:   %Op: @impl.%Op.type (%Op.type.b22) = struct_value () [symbolic = %Op (constants.%Op.d6d)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc11_14.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %Destroy.ref {
// CHECK:STDOUT:     %Op.decl: @impl.%Op.type (%Op.type.b22) = fn_decl @Op.2 [symbolic = @impl.%Op (constants.%Op.d6d)] {
// CHECK:STDOUT:       %self.patt: @Op.2.%T (%T) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Op.2.%T (%T) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Op.2.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%T.ref [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self: @Op.2.%T (%T) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Op.decl
// CHECK:STDOUT:     witness = file.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Destroy.%Self: %Destroy.type) {
// CHECK:STDOUT:   %Self: %Destroy.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_15.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.1.%Self.as_type.loc9_15.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.2(@impl.%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Op.2.%T (%T)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_15.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T.patt
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Destroy.facet) {
// CHECK:STDOUT:   %Self => constants.%Destroy.facet
// CHECK:STDOUT:   %Self.as_type.loc9_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:

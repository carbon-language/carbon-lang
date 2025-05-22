// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
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

//@dump-sem-ir-begin
fn F() {
  var no_addr: NoAddr;
  var explicit_return: ExplicitReturn;
  var with_addr: WithAddr;
}
//@dump-sem-ir-end

// --- nested_scope.carbon

library "[[@TEST_NAME]]";
import library "types";

//@dump-sem-ir-begin
fn F() {
  var no_addr: NoAddr;
  var explicit_return: ExplicitReturn;
  var with_addr: WithAddr;
  if (true) {
    var in_scope: NoAddr;
  }
}
//@dump-sem-ir-end

// --- temp.carbon

library "[[@TEST_NAME]]";
import library "types";

//@dump-sem-ir-begin
fn F() {
  // TODO: The scoping of these destroy calls is incorrect. Maybe we need to
  // establish statement scopes?
  NoAddr.Make();
  ExplicitReturn.Make();
  WithAddr.Make();
}
//@dump-sem-ir-end

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
library "[[@TEST_NAME]]";

class C(template T:! type) {}

//@dump-sem-ir-begin
fn F(template T:! type) {
  var v: C(T);
}
//@dump-sem-ir-end

fn G() { F({}); }

// CHECK:STDOUT: --- implicit_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %NoAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %WithAddr [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NoAddr: type = import_ref Main//types, NoAddr, loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.ExplicitReturn: type = import_ref Main//types, ExplicitReturn, loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.WithAddr: type = import_ref Main//types, WithAddr, loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %no_addr.patt: %pattern_type.88f = binding_pattern no_addr [concrete]
// CHECK:STDOUT:     %no_addr.var_patt: %pattern_type.88f = var_pattern %no_addr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %no_addr.var: ref %NoAddr = var %no_addr.var_patt
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %NoAddr.ref: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %no_addr: ref %NoAddr = bind_name no_addr, %no_addr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %explicit_return.patt: %pattern_type.611 = binding_pattern explicit_return [concrete]
// CHECK:STDOUT:     %explicit_return.var_patt: %pattern_type.611 = var_pattern %explicit_return.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %explicit_return.var: ref %ExplicitReturn = var %explicit_return.var_patt
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %explicit_return: ref %ExplicitReturn = bind_name explicit_return, %explicit_return.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %with_addr.patt: %pattern_type.f93 = binding_pattern with_addr [concrete]
// CHECK:STDOUT:     %with_addr.var_patt: %pattern_type.f93 = var_pattern %with_addr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %with_addr.var: ref %WithAddr = var %with_addr.var_patt
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %with_addr: ref %WithAddr = bind_name with_addr, %with_addr.var
// CHECK:STDOUT:   %addr: %ptr.b4e = addr_of %with_addr.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8: %ExplicitReturn = bind_value %explicit_return.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc7: %NoAddr = bind_value %no_addr.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %NoAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %WithAddr [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NoAddr: type = import_ref Main//types, NoAddr, loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.ExplicitReturn: type = import_ref Main//types, ExplicitReturn, loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.WithAddr: type = import_ref Main//types, WithAddr, loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %no_addr.patt: %pattern_type.88f = binding_pattern no_addr [concrete]
// CHECK:STDOUT:     %no_addr.var_patt: %pattern_type.88f = var_pattern %no_addr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %no_addr.var: ref %NoAddr = var %no_addr.var_patt
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %NoAddr.ref.loc7: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %no_addr: ref %NoAddr = bind_name no_addr, %no_addr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %explicit_return.patt: %pattern_type.611 = binding_pattern explicit_return [concrete]
// CHECK:STDOUT:     %explicit_return.var_patt: %pattern_type.611 = var_pattern %explicit_return.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %explicit_return.var: ref %ExplicitReturn = var %explicit_return.var_patt
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %explicit_return: ref %ExplicitReturn = bind_name explicit_return, %explicit_return.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %with_addr.patt: %pattern_type.f93 = binding_pattern with_addr [concrete]
// CHECK:STDOUT:     %with_addr.var_patt: %pattern_type.f93 = var_pattern %with_addr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %with_addr.var: ref %WithAddr = var %with_addr.var_patt
// CHECK:STDOUT:   <elided>
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
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %NoAddr.ref.loc11: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %in_scope: ref %NoAddr = bind_name in_scope, %in_scope.var
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc11: %NoAddr = bind_value %in_scope.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %addr: %ptr.b4e = addr_of %with_addr.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8: %ExplicitReturn = bind_value %explicit_return.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc7: %NoAddr = bind_value %no_addr.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- temp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %Make.type.bc9: type = fn_type @Make.1 [concrete]
// CHECK:STDOUT:   %Make.1be: %Make.type.bc9 = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %Make.type.378: type = fn_type @Make.2 [concrete]
// CHECK:STDOUT:   %Make.960: %Make.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %Make.type.e14: type = fn_type @Make.3 [concrete]
// CHECK:STDOUT:   %Make.b0a: %Make.type.e14 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NoAddr: type = import_ref Main//types, NoAddr, loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.ExplicitReturn: type = import_ref Main//types, ExplicitReturn, loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.WithAddr: type = import_ref Main//types, WithAddr, loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Main.import_ref.784: %Make.type.bc9 = import_ref Main//types, loc5_22, loaded [concrete = constants.%Make.1be]
// CHECK:STDOUT:   %Main.import_ref.8e0: %Make.type.378 = import_ref Main//types, loc10_30, loaded [concrete = constants.%Make.960]
// CHECK:STDOUT:   %Main.import_ref.974: %Make.type.e14 = import_ref Main//types, loc15_24, loaded [concrete = constants.%Make.b0a]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NoAddr.ref: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Make.ref.loc9: %Make.type.bc9 = name_ref Make, imports.%Main.import_ref.784 [concrete = constants.%Make.1be]
// CHECK:STDOUT:   %.loc9_15.1: ref %NoAddr = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Make.call.loc9: init %NoAddr = call %Make.ref.loc9() to %.loc9_15.1
// CHECK:STDOUT:   %.loc9_15.2: ref %NoAddr = temporary %.loc9_15.1, %Make.call.loc9
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Make.ref.loc10: %Make.type.378 = name_ref Make, imports.%Main.import_ref.8e0 [concrete = constants.%Make.960]
// CHECK:STDOUT:   %.loc10_23.1: ref %ExplicitReturn = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Make.call.loc10: init %ExplicitReturn = call %Make.ref.loc10() to %.loc10_23.1
// CHECK:STDOUT:   %.loc10_23.2: ref %ExplicitReturn = temporary %.loc10_23.1, %Make.call.loc10
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Make.ref.loc11: %Make.type.e14 = name_ref Make, imports.%Main.import_ref.974 [concrete = constants.%Make.b0a]
// CHECK:STDOUT:   %.loc11_17.1: ref %WithAddr = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Make.call.loc11: init %WithAddr = call %Make.ref.loc11() to %.loc11_17.1
// CHECK:STDOUT:   %.loc11_17.2: ref %WithAddr = temporary %.loc11_17.1, %Make.call.loc11
// CHECK:STDOUT:   %addr: %ptr.b4e = addr_of %.loc11_17.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_23.3: %ExplicitReturn = bind_value %.loc10_23.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_15.3: %NoAddr = bind_value %.loc9_15.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_breaking_generics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.f2e [template]
// CHECK:STDOUT:   %pattern_type.e5e: type = pattern_type %C.f2e [template]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.99a: type = pattern_type %C.7a7 [concrete]
// CHECK:STDOUT:   %inst.as_compatible: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.072: ref %C.7a7 = as_compatible @F.%v.var
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.432: <error> = splice_block <error> [concrete = <error>] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc8_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_15.1: type) {
// CHECK:STDOUT:   %T.loc8_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc9_13.2: type = class_type @C, @C(%T.loc8_15.2) [template = %C.loc9_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc9_13.2 [template = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc9_13.2 [template = %pattern_type (constants.%pattern_type.e5e)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.e5e) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.e5e) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%C.loc9_13.2 (%C.f2e) = var %v.var_patt
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc9: type = splice_block %C.loc9_13.1 [template = %C.loc9_13.2 (constants.%C.f2e)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc8_15.1 [template = %T.loc8_15.2 (constants.%T)]
// CHECK:STDOUT:       %C.loc9_13.1: type = class_type @C, @C(constants.%T) [template = %C.loc9_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%C.loc9_13.2 (%C.f2e) = bind_name v, %v.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc8_15.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc9_13.2 => constants.%C.7a7
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.99a
// CHECK:STDOUT:   %.3 => constants.%inst.as_compatible
// CHECK:STDOUT:   %.4 => constants.%inst.splice_block
// CHECK:STDOUT:   %.5 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

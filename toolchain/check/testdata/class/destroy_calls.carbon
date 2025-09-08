// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
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

// --- generic_class.carbon
library "[[@TEST_NAME]]";

class C {}

class D(template T:! type) {}

//@dump-sem-ir-begin
fn F() {
  var a: D(C);
}
//@dump-sem-ir-end

// --- generic_use_inside_generic.carbon
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %NoAddr: type = class_type @NoAddr [concrete]
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %NoAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %WithAddr [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.type: type = fn_type @WithAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op: %WithAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.type: type = fn_type @ExplicitReturn.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op: %ExplicitReturn.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b0a: type = ptr_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.type: type = fn_type @NoAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op: %NoAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4b8: type = ptr_type %NoAddr [concrete]
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
// CHECK:STDOUT:   %addr.loc9: %ptr.b4e = addr_of %with_addr.var
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %WithAddr.as.Destroy.impl.Op.bound(%addr.loc9)
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.bound: <bound method> = bound_method %explicit_return.var, constants.%ExplicitReturn.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc8: %ptr.b0a = addr_of %explicit_return.var
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.call: init %empty_tuple.type = call %ExplicitReturn.as.Destroy.impl.Op.bound(%addr.loc8)
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.bound: <bound method> = bound_method %no_addr.var, constants.%NoAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc7: %ptr.4b8 = addr_of %no_addr.var
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %NoAddr.as.Destroy.impl.Op.bound(%addr.loc7)
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
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %NoAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %WithAddr [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.type: type = fn_type @NoAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op: %NoAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4b8: type = ptr_type %NoAddr [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.type: type = fn_type @WithAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op: %WithAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.type: type = fn_type @ExplicitReturn.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op: %ExplicitReturn.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b0a: type = ptr_type %ExplicitReturn [concrete]
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
// CHECK:STDOUT:   %NoAddr.ref.loc7: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
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
// CHECK:STDOUT:   %NoAddr.ref.loc11: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %in_scope: ref %NoAddr = bind_name in_scope, %in_scope.var
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.bound.loc11: <bound method> = bound_method %in_scope.var, constants.%NoAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc11: %ptr.4b8 = addr_of %in_scope.var
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.call.loc11: init %empty_tuple.type = call %NoAddr.as.Destroy.impl.Op.bound.loc11(%addr.loc11)
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.bound: <bound method> = bound_method %with_addr.var, constants.%WithAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc9: %ptr.b4e = addr_of %with_addr.var
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %WithAddr.as.Destroy.impl.Op.bound(%addr.loc9)
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.bound: <bound method> = bound_method %explicit_return.var, constants.%ExplicitReturn.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc8: %ptr.b0a = addr_of %explicit_return.var
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.call: init %empty_tuple.type = call %ExplicitReturn.as.Destroy.impl.Op.bound(%addr.loc8)
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.bound.loc7: <bound method> = bound_method %no_addr.var, constants.%NoAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc7: %ptr.4b8 = addr_of %no_addr.var
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.call.loc7: init %empty_tuple.type = call %NoAddr.as.Destroy.impl.Op.bound.loc7(%addr.loc7)
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
// CHECK:STDOUT:   %NoAddr.Make.type: type = fn_type @NoAddr.Make [concrete]
// CHECK:STDOUT:   %NoAddr.Make: %NoAddr.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitReturn: type = class_type @ExplicitReturn [concrete]
// CHECK:STDOUT:   %ExplicitReturn.Make.type: type = fn_type @ExplicitReturn.Make [concrete]
// CHECK:STDOUT:   %ExplicitReturn.Make: %ExplicitReturn.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr: type = class_type @WithAddr [concrete]
// CHECK:STDOUT:   %WithAddr.Make.type: type = fn_type @WithAddr.Make [concrete]
// CHECK:STDOUT:   %WithAddr.Make: %WithAddr.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.type: type = fn_type @WithAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op: %WithAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b4e: type = ptr_type %WithAddr [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.type: type = fn_type @ExplicitReturn.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op: %ExplicitReturn.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b0a: type = ptr_type %ExplicitReturn [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.type: type = fn_type @NoAddr.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op: %NoAddr.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4b8: type = ptr_type %NoAddr [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NoAddr: type = import_ref Main//types, NoAddr, loaded [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Main.ExplicitReturn: type = import_ref Main//types, ExplicitReturn, loaded [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Main.WithAddr: type = import_ref Main//types, WithAddr, loaded [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Main.import_ref.784: %NoAddr.Make.type = import_ref Main//types, loc5_22, loaded [concrete = constants.%NoAddr.Make]
// CHECK:STDOUT:   %Main.import_ref.8e0: %ExplicitReturn.Make.type = import_ref Main//types, loc10_30, loaded [concrete = constants.%ExplicitReturn.Make]
// CHECK:STDOUT:   %Main.import_ref.974: %WithAddr.Make.type = import_ref Main//types, loc15_24, loaded [concrete = constants.%WithAddr.Make]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NoAddr.ref: type = name_ref NoAddr, imports.%Main.NoAddr [concrete = constants.%NoAddr]
// CHECK:STDOUT:   %Make.ref.loc9: %NoAddr.Make.type = name_ref Make, imports.%Main.import_ref.784 [concrete = constants.%NoAddr.Make]
// CHECK:STDOUT:   %.loc9_15.1: ref %NoAddr = temporary_storage
// CHECK:STDOUT:   %NoAddr.Make.call: init %NoAddr = call %Make.ref.loc9() to %.loc9_15.1
// CHECK:STDOUT:   %.loc9_15.2: ref %NoAddr = temporary %.loc9_15.1, %NoAddr.Make.call
// CHECK:STDOUT:   %ExplicitReturn.ref: type = name_ref ExplicitReturn, imports.%Main.ExplicitReturn [concrete = constants.%ExplicitReturn]
// CHECK:STDOUT:   %Make.ref.loc10: %ExplicitReturn.Make.type = name_ref Make, imports.%Main.import_ref.8e0 [concrete = constants.%ExplicitReturn.Make]
// CHECK:STDOUT:   %.loc10_23.1: ref %ExplicitReturn = temporary_storage
// CHECK:STDOUT:   %ExplicitReturn.Make.call: init %ExplicitReturn = call %Make.ref.loc10() to %.loc10_23.1
// CHECK:STDOUT:   %.loc10_23.2: ref %ExplicitReturn = temporary %.loc10_23.1, %ExplicitReturn.Make.call
// CHECK:STDOUT:   %WithAddr.ref: type = name_ref WithAddr, imports.%Main.WithAddr [concrete = constants.%WithAddr]
// CHECK:STDOUT:   %Make.ref.loc11: %WithAddr.Make.type = name_ref Make, imports.%Main.import_ref.974 [concrete = constants.%WithAddr.Make]
// CHECK:STDOUT:   %.loc11_17.1: ref %WithAddr = temporary_storage
// CHECK:STDOUT:   %WithAddr.Make.call: init %WithAddr = call %Make.ref.loc11() to %.loc11_17.1
// CHECK:STDOUT:   %.loc11_17.2: ref %WithAddr = temporary %.loc11_17.1, %WithAddr.Make.call
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc11_17.2, constants.%WithAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc11: %ptr.b4e = addr_of %.loc11_17.2
// CHECK:STDOUT:   %WithAddr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %WithAddr.as.Destroy.impl.Op.bound(%addr.loc11)
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_23.2, constants.%ExplicitReturn.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc10: %ptr.b0a = addr_of %.loc10_23.2
// CHECK:STDOUT:   %ExplicitReturn.as.Destroy.impl.Op.call: init %empty_tuple.type = call %ExplicitReturn.as.Destroy.impl.Op.bound(%addr.loc10)
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc9_15.2, constants.%NoAddr.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc9: %ptr.4b8 = addr_of %.loc9_15.2
// CHECK:STDOUT:   %NoAddr.as.Destroy.impl.Op.call: init %empty_tuple.type = call %NoAddr.as.Destroy.impl.Op.bound(%addr.loc9)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.113: type = class_type @D, @D(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.c95: type = pattern_type %D.113 [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type.602: type = fn_type @D.as.Destroy.impl.Op, @D.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.218: %D.as.Destroy.impl.Op.type.602 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.22d: type = ptr_type %D.113 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c95 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c95 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %D.113 = var %a.var_patt
// CHECK:STDOUT:   %.loc9: type = splice_block %D [concrete = constants.%D.113] {
// CHECK:STDOUT:     %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %D: type = class_type @D, @D(constants.%C) [concrete = constants.%D.113]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %D.113 = bind_name a, %a.var
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%D.as.Destroy.impl.Op.218
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.var, %D.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.22d = addr_of %a.var
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_use_inside_generic.carbon
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
// CHECK:STDOUT:   %Destroy.impl_witness.a08: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T.8b3d5d.1) [template]
// CHECK:STDOUT:   %ptr.7d2: type = ptr_type %C.f2e [template]
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
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.99a: type = pattern_type %C.7a7 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.b34: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Destroy.facet.5e7: %Destroy.type = facet_value %C.7a7, (%Destroy.impl_witness.b34) [concrete]
// CHECK:STDOUT:   %.ec8: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.5e7 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.f57: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.a9a: %C.as.Destroy.impl.Op.type.f57 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.308: type = ptr_type %C.7a7 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.d47: <specific function> = specific_function %C.as.Destroy.impl.Op.a9a, @C.as.Destroy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type.903: <witness> = complete_type_witness %ptr.308 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc6_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc6_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_15.2: type) {
// CHECK:STDOUT:   %T.loc6_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc6_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_13.2: type = class_type @C, @C(%T.loc6_15.1) [template = %C.loc7_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %require_complete.loc7_13: <witness> = require_complete_type %C.loc7_13.2 [template = %require_complete.loc7_13 (constants.%require_complete.389)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc7_13.2 [template = %pattern_type (constants.%pattern_type.e5e)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T.loc6_15.1) [template = %Destroy.impl_witness (constants.%Destroy.impl_witness.a08)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.loc7_13.2, (%Destroy.impl_witness) [template = %Destroy.facet (constants.%Destroy.facet.d7f)]
// CHECK:STDOUT:   %.loc7_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [template = %.loc7_3 (constants.%.a2f)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T.loc6_15.1) [template = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type.7a1)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @F.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type.7a1) = struct_value () [template = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.d65)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %C.as.Destroy.impl.Op, @C.as.Destroy.impl.Op(%T.loc6_15.1) [template = %C.as.Destroy.impl.Op.specific_fn (constants.%C.as.Destroy.impl.Op.specific_fn.67e)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C.loc7_13.2 [template = %ptr (constants.%ptr.7d2)]
// CHECK:STDOUT:   %require_complete.loc7_3: <witness> = require_complete_type %ptr [template = %require_complete.loc7_3 (constants.%require_complete.448)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.e5e) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.e5e) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%C.loc7_13.2 (%C.f2e) = var %v.var_patt
// CHECK:STDOUT:     %.loc7_13: type = splice_block %C.loc7_13.1 [template = %C.loc7_13.2 (constants.%C.f2e)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_15.2 [template = %T.loc6_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:       %C.loc7_13.1: type = class_type @C, @C(constants.%T.8b3d5d.1) [template = %C.loc7_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%C.loc7_13.2 (%C.f2e) = bind_name v, %v.var
// CHECK:STDOUT:     %impl.elem0: @F.%.loc7_3 (%.a2f) = impl_witness_access constants.%Destroy.impl_witness.a08, element0 [template = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.d65)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Destroy.impl.Op(constants.%T.8b3d5d.1) [template = %C.as.Destroy.impl.Op.specific_fn (constants.%C.as.Destroy.impl.Op.specific_fn.67e)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_fn
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.7d2) = addr_of %v.var
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3d5d.1) {
// CHECK:STDOUT:   %T.loc6_15.1 => constants.%T.8b3d5d.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc6_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_13.2 => constants.%C.7a7
// CHECK:STDOUT:   %require_complete.loc7_13 => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.99a
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.b34
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.5e7
// CHECK:STDOUT:   %.loc7_3 => constants.%.ec8
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.f57
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.a9a
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn => constants.%C.as.Destroy.impl.Op.specific_fn.d47
// CHECK:STDOUT:   %ptr => constants.%ptr.308
// CHECK:STDOUT:   %require_complete.loc7_3 => constants.%complete_type.903
// CHECK:STDOUT: }
// CHECK:STDOUT:

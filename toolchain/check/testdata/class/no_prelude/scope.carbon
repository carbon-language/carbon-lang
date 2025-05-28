// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/scope.carbon

// --- scope.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F() -> () {
    return ();
  }

  fn G() -> () {
    return F();
  }
}

//@dump-sem-ir-begin
fn F() -> () {
  return ();
}
//@dump-sem-ir-end

fn Run() {
  var a: () = F();
  var b: () = Class.F();
}

// --- reenter_scope.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F() -> ();
  fn G() -> ();
}

//@dump-sem-ir-begin
fn Class.F() -> () {
  Self.G();
  return G();
}
//@dump-sem-ir-end

// --- fail_out_of_scope.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F() -> () {
    return ();
  }
}

fn G() -> () {
  // CHECK:STDERR: fail_out_of_scope.carbon:[[@LINE+4]]:10: error: name `F` not found [NameNotFound]
  // CHECK:STDERR:   return F();
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return F();
}

// CHECK:STDOUT: --- scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc14_12.2: type = converted %.loc14_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_11: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_12: %empty_tuple.type = converted %.loc15_11, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc15_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- reenter_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_18.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_18.2: type = converted %.loc9_18.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param.loc9: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return.loc9: ref %empty_tuple.type = return_slot %return.param.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:   %G.ref.loc10: %G.type = name_ref G, @Class.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call.loc10: init %empty_tuple.type = call %G.ref.loc10()
// CHECK:STDOUT:   %G.ref.loc11: %G.type = name_ref G, @Class.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call.loc11: init %empty_tuple.type = call %G.ref.loc11()
// CHECK:STDOUT:   %.loc11_12.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc11_12.2: ref %empty_tuple.type = temporary %.loc11_12.1, %G.call.loc11
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_13: %empty_tuple.type = converted %G.call.loc11, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc11_13
// CHECK:STDOUT: }
// CHECK:STDOUT:

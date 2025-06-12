// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/var_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/var_pattern.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

fn F() {
  let var x: () = ();
}

// --- var_in_tuple.carbon

library "[[@TEST_NAME]]";

fn F() {
  let (x: (), var y: ()) = ((), ());
}

// --- tuple_in_var.carbon

library "[[@TEST_NAME]]";

fn F() {
  var (x: (), y: ()) = ((), ());
}

// --- tuple_in_let_var.carbon

library "[[@TEST_NAME]]";

fn F() {
  let var (x: (), y: ()) = ((), ());
}

// --- function.carbon

library "[[@TEST_NAME]]";

fn F(x: (), var y: ());

fn G() {
  var v: () = ();
  F(v, ());
}

// --- public_function.carbon

package P;

fn F(x: (), var y: ());

// --- public_function.impl.carbon

impl package P;

fn F(x: (), var y: ()) {}

// --- use_public_function.carbon

library "[[@TEST_NAME]]";

import P;

fn G() {
  P.F((), ());
}

// --- fail_nested.carbon

library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:27: error: `var` nested within another `var` [NestedVar]
  // CHECK:STDERR:   let (x: (), var (y: (), var z: ())) = ((), ((), ()));
  // CHECK:STDERR:                           ^~~
  // CHECK:STDERR:
  let (x: (), var (y: (), var z: ())) = ((), ((), ()));
}

// --- fail_compile_time.carbon

library "[[@TEST_NAME]]";

fn f() {
  // CHECK:STDERR: fail_compile_time.carbon:[[@LINE+8]]:7: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var T:! type;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_compile_time.carbon:[[@LINE+4]]:15: error: `var` pattern cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
  // CHECK:STDERR:   var T:! type;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  var T:! type;
  let (x: (), T:! type) = ((), ());
}

// --- fail_implicit.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:10: error: implicit parameters of functions must be constant or `self` [ImplictParamMustBeConstant]
// CHECK:STDERR: fn F[var u: ()]();
// CHECK:STDERR:          ^~~~~
// CHECK:STDERR:
fn F[var u: ()]();

// CHECK:STDERR: fail_implicit.carbon:[[@LINE+8]]:10: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: fn G[var T:! type]();
// CHECK:STDERR:          ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:18: error: `var` pattern cannot declare a compile-time binding [CompileTimeBindingInVarDecl]
// CHECK:STDERR: fn G[var T:! type]();
// CHECK:STDERR:                  ^
// CHECK:STDERR:
fn G[var T:! type]();

// --- var_self.carbon

library "[[@TEST_NAME]]";

class C {
  fn F[var self: Self]();
}

// --- nested_match.carbon

library "[[@TEST_NAME]]";

fn F() -> ();

fn G() {
  // The SemIR should contain only two temporary_storage insts, not three.
  let (x: (), var y: (), z: ()) = (F(), F(), F());
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt
// CHECK:STDOUT:   %.loc5_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_20.2: init %empty_tuple.type = tuple_init () to %x.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_7: init %empty_tuple.type = converted %.loc5_20.1, %.loc5_20.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var, %.loc5_7
// CHECK:STDOUT:   %.loc5_15.1: type = splice_block %.loc5_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_15.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_15.3: type = converted %.loc5_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_in_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.cb1 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:     %.loc5_24: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.var_patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var %y.var_patt
// CHECK:STDOUT:   %.loc5_30.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_34.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_35: %tuple.type = tuple_literal (%.loc5_30.1, %.loc5_34.1)
// CHECK:STDOUT:   %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_30.2: %empty_tuple.type = converted %.loc5_30.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_name x, %.loc5_30.2
// CHECK:STDOUT:   %.loc5_34.2: init %empty_tuple.type = tuple_init () to %y.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_15: init %empty_tuple.type = converted %.loc5_34.1, %.loc5_34.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %y.var, %.loc5_15
// CHECK:STDOUT:   %.loc5_23.1: type = splice_block %.loc5_23.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_23.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_23.3: type = converted %.loc5_23.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = bind_name y, %y.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_in_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = binding_pattern y [concrete]
// CHECK:STDOUT:     %.loc5_20: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:     %.var_patt: %pattern_type.5b8 = var_pattern %.loc5_20 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type = var %.var_patt
// CHECK:STDOUT:   %.loc5_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_30.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_31.1: %tuple.type = tuple_literal (%.loc5_26.1, %.loc5_30.1)
// CHECK:STDOUT:   %tuple.elem0.loc5_31: ref %empty_tuple.type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc5_26.2: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc5_31 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_31.2: init %empty_tuple.type = converted %.loc5_26.1, %.loc5_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc5_31: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc5_30.2: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc5_31 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_31.3: init %empty_tuple.type = converted %.loc5_30.1, %.loc5_30.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_31.4: init %tuple.type = tuple_init (%.loc5_31.2, %.loc5_31.3) to %.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc5_3: init %tuple.type = converted %.loc5_31.1, %.loc5_31.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc5_3
// CHECK:STDOUT:   %tuple.elem0.loc5_3: ref %empty_tuple.type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc5_3: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc5_12.1: type = splice_block %.loc5_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_12.3: type = converted %.loc5_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %tuple.elem0.loc5_3
// CHECK:STDOUT:   %.loc5_19.1: type = splice_block %.loc5_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_19.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_19.3: type = converted %.loc5_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = bind_name y, %tuple.elem1.loc5_3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_in_let_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = binding_pattern y [concrete]
// CHECK:STDOUT:     %.loc5_24: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:     %.var_patt: %pattern_type.5b8 = var_pattern %.loc5_24 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type = var %.var_patt
// CHECK:STDOUT:   %.loc5_30.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_34.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_35.1: %tuple.type = tuple_literal (%.loc5_30.1, %.loc5_34.1)
// CHECK:STDOUT:   %tuple.elem0.loc5_35: ref %empty_tuple.type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc5_30.2: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc5_35 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_35.2: init %empty_tuple.type = converted %.loc5_30.1, %.loc5_30.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc5_35: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc5_34.2: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc5_35 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_35.3: init %empty_tuple.type = converted %.loc5_34.1, %.loc5_34.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_35.4: init %tuple.type = tuple_init (%.loc5_35.2, %.loc5_35.3) to %.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc5_7: init %tuple.type = converted %.loc5_35.1, %.loc5_35.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc5_7
// CHECK:STDOUT:   %tuple.elem0.loc5_7: ref %empty_tuple.type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc5_7: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc5_16.1: type = splice_block %.loc5_16.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_16.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_16.3: type = converted %.loc5_16.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %tuple.elem0.loc5_7
// CHECK:STDOUT:   %.loc5_23.1: type = splice_block %.loc5_23.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_23.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_23.3: type = converted %.loc5_23.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = bind_name y, %tuple.elem1.loc5_7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type = ref_param_pattern %y.patt, call_param1 [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:     %y.param: ref %empty_tuple.type = ref_param call_param1
// CHECK:STDOUT:     %.loc4_21.1: type = splice_block %.loc4_21.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_21.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc4_21.3: type = converted %.loc4_21.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: ref %empty_tuple.type = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %empty_tuple.type, %y.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %empty_tuple.type = var %v.var_patt
// CHECK:STDOUT:   %.loc7_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_16.2: init %empty_tuple.type = tuple_init () to %v.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_3: init %empty_tuple.type = converted %.loc7_16.1, %.loc7_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %v.var, %.loc7_3
// CHECK:STDOUT:   %.loc7_11.1: type = splice_block %.loc7_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_11.3: type = converted %.loc7_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %empty_tuple.type = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %v.ref: ref %empty_tuple.type = name_ref v, %v
// CHECK:STDOUT:   %.loc8_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_5: %empty_tuple.type = converted %v.ref, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_13.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc8_9.2: init %empty_tuple.type = tuple_init () to %.loc4_13.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_13.2: init %empty_tuple.type = converted %.loc8_9.1, %.loc8_9.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc4_13.1, %.loc4_13.2
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc8_5, %.loc4_13.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- public_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type = ref_param_pattern %y.patt, call_param1 [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:     %y.param: ref %empty_tuple.type = ref_param call_param1
// CHECK:STDOUT:     %.loc4_21.1: type = splice_block %.loc4_21.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_21.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc4_21.3: type = converted %.loc4_21.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: ref %empty_tuple.type = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %empty_tuple.type, %y.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: --- public_function.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P.import = import P
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type = ref_param_pattern %y.patt, call_param1 [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:     %y.param: ref %empty_tuple.type = ref_param call_param1
// CHECK:STDOUT:     %.loc4_21.1: type = splice_block %.loc4_21.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_21.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc4_21.3: type = converted %.loc4_21.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: ref %empty_tuple.type = bind_name y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %empty_tuple.type, %y.param: %empty_tuple.type) [from "public_function.carbon"] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_public_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P: <namespace> = namespace file.%P.import, [concrete] {
// CHECK:STDOUT:     .F = %P.F
// CHECK:STDOUT:     import P//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P.F: %F.type = import_ref P//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .P = imports.%P
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %P.import = import P
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %P.ref: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%P.F [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7_8.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_8.2: %empty_tuple.type = converted %.loc7_8.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc7_12.2: init %empty_tuple.type = tuple_init () to %.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.2: init %empty_tuple.type = converted %.loc7_12.1, %.loc7_12.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.1, %.2
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc7_8.2, %.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "public_function.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type.bcd: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type.bcd [concrete]
// CHECK:STDOUT:   %tuple.type.a21: type = tuple_type (%empty_tuple.type, %tuple.type.bcd) [concrete]
// CHECK:STDOUT:   %pattern_type.c6c: type = pattern_type %tuple.type.a21 [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.bcd = tuple_value (%empty_tuple, %empty_tuple) [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = binding_pattern y [concrete]
// CHECK:STDOUT:     %z.patt: %pattern_type.cb1 = binding_pattern z [concrete]
// CHECK:STDOUT:     %z.var_patt: %pattern_type.cb1 = var_pattern %z.patt [concrete]
// CHECK:STDOUT:     %.loc9_36: %pattern_type.5b8 = tuple_pattern (%y.patt, %z.var_patt) [concrete]
// CHECK:STDOUT:     %.var_patt: %pattern_type.5b8 = var_pattern %.loc9_36 [concrete]
// CHECK:STDOUT:     %.loc9_37: %pattern_type.c6c = tuple_pattern (%x.patt, %.var_patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %z.var: ref %empty_tuple.type = var %z.var_patt
// CHECK:STDOUT:   %.var: ref %tuple.type.bcd = var %.var_patt
// CHECK:STDOUT:   %.loc9_43.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_48.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_52.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_53.1: %tuple.type.bcd = tuple_literal (%.loc9_48.1, %.loc9_52.1)
// CHECK:STDOUT:   %.loc9_54: %tuple.type.a21 = tuple_literal (%.loc9_43.1, %.loc9_53.1)
// CHECK:STDOUT:   %.loc9_12.1: type = splice_block %.loc9_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc9_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_12.3: type = converted %.loc9_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_43.2: %empty_tuple.type = converted %.loc9_43.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_name x, %.loc9_43.2
// CHECK:STDOUT:   %tuple.elem0.loc9_53: ref %empty_tuple.type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc9_48.2: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc9_53 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_53.2: init %empty_tuple.type = converted %.loc9_48.1, %.loc9_48.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc9_53: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc9_52.2: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc9_53 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_53.3: init %empty_tuple.type = converted %.loc9_52.1, %.loc9_52.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_53.4: init %tuple.type.bcd = tuple_init (%.loc9_53.2, %.loc9_53.3) to %.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc9_15.1: init %tuple.type.bcd = converted %.loc9_53.1, %.loc9_53.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc9_15.1
// CHECK:STDOUT:   %tuple.elem0.loc9_15: ref %empty_tuple.type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc9_15: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc9_24.1: type = splice_block %.loc9_24.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc9_24.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_24.3: type = converted %.loc9_24.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = bind_name y, %tuple.elem0.loc9_15
// CHECK:STDOUT:   %.loc9_15.2: init %empty_tuple.type = tuple_init () to %z.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_27: init %empty_tuple.type = converted %tuple.elem1.loc9_15, %.loc9_15.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %z.var, %.loc9_27
// CHECK:STDOUT:   %.loc9_35.1: type = splice_block %.loc9_35.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc9_35.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_35.3: type = converted %.loc9_35.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %z: ref %empty_tuple.type = bind_name z, %z.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_compile_time.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @f();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %self.var_patt: %pattern_type = var_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: ref %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%self.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_match.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.8c1: type = pattern_type %tuple.type [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_12.2: type = converted %.loc4_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.cb1 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:     %z.patt: %pattern_type.cb1 = binding_pattern z [concrete]
// CHECK:STDOUT:     %.loc8_31: %pattern_type.8c1 = tuple_pattern (%x.patt, %y.var_patt, %z.patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var %y.var_patt
// CHECK:STDOUT:   %F.ref.loc8_36: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call.loc8_38: init %empty_tuple.type = call %F.ref.loc8_36()
// CHECK:STDOUT:   %F.ref.loc8_41: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call.loc8_43: init %empty_tuple.type = call %F.ref.loc8_41()
// CHECK:STDOUT:   %F.ref.loc8_46: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call.loc8_48: init %empty_tuple.type = call %F.ref.loc8_46()
// CHECK:STDOUT:   %.loc8_49: %tuple.type = tuple_literal (%F.call.loc8_38, %F.call.loc8_43, %F.call.loc8_48)
// CHECK:STDOUT:   %.loc8_12.1: type = splice_block %.loc8_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_12.3: type = converted %.loc8_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_38.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc8_38.2: ref %empty_tuple.type = temporary %.loc8_38.1, %F.call.loc8_38
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %.loc8_38.2
// CHECK:STDOUT:   assign %y.var, %F.call.loc8_43
// CHECK:STDOUT:   %.loc8_23.1: type = splice_block %.loc8_23.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_23.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_23.3: type = converted %.loc8_23.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = bind_name y, %y.var
// CHECK:STDOUT:   %.loc8_30.1: type = splice_block %.loc8_30.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_30.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_30.3: type = converted %.loc8_30.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_48.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc8_48.2: ref %empty_tuple.type = temporary %.loc8_48.1, %F.call.loc8_48
// CHECK:STDOUT:   %z: ref %empty_tuple.type = bind_name z, %.loc8_48.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

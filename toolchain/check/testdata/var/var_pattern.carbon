// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/var_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/var_pattern.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

fn F() {
  let var unused x: () = ();
}

// --- var_in_tuple.carbon

library "[[@TEST_NAME]]";

fn F() {
  let (unused x: (), var unused y: ()) = ((), ());
}

// --- tuple_in_var.carbon

library "[[@TEST_NAME]]";

fn F() {
  var (unused x: (), unused y: ()) = ((), ());
}

// --- tuple_in_let_var.carbon

library "[[@TEST_NAME]]";

fn F() {
  let var (unused x: (), unused y: ()) = ((), ());
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

fn F(unused x: (), var unused y: ()) {}

// --- use_public_function.carbon

library "[[@TEST_NAME]]";

import P;

fn G() {
  P.F((), ());
}

// --- fail_nested.carbon

library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_nested.carbon:[[@LINE+8]]:41: error: `var` nested within another `var` [NestedVar]
  // CHECK:STDERR:   let (unused x: (), var (unused y: (), var unused z: ())) = ((), ((), ()));
  // CHECK:STDERR:                                         ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:41: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   let (unused x: (), var (unused y: (), var unused z: ())) = ((), ((), ()));
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let (unused x: (), var (unused y: (), var unused z: ())) = ((), ((), ()));
}

// --- fail_compile_time.carbon

library "[[@TEST_NAME]]";

fn f() {
  // CHECK:STDERR: fail_compile_time.carbon:[[@LINE+8]]:7: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
  // CHECK:STDERR:   var T:! type;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_compile_time.carbon:[[@LINE+4]]:15: error: found `:!` pattern inside `var` pattern [NonRegularBindingInVarDecl]
  // CHECK:STDERR:   var T:! type;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  var T:! type;
  let (x: (), T:! type) = ((), ());
}

// --- fail_implicit.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:10: error: implicit parameters of functions must be constant [ImplictParamMustBeConstant]
// CHECK:STDERR: fn F[var u: ()]();
// CHECK:STDERR:          ^~~~~
// CHECK:STDERR:
fn F[var u: ()]();

// CHECK:STDERR: fail_implicit.carbon:[[@LINE+8]]:10: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: fn G[var T:! type]();
// CHECK:STDERR:          ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_implicit.carbon:[[@LINE+4]]:18: error: found `:!` pattern inside `var` pattern [NonRegularBindingInVarDecl]
// CHECK:STDERR: fn G[var T:! type]();
// CHECK:STDERR:                  ^
// CHECK:STDERR:
fn G[var T:! type]();

// --- var_self.carbon

library "[[@TEST_NAME]]";

class C {
  fn F(var self);
}

// --- nested_match.carbon

library "[[@TEST_NAME]]";

fn F() -> ();

fn G() {
  // The SemIR should contain only two temporary_storage insts, not three.
  let (unused x: (), var unused y: (), unused z: ()) = (F(), F(), F());
}

// --- var_param_from_init.carbon

library "[[@TEST_NAME]]";

class X {}

//@dump-sem-ir-begin
fn TakeVar(var a: X);
//@dump-sem-ir-end

fn MakeInit() -> X;

fn PassInitToVar() {
  //@dump-sem-ir-begin
  TakeVar(MakeInit());
  //@dump-sem-ir-end
}

// --- var_param_from_int_literal.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn TakeVar(var n: i32);
//@dump-sem-ir-end

fn Call() {
  //@dump-sem-ir-begin
  TakeVar(123);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.cb1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt
// CHECK:STDOUT:   %.loc5_27.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_27.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_7: init %empty_tuple.type = converted %.loc5_27.1, %.loc5_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var, %.loc5_7
// CHECK:STDOUT:   %.loc5_22.1: type = splice_block %.loc5_22.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_22.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_22.3: type = converted %.loc5_22.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = wrapper_binding x, %x.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %x.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%x.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_in_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.var_patt: %pattern_type.cb1 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.5ae: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.var_patt) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.cb1 = var_pattern %y.patt [concrete = constants.%y.var_patt]
// CHECK:STDOUT:     %.loc5_38: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.var_patt) [concrete = constants.%.5ae]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var %y.var_patt
// CHECK:STDOUT:   %.loc5_44.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_48.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49: %tuple.type = tuple_literal (%.loc5_44.1, %.loc5_48.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_44.2: %empty_tuple.type = converted %.loc5_44.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_19.1: type = splice_block %.loc5_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_19.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_19.3: type = converted %.loc5_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %empty_tuple.type = wrapper_binding x, %.loc5_44.2
// CHECK:STDOUT:   %.loc5_48.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_22: init %empty_tuple.type = converted %.loc5_48.1, %.loc5_48.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %y.var, %.loc5_22
// CHECK:STDOUT:   %.loc5_37.1: type = splice_block %.loc5_37.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_37.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_37.3: type = converted %.loc5_37.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = wrapper_binding y, %y.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %y.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%y.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_in_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.8b4: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:   %.var_patt: %pattern_type.5b8 = var_pattern %.8b4 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc5_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %.loc5_34: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.patt) [concrete = constants.%.8b4]
// CHECK:STDOUT:     %.var_patt: %pattern_type.5b8 = var_pattern %.loc5_34 [concrete = constants.%.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type = var %.var_patt
// CHECK:STDOUT:   %.loc5_40.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_44.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_45.1: %tuple.type = tuple_literal (%.loc5_40.1, %.loc5_44.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc5_45: ref %empty_tuple.type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc5_40.2: init %empty_tuple.type to %tuple.elem0.loc5_45 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_45.2: init %empty_tuple.type = converted %.loc5_40.1, %.loc5_40.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_45.3: init %empty_tuple.type to %tuple.elem0.loc5_45 = in_place_init %.loc5_45.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc5_45: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc5_44.2: init %empty_tuple.type to %tuple.elem1.loc5_45 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_45.4: init %empty_tuple.type = converted %.loc5_44.1, %.loc5_44.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_45.5: init %empty_tuple.type to %tuple.elem1.loc5_45 = in_place_init %.loc5_45.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_45.6: init %tuple.type to %.var = tuple_init (%.loc5_45.3, %.loc5_45.5) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc5_3: init %tuple.type = converted %.loc5_45.1, %.loc5_45.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc5_3
// CHECK:STDOUT:   %tuple.elem0.loc5_3: ref %empty_tuple.type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc5_3: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc5_19.1: type = splice_block %.loc5_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_19.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_19.3: type = converted %.loc5_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = wrapper_binding x, %tuple.elem0.loc5_3
// CHECK:STDOUT:   %.loc5_33.1: type = splice_block %.loc5_33.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_33.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_33.3: type = converted %.loc5_33.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = wrapper_binding y, %tuple.elem1.loc5_3
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_3.1(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_3.2(%self.param: ref %tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_in_let_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.8b4: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:   %.var_patt: %pattern_type.5b8 = var_pattern %.8b4 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc5_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %.loc5_38: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.patt) [concrete = constants.%.8b4]
// CHECK:STDOUT:     %.var_patt: %pattern_type.5b8 = var_pattern %.loc5_38 [concrete = constants.%.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type = var %.var_patt
// CHECK:STDOUT:   %.loc5_44.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_48.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49.1: %tuple.type = tuple_literal (%.loc5_44.1, %.loc5_48.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc5_49: ref %empty_tuple.type = tuple_access %.var, element0
// CHECK:STDOUT:   %.loc5_44.2: init %empty_tuple.type to %tuple.elem0.loc5_49 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49.2: init %empty_tuple.type = converted %.loc5_44.1, %.loc5_44.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49.3: init %empty_tuple.type to %tuple.elem0.loc5_49 = in_place_init %.loc5_49.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc5_49: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc5_48.2: init %empty_tuple.type to %tuple.elem1.loc5_49 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49.4: init %empty_tuple.type = converted %.loc5_48.1, %.loc5_48.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49.5: init %empty_tuple.type to %tuple.elem1.loc5_49 = in_place_init %.loc5_49.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_49.6: init %tuple.type to %.var = tuple_init (%.loc5_49.3, %.loc5_49.5) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc5_7: init %tuple.type = converted %.loc5_49.1, %.loc5_49.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %.var, %.loc5_7
// CHECK:STDOUT:   %tuple.elem0.loc5_7: ref %empty_tuple.type = tuple_access %.var, element0
// CHECK:STDOUT:   %tuple.elem1.loc5_7: ref %empty_tuple.type = tuple_access %.var, element1
// CHECK:STDOUT:   %.loc5_23.1: type = splice_block %.loc5_23.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_23.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_23.3: type = converted %.loc5_23.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = wrapper_binding x, %tuple.elem0.loc5_7
// CHECK:STDOUT:   %.loc5_37.1: type = splice_block %.loc5_37.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_37.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_37.3: type = converted %.loc5_37.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = wrapper_binding y, %tuple.elem1.loc5_7
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_7.1(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_7.2(%self.param: ref %tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.param_patt: %pattern_type.cb1 = var_param_pattern %y.patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.cb1 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.cb1 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.cb1 = var_param_pattern %y.patt [concrete = constants.%y.param_patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:     %y.param: ref %empty_tuple.type = ref_param call_param1
// CHECK:STDOUT:     %.loc4_21.1: type = splice_block %.loc4_21.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_21.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc4_21.3: type = converted %.loc4_21.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: ref %empty_tuple.type = wrapper_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %empty_tuple.type, %y.param: ref %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.cb1 = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.cb1 = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %empty_tuple.type = var %v.var_patt
// CHECK:STDOUT:   %.loc7_16.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_16.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_3: init %empty_tuple.type = converted %.loc7_16.1, %.loc7_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %v.var, %.loc7_3
// CHECK:STDOUT:   %.loc7_11.1: type = splice_block %.loc7_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_11.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_11.3: type = converted %.loc7_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %empty_tuple.type = wrapper_binding v, %v.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %v.ref: ref %empty_tuple.type = name_ref v, %v
// CHECK:STDOUT:   %.loc8_9.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_5: %empty_tuple.type = converted %v.ref, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var @F.%y.param_patt
// CHECK:STDOUT:   %.loc8_9.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4: init %empty_tuple.type = converted %.loc8_9.1, %.loc8_9.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %y.var, %.loc4
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc8_5, %y.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc4: <bound method> = bound_method %y.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc4: init %empty_tuple.type = call %Destroy.Op.bound.loc4(%y.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc7: <bound method> = bound_method %v.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc7: init %empty_tuple.type = call %Destroy.Op.bound.loc7(%v.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- public_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.param_patt: %pattern_type = var_param_pattern %y.patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.patt: %pattern_type = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.param_patt: %pattern_type = var_param_pattern %y.patt [concrete = constants.%y.param_patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:     %y.param: ref %empty_tuple.type = ref_param call_param1
// CHECK:STDOUT:     %.loc4_21.1: type = splice_block %.loc4_21.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_21.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc4_21.3: type = converted %.loc4_21.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: ref %empty_tuple.type = wrapper_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %empty_tuple.type, %y.param: ref %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: --- public_function.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.7ba44d.1: %pattern_type = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %y.patt.d25273.1: %pattern_type = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.param_patt.ce1148.1: %pattern_type = var_param_pattern %y.patt.d25273.1 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P.import = import P
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.7ba44d.1]
// CHECK:STDOUT:     %y.patt: %pattern_type = ref_binding_pattern y [concrete = constants.%y.patt.d25273.1]
// CHECK:STDOUT:     %y.param_patt: %pattern_type = var_param_pattern %y.patt [concrete = constants.%y.param_patt.ce1148.1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc4_17.3: type = converted %.loc4_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:     %y.param: ref %empty_tuple.type = ref_param call_param1
// CHECK:STDOUT:     %.loc4_35.1: type = splice_block %.loc4_35.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc4_35.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc4_35.3: type = converted %.loc4_35.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: ref %empty_tuple.type = wrapper_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %empty_tuple.type, %y.param: ref %empty_tuple.type) [from "public_function.carbon"] {
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
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.param_patt: %pattern_type.cb1 = var_param_pattern %y.patt [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9: <witness> = custom_witness (%Destroy.Op), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.5ad: %Destroy.type = facet_value %empty_tuple.type, (%custom_witness.df9) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.869: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.5ad) [concrete]
// CHECK:STDOUT:   %.6ff: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.869, %Destroy.facet.5ad [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P: <namespace> = namespace file.%P.import, [concrete] {
// CHECK:STDOUT:     .F = %P.F
// CHECK:STDOUT:     import P//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P.F: %F.type = import_ref P//default, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.c64: %Destroy.assoc_type = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.import_ref.918: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.d3e) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)]
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
// CHECK:STDOUT:   %.loc7_8.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_8.2: %empty_tuple.type = converted %.loc7_8.1, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var constants.%y.param_patt
// CHECK:STDOUT:   %.loc7_12.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.1: init %empty_tuple.type = converted %.loc7_12.1, %.loc7_12.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %y.var, %.1
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc7_8.2, %y.var)
// CHECK:STDOUT:   %Op.ref: %Destroy.assoc_type = name_ref Op, imports.%Core.import_ref.c64 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0: %.6ff = impl_witness_access constants.%custom_witness.df9, element0 [concrete = constants.%Destroy.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %y.var, %impl.elem0
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %bound_method(%y.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F [from "public_function.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_compile_time.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: fn @f();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.303: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt.303 [concrete]
// CHECK:STDOUT:   %self.param_patt.b69: %pattern_type = var_param_pattern %self.patt [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     %self.param_patt.loc5_12: %pattern_type = value_param_pattern [concrete = constants.%self.param_patt.303]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt.loc5_12 [concrete = constants.%self.patt]
// CHECK:STDOUT:     %self.param_patt.loc5_8: %pattern_type = var_param_pattern %self.patt [concrete = constants.%self.param_patt.b69]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc5_8: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self.param.loc5_12: %C = value_param call_param1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = wrapper_binding self, %self.param.loc5_12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %C.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.F(%self.param.loc5_8: ref %C, %self.param.loc5_12: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_match.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %y.var_patt: %pattern_type.cb1 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   %z.patt: %pattern_type.cb1 = value_binding_pattern z [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.8c1: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.b14: %pattern_type.8c1 = tuple_pattern (%x.patt, %y.var_patt, %z.patt) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc4_12.2 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc4_12.2: type = converted %.loc4_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc4_12.3: Core.Form = init_form %.loc4_12.2 [concrete = constants.%.262]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.cb1 = var_pattern %y.patt [concrete = constants.%y.var_patt]
// CHECK:STDOUT:     %z.patt: %pattern_type.cb1 = value_binding_pattern z [concrete = constants.%z.patt]
// CHECK:STDOUT:     %.loc8_52: %pattern_type.8c1 = tuple_pattern (%x.patt, %y.var_patt, %z.patt) [concrete = constants.%.b14]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var %y.var_patt
// CHECK:STDOUT:   %F.ref.loc8_57: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call.loc8_59: init %empty_tuple.type = call %F.ref.loc8_57()
// CHECK:STDOUT:   %F.ref.loc8_62: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call.loc8_64: init %empty_tuple.type = call %F.ref.loc8_62()
// CHECK:STDOUT:   %F.ref.loc8_67: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call.loc8_69: init %empty_tuple.type = call %F.ref.loc8_67()
// CHECK:STDOUT:   %.loc8_70: %tuple.type = tuple_literal (%F.call.loc8_59, %F.call.loc8_64, %F.call.loc8_69)
// CHECK:STDOUT:   %.loc8_59.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc8_59.2: ref %empty_tuple.type = temporary %.loc8_59.1, %F.call.loc8_59
// CHECK:STDOUT:   %tuple.loc8_59: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_59.3: %empty_tuple.type = converted %F.call.loc8_59, %tuple.loc8_59 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_19.1: type = splice_block %.loc8_19.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_19.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_19.3: type = converted %.loc8_19.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %empty_tuple.type = wrapper_binding x, %.loc8_59.3
// CHECK:STDOUT:   assign %y.var, %F.call.loc8_64
// CHECK:STDOUT:   %.loc8_37.1: type = splice_block %.loc8_37.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_37.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_37.3: type = converted %.loc8_37.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = wrapper_binding y, %y.var
// CHECK:STDOUT:   %.loc8_69.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc8_69.2: ref %empty_tuple.type = temporary %.loc8_69.1, %F.call.loc8_69
// CHECK:STDOUT:   %tuple.loc8_69: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_69.3: %empty_tuple.type = converted %F.call.loc8_69, %tuple.loc8_69 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_51.1: type = splice_block %.loc8_51.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_51.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_51.3: type = converted %.loc8_51.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %z: %empty_tuple.type = wrapper_binding z, %.loc8_69.3
// CHECK:STDOUT:   %Destroy.Op.bound.loc8_69: <bound method> = bound_method %.loc8_69.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc8_69: init %empty_tuple.type = call %Destroy.Op.bound.loc8_69(%.loc8_69.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8_59: <bound method> = bound_method %.loc8_59.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc8_59: init %empty_tuple.type = call %Destroy.Op.bound.loc8_59(%.loc8_59.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc8_22: <bound method> = bound_method %y.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc8_22: init %empty_tuple.type = call %Destroy.Op.bound.loc8_22(%y.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_param_from_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6cb = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.6cb = var_param_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakeVar.type: type = fn_type @TakeVar [concrete]
// CHECK:STDOUT:   %TakeVar: %TakeVar.type = struct_value () [concrete]
// CHECK:STDOUT:   %MakeInit.type: type = fn_type @MakeInit [concrete]
// CHECK:STDOUT:   %MakeInit: %MakeInit.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc7_12.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TakeVar.decl: %TakeVar.type = fn_decl @TakeVar [concrete = constants.%TakeVar] {
// CHECK:STDOUT:     %a.patt: %pattern_type.6cb = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.6cb = var_param_pattern %a.patt [concrete = constants.%a.param_patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %a: ref %X = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TakeVar(%a.param: ref %X);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassInitToVar() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeVar.ref: %TakeVar.type = name_ref TakeVar, file.%TakeVar.decl [concrete = constants.%TakeVar]
// CHECK:STDOUT:   %MakeInit.ref: %MakeInit.type = name_ref MakeInit, file.%MakeInit.decl [concrete = constants.%MakeInit]
// CHECK:STDOUT:   %a.var: ref %X = var @TakeVar.%a.param_patt
// CHECK:STDOUT:   %MakeInit.call: init %X to %a.var = call %MakeInit.ref()
// CHECK:STDOUT:   assign %a.var, %MakeInit.call
// CHECK:STDOUT:   %TakeVar.call: init %empty_tuple.type = call %TakeVar.ref(%a.var)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_12.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_12.2(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_param_from_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.6b6 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.6b6 = var_param_pattern %n.patt [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %TakeVar.type: type = fn_type @TakeVar [concrete]
// CHECK:STDOUT:   %TakeVar: %TakeVar.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_123.fff: Core.IntLiteral = int_value 123 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_123.fff, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_123.2bf: %i32 = int_value 123 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc5_12.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TakeVar.decl: %TakeVar.type = fn_decl @TakeVar [concrete = constants.%TakeVar] {
// CHECK:STDOUT:     %n.patt: %pattern_type.6b6 = ref_binding_pattern n [concrete = constants.%n.patt]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.6b6 = var_param_pattern %n.patt [concrete = constants.%n.param_patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: ref %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TakeVar(%n.param: ref %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeVar.ref: %TakeVar.type = name_ref TakeVar, file.%TakeVar.decl [concrete = constants.%TakeVar]
// CHECK:STDOUT:   %int_123: Core.IntLiteral = int_value 123 [concrete = constants.%int_123.fff]
// CHECK:STDOUT:   %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc5_12.1: <bound method> = bound_method %int_123, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_12.2: <bound method> = bound_method %int_123, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_12.2(%int_123) [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %.loc5: init %i32 = converted %int_123, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_123.2bf]
// CHECK:STDOUT:   %n.var: ref %i32 = var @TakeVar.%n.param_patt
// CHECK:STDOUT:   assign %n.var, %.loc5
// CHECK:STDOUT:   %TakeVar.call: init %empty_tuple.type = call %TakeVar.ref(%n.var)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %n.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_12.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc5_12.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// The default behavior should be `--dump-sem-ir-ranges=if-present`, although
// tests set it to `only`. This explicitly tests the default of the toolchain,
// not of file_test.
// ARGS: compile --phase=check --dump-sem-ir %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/dump_sem_ir_ranges.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/dump_sem_ir_ranges.carbon

// --- function.carbon

library "[[@TEST_NAME]]";

fn A() {
  var a: ();
}

fn B() -> () {
  var b: ();
  //@dump-sem-ir-begin
  b = A();
  //@dump-sem-ir-end
  return b;
}

//@dump-sem-ir-begin
fn C() -> () {
  var c: ();
  c = B();
  return c;
}
//@dump-sem-ir-end

// --- class.carbon

library "[[@TEST_NAME]]";

class A {
  fn F();

  //@dump-sem-ir-begin
  fn G();
  //@dump-sem-ir-end
}

class B {
  fn H();
}

//@dump-sem-ir-begin
class C {
  fn I();
  //@dump-sem-ir-end

  fn J();
}

// --- call_params.carbon

library "[[@TEST_NAME]]";

fn F(a: (), b: (), c: ());

fn A();
fn B();
fn C();

fn G() {
  F(
    //@dump-sem-ir-begin
    A(),
    //@dump-sem-ir-end
    B(),
    //@dump-sem-ir-begin
    C()
    //@dump-sem-ir-end
  );
}

// --- file_without_ranges.carbon

library "[[@TEST_NAME]]";

fn F();

// CHECK:STDOUT: --- function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Op.type.a63: type = fn_type @Op.2, @Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.ea3: %Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_12.2: type = converted %.loc17_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %b.ref.loc11: ref %empty_tuple.type = name_ref b, %b
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign %b.ref.loc11, %A.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.cb1 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.cb1 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var %c.var_patt
// CHECK:STDOUT:   %.loc18_11.1: type = splice_block %.loc18_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc18_11.3: type = converted %.loc18_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %empty_tuple.type = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc19: ref %empty_tuple.type = name_ref c, %c
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %B.call: init %empty_tuple.type = call %B.ref()
// CHECK:STDOUT:   assign %c.ref.loc19, %B.call
// CHECK:STDOUT:   %c.ref.loc20: ref %empty_tuple.type = name_ref c, %c
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc20: %empty_tuple.type = converted %c.ref.loc20, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %c.var, constants.%Op.ea3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.var, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.843 = addr_of %c.var
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %.loc20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = fn_type @I [concrete]
// CHECK:STDOUT:   %I: %I.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %I.decl: %I.type = fn_decl @I [concrete = constants.%I] {} {}
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT:   .J = %J.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I();
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Op.type.a63: type = fn_type @Op.2, @Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.ea3: %Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.call: init %empty_tuple.type = call %C.ref()
// CHECK:STDOUT:   %.loc13_7.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc13_7.2: ref %empty_tuple.type = temporary %.loc13_7.1, %A.call
// CHECK:STDOUT:   %tuple.loc13: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_7.3: %empty_tuple.type = converted %A.call, %tuple.loc13 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc17_7.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc17_7.2: ref %empty_tuple.type = temporary %.loc17_7.1, %C.call
// CHECK:STDOUT:   %tuple.loc17: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_7.3: %empty_tuple.type = converted %C.call, %tuple.loc17 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.bound.loc17: <bound method> = bound_method %.loc17_7.1, constants.%Op.ea3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc17: <bound method> = bound_method %.loc17_7.1, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc17: %ptr.843 = addr_of %.loc17_7.1
// CHECK:STDOUT:   %no_op.loc17: init %empty_tuple.type = call %bound_method.loc17(%addr.loc17)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Op.bound.loc13: <bound method> = bound_method %.loc13_7.1, constants.%Op.ea3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %.loc13_7.1, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc13: %ptr.843 = addr_of %.loc13_7.1
// CHECK:STDOUT:   %no_op.loc13: init %empty_tuple.type = call %bound_method.loc13(%addr.loc13)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- file_without_ranges.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:

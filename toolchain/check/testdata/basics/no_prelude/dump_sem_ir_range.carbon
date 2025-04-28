// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/dump_sem_ir_range.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/dump_sem_ir_range.carbon

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

// CHECK:STDOUT: --- function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0
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
// CHECK:STDOUT:   %b.ref.loc11: ref %empty_tuple.type = name_ref b, %b
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign %b.ref.loc11, %A.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type = binding_pattern c
// CHECK:STDOUT:     %.loc18_3: %pattern_type = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var c
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
// CHECK:STDOUT:   return %.loc20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.call: init %empty_tuple.type = call %C.ref()
// CHECK:STDOUT:   %.loc13_7.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc13_7.2: ref %empty_tuple.type = temporary %.loc13_7.1, %A.call
// CHECK:STDOUT:   %tuple.loc13: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13_7.3: %empty_tuple.type = converted %A.call, %tuple.loc13 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_7.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc17_7.2: ref %empty_tuple.type = temporary %.loc17_7.1, %C.call
// CHECK:STDOUT:   %tuple.loc17: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_7.3: %empty_tuple.type = converted %C.call, %tuple.loc17 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc13_7.3, %.loc15_7.3, %.loc17_7.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

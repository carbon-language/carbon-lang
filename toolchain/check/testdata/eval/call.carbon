// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/call.carbon

// --- call.carbon

library "[[@TEST_NAME]]";

eval fn F() -> i32 { return 3; }

var a: array(i32, F()) = (0, 1, 2);

// --- fail_call_wrong_value.carbon

library "[[@TEST_NAME]]";

eval fn F() -> i32 { return 3; }

// Ensure we get an error about the over-sized initializer and aren't just
// treating all `eval fn` calls as silent errors.
// CHECK:STDERR: fail_call_wrong_value.carbon:[[@LINE+4]]:26: error: cannot initialize array of 3 elements from 4 initializers [ArrayInitFromLiteralArgCountMismatch]
// CHECK:STDERR: var a: array(i32, F()) = (0, 1, 2, 3);
// CHECK:STDERR:                          ^~~~~~~~~~~~
// CHECK:STDERR:
var a: array(i32, F()) = (0, 1, 2, 3);

// --- fail_call_fn_not_eval.carbon

library "[[@TEST_NAME]]";

fn F() -> i32 { return 3; }

// CHECK:STDERR: fail_call_fn_not_eval.carbon:[[@LINE+4]]:19: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var a: array(i32, F()) = (0, 1, 2);
// CHECK:STDERR:                   ^~~
// CHECK:STDERR:
var a: array(i32, F()) = (0, 1, 2);

// --- fail_call_eval_fn_not_defined.carbon

library "[[@TEST_NAME]]";

eval fn F() -> i32;

// TODO: We should be able to diagnose this better.
// CHECK:STDERR: fail_call_eval_fn_not_defined.carbon:[[@LINE+4]]:19: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var a: array(i32, F()) = (0, 1, 2);
// CHECK:STDERR:                   ^~~
// CHECK:STDERR:
var a: array(i32, F()) = (0, 1, 2);

// --- param_and_return.carbon

library "[[@TEST_NAME]]";

eval fn F(x: i32) -> i32 { return x; }

var a: array(i32, F(3)) = (1, 2, 3);

// --- fail_todo_dependent_call.carbon

library "[[@TEST_NAME]]";

eval fn F(x: i32) -> i32 { return x; }

fn G(N:! i32) {
  // CHECK:STDERR: fail_todo_dependent_call.carbon:[[@LINE+4]]:17: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var unused a: array(i32, F(N)) = (0, 1, 2);
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused a: array(i32, F(N)) = (0, 1, 2);
}

fn H() { G(3); }

// --- fail_todo_dependent_call_type.carbon

library "[[@TEST_NAME]]";

class C {}
eval fn F(_: i32) -> type {
  return C;
}

fn UseFGenerically(X:! i32) {
  // CHECK:STDERR: fail_todo_dependent_call_type.carbon:[[@LINE+12]]:3: error: member name of type `<dependent type>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:   var unused v: F(X) = {};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_dependent_call_type.carbon:[[@LINE+8]]:3: error: value of type `<dependent type>` is not callable [CallToNonCallable]
  // CHECK:STDERR:   var unused v: F(X) = {};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_dependent_call_type.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.Destroy` in type `<cannot stringify inst7C00018B: {kind: Call, arg0: inst7C00003E, arg1: inst_block7C00008A, type: type(TypeType)}>` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   var unused v: F(X) = {};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused v: F(X) = {};
}

fn UseFSpecifically() {
  UseFGenerically(3);
}

// --- return_in_place_discarded.carbon

library "[[@TEST_NAME]]";

eval fn F() -> (i32, i32, i32) { return (1, 2, 3); }

fn G() {
  //@dump-sem-ir-begin
  F();
  //@dump-sem-ir-end
}

// --- return_in_place_init.carbon

library "[[@TEST_NAME]]";

eval fn F() -> (i32, i32, i32) { return (1, 2, 3); }

fn G() {
  //@dump-sem-ir-begin
  var unused a: (i32, i32, i32) = F();
  //@dump-sem-ir-end
}

// --- fail_todo_return_in_place_tuple.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
eval fn F() -> (i32, i32, i32) { return (1, 2, 3); }
//@dump-sem-ir-end

eval fn G(v: (i32, i32, i32)) -> i32 { return v.1; }

fn H() {
  // TODO: The call to the function is a constant, but the `temporary` instruction wrapping it is not.
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_return_in_place_tuple.carbon:[[@LINE+4]]:28: error: array bound is not a constant [InvalidArrayExpr]
  // CHECK:STDERR:   var unused a: array(i32, G(F())) = (1, 2);
  // CHECK:STDERR:                            ^~~~~~
  // CHECK:STDERR:
  var unused a: array(i32, G(F())) = (1, 2);
  //@dump-sem-ir-end
}

// --- return_in_place_class.carbon

library "[[@TEST_NAME]]";

class C {}
//@dump-sem-ir-begin
eval fn F() -> C { return {}; }
//@dump-sem-ir-end

fn G(_:! C) {}

fn H() {
  //@dump-sem-ir-begin
  G(F());
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- return_in_place_discarded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.a64: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.4ff: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.49a: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.8d2: %tuple.type.a64 = tuple_value (%int_1.d5e, %int_2.4ff, %int_3.49a) [concrete]
// CHECK:STDOUT:   %.041: ref %tuple.type.a64 = temporary invalid, %tuple.8d2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc8_5.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.041, %Destroy.Op.1dc86d.3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc8_5.1: ref %tuple.type.a64 = temporary_storage
// CHECK:STDOUT:   %F.call: init %tuple.type.a64 to %.loc8_5.1 = call %F.ref() [concrete = constants.%tuple.8d2]
// CHECK:STDOUT:   %.loc8_5.2: ref %tuple.type.a64 = temporary %.loc8_5.1, %F.call [concrete = constants.%.041]
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.041)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_5.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_5.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_5.3(%self.param: ref %tuple.type.a64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_in_place_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.bed: %tuple.type.ff9 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.a64: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.3ff: type = pattern_type %tuple.type.a64 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.4ff: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.49a: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.8d2: %tuple.type.a64 = tuple_value (%int_1.d5e, %int_2.4ff, %int_3.49a) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.3ff = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.3ff = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %tuple.type.a64 = var %a.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc8_3: ref %tuple.type.a64 = splice_block %a.var {}
// CHECK:STDOUT:   %F.call: init %tuple.type.a64 to %.loc8_3 = call %F.ref() [concrete = constants.%tuple.8d2]
// CHECK:STDOUT:   assign %a.var, %F.call
// CHECK:STDOUT:   %.loc8_31.1: type = splice_block %.loc8_31.3 [concrete = constants.%tuple.type.a64] {
// CHECK:STDOUT:     %i32.loc8_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc8_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc8_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_31.2: %tuple.type.ff9 = tuple_literal (%i32.loc8_18, %i32.loc8_23, %i32.loc8_28) [concrete = constants.%tuple.bed]
// CHECK:STDOUT:     %.loc8_31.3: type = converted %.loc8_31.2, constants.%tuple.type.a64 [concrete = constants.%tuple.type.a64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %tuple.type.a64 = ref_binding a, %a.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.3(%self.param: ref %tuple.type.a64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_return_in_place_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.bed: %tuple.type.ff9 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.a64: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %.f83: Core.Form = init_form %tuple.type.a64 [concrete]
// CHECK:STDOUT:   %pattern_type.3ff: type = pattern_type %tuple.type.a64 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.33d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.cc3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.4ff: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.353: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.49a: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.8d2: %tuple.type.a64 = tuple_value (%int_1.d5e, %int_2.4ff, %int_3.49a) [concrete]
// CHECK:STDOUT:   %.041: ref %tuple.type.a64 = temporary invalid, %tuple.8d2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc17_32.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.041, %Destroy.Op.1dc86d.3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.3ff = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.3ff = return_slot_pattern %return.param_patt, %.loc5_30.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc5_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc5_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_30.1: %tuple.type.ff9 = tuple_literal (%i32.loc5_17, %i32.loc5_22, %i32.loc5_27) [concrete = constants.%tuple.bed]
// CHECK:STDOUT:     %.loc5_30.2: type = converted %.loc5_30.1, constants.%tuple.type.a64 [concrete = constants.%tuple.type.a64]
// CHECK:STDOUT:     %.loc5_30.3: Core.Form = init_form %.loc5_30.2 [concrete = constants.%.f83]
// CHECK:STDOUT:     %return.param: ref %tuple.type.a64 = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.a64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %tuple.type.a64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc5_49.1: %tuple.type.37f = tuple_literal (%int_1, %int_2, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %impl.elem0.loc5_49.1: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc5_49.1: <bound method> = bound_method %int_1, %impl.elem0.loc5_49.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c0d]
// CHECK:STDOUT:   %specific_fn.loc5_49.1: <specific function> = specific_function %impl.elem0.loc5_49.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_49.2: <bound method> = bound_method %int_1, %specific_fn.loc5_49.1 [concrete = constants.%bound_method.33d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.1: init %i32 = call %bound_method.loc5_49.2(%int_1) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc5_49.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc5_49.3: init %i32 to %tuple.elem0 = in_place_init %.loc5_49.2 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %impl.elem0.loc5_49.2: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc5_49.3: <bound method> = bound_method %int_2, %impl.elem0.loc5_49.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.22e]
// CHECK:STDOUT:   %specific_fn.loc5_49.2: <specific function> = specific_function %impl.elem0.loc5_49.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_49.4: <bound method> = bound_method %int_2, %specific_fn.loc5_49.2 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.2: init %i32 = call %bound_method.loc5_49.4(%int_2) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc5_49.4: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.2 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc5_49.5: init %i32 to %tuple.elem1 = in_place_init %.loc5_49.4 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %impl.elem0.loc5_49.3: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc5_49.5: <bound method> = bound_method %int_3, %impl.elem0.loc5_49.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a]
// CHECK:STDOUT:   %specific_fn.loc5_49.3: <specific function> = specific_function %impl.elem0.loc5_49.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_49.6: <bound method> = bound_method %int_3, %specific_fn.loc5_49.3 [concrete = constants.%bound_method.353]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.3: init %i32 = call %bound_method.loc5_49.6(%int_3) [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %.loc5_49.6: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.3 [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %return.param, element2
// CHECK:STDOUT:   %.loc5_49.7: init %i32 to %tuple.elem2 = in_place_init %.loc5_49.6 [concrete = constants.%int_3.49a]
// CHECK:STDOUT:   %.loc5_49.8: init %tuple.type.a64 to %return.param = tuple_init (%.loc5_49.3, %.loc5_49.5, %.loc5_49.7) [concrete = constants.%tuple.8d2]
// CHECK:STDOUT:   %.loc5_50: init %tuple.type.a64 = converted %.loc5_49.1, %.loc5_49.8 [concrete = constants.%tuple.8d2]
// CHECK:STDOUT:   return %.loc5_50 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc17_43: %tuple.type.f94 = tuple_literal (%int_1, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a: ref <error> = ref_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.041)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc17_32.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc17_32.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc17_32.3(%self.param: ref %tuple.type.a64) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_in_place_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.8c2: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.6f6: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.674: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%C.val) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc13_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.674, %Destroy.Op.1dc86d.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6f6 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6f6 = return_slot_pattern %return.param_patt, %C.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc6_16: Core.Form = init_form %C.ref [concrete = constants.%.8c2]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_28.2: init %C to %return.param = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_29: init %C = converted %.loc6_28.1, %.loc6_28.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc6_29 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc13_7.1: ref %C = temporary_storage
// CHECK:STDOUT:   %F.call: init %C to %.loc13_7.1 = call %F.ref() [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_7.2: ref %C = temporary %.loc13_7.1, %F.call [concrete = constants.%.674]
// CHECK:STDOUT:   %.loc13_7.3: %C = acquire_value %.loc13_7.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%C.val) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.674)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_7.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/terse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/terse.carbon

// --- basic.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn Id(x: i32) => x;

fn Unit() => ();

fn Pair(x: i32, y: i32) => {.a = x, .b = y};

fn TestBasic() {
  let a: i32 = Id(42);
  Unit();
  let p: {.a: i32, .b: i32} = Pair(a, 10);
  let unused b: i32 = p.a;
}
//@dump-sem-ir-end

eval fn EvalId(x: i32) => x;
musteval fn MustEvalId(x: i32) => x;

fn TestEval() {
  var a: array(i32, EvalId(3));
  var b: array(i32, MustEvalId(4));
  let _: (array(i32, 3), array(i32, 4)) = (a, b);
}

class WithMethods {
  fn Get(self: Self) => self.x;
  fn Static() => 7 as i32;
  var x: i32;
}

fn TestMethods(c: WithMethods) {
  let unused a: i32 = c.Get();
  let unused b: i32 = WithMethods.Static();
}

// --- generic_non_dependent.carbon
library "[[@TEST_NAME]]";

fn GenericNonDependent[T: type](unused x: T) => 42 as i32;

fn TestGenericNonDependent() {
  let unused a: i32 = GenericNonDependent(true);
}

// --- main_run_i32.carbon
library "[[@TEST_NAME]]";

fn Run() => 0 as i32;

// --- main_run_unit.carbon
library "[[@TEST_NAME]]";

fn Run() => ();

// --- fail_main_run_bool.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_main_run_bool.carbon:[[@LINE+2]]:1-11: error: invalid return type for `Main.Run` function; expected `fn (...)` or `fn (...) -> i32` [InvalidMainRunReturnType]
// CHECK:STDERR:
fn Run() => true;

// --- fail_return_type.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_return_type.carbon:[[@LINE+2]]:14-19: error: cannot specify a return type with `=>` [ReturnTypeInTerseFunction]
// CHECK:STDERR:
fn G(x: i32) -> i32 => x;

// CHECK:STDERR: fail_return_type.carbon:[[@LINE+2]]:18-34: error: cannot specify a return form with `=>` [ReturnTypeInTerseFunction]
// CHECK:STDERR:
fn H(ref x: i32) ->? form(ref i32) => x;

// --- fail_redecl.carbon
library "[[@TEST_NAME]]";

fn DeclaredI32() -> i32;
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+3]]:1-19: error: redeclaration of function with deduced return type [DeducedReturnTypeRedecl]
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-2]]:1-24: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR:
fn DeclaredI32() => 0 as i32;

fn DeclaredUnit();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+3]]:1-20: error: redeclaration of function with deduced return type [DeducedReturnTypeRedecl]
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-2]]:1-18: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR:
fn DeclaredUnit() => ();

fn TerseFirst() => 0 as i32;
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+3]]:1-18: error: redeclaration of function with deduced return type [DeducedReturnTypeRedecl]
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-2]]:1-18: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR:
fn TerseFirst() => 1 as i32;

// --- fail_incomplete_return.carbon
library "[[@TEST_NAME]]";

class Incomplete;

// CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE+3]]:39-40: error: function returns incomplete type `Incomplete` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE-3]]:1-17: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:
fn DerefIncomplete(p: Incomplete*) => *p;

// --- fail_unused.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_unused.carbon:[[@LINE+5]]:15-20: warning: binding `x` unused [UnusedBinding]
// CHECK:STDERR:
// CHECK:STDERR: fail_unused.carbon:[[@LINE+3]]:30-35: error: variable `y` marked `unused` but used [UnusedButUsed]
// CHECK:STDERR: fail_unused.carbon:[[@LINE+2]]:41: note: usage here [UnusedButUsedHere]
// CHECK:STDERR:
fn UnusedTest(x: i32, unused y: i32) => y;

// --- dependent_return.carbon
library "[[@TEST_NAME]]";

fn GenericId[T: Core.Copy](x: T) => x;

// --- fail_call_before_deduced.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_call_before_deduced.carbon:[[@LINE+3]]:23-32: error: call to `Recurse` before its return type is known [CallBeforeReturnTypeDeduced]
// CHECK:STDERR: fail_call_before_deduced.carbon:[[@LINE+2]]:1-21: note: return type of `Recurse` is deduced from its definition [ReturnTypeDeducedFromDefinition]
// CHECK:STDERR:
fn Recurse(x: i32) => Recurse(x);

class EarlyCall {
  fn M() => 1 as i32;
  // CHECK:STDERR: fail_call_before_deduced.carbon:[[@LINE+3]]:16-18: error: call to `M` before its return type is known [CallBeforeReturnTypeDeduced]
  // CHECK:STDERR: fail_call_before_deduced.carbon:[[@LINE-2]]:3-11: note: return type of `M` is deduced from its definition [ReturnTypeDeducedFromDefinition]
  // CHECK:STDERR:
  var x: i32 = M();
}

// --- fail_todo_thunk_before_deduced.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F() -> i32;
}

class C {
  impl as I {
    // CHECK:STDERR: fail_todo_thunk_before_deduced.carbon:[[@LINE+2]]:5-13: error: semantics TODO: `deduced return type for `impl` member` [SemanticsTodo]
    // CHECK:STDERR:
    fn F() => 0 as i32;
  }
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %Id.type: type = fn_type @Id [concrete]
// CHECK:STDOUT:   %Id: %Id.type = struct_value () [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Unit.type: type = fn_type @Unit [concrete]
// CHECK:STDOUT:   %Unit: %Unit.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt.900: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.427: %pattern_type.cb1 = return_slot_pattern %return.param_patt.900, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %y.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.d94 = wrapper_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:   %Pair.type: type = fn_type @Pair [concrete]
// CHECK:STDOUT:   %Pair: %Pair.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %.093: Core.Form = init_form %struct_type.a.b [concrete]
// CHECK:STDOUT:   %pattern_type.290: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %return.param_patt.cfa: %pattern_type.290 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.380: %pattern_type.290 = return_slot_pattern %return.param_patt.cfa, %struct_type.a.b [concrete]
// CHECK:STDOUT:   %TestBasic.type: type = fn_type @TestBasic [concrete]
// CHECK:STDOUT:   %TestBasic: %TestBasic.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt.413: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.81c: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.98e: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.db5: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.290 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.02d: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.002: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.6a3: %i32 = int_value 10 [concrete]
// CHECK:STDOUT:   %b.patt.094d65.1: %pattern_type.d94 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_41.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_41.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c2b: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.582: %SubobjectDestroy.type = facet_value %struct_type.a.b, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6b7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.582) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6bf: %Destroy.type = facet_value %struct_type.a.b, (%DestroyImpl.impl_witness.6b7, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Id.decl: %Id.type = fn_decl @Id [concrete = constants.%Id] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc4_18 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc4_10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %i32.loc4_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_18.2: Core.Form = init_form %i32.loc4_18 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Unit.decl: %Unit.type = fn_decl @Unit [concrete = constants.%Unit] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt.900]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc6_15.5 [concrete = constants.%return.patt.427]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_15.5: type = type_literal constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc6_15.6: Core.Form = init_form %.loc6_15.5 [concrete = constants.%.262]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Pair.decl: %Pair.type = fn_decl @Pair [concrete = constants.%Pair] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%y.param_patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.d94 = wrapper_binding_pattern y, %y.param_patt [concrete = constants.%y.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.290 = out_param_pattern [concrete = constants.%return.param_patt.cfa]
// CHECK:STDOUT:     %return.patt: %pattern_type.290 = return_slot_pattern %return.param_patt, %.loc8_43.9 [concrete = constants.%return.patt.380]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc8_12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %y.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc8_20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %y: %i32 = wrapper_binding y, %y.param
// CHECK:STDOUT:     %.loc8_43.9: type = type_literal constants.%struct_type.a.b [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:     %.loc8_43.10: Core.Form = init_form %.loc8_43.9 [concrete = constants.%.093]
// CHECK:STDOUT:     %return.param: ref %struct_type.a.b = out_param call_param2
// CHECK:STDOUT:     %return: ref %struct_type.a.b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestBasic.decl: %TestBasic.type = fn_decl @TestBasic [concrete = constants.%TestBasic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Id(%x.param: %i32) -> out %return.param: %i32 [deduced] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc4_18.1: init %i32 = splice_block %Int.as.Copy.impl.Op.call {
// CHECK:STDOUT:     %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:     %bound_method.loc4_18.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc4_18.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc4_18.2(%x.ref)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return %.loc4_18.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Unit() -> out %return.param: %empty_tuple.type [deduced] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_15.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_15.2: init %empty_tuple.type = splice_block %.loc6_15.4 [concrete = constants.%empty_tuple] {
// CHECK:STDOUT:     %.loc6_15.3: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_15.4: init %empty_tuple.type = converted %.loc6_15.1, %.loc6_15.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return %.loc6_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Pair(%x.param: %i32, %y.param: %i32) -> out %return.param: %struct_type.a.b [deduced] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %y.ref: %i32 = name_ref y, %y
// CHECK:STDOUT:   %.loc8_43.1: %struct_type.a.b = struct_literal (%x.ref, %y.ref)
// CHECK:STDOUT:   %.loc8_43.2: init %struct_type.a.b = splice_block %.loc8_43.8 {
// CHECK:STDOUT:     %impl.elem0.loc8_34: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:     %bound_method.loc8_34.1: <bound method> = bound_method %x.ref, %impl.elem0.loc8_34
// CHECK:STDOUT:     %specific_fn.loc8_34: <specific function> = specific_function %impl.elem0.loc8_34, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_34.2: <bound method> = bound_method %x.ref, %specific_fn.loc8_34
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call.loc8_34: init %i32 = call %bound_method.loc8_34.2(%x.ref)
// CHECK:STDOUT:     %.loc8_43.3: ref %i32 = struct_access %return.param, element0
// CHECK:STDOUT:     %.loc8_43.4: init %i32 to %.loc8_43.3 = in_place_init %Int.as.Copy.impl.Op.call.loc8_34
// CHECK:STDOUT:     %impl.elem0.loc8_42: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:     %bound_method.loc8_42.1: <bound method> = bound_method %y.ref, %impl.elem0.loc8_42
// CHECK:STDOUT:     %specific_fn.loc8_42: <specific function> = specific_function %impl.elem0.loc8_42, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_42.2: <bound method> = bound_method %y.ref, %specific_fn.loc8_42
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call.loc8_42: init %i32 = call %bound_method.loc8_42.2(%y.ref)
// CHECK:STDOUT:     %.loc8_43.5: ref %i32 = struct_access %return.param, element1
// CHECK:STDOUT:     %.loc8_43.6: init %i32 to %.loc8_43.5 = in_place_init %Int.as.Copy.impl.Op.call.loc8_42
// CHECK:STDOUT:     %.loc8_43.7: init %struct_type.a.b to %return.param = struct_init (%.loc8_43.4, %.loc8_43.6)
// CHECK:STDOUT:     %.loc8_43.8: init %struct_type.a.b = converted %.loc8_43.1, %.loc8_43.7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return %.loc8_43.2 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBasic() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Id.ref: %Id.type = name_ref Id, file.%Id.decl [concrete = constants.%Id]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0.loc11: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc11_19.1: <bound method> = bound_method %int_42, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.81c]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_19.2: <bound method> = bound_method %int_42, %specific_fn.loc11 [concrete = constants.%bound_method.98e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_19.2(%int_42) [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int_42, %.loc11_19.1 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %Id.call: init %i32 = call %Id.ref(%.loc11_19.2)
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %Id.call
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %Id.call, %.loc11_21.1
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc11_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt.413]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Unit.ref: %Unit.type = name_ref Unit, file.%Unit.decl [concrete = constants.%Unit]
// CHECK:STDOUT:   %Unit.call: init %empty_tuple.type = call %Unit.ref()
// CHECK:STDOUT:   %Pair.ref: %Pair.type = name_ref Pair, file.%Pair.decl [concrete = constants.%Pair]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %.loc13_41.1: ref %struct_type.a.b = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc13: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc13_39.1: <bound method> = bound_method %int_10, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.02d]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_39.2: <bound method> = bound_method %int_10, %specific_fn.loc13 [concrete = constants.%bound_method.002]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_39.2(%int_10) [concrete = constants.%int_10.6a3]
// CHECK:STDOUT:   %.loc13_39.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_10.6a3]
// CHECK:STDOUT:   %.loc13_39.2: %i32 = converted %int_10, %.loc13_39.1 [concrete = constants.%int_10.6a3]
// CHECK:STDOUT:   %Pair.call: init %struct_type.a.b to %.loc13_41.1 = call %Pair.ref(%a.ref, %.loc13_39.2)
// CHECK:STDOUT:   %.loc13_41.2: ref %struct_type.a.b = temporary %.loc13_41.1, %Pair.call
// CHECK:STDOUT:   %.loc13_41.3: ref %i32 = struct_access %.loc13_41.2, element0
// CHECK:STDOUT:   %.loc13_41.4: %i32 = acquire_value %.loc13_41.3
// CHECK:STDOUT:   %.loc13_41.5: ref %i32 = struct_access %.loc13_41.2, element1
// CHECK:STDOUT:   %.loc13_41.6: %i32 = acquire_value %.loc13_41.5
// CHECK:STDOUT:   %struct: %struct_type.a.b = struct_value (%.loc13_41.4, %.loc13_41.6)
// CHECK:STDOUT:   %.loc13_41.7: %struct_type.a.b = converted %Pair.call, %struct
// CHECK:STDOUT:   %.loc13_27: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b] {
// CHECK:STDOUT:     %i32.loc13_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc13_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %struct_type.a.b = wrapper_binding p, %.loc13_41.7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.290 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.ref: %struct_type.a.b = name_ref p, %p
// CHECK:STDOUT:   %.loc14: %i32 = struct_access %p.ref, element0
// CHECK:STDOUT:   %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = wrapper_binding b, %.loc14
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete = constants.%b.patt.094d65.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc13_41.1: %Destroy.type = facet_value constants.%struct_type.a.b, (constants.%DestroyImpl.impl_witness.6b7, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.6bf]
// CHECK:STDOUT:   %.loc13_41.8: %Destroy.type = converted constants.%struct_type.a.b, %Destroy.facet.loc13_41.1 [concrete = constants.%Destroy.facet.6bf]
// CHECK:STDOUT:   %Destroy.facet.loc13_41.2: %Destroy.type = facet_value constants.%struct_type.a.b, (constants.%DestroyImpl.impl_witness.6b7, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.6bf]
// CHECK:STDOUT:   %.loc13_41.9: %Destroy.type = converted constants.%struct_type.a.b, %Destroy.facet.loc13_41.2 [concrete = constants.%Destroy.facet.6bf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc13_41.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_41.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_41.2(%self.param: ref %struct_type.a.b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_41.1: ref %i32 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc13_41.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc13_41.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc13_41.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc13_41.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc13_41.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc13_41.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc13_41.1)
// CHECK:STDOUT:   %.loc13_41.4: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc13_41.3: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc13_41.5: %Destroy.type = converted constants.%i32, %Destroy.facet.loc13_41.3 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc13_41.4: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.1) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc13_41.6: %Destroy.type = converted constants.%i32, %Destroy.facet.loc13_41.4 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc13_41.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
